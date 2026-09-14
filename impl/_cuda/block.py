"""TransformerBlock — CUDA assembly of all CUDA primitives.

Assembles:
  - RMSNorm     → impl/_cuda/layernorm.rmsnorm    (warp-reduction kernel)
  - MHA         → impl/_cuda/attention.scaled_dot_product_attention (CUDA softmax/weighted-sum)
  - SwiGLU FFN  → impl/_cuda/ffn.swiglu_ffn       (CUDA SiLU kernel inside SwiGLU)
  - RoPE        → impl/_cuda/rope.apply_rope       (CUDA rotary embedding kernel)
  - MoE         → torch routing (same math as the other tracks) + per-expert
                 CUDA SwiGLU FFN

Architecture (LLaMA-style pre-norm, identical to the NumPy/PyTorch tracks):
    Input:  x [B, S, D]
    │
    ├─ Stream 1: Attention ──────────────────────────────────────────────
    │   1. xn = RMSNorm(x, input_ln)                    # (B, S, D)
    │   2. q = xn @ Wq; k = xn @ Wk; v = xn @ Wv        # (B, ·, S, hd)
    │   3. q, k ← RoPE(q), RoPE(k)                      # CUDA rope kernel
    │   4. if G < H: repeat k, v to H heads (GQA)
    │   5. attn = SDPA(q, k, v) @ Wo                    # (B, S, D)
    │   6. h = x + attn                                  # residual
    │
    ├─ Stream 2: Feed-forward (dense SwiGLU or MoE) ─────────────────────
    │   7. hn = RMSNorm(h, post_ln)                     # (B, S, D)
    │   8. ff = SwiGLU(hn) | MoE(hn)                    # (B, S, D)
    │   9. out = h + ff                                  # residual
    │
    Output: out [B, S, D]

Reference
---------
Vaswani et al. "Attention Is All You Need" (2017)
https://arxiv.org/abs/1706.03762

Shazeer, "GLU Variants Improve Transformer" (2020)
https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import logging
import math

import torch

from impl._cuda.attention import scaled_dot_product_attention as cuda_sdp_attention
from impl._cuda.ffn import swiglu_ffn
from impl._cuda.layernorm import rmsnorm
from impl._cuda.rope import apply_rope
from shared.config import TransformerConfig

logger = logging.getLogger(__name__)

# ── Weight initialization helpers ──────────────────────────────────────────────


def _init_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Xavier/uniform initialization for attention/FFN weights.

    Returns a (rows, cols) float32 tensor with values drawn from
    U(-limit, limit) where limit = sqrt(6 / (rows + cols)).
    """
    bound = math.sqrt(6.0 / (rows + cols))
    tensor = torch.empty(rows, cols, dtype=torch.float32)
    torch.nn.init.uniform_(tensor, -bound, bound, generator=torch.Generator().manual_seed(seed))
    return tensor


def _init_zeros(shape: tuple[int, ...]) -> torch.Tensor:
    """Initialize a tensor to zeros.

    Returns
    -------
    torch.Tensor
        Zeros tensor of the given shape, float32.
    """
    return torch.zeros(shape)


# ── CUDA block assembly ────────────────────────────────────────────────────────


class CuTransformerBlock:
    """CUDA TransformerBlock — assembly of all CUDA primitives.

    Mirrors ``impl._np.block.TransformerBlock``: same pre-norm layout,
    same GQA, same RoPE, same MoE routing math — with the heavy compute
    delegated to CUDA kernels.

    It does NOT inherit from nn.Module — weights are stored as plain tensors
    for parity checking against the NumPy implementation (easy to extract
    raw arrays for comparison).

    Attributes
    ----------
    input_layernorm_gamma : torch.Tensor, shape (D,)
        RMSNorm gain before attention (LLaMA naming).
    post_attention_layernorm_gamma : torch.Tensor, shape (D,)
        RMSNorm gain before the feed-forward.
    q_proj / k_proj / v_proj / o_proj : torch.Tensor
        Attention projections (in, out) layout: (D, H·hd), (D, G·hd),
        (D, G·hd), (H·hd, D).
    Dense FFN: gate_proj (D, FF), up_proj (D, FF), down_proj (FF, D).
    MoE: router (D, E); experts stacked as (E, D, FF), (E, D, FF), (E, FF, D).
    """

    # ------------------------------------------------------------------ init
    def __init__(self, config: TransformerConfig, seed: int = 0) -> None:
        self.config = config
        D, H, G = config.embed_dim, config.n_heads, config.kv_heads
        hd, FF, E = config.head_dim, config.expert_dim, config.n_experts
        self.n_heads = H
        self.n_groups = G
        self.n_experts = E
        self.head_dim = hd
        self.rope_dim = config.rope_dim

        # RMSNorm gains (D,) — identity at init
        self.input_layernorm_gamma = torch.ones(D, dtype=torch.float32)
        self.input_layernorm_gamma.requires_grad_(True)
        self.post_attention_layernorm_gamma = torch.ones(D, dtype=torch.float32)
        self.post_attention_layernorm_gamma.requires_grad_(True)

        # ── Attention projections (no bias, Llama convention) ──────────
        self.q_proj = _init_weight(D, H * hd, seed=seed + 2)
        self.k_proj = _init_weight(D, G * hd, seed=seed + 3)
        self.v_proj = _init_weight(D, G * hd, seed=seed + 4)
        self.o_proj = _init_weight(H * hd, D, seed=seed + 5)
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            p.requires_grad_(True)

        if config.has_moe():
            # ── MoE: router + SwiGLU experts ───────────────────────────
            # Router: (D, E), no bias (Mixtral convention)
            self.router = _init_weight(D, E, seed=seed + 8)
            self.router.requires_grad_(True)
            # Experts: stacked SwiGLU projections
            bound = math.sqrt(6.0 / (D + FF))
            gen = torch.Generator().manual_seed(seed + 7)
            self.expert_gate_proj = torch.empty(E, D, FF, dtype=torch.float32)
            torch.nn.init.uniform_(self.expert_gate_proj, -bound, bound, generator=gen)
            self.expert_up_proj = torch.empty(E, D, FF, dtype=torch.float32)
            torch.nn.init.uniform_(self.expert_up_proj, -bound, bound, generator=gen)
            self.expert_down_proj = torch.empty(E, FF, D, dtype=torch.float32)
            torch.nn.init.uniform_(self.expert_down_proj, -bound, bound, generator=gen)
            for p in (self.expert_gate_proj, self.expert_up_proj, self.expert_down_proj):
                p.requires_grad_(True)
        else:
            # ── Dense SwiGLU feed-forward ──────────────────────────────
            self.gate_proj = _init_weight(D, FF, seed=seed + 9)
            self.up_proj = _init_weight(D, FF, seed=seed + 10)
            self.down_proj = _init_weight(FF, D, seed=seed + 11)
            for p in (self.gate_proj, self.up_proj, self.down_proj):
                p.requires_grad_(True)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the TransformerBlock.

        Pre-norm layout (identical math to the NumPy/PyTorch tracks):

            xn = RMSNorm(x);  h   = x + SDPA(RoPE(q), RoPE(k), v) @ Wo
            hn = RMSNorm(h);  out = h + FFN(hn)          # FFN = SwiGLU | MoE

        Shape flow:
          x: (B, S, D)
            → q: (B, S, H·hd) → (B, H, S, hd); k, v: (B, G, S, hd)
            → RoPE: (B, ·, S, hd); GQA repeat → (B, H, S, hd)
            → SDPA: (B, H, S, hd) → (B, S, H·hd) → @Wo: (B, S, D)
            → residual, pre-norm, FFN, residual: (B, S, D)
        """
        B, S, D = x.shape
        device = x.device
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        # Move weights to the input's device (caller may have built them on CPU)
        Wq, Wk, Wv, Wo = (
            self.q_proj.to(device),
            self.k_proj.to(device),
            self.v_proj.to(device),
            self.o_proj.to(device),
        )
        ln1 = self.input_layernorm_gamma.to(device)
        ln2 = self.post_attention_layernorm_gamma.to(device)

        # ── Stream 1: attention (pre-norm) ─────────────────────────────
        xn = rmsnorm(x, ln1, eps=self.config.norm_eps)  # (B, S, D)

        # Q, K, V projections: (B, S, D) @ (D, ·) → (B, S, ·)
        q = xn @ Wq  # (B, S, H·hd)
        k = xn @ Wk  # (B, S, G·hd)
        v = xn @ Wv  # (B, S, G·hd)

        # Split into heads: (B, S, ·) → (B, S, H, hd)
        q = q.view(B, S, H, hd)  # (B, S, H, hd)
        k = k.view(B, S, G, hd)  # (B, S, G, hd)
        v = v.view(B, S, G, hd)
        v = v.transpose(1, 2).contiguous()  # (B, G, S, hd) — attention-kernel layout

        # RoPE on q and k (rope_dim=0 rotates all hd dims — the standard case).
        # The RoPE kernel expects (B, S, H, D) layout; transpose to the
        # attention kernel's (B, H, S, hd) layout afterwards.
        if positions is None:
            positions = torch.arange(S, device=x.device, dtype=torch.long)
        q = apply_rope(q, positions, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, H, S, hd)
        k = apply_rope(k, positions, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, G, S, hd)
        # GQA: broadcast each K/V group to its H // G query heads
        if G != H:
            k = k.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
            v = v.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)

        # Scaled dot-product attention — CUDA softmax + weighted-sum kernels
        attn = cuda_sdp_attention(q, k, v, is_causal=True)  # (B, H, S, hd)

        # Output projection: (B, H, S, hd) → (B, S, H·hd) → (B, S, D)
        attn_out = attn.transpose(1, 2).contiguous().view(B, S, H * hd) @ Wo  # (B, S, D)

        # Residual
        h = x + attn_out  # (B, S, D)

        # ── Stream 2: feed-forward (pre-norm) ──────────────────────────
        hn = rmsnorm(h, ln2, eps=self.config.norm_eps)  # (B, S, D)

        if self.config.has_moe():
            ff_out = self._moe_forward(hn, device)  # (B, S, D)
        else:
            gp = self.gate_proj.to(device)
            up = self.up_proj.to(device)
            dn = self.down_proj.to(device)
            ff_out = swiglu_ffn(hn, gp, up, dn)  # (B, S, D)

        out = h + ff_out  # (B, S, D)
        return out

    def _moe_forward(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """MoE forward: torch routing (identical to the other tracks) + CUDA SwiGLU experts.

        scores = x @ router            (B, S, E)
        probs  = softmax(scores)       (B, S, E)
        top-k mask + renormalize       (B, S, E)
        out    = Σ_j probs_j · SwiGLU_j(x)   (B, S, D)

        Every expert is computed and multiplied by its (possibly zero)
        weight — the math is identical to the gathered-token variant.
        """
        E = self.n_experts
        router = self.router.to(device)
        eg = self.expert_gate_proj.to(device)
        eu = self.expert_up_proj.to(device)
        ed = self.expert_down_proj.to(device)

        # Router scores and softmax over all experts: (B, S, E)
        scores = x @ router  # (B, S, E)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores)
        probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)  # (B, S, E)

        # Top-k mask: keep the k largest probs per token, zero the rest.
        if self.config.top_k < E:
            kth_vals, _ = torch.topk(probs, self.config.top_k, dim=-1)  # (B, S, k)
            threshold = kth_vals[..., -1:].expand_as(probs)  # (B, S, E)
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
            probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-8)  # renormalize

        # Weighted sum of expert outputs (CUDA SwiGLU per expert).
        out = torch.zeros_like(x)  # (B, S, D)
        for expert_idx in range(E):
            w = probs[..., expert_idx : expert_idx + 1]  # (B, S, 1)
            expert_out = swiglu_ffn(x, eg[expert_idx], eu[expert_idx], ed[expert_idx])  # (B, S, D)
            out = out + w * expert_out
        return out
