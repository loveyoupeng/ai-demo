"""Triton track: Python wiring around the Triton attention/FFN kernels.

Mirrors the NumPy/PyTorch tracks operator for operator (same block layout,
same key scheme, same math) so the cross-backend parity tests can load
weights across tracks. The attention core runs on the Triton
``scaled_dot_product_attention`` kernel and the feed-forward on the Triton
``swiglu_ffn`` kernel; everything else is plain PyTorch wiring.

Shared-seam note: ``RoPE`` is imported from ``impl._torch.layers`` — the
triton track reuses the torch building blocks it does not kernelize
(a deliberate seam, documented; see docs).
"""

import logging
import math

import torch
import torch.nn as nn

from impl._torch.layers import RoPE  # shared-seam: torch building block reused by triton
from impl._triton.attn import scaled_dot_product_attention
from impl._triton.ffn import swiglu_ffn
from shared.config import TransformerConfig

logger = logging.getLogger(__name__)


class TritonMultiHeadAttention(nn.Module):
    """Multi-head attention with GQA and RoPE, on the Triton SDPA kernel.

    Same math as ``impl._torch.layers.MultiHeadAttention``:

        q = x @ Wq   (B, S, H*hd)  → (B, H, S, hd)
        k = x @ Wk   (B, S, G*hd)  → (B, G, S, hd)
        v = x @ Wv   (B, S, G*hd)  → (B, G, S, hd)
        q, k ← RoPE(q), RoPE(k)
        if G < H: repeat k, v so head h uses group h % G  → (B, H, S, hd)
        ctx = triton_sdpa(q, k, v)      (B, H, S, hd)
        out = ctx @ Wo                  (B, S, D)

    The Triton kernel computes scaled dot-product attention (Q @ K^T /
    sqrt(hd) → softmax → @ V) without materializing the (S, S) score matrix
    in the same memory layout as a naive implementation (tiled).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        D, H, G = config.embed_dim, config.n_heads, config.kv_heads
        hd = config.head_dim
        self.n_heads = H
        self.n_groups = G
        self.head_dim = hd
        self.rope_dim = config.rope_dim

        # No-bias projections (Llama convention); nn.Linear stores (out, in)
        self.q_proj = nn.Linear(D, H * hd, bias=False)
        self.k_proj = nn.Linear(D, G * hd, bias=False)
        self.v_proj = nn.Linear(D, G * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, D, bias=False)
        self.rope = RoPE()

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move projection weights to x's device/dtype (triton kernels need it)."""
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is None:
            for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                module.to(device)
        else:
            for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                module.to(device, dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """MHA forward. x: (B, S, D) → out: (B, S, D)."""
        self._move_to_device(x)
        B, S, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        # (B, S, D) → (B, S, {H,G}*hd)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (B, S, {H,G}*hd) → (B, {H,G}, S, hd)
        q = q.view(B, S, H, hd).permute(0, 2, 1, 3)  # (B, H, S, hd)
        k = k.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)
        v = v.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)

        # RoPE on q and k (RoPE's contract shape is (B, S, H, D)).
        if positions is None:
            positions = torch.arange(S, device=x.device, dtype=torch.long)
        q = self.rope(q.permute(0, 2, 1, 3), positions, rope_dim=self.rope_dim).permute(0, 2, 1, 3)
        k = self.rope(k.permute(0, 2, 1, 3), positions, rope_dim=self.rope_dim).permute(0, 2, 1, 3)

        # GQA: broadcast each K/V group to its H // G query heads.
        if G != H:
            k = k.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
            v = v.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)

        # Triton SDPA kernel: (B, H, S, hd) → (B, H, S, hd)
        ctx = scaled_dot_product_attention(q, k, v, is_causal=True)

        # Merge heads: (B, H, S, hd) → (B, S, H*hd) → (B, S, D)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, S, H * hd)
        return self.o_proj(ctx)  # (B, S, D)


class TritonSwiGLUFFN(nn.Module):
    """Dense SwiGLU feed-forward, computed by the triton ``swiglu_ffn`` kernel.

    gate = SiLU(x @ gate_proj); up = x @ up_proj; out = (gate * up) @ down_proj.
    Weights stored in (in, out) layout (matching the NumPy reference):
    gate_proj, up_proj: (D, FF); down_proj: (FF, D).
    """

    __slots__ = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, embed_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.up_proj = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.down_proj = nn.Parameter(torch.empty(ff_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all weights with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))

    def _move_to_device(self, x: torch.Tensor) -> None:
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is None:
            for p in [self.gate_proj, self.up_proj, self.down_proj]:
                p.data = p.data.to(device)
        else:
            for p in [self.gate_proj, self.up_proj, self.down_proj]:
                p.data = p.data.to(device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward via the triton kernel. x: (..., D) → out: (..., D)."""
        self._move_to_device(x)
        # kernel arg order: (x, gate_w, up_w, down_w)
        return swiglu_ffn(x, self.gate_proj, self.up_proj, self.down_proj)


class TritonExpert(nn.Module):
    """One SwiGLU expert of the MoE (same weights as ``TritonSwiGLUFFN``)."""

    __slots__ = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, embed_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.up_proj = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.down_proj = nn.Parameter(torch.empty(ff_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all weights with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expert forward via the triton kernel. x: (..., D) → out: (..., D)."""
        return swiglu_ffn(x, self.gate_proj, self.up_proj, self.down_proj)


class TritonMixtureOfExperts(nn.Module):
    """Mixture of Experts on triton FFN kernels.

    Same routing math as the NumPy/PyTorch tracks: router softmax over all
    experts, top-k mask, renormalize, weighted sum of expert outputs.
    Router: gate = nn.Linear(D, E, bias=False) (Mixtral convention).
    """

    def __init__(self, embed_dim: int, n_experts: int, ff_dim: int, top_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(embed_dim, n_experts, bias=False)
        # Typed view for pyright; nn.ModuleList registers parameters for torch
        self.expert_list: list[TritonExpert] = [TritonExpert(embed_dim, ff_dim) for _ in range(n_experts)]
        self.experts = nn.ModuleList(self.expert_list)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        for expert in self.expert_list:
            expert.reset_parameters()

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move router + expert weights to x's device/dtype."""
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is None:
            self.gate.to(device)
            for expert in self.expert_list:
                for p in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                    p.data = p.data.to(device)
        else:
            self.gate.to(device, dtype)
            for expert in self.expert_list:
                for p in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                    p.data = p.data.to(device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE forward. x: (B, S, D) → out: (B, S, D)."""
        self._move_to_device(x)
        E = self.n_experts

        # Router scores and softmax over all experts: (B, S, E)
        scores = self.gate(x)  # (B, S, E)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores)
        probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)  # (B, S, E)

        # Top-k mask: keep the k largest probs per token, zero the rest.
        if self.top_k < E:
            kth_vals, _ = torch.topk(probs, self.top_k, dim=-1)  # (B, S, k)
            threshold = kth_vals[..., -1:].expand_as(probs)  # (B, S, E)
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
            probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-8)  # renormalize

        # Weighted sum of expert outputs (all experts computed; zeros masked).
        out = torch.zeros_like(x)  # (B, S, D)
        for expert_idx, expert in enumerate(self.experts):
            w = probs[..., expert_idx : expert_idx + 1]  # (B, S, 1)
            out = out + w * expert(x)  # (B, S, D)
        return out


class TritonTransformerBlock(nn.Module):
    """One decoder-only transformer block (LLaMA-style, pre-norm).

        h   = x + attention(rms_norm(x))       # (B, S, D)
        out = h + feed_forward(rms_norm(h))    # (B, S, D)

    ``feed_forward`` is the dense triton SwiGLU by default, or the MoE when
    ``config.has_moe()``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        D = config.embed_dim

        # nn.RMSNorm exposes its gain as ``weight`` (D,), init ones
        self.input_layernorm = nn.RMSNorm(D, eps=config.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(D, eps=config.norm_eps)
        self.self_attn = TritonMultiHeadAttention(config)
        if config.has_moe():
            self.mlp: TritonSwiGLUFFN | TritonMixtureOfExperts = TritonMixtureOfExperts(
                embed_dim=D, n_experts=config.n_experts, ff_dim=config.expert_dim, top_k=config.top_k
            )
        else:
            self.mlp = TritonSwiGLUFFN(embed_dim=D, ff_dim=config.expert_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Block forward. x: (B, S, D) → out: (B, S, D)."""
        self._move_to_device(x)
        # Stream 1: attention with pre-norm and residual.
        attn_out = self.self_attn(self.input_layernorm(x), positions)  # (B, S, D)
        h = x + attn_out  # (B, S, D)

        # Stream 2: feed-forward (dense or MoE) with pre-norm and residual.
        ff_out = self.mlp(self.post_attention_layernorm(h))  # (B, S, D)
        return h + ff_out  # (B, S, D)

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move all block parameters to x's device/dtype."""
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is None:
            self.input_layernorm.to(device)
            self.post_attention_layernorm.to(device)
        else:
            self.input_layernorm.to(device, dtype)
            self.post_attention_layernorm.to(device, dtype)
        self.self_attn._move_to_device(x)
        self.mlp._move_to_device(x)


class TritonDecoderStack(nn.Module):
    """Stack of TritonTransformerBlocks (the "body" of the decoder).

    out = block_{n-1}( ... block_1(block_0(x)) ... ), x: (B, S, D) → (B, S, D).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        # Typed view for pyright; nn.ModuleList registers parameters for torch
        self.blocks: list[TritonTransformerBlock] = [TritonTransformerBlock(config) for _ in range(config.n_layers)]
        self.layers = nn.ModuleList(self.blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.self_attn.q_proj.reset_parameters()
            block.self_attn.k_proj.reset_parameters()
            block.self_attn.v_proj.reset_parameters()
            block.self_attn.o_proj.reset_parameters()
            block.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Forward through all blocks. x: (B, S, D) → out: (B, S, D)."""
        self._move_to_device(x)
        out = x
        for block in self.blocks:
            out = block(out, positions)
        return out

    def _move_to_device(self, x: torch.Tensor) -> None:
        for block in self.blocks:
            block._move_to_device(x)
