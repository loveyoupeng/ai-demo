"""PyTorch implementation of the decoder-only transformer.

This track mirrors the NumPy reference (``impl._np``) operator for operator:
the same block layout, the same key scheme (``shared.constants.Keys``), and
the same math — the cross-backend parity tests load one track's parameters
into the other and compare logits.

Block layout (LLaMA-style pre-norm):

    x → RMSNorm → MultiHeadAttention → x + attn
    → RMSNorm → SwiGLUFFN | MixtureOfExperts → h + ffn

Model: embed → DecoderStack → final RMSNorm → lm_head (D → V).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.config import TransformerConfig
from shared.constants import Attn, Keys, LayerNorm, Mlp
from shared.registry import ParameterRegistry

logger = logging.getLogger(__name__)


class Embedding(nn.Module):
    """Token embedding: maps token IDs to dense vectors.

    ``input_ids [B, S]`` → row lookup in the (V, D) weight table → (B, S, D).
    """

    __slots__ = ("weight",)

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        # (V, D) — one learned vector per token
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings: (B, S) → (B, S, D)."""
        return F.embedding(input_ids, self.weight)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    out = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * gamma

    No mean-centering (unlike LayerNorm); each row is scaled to unit
    root-mean-square, then scaled per-dimension by the learned gain ``gamma``
    (D,), initialized to ones.

    Input: (..., D)  →  Output: (..., D)
    """

    __slots__ = ("gamma", "eps")

    def __init__(self, embed_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # (D,) — learned per-dimension gain, identity at init
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm. x: (..., D) → out: (..., D)."""
        # x**2: (..., D); mean(x**2, dim=-1): (..., 1)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)) + self.eps  # (..., 1)
        return x / rms * self.weight  # (..., D)


class RoPE(nn.Module):
    """Rotary Position Embedding (Su et al., 2021, "RoFormer").

    Injects position by rotating pairs of head dimensions. For pair (m, m+1)
    at position pos with pair index k:

        angle = pos * theta_k,  theta_k = 10000 ** (-2k / D_rot)
        y_m    = x_m * cos(angle) - x_{m+1} * sin(angle)
        y_{m+1} = x_m * sin(angle) + x_{m+1} * cos(angle)

    The rotation is a similarity transform, so it preserves vector lengths
    and makes the q·k inner product depend only on the *relative* position —
    which is the property attention needs.

    ``rope_dim``: 0 → rotate all D dimensions (standard); 0 < rope_dim < D →
    rotate the first rope_dim dims, the rest pass through.

    Input: (B, S, H, D), positions: (S,)  →  Output: (B, S, H, D)
    """

    def forward(self, x: torch.Tensor, positions: torch.Tensor, rope_dim: int = 0) -> torch.Tensor:
        """Apply RoPE to q or k. x: (B, S, H, D)."""
        # Split rotated prefix from pass-through suffix.
        d = x.shape[-1]
        if rope_dim > 0 and rope_dim < d:
            x_rot = x[..., :rope_dim]
            x_pass = x[..., rope_dim:]
        else:
            x_rot = x
            x_pass = None

        batch_size, seq_len, n_heads = x_rot.shape[0], x_rot.shape[1], x_rot.shape[2]
        pair_dim = x_rot.shape[-1] // 2  # number of (even, odd) pairs

        positions = positions.to(x.device)

        # theta_k = 10000 ** (-2k / d) — (pair_dim,)
        freqs = 1.0 / (
            10000.0 ** (torch.arange(pair_dim, device=x.device, dtype=torch.float32) * 2.0 / x_rot.shape[-1])
        )
        # angles: (S, pair_dim)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
        cos = torch.cos(angles)  # (S, pair_dim)
        sin = torch.sin(angles)  # (S, pair_dim)

        # (B, S, H, pair_dim, 2) — last dim is the (even, odd) pair
        x_flat = x_rot.reshape(batch_size, seq_len, n_heads, pair_dim, 2)
        x_even = x_flat[..., 0]  # (B, S, H, pair_dim)
        x_odd = x_flat[..., 1]  # (B, S, H, pair_dim)

        # Broadcast cos/sin over batch & heads: (S, pair_dim) → (S, 1, pair_dim)
        cos_b = cos[:, None, :]
        sin_b = sin[:, None, :]
        y_even = x_even * cos_b - x_odd * sin_b  # (B, S, H, pair_dim)
        y_odd = x_even * sin_b + x_odd * cos_b  # (B, S, H, pair_dim)
        rotated = torch.stack([y_even, y_odd], dim=-1).reshape(x_rot.shape)  # (B, S, H, d_rot)

        if x_pass is not None:
            return torch.cat([rotated, x_pass], dim=-1)  # (B, S, H, D)
        return rotated


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention with Grouped-Query Attention.

    Mirrors ``impl._np.attention.MultiHeadAttention`` exactly (same shapes,
    same RoPE, same causal mask), with the K/V head count following the
    config's ``n_groups``:

        q = x @ Wq   (B, S, H*hd)  → (B, H, S, hd)
        k = x @ Wk   (B, S, G*hd)  → (B, G, S, hd)
        v = x @ Wv   (B, S, G*hd)  → (B, G, S, hd)
        q, k ← RoPE(q), RoPE(k)
        ctx  = SDPA(q, k, v, is_causal=True)   (B, H, S, hd)
        out  = ctx @ Wo             (B, S, D)

    The scaled-dot-product computation uses the framework's
    ``F.scaled_dot_product_attention`` (which dispatches to flash/efficient
    kernels on GPU) instead of a hand-rolled max-subtract softmax. GQA: when
    G < H, the K/V heads are repeated to match the query head count (this
    PyTorch build requires matching head counts); the repeat is a no-op when
    G == H.

    GQA (Ainslie et al., 2023): when G < H, each K/V head is shared by
    H // G query heads, shrinking the KV cache by the same factor at
    inference. G == H is ordinary multi-head attention.

    Projections use no bias (Llama convention).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_groups: int,
        rope_dim: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_dim = embed_dim // n_heads
        self.rope_dim = rope_dim

        hd = self.head_dim
        # (out, in) per nn.Linear convention: q: D → H*hd
        self.q_proj = nn.Linear(embed_dim, n_heads * hd, bias=False)
        # k/v: D → G*hd (smaller than q when G < H — the whole point of GQA)
        self.k_proj = nn.Linear(embed_dim, n_groups * hd, bias=False)
        self.v_proj = nn.Linear(embed_dim, n_groups * hd, bias=False)
        # o: H*hd → D (mixes the heads back into one vector)
        self.o_proj = nn.Linear(n_heads * hd, embed_dim, bias=False)

        self.rope = RoPE()

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Multi-head attention forward. x: (B, S, D) → out: (B, S, D)."""
        batch_size, seq_len, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        # (B, S, D) @ (D, H*hd) → (B, S, H*hd)
        q = self.q_proj(x)  # (B, S, H*hd)
        k = self.k_proj(x)  # (B, S, G*hd)
        v = self.v_proj(x)  # (B, S, G*hd)

        # Split into heads: (B, S, H*hd) → (B, H, S, hd)
        q = q.view(batch_size, seq_len, H, hd).permute(0, 2, 1, 3)  # (B, H, S, hd)
        k = k.view(batch_size, seq_len, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)
        v = v.view(batch_size, seq_len, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)

        # RoPE on q and k (RoPE's contract shape is (B, S, H, D)).
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        q = self.rope(q.permute(0, 2, 1, 3), positions, rope_dim=self.rope_dim).permute(0, 2, 1, 3)
        k = self.rope(k.permute(0, 2, 1, 3), positions, rope_dim=self.rope_dim).permute(0, 2, 1, 3)

        # GQA: broadcast each K/V group to its H // G query heads (SDPA
        # requires the K/V head count to match the query head count on this
        # PyTorch build).
        if G != H:
            k = k.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
            v = v.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)

        # Scaled dot-product attention via the framework's SDPA (dispatches to
        ctx = F.scaled_dot_product_attention(
            q,  # (B, H, S, hd)
            k,  # (B, G, S, hd)
            v,  # (B, G, S, hd)
            is_causal=True,
        )  # (B, H, S, hd)

        # Merge heads: (B, H, S, hd) → (B, S, H*hd) → project to (B, S, D)
        ctx = ctx.permute(0, 2, 1, 3).reshape(batch_size, seq_len, H * hd)
        return self.o_proj(ctx)  # (B, S, D)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (GLU: Dauphin et al. 2016; SwiGLU: Shazeer 2020).

        gate = SiLU(x @ gate_proj)     (..., FF)
        up   = x @ up_proj              (..., FF)
        out  = (gate * up) @ down_proj  (..., D)

    Weights: gate_proj, up_proj: (D, FF); down_proj: (FF, D). Stored in
    (in, out) layout (matching the NumPy reference) so cross-track loading
    needs no transpose for FFN weights.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward. x: (..., D) → out: (..., D)."""
        # x @ gate_proj: (..., FF) — gate branch
        gate = F.silu(x @ self.gate_proj.to(x.dtype))  # (..., FF)
        # x @ up_proj: (..., FF) — up branch
        up = x @ self.up_proj.to(x.dtype)  # (..., FF)
        # element-wise gating, then back to model width
        return (gate * up) @ self.down_proj.to(x.dtype)  # (..., D)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts (Shazeer et al. 2017; Switch/Mixtral style).

    Mirrors ``impl._np.moe.MixtureOfExperts``: a router selects top-k
    of E SwiGLU experts per token.

        scores = x @ W_gate              (B, S, E)   raw router logits
        probs  = softmax(scores)       (B, S, E)   over ALL experts
        mask   = keep only top-k probs per token (rest → 0)
        weights = mask / sum(mask)     (B, S, E)   renormalized to sum 1
        out    = sum_j weights_j * expert_j(x)     (B, S, D)

    Note (reference implementation): every expert's output is computed and
    multiplied by its (possibly zero) weight; real systems gather tokens per
    expert to skip the zero-weight compute. The math is identical.

    Router: nn.Linear(D, E) with no bias (Mixtral convention); experts: SwiGLUFFN.
    """

    def __init__(self, embed_dim: int, n_experts: int, ff_dim: int, top_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # Router: (out=E, in=D) per nn.Linear convention, no bias
        self.gate = nn.Linear(embed_dim, n_experts, bias=False)
        # One SwiGLU expert per index (Mixtral-style)
        # Typed list for pyright; nn.ModuleList registers parameters for torch
        self.expert_list: list[SwiGLUFFN] = [SwiGLUFFN(embed_dim, ff_dim) for _ in range(n_experts)]
        self.experts = nn.ModuleList(self.expert_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE forward. x: (B, S, D) → out: (B, S, D)."""
        E = self.n_experts

        # Router scores and softmax over all experts: (B, S, E)
        scores = self.gate(x)  # (B, S, E)
        probs = F.softmax(scores, dim=-1)  # (B, S, E)

        # Top-k mask: keep the k largest probs per token, zero the rest.
        if self.top_k < E:
            # kth = the k-th largest prob value per token: (B, S, 1)
            kth_vals, _ = torch.topk(probs, self.top_k, dim=-1)  # (B, S, k)
            threshold = kth_vals[..., -1:].expand_as(probs)  # (B, S, E)
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
            probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-8)  # renormalize

        # Weighted sum of expert outputs (all experts computed; zeros masked).
        out = torch.zeros_like(x)  # (B, S, D)
        for expert_idx, expert in enumerate(self.expert_list):
            w = probs[..., expert_idx : expert_idx + 1]  # (B, S, 1)
            out = out + w * expert(x)  # (B, S, D)
        return out


class TransformerBlock(nn.Module):
    """One decoder-only transformer block (LLaMA-style, pre-norm).

    The canonical block (LLaMA-2/3, HuggingFace ``LlamaDecoderLayer``):

        h   = x + attention(rms_norm(x))       # (B, S, D)
        out = h + feed_forward(rms_norm(h))    # (B, S, D)

    ``feed_forward`` is the dense SwiGLU by default, or the MoE when
    ``config.has_moe()``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        D, H, G = config.embed_dim, config.n_heads, config.kv_heads

        # Norms (LLama naming: input norm, norm after attention)
        self.input_layernorm = RMSNorm(D, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(D, eps=config.norm_eps)
        self.self_attn = MultiHeadAttention(embed_dim=D, n_heads=H, n_groups=G, rope_dim=config.rope_dim)
        if config.has_moe():
            self.mlp: SwiGLUFFN | MixtureOfExperts = MixtureOfExperts(
                embed_dim=D, n_experts=config.n_experts, ff_dim=config.expert_dim, top_k=config.top_k
            )
        else:
            self.mlp = SwiGLUFFN(embed_dim=D, ff_dim=config.expert_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Block forward. x: (B, S, D) → out: (B, S, D)."""
        # Stream 1: attention with pre-norm and residual.
        attn_out = self.self_attn(self.input_layernorm(x), positions)  # (B, S, D)
        h = x + attn_out  # (B, S, D)

        # Stream 2: feed-forward (dense or MoE) with pre-norm and residual.
        ff_out = self.mlp(self.post_attention_layernorm(h))  # (B, S, D)
        return h + ff_out  # (B, S, D)


class DecoderStack(nn.Module):
    """Stack of n_layers TransformerBlocks (the "body" of the decoder).

    out = block_{n-1}( ... block_1(block_0(x)) ... ), x: (B, S, D) → (B, S, D).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        # Typed view of the blocks: nn.ModuleList registers parameters for
        # torch; the plain list keeps per-block attribute types for pyright.
        self.blocks: list[TransformerBlock] = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.layers = nn.ModuleList(self.blocks)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Forward through all blocks. x: (B, S, D) → out: (B, S, D)."""
        out = x
        for block in self.blocks:
            out = block(out, positions)
        return out


class TorchModel(nn.Module):
    """Complete decoder-only transformer in PyTorch.

    Forward: tokens → embed_tokens → DecoderStack → final RMSNorm → lm_head
    (D → V, no bias) → logits (B, S, V).

    Parameters are addressed by the ``shared.constants.Keys`` scheme — the
    same flat keys as every other track, which is the cross-backend
    checkpoint contract.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Build the model from a :class:`TransformerConfig`."""
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim

        torch.manual_seed(config.seed)

        self.embedding = Embedding(config.vocab_size, config.embed_dim)
        self.stack = DecoderStack(config)
        self.final_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        # lm_head: (in=D, out=V) per nn.Linear convention
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, S) int → logits: (B, S, V)."""
        # (B, S) → (B, S, D)
        x = self.embedding(x)
        # (B, S, D) → (B, S, D)
        x = self.stack(x)
        # (B, S, D) → (B, S, D)
        x = self.final_norm(x)
        # (B, S, D) → (B, S, V)
        return self.lm_head(x)

    def _param_tensors(self) -> dict[str, torch.Tensor]:
        """Storage binding: registry key → owning tensor (the track's only traversal)."""
        t: dict[str, torch.Tensor] = {
            Keys.embed(): self.embedding.weight,
            Keys.final_norm(): self.final_norm.weight,
            Keys.lm_head(): self.lm_head.weight,
        }
        for layer_idx, block in enumerate(self.stack.blocks):
            t[Keys.ln(layer_idx, LayerNorm.INPUT)] = block.input_layernorm.weight
            t[Keys.ln(layer_idx, LayerNorm.POST_ATTENTION)] = block.post_attention_layernorm.weight
            for proj, module in (
                (Attn.Q_PROJ, block.self_attn.q_proj),
                (Attn.K_PROJ, block.self_attn.k_proj),
                (Attn.V_PROJ, block.self_attn.v_proj),
                (Attn.O_PROJ, block.self_attn.o_proj),
            ):
                t[Keys.attn(layer_idx, proj)] = module.weight
            mlp = block.mlp
            if isinstance(mlp, MixtureOfExperts):
                t[Keys.moe_gate(layer_idx)] = mlp.gate.weight
                for expert_idx, expert in enumerate(mlp.expert_list):
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.GATE_PROJ)] = expert.gate_proj
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.UP_PROJ)] = expert.up_proj
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.DOWN_PROJ)] = expert.down_proj
            else:
                t[Keys.ffn(layer_idx, Mlp.GATE_PROJ)] = mlp.gate_proj
                t[Keys.ffn(layer_idx, Mlp.UP_PROJ)] = mlp.up_proj
                t[Keys.ffn(layer_idx, Mlp.DOWN_PROJ)] = mlp.down_proj
        return t

    def save_as_numpy(self) -> dict[str, Any]:  # Returns dict[str, np.ndarray]
        """Save all parameters as a NumPy-compatible flat dict (Keys scheme).

        Registry-driven: the key set, expected shapes, and the PyTorch
        ``nn.Linear`` ``(out, in) → (in, out)`` transpose rule all come
        from ``ParameterRegistry`` — this track only supplies storage.
        """
        tensors = self._param_tensors()
        params: dict[str, Any] = {}
        for entry in ParameterRegistry(self.config).entries:
            array = tensors[entry.key].detach().cpu().numpy()
            params[entry.key] = array.T if entry.torch_transpose else array
        return params  # type: ignore[return-value]

    def get_all_parameters(self) -> dict[str, Any]:
        """All parameters as a flat numpy dict (interface parity with NumPy/CUDA)."""
        return self.save_as_numpy()

    def load_from_numpy_dict(self, params_dict: dict[str, Any]) -> None:
        """Load parameters from a flat dict (inverse of ``save_as_numpy``).

        Validates the dict against the registry first, so stale or
        mismatched checkpoints fail fast instead of half-loading.
        """
        registry = ParameterRegistry(self.config)
        registry.validate(params_dict)
        tensors = self._param_tensors()
        for entry in registry.entries:
            loaded = torch.from_numpy(params_dict[entry.key])
            if entry.torch_transpose:
                loaded = loaded.T  # checkpoint (in, out) → nn.Linear (out, in)
            loaded = loaded.contiguous().to(tensors[entry.key].dtype)
            tensors[entry.key].data.copy_(loaded)
