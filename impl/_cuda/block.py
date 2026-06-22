"""TransformerBlock — CUDA assembly of all CUDA primitives.

Assembles:
  - RMSNorm     → impl/_cuda/layernorm.rmsnorm    (warp-reduction kernel)
  - MHA         → impl/_cuda/attention.scaled_dot_product_attention (CUDA softmax/weighted-sum)
  - MoE         → impl/_cuda/moe.moe_forward       (CUDA score + weighted-sum kernels)
  - RoPE        → impl/_cuda/rope.apply_rope       (CUDA rotary embedding kernel)
  - SiLU        → impl/_cuda/ffn._CUDASiLU         (CUDA SiLU kernel inside SwiGLU)
  - gated residual + dropout

Architecture (post-norm, gated residual):
    Input:  x [B, S, D]
    │
    ├─ Stream 1: Attention ──────────────────────────────────────────────
    │   1. attn_out = MHA(x)                         # (B, S, D)
    │   2. h = x + attn_out                          # residual FIRST
    │   3. h = RMSNorm(h, ln1_gamma)                 # post-norm
    │   4. h = h + sigmoid(gate1) * h                # gated residual
    │   5. h = dropout(h)                           # training only
    │
    ├─ Stream 2: MoE ────────────────────────────────────────────────────
    │   6. moe_out = MoE(h)                          # (B, S, D)
    │   7. out = h + moe_out                         # residual FIRST
    │   8. out = RMSNorm(out, ln2_gamma)             # post-norm
    │   9. out = out + sigmoid(gate2) * out          # gated residual
    │  10. out = dropout(out)                        # training only
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

import torch

from impl._cuda.attention import scaled_dot_product_attention as cuda_sdp_attention
from impl._cuda.layernorm import rmsnorm
from impl._cuda.moe import moe_forward
from impl._cuda.rope import apply_rope

# ── Weight initialization helpers ──────────────────────────────────────────────


def _init_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Xavier/uniform initialization for attention/FFN weights.

    Uses Kaiming (Xavier) uniform initialization with the bound:
        bound = sqrt(6 / (fan_in + fan_out))
    This is the same formula used by torch.nn.Linear default.

    Parameters
    ----------
    rows : int
        Input dimension (fan_in).
    cols : int
        Output dimension (fan_out).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Initialized weight matrix of shape (rows, cols) on CPU,
        values in [-bound, +bound].
    """
    bound = (6.0 / (rows + cols)) ** 0.5
    tensor = torch.empty(rows, cols, dtype=torch.float32)
    torch.nn.init.uniform_(tensor, -bound, bound, generator=torch.Generator().manual_seed(seed))
    return tensor


def _init_zeros(shape: tuple[int, ...]) -> torch.Tensor:
    """Initialize a tensor to zeros.

    Parameters
    ----------
    shape : tuple[int, ...]
        Desired shape.

    Returns
    -------
    torch.Tensor
        Zero-initialized tensor on CPU.
    """
    return torch.zeros(shape)


# ── CUDA block assembly ────────────────────────────────────────────────────────


class CuTransformerBlock:
    """CUDA TransformerBlock — assembly of all CUDA primitives.

    This class follows the same architecture as the NumPy/PyTorch TransformerBlock
    but delegates all heavy computation to CUDA kernels.

    It does NOT inherit from nn.Module — weights are stored as plain tensors
    (attributes or __dict__) for parity checking against the NumPy implementation.
    This makes it easy to extract raw parameter arrays for comparison.

    Parameters
    ----------
    embed_dim : int
        Input/output embedding dimension.
    n_heads : int
        Number of attention heads.
    n_experts : int
        Number of MoE experts.
    ff_dim : int
        Hidden dimension per MoE expert.
    k : int
        Number of top experts to activate per token (default: 2).
    rope_dim : int
        Number of head dimensions for RoPE (0 = disabled).
    seed : int
        Random seed for weight initialization (default: 0).

    Attributes
    ----------
    ln1_gamma : torch.Tensor, shape (D,)
        Learnable RMSNorm scale parameter for attention post-norm.
    ln2_gamma : torch.Tensor, shape (D,)
        Learnable RMSNorm scale parameter for MoE post-norm.
    gate1 : torch.Tensor, shape (1,)
        Learnable scalar gate for attention residual (initialized to 0).
    gate2 : torch.Tensor, shape (1,)
        Learnable scalar gate for MoE residual (initialized to 0).
    Wq : torch.Tensor, shape (D, D)
        Query projection weights.
    Wk : torch.Tensor, shape (D, D)
        Key projection weights.
    Wv : torch.Tensor, shape (D, D)
        Value projection weights.
    Wo : torch.Tensor, shape (D, D)
        Attention output projection weights.
    expert_weights : torch.Tensor, shape (N, D, D)
        Expert weight matrices — all experts share the same W (D, D).
    expert_bias : torch.Tensor, shape (N, D)
        Expert bias vectors — all experts share the same bias (D,).
    routing_weights : torch.Tensor, shape (N, D)
        Routing score weights for expert selection.

    Forward
    -------
    x : torch.Tensor, shape (batch_size, seq_len, embed_dim) on CUDA device

    Returns
    -------
    out : torch.Tensor, shape (batch_size, seq_len, embed_dim) on CUDA device

    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        rope_dim: int = 0,
        seed: int = 0,
    ) -> None:
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.k = k
        self.rope_dim = rope_dim
        self.head_dim = embed_dim // n_heads

        # RMSNorm weights — gamma for layer normalization
        # Shape: (D,) per layer
        self.ln1_gamma = torch.ones(embed_dim)
        self.ln1_gamma.requires_grad_(True)
        self.ln2_gamma = torch.ones(embed_dim)
        self.ln2_gamma.requires_grad_(True)

        # Gated residuals — learnable scalar gates initialized to zero
        # Initialized to zero → identity at start → gates learn to open
        self.gate1 = torch.zeros(1)
        self.gate1.requires_grad_(True)
        self.gate2 = torch.zeros(1)
        self.gate2.requires_grad_(True)

        # ── Multi-Head Attention weights ───────────────────────────────
        # Q, K, V projections: D → D (one projection per head = D)
        # Output projection: D → D
        self.Wq = _init_weight(embed_dim, embed_dim, seed=seed + 2)
        self.Wq.requires_grad_(True)
        self.Wk = _init_weight(embed_dim, embed_dim, seed=seed + 3)
        self.Wk.requires_grad_(True)
        self.Wv = _init_weight(embed_dim, embed_dim, seed=seed + 4)
        self.Wv.requires_grad_(True)
        self.Wo = _init_weight(embed_dim, embed_dim, seed=seed + 5)
        self.Wo.requires_grad_(True)

        # ── MoE weights ────────────────────────────────────────────────
        # Each expert has: W (D, D), bias (D,), routing (D,)
        # All experts share the same W and bias (like NumPy impl)
        self.expert_weights = torch.zeros(n_experts, embed_dim, embed_dim)
        self.expert_weights.requires_grad_(True)
        self.expert_bias = torch.zeros(n_experts, embed_dim)
        self.expert_bias.requires_grad_(True)
        self.routing_weights = torch.zeros(n_experts, embed_dim)
        self.routing_weights.requires_grad_(True)

        # Compute RoPE tables cache (shared across all calls for one block)
        self._rope_cos = None
        self._rope_sin = None

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        dropout: float = 0.0,
        training: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the TransformerBlock.

        Implements the post-norm gated residual architecture with MoE:
            attn_out = MHA(x)
            h = x + attn_out → RMSNorm → gated → (dropout)
            moe_out = MoE(h)
            out = h + moe_out → RMSNorm → gated → (dropout)

        Parameters
        ----------
        x : torch.Tensor, shape (B, S, D)
            Input activations on CUDA device. Weights must be on the
            same device.
        positions : torch.Tensor, shape (S,) or None
            Position indices for RoPE. If None, uses arange(S).
        dropout : float
            Dropout rate (default: 0.0, no dropout).
        training : bool
            Whether in training mode (dropout active).

        Returns
        -------
        torch.Tensor, shape (B, S, D)
            Block output with same shape as input.

        Shape flow
        ----------
          x: (B, S, D)
            → Wq/Wk/Wv proj: (B, S, D)
            → reshape: (B, S, H, hd) → (B, H, S, hd)
            → RoPE: (B, H, S, hd)
            → QK^T: (B, H, S, S) — CUDA softmax + weighted sum
            → @V: (B, H, S, hd)
            → reshape/transpose: (B, S, D)
            → Wo proj: (B, S, D) = attn_out

          h = x + attn_out       # (B, S, D)
          h = RMSNorm(h, ln1)    # (B, S, D)
          h = h + gate1 * h      # (B, S, D)
          h = dropout(h)         # (B, S, D)

          moe_out = MoE(h)       # (B, S, D)
          out = h + moe_out      # (B, S, D)
          out = RMSNorm(out, ln2)# (B, S, D)
          out = out + gate2 * out# (B, S, D)
          out = dropout(out)     # (B, S, D)

        """
        B, S, D = x.shape
        device = x.device

        # Ensure all weights are on the same device as input
        # Use .to(device) to handle cases where caller forgot to move weights
        Wq = self.Wq.to(device)
        Wk = self.Wk.to(device)
        Wv = self.Wv.to(device)
        Wo = self.Wo.to(device)
        ln1_gamma = self.ln1_gamma.to(device)
        ln2_gamma = self.ln2_gamma.to(device)
        gate1 = self.gate1.to(device)
        gate2 = self.gate2.to(device)
        expert_weights = self.expert_weights.to(device)
        expert_bias = self.expert_bias.to(device)
        routing_weights = self.routing_weights.to(device)

        # ── Stream 1: Attention ──────────────────────────────────────────
        # Q, K, V projections: (B, S, D) → (B, S, D)
        q = x @ Wq  # (B, S, D)
        k = x @ Wk  # (B, S, D)
        v = x @ Wv  # (B, S, D)

        # Reshape: (B, S, D) → (B, S, H, hd) → (B, H, S, hd)
        # Note: .transpose() creates non-contiguous tensor — .contiguous() ensures
        # the RoPE kernel can safely use .view() without stride errors
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2).contiguous()  # (B, H, S, hd)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2).contiguous()  # (B, H, S, hd)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2).contiguous()  # (B, H, S, hd)

        # Apply RoPE if enabled
        if self.rope_dim > 0:
            if positions is None:
                positions = torch.arange(S, device=x.device)
            q = apply_rope(q, positions, rope_dim=self.rope_dim)
            k = apply_rope(k, positions, rope_dim=self.rope_dim)

        # Scaled dot-product attention — CUDA softmax + weighted-sum kernels
        # q, k, v: (B, H, S, hd) — no causal mask (caller can pre-mask if needed)
        attn_out = cuda_sdp_attention(q, k, v)  # (B, H, S, hd)

        # Output projection: (B, H, S, hd) → (B, S, D)
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(B, S, self.embed_dim) @ Wo  # (B, S, D)
        )

        # Residual FIRST (post-norm): x + attn_out
        h = x + attn_out  # (B, S, D)

        # Post-norm: RMSNorm via CUDA kernel
        h = rmsnorm(h, ln1_gamma)  # (B, S, D)

        # Gated residual: h = h + sigmoid(gate1) * h
        gate1_sigmoid = torch.sigmoid(gate1)
        h = h + gate1_sigmoid * h  # (B, S, D)

        # Dropout (training only)
        if training and dropout > 0.0:
            h = torch.nn.functional.dropout(h, p=dropout, training=True)

        # ── Stream 2: MoE ────────────────────────────────────────────────
        # MoE forward — CUDA kernel for scoring and weighted sum
        # Returns: (output, indices, weights) — all on CUDA
        # output: (B, S, D), indices: (B, S, k), weights: (B, S, k)
        moe_out, _, _ = moe_forward(
            tokens=h,
            expert_weights=expert_weights,
            expert_bias=expert_bias,
            routing_weights=routing_weights,
            top_k=self.k,
        )
        # moe_out: (B, S, D) — weighted sum of expert outputs via CUDA kernel

        # Residual: h + moe_out
        out = h + moe_out  # (B, S, D)

        # Post-norm: RMSNorm via CUDA kernel
        out = rmsnorm(out, ln2_gamma)  # (B, S, D)

        # Gated residual: out = out + sigmoid(gate2) * out
        gate2_sigmoid = torch.sigmoid(gate2)
        out = out + gate2_sigmoid * out  # (B, S, D)

        # Dropout (training only)
        if training and dropout > 0.0:
            out = torch.nn.functional.dropout(out, p=dropout, training=True)

        return out
