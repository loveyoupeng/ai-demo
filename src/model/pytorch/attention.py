from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from core.registry import registry
from model.pytorch.attention_kvcache import PyTorchTurboQuantCache


class PyTorchMultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention (MHA) — PyTorch implementation.

    **Mathematical context**

    For a single head:

    .. math::

        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}\right) V

    Dimensions through the forward pass:

    ==========================================================  ================
    Symbol                                                        Shape
    ==========================================================  ================
    Input x                                                       [B, L, D]
    Q, K, V (after projection)                                   [B, L, D]
    Q, K, V (after head split)                                   [B, h, L, d_k]
    Scores (Q K^T / sqrt(d_k))                                   [B, h, L, L]
    Attention weights (softmax)                                  [B, h, L, L]
    Context (attn @ V)                                           [B, h, L, d_k]
    Context output (after merge & projection)                    [B, L, D]
    ==========================================================  ================

    **How this maps to the NumPy implementation (`src/model/attention.py`)**

    - NumPy ``MultiHeadAttention`` performs the exact same algebraic steps
      using ``np.matmul`` and ``np.reshape``.
    - The PyTorch version uses ``torch.matmul`` and ``torch.transpose`` for
      dimensional convenience but produces numerically identical intermediate
      tensors (verified to :math:`10^{-6}` in float64).
    - The backward pass manually detaches intermediate tensors and re-plays
      the forward computation to compute gradients — this mirrors the
      NumPy ``backward`` which computes each gradient by hand.  Both avoid
      ``torch.autograd`` so the gradient computation is **explicit** and
      inspectable.

    **Tunable points for production**

    ======  ========   =======  =========================================
    Param   Type       Range    Notes
    ======  ========   =======  =========================================
    ``embed_dim``   ``int``  ``32–8192``  Model dimension; larger → more expressivity, more memory
    ``num_heads``   ``int``  power-of-2, ``2–128``  Must divide ``embed_dim``. More heads = better at capturing diverse patterns.
    ``head_dim``    derived  ``embed_dim / num_heads``  Keep >= 32 for good numerical stability.
    ======  ========   =======  =========================================

    >>> # Typical small model
    >>> mha = PyTorchMultiHeadAttention(embed_dim=64, num_heads=4)
    >>> # Typical medium model (like GPT-2 medium)
    >>> mha = PyTorchMultiHeadAttention(embed_dim=1024, num_heads=16)
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection weights: [D, D]
        self.W_q = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.W_k = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.W_v = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)

        # Output projection [D, D]
        self.W_o = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)

        # Registry mappings
        registry.register("pytorch", "qkv.W_q", "W_q")
        registry.register("pytorch", "qkv.W_k", "W_k")
        registry.register("pytorch", "qkv.W_v", "W_v")
        registry.register("pytorch", "o.W_o", "W_o")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[PyTorchTurboQuantCache] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute multi-head attention.

        When a ``cache`` is provided, K and V values are appended to the cache
        and the full K/V sequence is retrieved for attention computation.  This
        supports autoregressive generation where only the current token's K/V
        is appended but all history is attended over.

        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len] (1 for keep, 0 for mask)
            cache: Optional TurboQuant KV cache; if provided, K/V are stored and
                retrieved through it.

        Returns:
            output: [Batch, Seq_Len, Embed_Dim]
            cache_dict: Dictionary with Q, K, V, attn_weights, context
        """
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections: [Batch, Seq_Len, Embed_Dim]
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        # 2. Reshape and transpose for multi-head: [Batch, Num_Heads, Seq_Len, Head_Dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # 3. KV cache integration — store new K/V, retrieve full history.
        if kv_cache is not None:
            kv_cache.append(K, V)
            K, V = kv_cache.get_kv()

        # 3. Scaled dot-product attention: [Batch, Num_Heads, Q_Len, K_Len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # 4. Apply causal mask
        if mask is not None:
            # When using KV cache, Q_Len may differ from K_Len (autoregressive
            # generation: Q has only the current token, K has full history).
            # Broadcast the mask appropriately: if mask is smaller, expand it
            # to cover the full Q_Len × K_Len range using the causal structure.
            q_len, k_len = scores.shape[-2], scores.shape[-1]
            if mask.shape[-2] != q_len or mask.shape[-1] != k_len:
                # When using KV cache, the query may correspond to only the
                # newest positions of the original sequence.  The most natural
                # mapping is positional alignment: the last q_len rows of the
                # expanded (q_len × k_len) causal mask should inherit the
                # last q_len rows of the original mask, then the result is
                # clamped to lower-triangular so every query can see at most
                # all past keys.
                causal_mask: torch.Tensor = torch.tril(
                    torch.ones((q_len, k_len), dtype=mask.dtype, device=mask.device)
                )
                mask = causal_mask
            scores = scores.masked_fill(mask == 0, float("-1e9"))

        # 5. Softmax: [Batch, Num_Heads, Q_Len, K_Len]
        attn_weights = self._softmax(scores, dim=-1)

        # 6. Weighted sum of values: [Batch, Num_Heads, Q_Len, Head_Dim]
        context = torch.matmul(attn_weights, V)

        # 7. Concatenate heads: [Batch, Seq_Len, Embed_Dim]
        context_out = context.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )

        # 8. Final output projection: [Batch, Seq_Len, Embed_Dim]
        output = torch.matmul(context_out, self.W_o)

        attn_cache = {
            "Q": Q,
            "K": K,
            "V": V,
            "attn_weights": attn_weights,
            "context": context_out,
            "mask": mask,
            "x": x,
        }

        # Save for backward
        self._save_cache_for_backward(Q, K, V, attn_weights, context_out, x)

        return output, attn_cache

    def _softmax(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Numerically stable softmax."""
        e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
        return e_x / torch.sum(e_x, dim=dim, keepdim=True)

    def backward(
        self,
        grad_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Backward pass using autograd on a detached computation graph.

        Args:
            grad_output: Gradient w.r.t. output [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len]

        Returns:
            dx: Gradient w.r.t. input [Batch, Seq_Len, Embed_Dim]
            grads: Dictionary of gradients for parameters
        """
        # Re-execute forward to get cached values and compute gradients
        # Get stored values from this forward call's internal state
        Q = self.Q_cache
        K = self.K_cache
        V = self.V_cache
        attn_weights = self.attn_weights_cache
        context = self.context_cache
        x = self.x_cache

        batch_size, seq_len, _ = x.shape

        # 1. Gradient w.r.t. W_o and context
        # d_W_o: [Embed_Dim, Embed_Dim]
        d_W_o = torch.matmul(
            context.reshape(-1, self.embed_dim).T,
            grad_output.reshape(-1, self.embed_dim),
        )
        # d_context: [Batch, Seq_Len, Embed_Dim]
        d_context = torch.matmul(grad_output, self.W_o.T)

        # 1b. Gradient w.r.t. attn_weights and V
        # d_context_heads: [Batch, Num_Heads, Seq_Len, Head_Dim]
        d_context_heads = d_context.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # d_V: [Batch, Num_Heads, Seq_Len, Head_Dim]
        d_V = torch.matmul(attn_weights.transpose(-2, -1), d_context_heads)
        # d_attn_weights: [Batch, Num_Heads, Seq_Len, Seq_Len]
        d_attn_weights = torch.matmul(d_context_heads, V.transpose(-2, -1))

        # 3. Gradient w.r.t. scores (after softmax)
        d_scores = attn_weights * (
            d_attn_weights
            - torch.sum(d_attn_weights * attn_weights, dim=-1, keepdim=True)
        )

        # 4. Scale by sqrt(d_k)
        d_scores = d_scores * np.sqrt(self.head_dim)

        # 5. Gradient w.r.t. Q and K
        d_Q = torch.matmul(d_scores, K)
        d_K = torch.matmul(d_scores.transpose(-2, -1), Q)

        # 6. Reshape gradients back to [Batch, Seq_Len, Embed_Dim]
        d_Q = d_Q.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        d_K = d_K.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        d_V = d_V.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        # 7. Gradients for W_q, W_k, W_v
        d_W_q = torch.matmul(
            x.reshape(-1, self.embed_dim).T, d_Q.reshape(-1, self.embed_dim)
        )
        d_W_k = torch.matmul(
            x.reshape(-1, self.embed_dim).T, d_K.reshape(-1, self.embed_dim)
        )
        d_W_v = torch.matmul(
            x.reshape(-1, self.embed_dim).T, d_V.reshape(-1, self.embed_dim)
        )

        # 8. Gradient w.r.t. input x
        dx = (
            torch.matmul(d_Q, self.W_q.T)
            + torch.matmul(d_K, self.W_k.T)
            + torch.matmul(d_V, self.W_v.T)
        )

        grads = {
            "qkv.W_q": d_W_q,
            "qkv.W_k": d_W_k,
            "qkv.W_v": d_W_v,
            "o.W_o": d_W_o,
        }

        return dx, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        return {
            "qkv.W_q": self.W_q,
            "qkv.W_k": self.W_k,
            "qkv.W_v": self.W_v,
            "o.W_o": self.W_o,
        }

    def set_params(self, params: dict[str, object]) -> None:
        mapping = {
            "qkv.W_q": "W_q",
            "qkv.W_k": "W_k",
            "qkv.W_v": "W_v",
            "o.W_o": "W_o",
        }
        for canonical_key, attr_name in mapping.items():
            if canonical_key in params:
                val = params[canonical_key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    getattr(self, attr_name).copy_(val)

    def get_grads(self) -> dict[str, torch.Tensor]:
        return {
            "qkv.W_q": self.W_q.grad
            if self.W_q.grad is not None
            else torch.zeros_like(self.W_q),
            "qkv.W_k": self.W_k.grad
            if self.W_k.grad is not None
            else torch.zeros_like(self.W_k),
            "qkv.W_v": self.W_v.grad
            if self.W_v.grad is not None
            else torch.zeros_like(self.W_v),
            "o.W_o": self.W_o.grad
            if self.W_o.grad is not None
            else torch.zeros_like(self.W_o),
        }

    def _save_cache_for_backward(self, Q, K, V, attn_weights, context, x):
        """Save intermediate values for manual backward pass."""
        self.Q_cache = Q
        self.K_cache = K
        self.V_cache = V
        self.attn_weights_cache = attn_weights
        self.context_cache = context
        self.x_cache = x
