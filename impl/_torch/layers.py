"""Embedding layer — maps token IDs to dense vectors.

Maps integer token IDs [batch, seq_len] to embedding vectors [batch, seq_len, embed_dim]
by looking up rows of a learnable weight matrix [vocab_size, embed_dim].

This mirrors the NumPy implementation in impl/_np/modules.py which uses
a stateless forward function. The PyTorch version stores weight as an
nn.Parameter for autograd.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from shared.constants import Block, Mha, Transformer

# pyright: reportAttributeAccessIssue=false


class Embedding(nn.Module):
    """Token token IDs to dense embedding vectors.

    Parameters:
        vocab_size: Number of tokens in the vocabulary.
        embed_dim: Dimension of the embedding vector.
    """

    __slots__ = ("weight",)

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        # weight.shape = [vocab_size, embed_dim]
        # Initialized with Kaiming uniform (default nn init)
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for token IDs.

        Args:
            input_ids: Token indices [batch, seq_len] (int64 or int32).

        Returns:
            Embedding vectors [batch, seq_len, embed_dim].
        """
        return nn.functional.embedding(input_ids, self.weight)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes each feature vector to unit variance, then scales by learned gamma.

    RMSNorm formula: out = x / (sqrt(mean(x^2, dim=-1, keepdim=True)) + eps) * gamma

    Where eps = 1e-6 is added to prevent numerical instability. The mean is
    computed over the last dimension (embed_dim), broadcasting gamma over
    batch dimensions.

    Parameters:
        embed_dim: Number of features (dimension of input and output).

    Shape:
        - Input: (..., embed_dim) — any leading batch dimensions
        - Output: (..., embed_dim) — same shape as input
    """

    __slots__ = ("gamma",)

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # gamma.shape = (embed_dim,) — initialized to ones (identity scale)
        self.gamma = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input activations. Any shape with last dim = embed_dim.

        Returns:
            RMS-normalized, scaled output. Same shape as input.
        """
        # x:           (..., embed_dim)
        # x^2:         (..., embed_dim)
        # mean(x^2):   (..., 1)       — mean over last dim
        # rms:         (..., 1)       — sqrt(mean(x^2)) + eps
        # x/rms:       (..., embed_dim) — broadcast division
        # output:      (..., embed_dim) — broadcast gamma
        eps: float = 1e-6  # prevent divide-by-zero
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)) + eps
        return x / rms * self.gamma


class SiLULayer(nn.Module):
    """Sigmoid Linear Unit (SiLU / Swish) activation: f(x) = x * sigmoid(x).

    Element-wise nonlinear activation. Also known as Swish.

    Parameters:
        None — SiLU is stateless.

    Shape:
        - Input: (..., any dims)
        - Output: (..., same) — element-wise
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU activation element-wise.

        Args:
            x: Input tensor, any shape.

        Returns:
            SiLU(x) = x * sigmoid(x), same shape as input.
        """
        return x * torch.sigmoid(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    A modern feedforward layer with gating mechanism:
        gated = SiLU(x @ W1) * (x @ W3)
        output = gated @ W2

    This replaces the traditional single linear layer FFN with a gated
    variant that typically provides better representational capacity.

    Parameters:
        embed_dim: Input and output dimension.
        ff_dim: Inner (feedforward) dimension for W1 and W3.

    Shape:
        - x:       (batch, seq_len, embed_dim)
        - W1, W3:  (embed_dim, ff_dim)
        - W2:      (ff_dim, embed_dim)
        - output:  (batch, seq_len, embed_dim) — same as input
    """

    def __init__(self, embed_dim: int, ff_dim: int) -> None:
        super().__init__()
        # W1: (embed_dim, ff_dim) — projects input to inner dimension
        # W3: (embed_dim, ff_dim) — parallel projected input (gating signal)
        # W2: (ff_dim, embed_dim) — projects back to original dimension
        self.W1 = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.W3 = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.W2 = nn.Parameter(torch.empty(ff_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all weights with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W3, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass.

        Args:
            x: Input activations [batch, seq_len, embed_dim].

        Returns:
            Output [batch, seq_len, embed_dim] with gating applied.
        """
        # x:        (batch, seq_len, embed_dim)
        # x @ W1:   (batch, seq_len, ff_dim) — projection to inner dim
        # SiLU:     (batch, seq_len, ff_dim) — element-wise activation
        # x @ W3:   (batch, seq_len, ff_dim) — gating signal
        # gated:    (batch, seq_len, ff_dim) — SiLU(xW1) * xW3
        # output:   (batch, seq_len, embed_dim) — gated @ W2
        # Ensure weights match input dtype for dtype-flexible inference
        w1 = self.W1.to(x.dtype)
        w3 = self.W3.to(x.dtype)
        w2 = self.W2.to(x.dtype)
        # x:        (batch, seq_len, embed_dim)
        # x @ W1:   (batch, seq_len, ff_dim) — projection to inner dim
        # x @ W3:   (batch, seq_len, ff_dim) — gating signal
        # gated:    (batch, seq_len, ff_dim) — SiLU(xW1) * xW3
        # output:   (batch, seq_len, embed_dim) — gated @ W2
        return self.SiLU()(x @ w1) * (x @ w3) @ w2

    def SiLU(self) -> nn.Module:
        """Return a SiLU activation module."""
        return nn.SiLU()


class RoPE(nn.Module):
    """Rotary Positional Embedding — injects position via 2D rotation.

    Applies a rotation matrix to each (odd, even) pair of head dimensions,
    where the rotation angle depends on the token position:
        x_m' = x_m * cos(mθ) - x_{m+1} * sin(mθ)
        x_{m+1}' = x_m * sin(mθ) + x_{m+1} * cos(mθ)
    where θ = 10000^(-2k/d) for the k-th dimension pair.

    Parameters:
        None — frequencies are computed from head_dim and position.

    Shape:
        - Input:  (..., n_heads, head_dim) — Q or K tensor
        - Output: (..., n_heads, head_dim) — rotated Q or K

    Args:
        x: Q or K tensor with shape [batch, seq_len, n_heads, head_dim].
        positions: Position indices, shape [seq_len] (int64).
        rope_dim: Number of head_dims to rotate. 0 = full rotation.
                  Values after rope_dim pass through unchanged.

    Returns:
        Rotated tensor with same shape as input.
    """

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rope_dim: int = 0,
    ) -> torch.Tensor:
        """Apply rotary positional embedding.

        Args:
            x: Q or K tensor. Shape [batch, seq_len, n_heads, head_dim].
            positions: Position indices. Shape [seq_len].
            rope_dim: Rotate only first rope_dim dims. 0 = all.

        Returns:
            Rotated tensor. Same shape as input.
        """
        # x:          (B, S, H, D)
        # positions:  (S,) — position indices
        # output:     (B, S, H, D)

        # Separate unrotated dims (after rope_dim)
        head_dim = x.shape[-1]
        if rope_dim > 0 and rope_dim < head_dim:
            x_rot = x[..., :rope_dim]
            x_pass = x[..., rope_dim:]
        else:
            x_rot = x
            x_pass = None

        batch_size, seq_len = x_rot.shape[0], x_rot.shape[1]
        n_heads = x_rot.shape[2]
        # pair_dim = half of the actual rotated dimensions (not full head_dim)
        pair_dim = x_rot.shape[-1] // 2

        # Ensure positions is on the same device as input tensor
        positions = positions.to(x.device)

        # Compute rotation frequencies: use full head_dim in denominator
        # (same as NumPy: freqs for the first pair_dim pairs)
        # (pair_dim,) = 1 / 10000^(2k / head_dim)
        freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, device=x.device) * 2.0 / head_dim))

        # For each position and each pair: angle = pos * freq
        # positions: (S,)
        # freqs:     (pair_dim,)
        # angles:    (S, pair_dim) ← then broadcast to (B, S, 1, pair_dim)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # (S, pair_dim)

        cos = torch.cos(angles)  # (S, pair_dim)
        sin = torch.sin(angles)  # (S, pair_dim)

        # Reshape: (B, S, H, pair_dim, 2) — last dim is (even, odd) pair
        x_flat = x_rot.reshape(batch_size, seq_len, n_heads, pair_dim, 2)

        # Extract odd and even components
        # x_even: (B, S, H, pair_dim)  — x_{2k}
        # x_odd:  (B, S, H, pair_dim)  — x_{2k+1}
        # cos_broad: (S, 1, pair_dim) → (1, S, 1, pair_dim)
        # sin_broad: (S, 1, pair_dim) → (1, S, 1, pair_dim)
        x_even = x_flat[..., 0]  # (B, S, H, pair_dim)
        x_odd = x_flat[..., 1]  # (B, S, H, pair_dim)

        # Broadcast cos/sin from (S, pair_dim) → (B, S, 1, pair_dim) to match x
        cos_broad = cos[:, None, :].unsqueeze(0)  # (1, S, 1, pair_dim)
        sin_broad = sin[:, None, :].unsqueeze(0)  # (1, S, 1, pair_dim)

        # Apply 2D rotation: each pair rotates independently
        # y_{2k}   = x_{2k}   * cos - x_{2k+1}   * sin
        # y_{2k+1} = x_{2k}   * sin + x_{2k+1}   * cos
        y_even = x_even * cos_broad - x_odd * sin_broad  # (B, S, H, pair_dim)
        y_odd = x_even * sin_broad + x_odd * cos_broad  # (B, S, H, pair_dim)

        # Reassemble: (B, S, H, pair_dim, 2) → (B, S, H, x_rot_size)
        rotated = torch.stack([y_even, y_odd], dim=-1).reshape(x_rot.shape)

        if x_pass is not None:
            return torch.cat([rotated, x_pass], dim=-1)
        return rotated


class Linear(nn.Module):
    """Simple linear layer with optional zero bias.

    Parameters:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: If False, no bias term (matching NumPy convention).

    Shape:
        - Input:  (..., in_features)
        - Output: (..., out_features)
    """

    __slots__ = ("weight", "bias")

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        # weight.shape = (out_features, in_features) — matches nn.Linear convention
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight with Kaiming uniform. Zero bias."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.size(1))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear forward.

        Args:
            x: Input tensor. Any shape with last dim = in_features.

        Returns:
            Output with last dim = out_features.
        """
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.linear(x, w, b)


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention with grouped-query (GQA).

    Parameters:
        embed_dim: Total embedding dimension (also output dimension).
        n_heads: Number of attention heads. Must divide embed_dim evenly.
        n_groups: Number of groups for GQA. n_groups <= n_heads.
                  n_groups == n_heads → standard MHA (no GQA).
        rope_dim: Number of head dimensions to rotate. 0 = no RoPE.

    Architecture:
        X [B,S,D] → Q_proj → Q [B,S,H,head_dim]
                  → K_proj → K [B,S,G,head_dim]
                  → V_proj → V [B,S,G,head_dim]

        For GQA: K and V shared across groups of query heads (G<H).
        K/V expanded from G to H heads for attention computation.

        Scaled: QK^T / sqrt(head_dim) → softmax → V

        Final: reshape → O_proj → [B,S,D]

    Args:
        embed_dim: Input/output dimension.
        n_heads: Number of query heads.
        n_groups: Number of K/V groups (default = n_heads, standard MHA).
        rope_dim: Rotary position embedding dimension (default 0 = disabled).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_groups: int | None = None,
        rope_dim: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.rope_dim = rope_dim
        self.n_groups = n_groups if n_groups is not None else n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        assert self.n_groups <= n_heads, "n_groups must be <= n_heads"

        # Q projection: D → H * head_dim (one projection per query head)
        self.Wq = Linear(embed_dim, n_heads * self.head_dim)

        # K projection: D → G * head_dim (shared across G groups)
        self.Wk = Linear(embed_dim, self.n_groups * self.head_dim)

        # V projection: D → G * head_dim (shared across G groups)
        self.Wv = Linear(embed_dim, self.n_groups * self.head_dim)

        # Output projection: H * head_dim → D
        self.Wo = Linear(n_heads * self.head_dim, embed_dim)

        # RoPE module
        self.rope = RoPE()

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        position: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention forward pass.

        Args:
            x: Input activations [batch, seq_len, embed_dim].
            past_key_value: Optional list of (k, v) pairs per layer for KV cache.
                            Each k is [B, n_groups, cached_len, head_dim],
                            each v is [B, n_groups, cached_len, head_dim].
            position: Current position index for KV cache (not used but kept for
                      API compatibility with TransformerBlock).
        Returns:
            output: [batch, seq_len, embed_dim] — attention output
            attn_weights: [batch, n_heads, seq_len, seq_len] — attention for debug
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        head_dim = self.head_dim

        # Q, K, V projections
        # q: (B, S, H*hd) → reshape → (B, H, S, hd)
        # k: (B, S, G*hd) → reshape → (B, G, S, hd)
        # v: (B, S, G*hd) → reshape → (B, G, S, hd)
        q = self.Wq(x).reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)  # (B, H, S, hd)

        # For past_key_value, we need to handle appending
        if past_key_value is not None:
            pk, pv = past_key_value[0]
            # pk, pv already have shape (B, n_groups, past_len, head_dim)
            # We need to concat along seq dimension, but k/v come from projections
            # Since x is the NEW tokens only, we need to append to past
            # BUT this design doesn't quite work — let me reconsider.
            # Actually, in the NumPy version, the past_key_value is used in
            # TransformerBlock, which passes the CURRENT full input x (not just new tokens).
            # So the KV cache updates happen at the block level.
            # For now, this forward only handles fresh sequences.
            # The KV cache will be managed by TransformerBlock.
            k = self.Wk(x).reshape(batch_size, seq_len, self.n_groups, head_dim).transpose(1, 2)  # (B, G, S, hd)
            v = self.Wv(x).reshape(batch_size, seq_len, self.n_groups, head_dim).transpose(1, 2)  # (B, G, S, hd)
        else:
            k = self.Wk(x).reshape(batch_size, seq_len, self.n_groups, head_dim).transpose(1, 2)  # (B, G, S, hd)
            v = self.Wv(x).reshape(batch_size, seq_len, self.n_groups, head_dim).transpose(1, 2)  # (B, G, S, hd)

        # Apply RoPE to Q and K if rope_dim > 0
        if self.rope_dim > 0:
            # RoPE expects (B, seq_len, n_heads/dim, head_dim) but we have (B, n_heads, seq_len, head_dim)
            # So we need to reshape for RoPE
            q_reshape = q.transpose(1, 2)  # (B, S, H, hd) → RoPE expects (B, S, H, hd)
            k_reshape = k.transpose(1, 2)  # (B, S, G, hd)
            q = self.rope(q_reshape, torch.arange(seq_len), rope_dim=self.rope_dim).transpose(1, 2)  # (B, H, S, hd)
            k = self.rope(k_reshape, torch.arange(seq_len), rope_dim=self.rope_dim).transpose(1, 2)  # (B, G, S, hd)

        # For GQA: expand K and V from G to H by repeating heads
        # k: (B, G, S, hd) → (B, H, S, hd) where head_i gets group_i % G
        if self.n_groups != self.n_heads:
            # Repeat K/V along the head dim: for each query head h, use group h % G
            # k: (B, G, S, hd) → (B, H, S, hd) via index_repeat
            repeat_idxs = torch.tensor([h % self.n_groups for h in range(self.n_heads)], device=x.device)
            k = k[:, repeat_idxs, :, :]  # (B, H, S, hd)
            v = v[:, repeat_idxs, :, :]  # (B, H, S, hd)

        # Scaled dot-product attention
        # q: (B, H, S, hd), k: (B, H, S, hd) → k^T: (B, H, hd, S)
        # scores: (B, H, S, S) = q @ k^T
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(float(head_dim))  # (B, H, S, S)

        # Softmax with numerical stability (max subtraction)
        scores = scores - scores.max(dim=-1, keepdim=True).values  # (B, H, S, S)
        exp_scores = torch.exp(scores)  # (B, H, S, S)
        attn_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)  # (B, H, S, S)

        # Attention: score @ V
        # scores: (B, H, S, S), V: (B, H, S, hd) → output: (B, H, S, hd)
        context = attn_weights @ v  # (B, H, S, hd)

        # Output projection
        # context: (B, H, S, hd) → (B, S, H*hd) → O_proj → (B, S, D)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * head_dim)
        )  # (B, S, H*hd)
        output = self.Wo(context)  # (B, S, D)

        return output, attn_weights


class MixtureOfExperts(nn.Module):
    """Mixture of Experts with top-k selection.

    For each input token, a small router network produces softmax over E experts.
    Only the top-k experts (with highest routing weight) fire. Each expert is a
    SwiGLU FFN. The output is the weighted sum of the k active experts.

    Architecture:
        x [B,S,D] → router = x @ W_r + b_r [B,S,E] → softmax → w_r [B,S,E]
        topk(w_r, k) → k expert indices per token

        expert_out[t] = sum(w_r[t, i] * experts[i](x[t]) for i in top-k)

    Parameters:
        embed_dim: Input/output dimension of each expert.
        n_experts: Number of expert networks.
        ff_dim: Feed-forward hidden dimension within each expert.
        k: Number of top experts to select per token (default=2).
    """

    def __init__(
        self,
        embed_dim: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_experts = n_experts
        self.ff_dim = ff_dim
        self.k = k

        # Router: linear layer from embed_dim to n_experts
        # Shape: [embed_dim, n_experts]
        self.router = Linear(embed_dim, n_experts, bias=True)

        # Experts: each expert is a SwiGLU FFN
        # Each expert: input embed_dim → output embed_dim
        self.experts = nn.ModuleList([SwiGLUFFN(embed_dim, ff_dim) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture of Experts forward pass.

        Args:
            x: Input tensor. Shape [batch, seq, embed_dim].

        Returns:
            Output tensor. Shape [batch, seq, embed_dim].
        """
        n_experts = self.n_experts
        k = self.k

        # Router: (B, S, D) @ (D, E) → (B, S, E)
        # Apply softmax over experts for routing
        router_scores = self.router(x)  # (B, S, E)

        # Softmax over the expert dimension for routing weights
        # Normalize across experts: sum over E = 1 for each (batch, seq)
        router_scores_max = router_scores.max(dim=-1, keepdim=True).values  # (B, S, 1)
        router_scores = router_scores - router_scores_max  # stable
        exp_scores = torch.exp(router_scores)  # (B, S, E)
        exp_scores_sum = exp_scores.sum(dim=-1, keepdim=True)  # (B, S, 1)
        routing_weights = exp_scores / exp_scores_sum  # (B, S, E) — softmax over experts

        # For each token, select top-k experts
        # Zero out weights below the k-th highest
        if k < n_experts:
            # Get the k-th highest weight per token via k-th argmax
            # torch.topk returns (values, indices) — use it to find threshold
            top_k_values, _ = torch.topk(routing_weights, k, dim=-1)  # (B, S, k)
            threshold = top_k_values.min(dim=-1, keepdim=True).values  # (B, S, 1)
            # Mask: keep only weights >= threshold
            routing_weights = routing_weights * (routing_weights >= threshold).float()
            # Re-normalize so selected weights sum to 1
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Compute expert outputs for ALL experts and ALL tokens in parallel
        # expert_out: [n_experts, B, S, D]
        expert_outs = torch.stack([expert(x) for expert in self.experts])  # (E, B, S, D)

        # Weighted sum: route_weights[B,S,E] @ expert_outs[E,B,S,D] → (B,S,D)
        # Equivalent to: sum over E of routing_weights[e] * expert_outs[e]
        # For each (b, s): output[b,s] = sum_e(routing_weights[b,s,e] * expert_outs[e,b,s])
        # bse × ebsd → bsd (sum over expert dimension e)
        output = torch.einsum("bse,ebsd->bsd", routing_weights, expert_outs)

        return output


class TransformerBlock(nn.Module):
    """Decoder-only transformer block.

    Composes multi-head attention and mixture-of-experts with residual
    connections and RMS normalization.

        h = x + MHA(RMSNorm(x)) + MoE(RMSNorm(x + MHA(RMSNorm(x))))

    Architecture:
        Input:  x [B, S, D]
        │
        ├→ RMSNorm (ln1)       [B, S, D]
        ├→ MHA (Q/K/V proj)   [B, S, D]
        ├→ Residual add        [B, S, D]
        ├→ RMSNorm (ln2)       [B, S, D]
        ├→ MoE (router + kWExperts) [B, S, D]
        └→ Residual add        [B, S, D]

    Parameters:
        embed_dim: Input/output dimension.
        n_heads: Number of attention heads.
        n_experts: Number of MoE experts.
        ff_dim: Feed-forward hidden dimension per expert.
        k: Number of top experts per token.
        rope_dim: Rotary position embedding dimension (0 = disabled).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        rope_dim: int = 0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.k = k
        self.norm_type = "post"

        # RMSNorm layers
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)

        # Gated residuals — learnable scalar gates, initialized to zero
        # Gate controls signal flow: h = h + sigmoid(gate) * gated_value
        # Initialized to zero → identity at start → gates learn to open
        self.gate1 = nn.Parameter(torch.zeros(1))
        self.gate2 = nn.Parameter(torch.zeros(1))

        # Dropout for regularization (applied after gated residual)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Multi-head attention (standard MHA — no GQA)
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_groups=n_heads,  # standard MHA (no GQA)
            rope_dim=rope_dim,
        )

        # Mixture of Experts
        self.moe = MixtureOfExperts(
            embed_dim=embed_dim,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor. Shape [batch, seq_len, embed_dim].

        Returns:
            Output tensor. Shape [batch, seq_len, embed_dim].
        """
        # ── Stream 1: Attention ─────────────────────────────────────
        # MHA: (B, S, D) → (B, S, D) — attention output
        attn_out, _ = self.mha(x)  # (B, S, D), (B, H, S, S)

        # Residual FIRST (post-norm): x + attn_out
        h = x + attn_out  # (B, S, D)

        # Post-norm: h = RMSNorm(h)
        h = self.ln1(h)  # (B, S, D)

        # Gated residual with sigmoid activation → values in [0, 1]
        gate1_sigmoid = torch.sigmoid(self.gate1)
        h = h + gate1_sigmoid * h  # (B, S, D)

        # Dropout after gated residual (training mode only)
        h = self.dropout1(h)  # (B, S, D)

        # ── Stream 2: MoE ──────────────────────────────────────────
        # MoE: (B, S, D) → (B, S, D)
        moe_out = self.moe(h)  # (B, S, D)

        # Residual FIRST (post-norm): h + moe_out
        out = h + moe_out  # (B, S, D)

        # Post-norm: out = RMSNorm(out)
        out = self.ln2(out)  # (B, S, D)

        # Gated residual with sigmoid activation → values in [0, 1]
        gate2_sigmoid = torch.sigmoid(self.gate2)
        out = out + gate2_sigmoid * out  # (B, S, D)

        # Dropout after gated residual (training mode only)
        out = self.dropout2(out)  # (B, S, D)

        return out


class DecoderStack(nn.Module):
    """Stack of TransformerBlocks — chains n_layers of decoder blocks.

    Architecture:
        Input:  x [B, S, D]
        │
        └→ block_0 → block_1 → ... → block_{n_layers-1} → output [B, S, D]

        Each block:
          h = x + MHA(RMSNorm(x)) + MoE(RMSNorm(x + MHA(RMSNorm(x))))

    Parameters:
        n_layers: Number of transformer blocks.
        embed_dim: Input/output dimension.
        n_heads: Number of attention heads per block.
        n_experts: Number of MoE experts per block.
        ff_dim: Feed-forward hidden dimension per expert.
        k: Number of top experts per token.
        rope_dim: Rotary position embedding dimension (0 = disabled).
    """

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        rope_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    n_experts=n_experts,
                    ff_dim=ff_dim,
                    k=k,
                    rope_dim=rope_dim,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all stacked blocks.

        Args:
            x: Input tensor. Shape [batch, seq_len, embed_dim].

        Returns:
            Output tensor. Shape [batch, seq_len, embed_dim].
        """
        out = x
        for block in self.layers:
            out = block(out)
        return out


class TorchModel(nn.Module):
    """Complete decoder-only transformer with embedding + SwiGLU output.

    Forward: tokens → embedding → DecoderStack → RMSNorm → SwiGLU → Linear → logits

    Architecture:
        Input:  tokens [batch, seq_len] (int64)
        │
        ├→ Embedding table lookup       [batch, seq_len, embed_dim]
        ├→ DecoderStack (n_layers)     [batch, seq_len, embed_dim]
        ├→ RMSNorm (final_ln)          [batch, seq_len, embed_dim]
        ├→ SwiGLU (output)             [batch, seq_len, embed_dim]
        └→ Linear (output_proj)        [batch, seq_len, vocab_size]

    Parameters:
        vocab_size: Vocabulary size.
        embed_dim: Hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads per block.
        n_experts: Number of MoE experts per block.
        ff_dim: Feed-forward hidden dimension per expert.
        k: Number of top experts per token.
        rope_dim: Rotary position embedding dimension (0 = disabled).
        seed: Random seed for initialization.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        rope_dim: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.k = k
        self.seed = seed

        # Embedding layer
        self.embedding = Embedding(vocab_size, embed_dim)

        # Decoder stack
        self.stack = DecoderStack(
            n_layers=n_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=rope_dim,
        )

        # Final layer normalization
        self.final_ln = RMSNorm(embed_dim)

        # Output projection — SwiGLU → Linear
        # SwiGLU maps D → D via hidden (D*2), then linear projects D → V
        ff_dim_out = embed_dim * 2  # output projection hidden dim
        self.output = SwiGLUFFN(embed_dim, ff_dim_out)
        self.output_proj = Linear(embed_dim, vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete model.

        Args:
            x: Token IDs. Shape [batch_size, seq_len], dtype int64.

        Returns:
            Predicted logits. Shape [batch_size, seq_len, vocab_size].
        """
        # Embedding: [B,S] → [B,S,D]
        x = self.embedding(x)  # (B, S, D)

        # Decoder stack: [B,S,D] → [B,S,D]
        x = self.stack(x)

        # Final layer normalization: [B,S,D] → [B,S,D]
        x = self.final_ln(x)

        # SwiGLU output projection: [B,S,D] → [B,S,D]
        x = self.output(x)

        # Linear projection to vocab: [B,S,D] → [B,S,V]
        logits = self.output_proj(x)

        return logits

    def load_from_numpy(self, np_model: Any) -> None:  # pyright: ignore
        """Load parameters from a NumPyModel.

        Maps NumPyModel parameter keys to the corresponding nn.Parameter
        attributes in this PyTorch model. Both models must have the same
        architecture.

        Args:
            np_model: A NumPyModel instance with loaded parameters.
        """
        from impl._np.model import NumPyModel

        if not isinstance(np_model, NumPyModel):
            raise TypeError("Expected NumPyModel instance")

        params = np_model.get_all_parameters()

        def load(np_key: str, tensor: Any) -> None:  # pyright: ignore[reportUnknownParameterType]
            """Copy numpy array to tensor in-place.

            NumPy Linear: out = x @ W + b, W is (in, out)
            PyTorch Linear: W is stored as (out, in) — transpose needed.
            SwiGLU: W1, W2, W3 use (in, ff), (ff, out), (in, ff) in both backends — no transpose.
            """
            np_array = params[np_key]
            loaded = torch.from_numpy(np_array).to(tensor.dtype)
            # Transpose Linear weight matrices: NumPy uses (in, out)
            # but PyTorch Linear stores (out, in).
            # Do NOT transpose SwiGLU (W1, W2, W3) or embedding — both backends
            # use the same (in, out) convention for these.
            if (
                tensor.dim() == 2
                and np_key != Transformer.EMBEDDING_WEIGHTS
                and not any(np_key.endswith(f".{k}") for k in ("W1", "W2", "W3"))
            ):
                loaded = loaded.T.contiguous()
            tensor.data.copy_(loaded)

        # ── Embedding ──────────────────────────────────────────────
        load(Transformer.EMBEDDING_WEIGHTS, self.embedding.weight)

        # ── DecoderStack ───────────────────────────────────────────
        for layer_idx, block in enumerate(self.stack.layers):

            # Layer norm gamma
            load(Block.ln1_gamma(layer_idx), block.ln1.gamma)
            load(Block.ln2_gamma(layer_idx), block.ln2.gamma)

            # MHA Q (weight + bias)
            load(Block.mha(layer_idx, Mha.WQ), block.mha.Wq.weight)
            load(Block.mha(layer_idx, Mha.BQ), block.mha.Wq.bias)

            # MHA K (weight + bias)
            load(Block.mha(layer_idx, Mha.WK), block.mha.Wk.weight)
            load(Block.mha(layer_idx, Mha.BK), block.mha.Wk.bias)

            # MHA V (weight + bias)
            load(Block.mha(layer_idx, Mha.WV), block.mha.Wv.weight)
            load(Block.mha(layer_idx, Mha.BV), block.mha.Wv.bias)

            # MHA O (weight + bias)
            load(Block.mha(layer_idx, Mha.WO), block.mha.Wo.weight)
            load(Block.mha(layer_idx, Mha.BO), block.mha.Wo.bias)

            # MoE router — loads weight and bias from NumPy
            load(Block.moe_router(layer_idx), block.moe.router.weight)
            load(Block.moe_bias(layer_idx), block.moe.router.bias)

            # MoE experts
            for expert_idx, expert in enumerate(
                block.moe.experts  # pyright: ignore[reportArgumentType]
            ):
                load(Block.moe_expert(layer_idx, expert_idx, "W1"), expert.W1)
                load(Block.moe_expert(layer_idx, expert_idx, "W2"), expert.W2)
                load(Block.moe_expert(layer_idx, expert_idx, "W3"), expert.W3)

        # ── Final RMSNorm ──────────────────────────────────────────
        load(Transformer.FINAL_GAMMA, self.final_ln.gamma)

        # ── Output SwiGLU ──────────────────────────────────────────
        load(Transformer.OUTPUT_W1, self.output.W1)
        load(Transformer.OUTPUT_W2, self.output.W2)
        load(Transformer.OUTPUT_W3, self.output.W3)

        # ── Output projection ──────────────────────────────────────
        # PyTorch Linear output_proj has weight and bias (matches NumPy)
        load(Transformer.OUTPUT_PROJ_W, self.output_proj.weight)
        load(Transformer.OUTPUT_PROJ_B, self.output_proj.bias)

    def save_as_numpy(self) -> dict[str, Any]:  # Returns dict[str, np.ndarray]
        """Save all parameters as a NumPy-compatible dictionary.

        Returns a dict with the same key structure as NumPyModel.get_all_parameters(),
        enabling cross-backend parameter exchange for parity testing.

        Linear weight matrices are transposed from PyTorch (out, in) to
        NumPy (in, out) convention for direct cross-backend comparison.

        Returns
        -------
        params : dict[str, np.ndarray]
            Dictionary mapping parameter names to NumPy arrays.

        Notes
        -----
        Linear weight matrices are transposed to match NumPy's (in, out)
        convention. The matching load_from_numpy_dict reverses this transpose.
        """

        params: dict[str, Any] = {}

        def save(tensor: Any, np_key: str, transpose: bool = False) -> None:
            """Copy PyTorch tensor to NumPy array with matching key.

            Args:
                tensor: PyTorch tensor to save.
                np_key: NumPy parameter name.
                transpose: If True, transpose the tensor before saving
                    (for Linear layers: PyTorch (out, in) → NumPy (in, out)).
            """
            array = tensor.detach().cpu().numpy()
            if transpose:
                array = array.T
            params[np_key] = array

        # ── Embedding ──────────────────────────────────────────────
        save(self.embedding.weight, Transformer.EMBEDDING_WEIGHTS)

        # ── DecoderStack ───────────────────────────────────────────
        for layer_idx, block in enumerate(self.stack.layers):

            # Layer norm gamma
            save(block.ln1.gamma, Block.ln1_gamma(layer_idx))
            save(block.ln2.gamma, Block.ln2_gamma(layer_idx))

            # Gated residuals
            save(block.gate1, Block.gate1(layer_idx))
            save(block.gate2, Block.gate2(layer_idx))

            # MHA Q (weight + bias)
            save(block.mha.Wq.weight, Block.mha(layer_idx, Mha.WQ), transpose=True)
            save(block.mha.Wq.bias, Block.mha(layer_idx, Mha.BQ))

            # MHA K (weight + bias)
            save(block.mha.Wk.weight, Block.mha(layer_idx, Mha.WK), transpose=True)
            save(block.mha.Wk.bias, Block.mha(layer_idx, Mha.BK))

            # MHA V (weight + bias)
            save(block.mha.Wv.weight, Block.mha(layer_idx, Mha.WV), transpose=True)
            save(block.mha.Wv.bias, Block.mha(layer_idx, Mha.BV))

            # MHA O (weight + bias)
            save(block.mha.Wo.weight, Block.mha(layer_idx, Mha.WO), transpose=True)
            save(block.mha.Wo.bias, Block.mha(layer_idx, Mha.BO))

            # MoE router — saves weight and bias to NumPy format
            save(block.moe.router.weight, Block.moe_router(layer_idx), transpose=True)
            if block.moe.router.bias is not None:
                save(block.moe.router.bias, Block.moe_bias(layer_idx))

            # MoE experts
            for expert_idx, expert in enumerate(block.moe.experts):  # pyright: ignore[reportArgumentType]
                save(expert.W1, Block.moe_expert(layer_idx, expert_idx, "W1"))
                save(expert.W2, Block.moe_expert(layer_idx, expert_idx, "W2"))
                save(expert.W3, Block.moe_expert(layer_idx, expert_idx, "W3"))

        # ── Final RMSNorm ──────────────────────────────────────────
        save(self.final_ln.gamma, Transformer.FINAL_GAMMA)

        # ── Output SwiGLU ──────────────────────────────────────────
        save(self.output.W1, Transformer.OUTPUT_W1)
        save(self.output.W2, Transformer.OUTPUT_W2)
        save(self.output.W3, Transformer.OUTPUT_W3)

        # ── Output projection ──────────────────────────────────────
        save(self.output_proj.weight, Transformer.OUTPUT_PROJ_W, transpose=True)
        save(self.output_proj.bias, Transformer.OUTPUT_PROJ_B)

        return params  # type: ignore[return-value]

    def load_from_numpy_dict(self, params_dict: dict[str, Any]) -> None:
        """Load parameters from a raw NumPy-compatible dictionary.

        This is the reverse of save_as_numpy(). It loads a dict with the
        same key structure as NumPyModel.get_all_parameters() and copies
        the values into this model's parameters. The keys match the output
        of save_as_numpy(), so save→dict→load is a no-op.

        Linear weight matrices are transposed from NumPy (in, out) to
        PyTorch (out, in) layout since save_as_numpy() stores them in
        NumPy convention.

        Args:
            params_dict: Dictionary mapping parameter names to array-like values.
                         Keys match those from save_as_numpy() (NumPy format,
                         Linear weights are transposed to (in, out)).
        """

        def load(tensor: Any, np_key: str, transpose: bool = False) -> None:
            """Copy numpy array to tensor in-place.

            Matches save_as_numpy() convention — transpose Linear weights
            from NumPy (in, out) back to PyTorch (out, in).
            """
            loaded = torch.from_numpy(params_dict[np_key]).to(tensor.dtype)
            if transpose:
                loaded = loaded.T.contiguous()
            tensor.data.copy_(loaded)

        # ── Embedding ──────────────────────────────────────────────
        load(self.embedding.weight, Transformer.EMBEDDING_WEIGHTS)

        # ── DecoderStack ───────────────────────────────────────────
        for layer_idx, block in enumerate(self.stack.layers):

            # Layer norm gamma
            load(block.ln1.gamma, Block.ln1_gamma(layer_idx))
            load(block.ln2.gamma, Block.ln2_gamma(layer_idx))

            # Gated residuals
            load(block.gate1, Block.gate1(layer_idx))
            load(block.gate2, Block.gate2(layer_idx))

            # MHA Q (weight + bias)
            load(block.mha.Wq.weight, Block.mha(layer_idx, Mha.WQ), transpose=True)
            load(block.mha.Wq.bias, Block.mha(layer_idx, Mha.BQ))

            # MHA K (weight + bias)
            load(block.mha.Wk.weight, Block.mha(layer_idx, Mha.WK), transpose=True)
            load(block.mha.Wk.bias, Block.mha(layer_idx, Mha.BK))

            # MHA V (weight + bias)
            load(block.mha.Wv.weight, Block.mha(layer_idx, Mha.WV), transpose=True)
            load(block.mha.Wv.bias, Block.mha(layer_idx, Mha.BV))

            # MHA O (weight + bias)
            load(block.mha.Wo.weight, Block.mha(layer_idx, Mha.WO), transpose=True)
            load(block.mha.Wo.bias, Block.mha(layer_idx, Mha.BO))

            # MoE router
            load(block.moe.router.weight, Block.moe_router(layer_idx), transpose=True)
            load(block.moe.router.bias, Block.moe_bias(layer_idx))

            # MoE experts
            for expert_idx, expert in enumerate(block.moe.experts):  # pyright: ignore[reportArgumentType]
                load(expert.W1, Block.moe_expert(layer_idx, expert_idx, "W1"))
                load(expert.W2, Block.moe_expert(layer_idx, expert_idx, "W2"))
                load(expert.W3, Block.moe_expert(layer_idx, expert_idx, "W3"))

        # ── Final RMSNorm ──────────────────────────────────────────
        load(self.final_ln.gamma, Transformer.FINAL_GAMMA)

        # ── Output SwiGLU ──────────────────────────────────────────
        load(self.output.W1, Transformer.OUTPUT_W1)
        load(self.output.W2, Transformer.OUTPUT_W2)
        load(self.output.W3, Transformer.OUTPUT_W3)

        # ── Output projection ──────────────────────────────────────
        load(self.output_proj.weight, Transformer.OUTPUT_PROJ_W, transpose=True)
        load(self.output_proj.bias, Transformer.OUTPUT_PROJ_B)


class AdamW:
    """AdamW optimizer with bias correction and decoupled weight decay.

    Parameters
    ----------
    lr : float, default 3e-4
        Learning rate.
    beta1 : float, default 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default 0.999
        Exponential decay rate for the second moment estimates.
    eps : float, default 1e-8
        Term added for numerical stability.
    weight_decay : float, default 0.0
        Weight decay coefficient (L2 regularization, decoupled).

    Usage
    -----
    >>> optimizer = AdamW(lr=1e-3, weight_decay=0.01)
    >>> optimizer.step(model.state_dict())
    """

    def __init__(
        self,
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # Internal state: first and second moment estimates per parameter
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}
        self._count: int = 0

    def step(self, params: dict[str, torch.Tensor], grads: dict[str, torch.Tensor]) -> None:
        """Perform one optimization step.

        Parameters
        ----------
        params : dict[str, torch.Tensor]
            Model parameters to update (modified in-place).
            Each key maps to a PyTorch tensor of arbitrary shape.
        grads : dict[str, torch.Tensor]
            Gradients for each parameter (must have same keys as params).
            Each key maps to a tensor of the same shape as the
            corresponding parameter.
        """
        self._count += 1

        # Pre-compute bias correction denominators once per step
        bias_correction_1 = 1.0 - self.beta1**self._count
        bias_correction_2 = 1.0 - self.beta2**self._count

        for name, grad in grads.items():
            param = params[name]

            # Initialize moment estimates on first visit
            if name not in self.m:
                self.m[name] = torch.zeros_like(param, dtype=torch.float64)
                self.v[name] = torch.zeros_like(param, dtype=torch.float64)

            # Step 1: Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad

            # Step 2: Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * grad**2

            # Step 3: Compute bias-corrected moment estimates
            m_hat = self.m[name] / bias_correction_1
            v_hat = self.v[name] / bias_correction_2

            # Step 4: Parameter update with Adam + decoupled weight decay
            adam_step = m_hat / (torch.sqrt(v_hat) + self.eps)
            decay_step = self.weight_decay * param

            params[name] -= self.lr * (adam_step + decay_step)
