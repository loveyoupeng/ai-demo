"""Embedding layer — maps token IDs to dense vectors.

Maps integer token IDs [batch, seq_len] to embedding vectors [batch, seq_len, embed_dim]
by looking up rows of a learnable weight matrix [vocab_size, embed_dim].

This mirrors the NumPy implementation in impl/_np/modules.py which uses
a stateless forward function. The PyTorch version stores weight as an
nn.Parameter for autograd.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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

        # Compute rotation frequencies: use full head_dim in denominator
        # (same as NumPy: freqs for the first pair_dim pairs)
        # (pair_dim,) = 1 / 10000^(2k / head_dim)
        freqs = 1.0 / (
            10000.0
            ** (torch.arange(pair_dim, device=x.device) * 2.0 / head_dim)
        )

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

        # Reassemble: (B, S, H, pair_dim, 2) → (B, S, H, rope_dim)
        rotated = torch.stack([y_even, y_odd], dim=-1).reshape(x_rot.shape)

        if x_pass is not None:
            return torch.cat([rotated, x_pass], dim=-1)
        return rotated
