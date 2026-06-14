from __future__ import annotations

import torch
import torch.nn as nn


def compute_theta(positions: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Compute rotation angles for RoPE (PyTorch version).

    Formula (from the original RoPE paper, Su et al. 2021):
    theta(pos, i) = pos * base^(-2i/d)

    Dimension tracking:
    - Input positions: [Num_Positions] (e.g., [1, 2, 3, ...] for absolute positions)
    - Output theta: [Num_Positions, Dim//2] (rotation angle per dimension pair)
    """
    pair_count = dim // 2
    power = torch.arange(pair_count, dtype=positions.dtype, device=positions.device) * -2.0 / dim
    div_term = base ** power

    theta = positions.unsqueeze(-1) * div_term.unsqueeze(0)
    return theta


def apply_rope(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to a tensor of shape [Batch, Seq_Len, Embed_Dim] or
    [Batch, Num_Heads, Seq_Len, Head_Dim].

    For each position *l* at absolute position *m*, rotate pairs of dimensions:

    .. math::
       \begin{bmatrix} x'(2i) \\ x'(2i+1) \end{bmatrix}
       = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
         \begin{bmatrix} x(2i) \\ x(2i+1) \end{bmatrix}

    Same shape handling as NumPy: theta is [L, D//2], broadcast across
    batch and heads (if 4D).
    """
    cos = torch.cos(theta)  # [L, D//2]
    sin = torch.sin(theta)  # [L, D//2]

    # Expand dims for correct broadcasting
    if x.ndim == 3:
        # [B, L, D] — cos/sin: [1, L, 1, D//2] → [1, L, D//2]
        cos = cos.unsqueeze(0)        # [1, L, D//2]
        sin = sin.unsqueeze(0)
    else:  # 4D: [B, H, L, D]
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, L, D//2]
        sin = sin.unsqueeze(0).unsqueeze(1)

    x_even = x[..., 0::2]  # [B, L, D//2] or [B, H, L, D//2]
    x_odd = x[..., 1::2]   # [B, L, D//2] or [B, H, L, D//2]

    x_rope_even = cos * x_even - sin * x_odd
    x_rope_odd = sin * x_even + cos * x_odd

    x_rope = torch.empty_like(x)
    x_rope[..., 0::2] = x_rope_even
    x_rope[..., 1::2] = x_rope_odd

    return x_rope


def reverse_rope(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Reverse RoPE — inverse rotation. R^{-1} = R(-theta).
    Swaps -sin <-> +sin. Returns same shape as input.
    """
    cos = torch.cos(theta)  # [L, D//2]
    sin = torch.sin(theta)  # [L, D//2]

    if x.ndim == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    else:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Inverse: swap signs
    x_rope_even = cos * x_even + sin * x_odd
    x_rope_odd = -sin * x_even + cos * x_odd

    x_rope = torch.empty_like(x)
    x_rope[..., 0::2] = x_rope_even
    x_rope[..., 1::2] = x_rope_odd

    return x_rope
