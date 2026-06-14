from __future__ import annotations

import numpy as np


def compute_theta(positions: np.ndarray, dim: int, base: float = 10000.0) -> np.ndarray:
    """
    Compute rotation angles for RoPE.

    Formula (from the original RoPE paper, Su et al. 2021):
    theta(pos, i) = pos * base^(-2i/d)

    where d is the embedding dimension and i indexes the dimension pair.

    Dimension tracking:
    - Input positions: [Num_Positons] (e.g., [1, 2, 3, ...] for absolute positions)
    - Output theta: [Num_Positons, Dim//2] (rotation angle per dimension pair)

    >>> pos = np.array([0, 1, 2], dtype=np.float64)
    >>> theta = compute_theta(pos, 8)
    >>> theta.shape
    (3, 4)
    >>> # Position 0 has no rotation
    >>> np.allclose(theta[0], 0)
    True
    """
    # Number of dimension pairs
    pair_count = dim // 2

    # div_term for each dimension pair (unchanging across positions):
    # [Dim//2] = [base^(-2*0/d), base^(-2*1/d), ..., base^(-2*(pair_count-1)/d)]
    power = np.arange(pair_count, dtype=np.float64) * -2.0 / dim
    div_term = base**power

    # theta[pos, i] = position * div_term[i]
    # shapes: [Num_Positons, 1] * [Dim//2] → [Num_Positons, Dim//2]
    theta = positions.reshape(-1, 1) * div_term

    return theta


def reverse_rope(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    r"""
    Reverse RoPE — applies the inverse rotation.

    Since rotation matrix R(\theta) is orthogonal, R^{-1} = R(-\theta) = R^T.
    Replace \sin with -\sin in the forward rotation.

    Same I/O shapes as apply_rope.
    """
    cos = np.cos(theta)
    sin = np.sin(theta)

    if x.ndim == 3:
        cos = cos[None, :, :]
        sin = sin[None, :, :]
    else:  # 4D
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Inverse: swap -/ + in the rotation
    x_rope_even = cos * x_even + sin * x_odd
    x_rope_odd = -sin * x_even + cos * x_odd

    x_rope = np.empty_like(x)
    x_rope[..., 0::2] = x_rope_even
    x_rope[..., 1::2] = x_rope_odd

    return x_rope


def apply_rope(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    r"""
    Apply RoPE to a tensor of shape ``[Batch, Seq_Len, Embed_Dim]`` or
    ``[Batch, Seq_Len, Head_Len, Head_Dim]``.

    For each position *l* at absolute position *m*, rotate pairs of dimensions:

    .. math::
       \begin{bmatrix} x'(2i) \\ x'(2i+1) \end{bmatrix}
       = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
         \begin{bmatrix} x(2i) \\ x(2i+1) \end{bmatrix}

    where ``theta[l, i]`` is the rotation angle for position *l*, pair *i*.

    Dimension tracking (:math:`n` = tensor dim count):

    ====================  =========================  =========================
    Symbol                Input (x)                  Input (theta) / Output
    ====================  =========================  =========================
    ``x``                 | ``[B, L, D]``            | ``[B, L, D]``
                          | ``[B, L, H, D]``         | ``[B, L, H, D]``
    ``theta``             | ``[L, D//2]``            | used internally
    ====================  =========================  =========================

    >>> x = np.array([[[0., 1., 0., 1.]]], dtype=np.float64)  # [1, 1, 4]
    >>> theta = np.array([[0., 0.]], dtype=np.float64)        # [1, 2] at pos 0
    >>> result = apply_rope(x, theta)
    >>> np.allclose(result, x)
    True
    """
    # x: [B, L, D] or [B, L, H, D]
    # theta: [L, D//2]
    # Extract cos and sin from theta, add dims for correct broadcasting:
    # [L, D//2] → [L, 1, D//2] for 3D x, [L, 1, 1, D//2] for 4D x
    cos = np.cos(theta)
    sin = np.sin(theta)

    if x.ndim == 3:
        # [L, D//2] → [1, L, D//2]
        cos = cos[None, :, :]
        sin = sin[None, :, :]
    else:  # 4D
        # [L, D//2] → [1, L, 1, D//2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

    # Split even and odd dimensions
    x_even = x[..., 0::2]  # [B, L, D//2] or [B, L, H, D//2]
    x_odd = x[..., 1::2]  # [B, L, D//2] or [B, L, H, D//2]

    # Apply rotation — (1, L, 1, D//2) broadcasts correctly over x_even/odd
    # [B, L, H, D//2] × [1, L, 1, D//2] → element-wise
    x_rope_even = cos * x_even - sin * x_odd
    x_rope_odd = sin * x_even + cos * x_odd

    # Reconstruct: interleave even and odd back
    x_rope = np.empty_like(x)
    x_rope[..., 0::2] = x_rope_even
    x_rope[..., 1::2] = x_rope_odd

    return x_rope
