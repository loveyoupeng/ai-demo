"""Rotary Position Embedding for the NumPy reference implementation.

RoPE (Su et al., 2021, "RoFormer") with the analytic backward. RoPE has no
learned parameters, so its backward is a pure input gradient.

**Intuition:** attention has no way to tell "token 3" from "token 30"
without position. RoPE rotates each (x_m, x_{m+1}) pair by an angle
p·θ_m — where p is the absolute position and θ_m = 10000^(-2m/d) is the
pair's rotation frequency. High-frequency pairs (small m) spin fast
position by position; low-frequency pairs (large m) spin slowly across
the whole context. Absolute position becomes a smooth, multi-scale
rotation instead of a rigid offset, and the q·k inner product ends up
depending only on the relative distance between two tokens.
"""

from __future__ import annotations

import numpy as np


class RoPE:
    """Rotary Position Embedding (Su et al., 2021, "RoFormer").

    Position is injected by *rotating* pairs of dimensions of q and k. For
    dimension pair (m, m+1) at position pos and pair index k:

        angle = pos * theta_k,   theta_k = 10000 ** (-2k / D)
        y_m    = x_m   * cos(angle) - x_{m+1} * sin(angle)
        y_{m+1} = x_m   * sin(angle) + x_{m+1} * cos(angle)

    A 2x2 rotation is a similarity transform: it preserves vector lengths and
    changes the inner product q·k in a way that depends only on the *relative*
    position (pos_q - pos_k). That is the property transformers need —
    attention can then "see" how far apart two tokens are.

    ``rope_dim`` selects which dimensions rotate:
        rope_dim == 0            → rotate all D dimensions (standard)
        0 < rope_dim < D         → rotate the first rope_dim dims, the rest
                                    pass through unchanged (NTK-style partial)
    rope_dim must be even (rotation happens in pairs).

    Forward
    -------
    x : (B, S, H, D)   (query or key tensor, any leading dims: (..., H, D))
    positions : (S,) int array (or (B, S)); the token index of each position
    Returns: (B, S, H, D), same shape as x.

    Step shapes:
        freqs   : (D//2,)                     — one theta per pair
        angles  : (B, S, D//2)                — pos * theta per pair
        cos/sin : (B, S, D//2)
        x_even  : (B, S, H, D//2), x_odd: (B, S, H, D//2)
        output  : re-assembled to (B, S, H, D)

    Backward
    --------
    Each 2x2 rotation matrix R = [[c, -s], [s, c]] is orthonormal, so its
    inverse is its transpose:

        x_m    =  y_m * cos(angle) + y_{m+1} * sin(angle)
        x_{m+1} = -y_m * sin(angle) + y_{m+1} * cos(angle)

    The pass-through suffix (rope_dim < D) contributes its upstream gradient
    unchanged.
    """

    def forward(self, x: np.ndarray, positions: np.ndarray, rope_dim: int = 0) -> np.ndarray:
        """Apply RoPE to q or k. x: (..., H, D), positions: (S,) or (B, S)."""
        rotated, _state = self._forward_state(x, positions, rope_dim)
        return rotated

    def _forward_state(self, x: np.ndarray, positions: np.ndarray, rope_dim: int = 0) -> tuple[np.ndarray, dict]:
        """Apply RoPE and return the rotation state (the intermediates of forward).

        Returns (rotated, state) where state holds:
            freqs : (D//2,)      — one theta per pair
            angles: (B, S, D//2) — pos * theta per pair
            cos   : (B, S, D//2)
            sin   : (B, S, D//2)
        """
        d = x.shape[-1]

        # Split the rotated prefix from the pass-through suffix.
        if rope_dim > 0 and rope_dim < d:
            x_rot = x[..., :rope_dim]
            x_pass = x[..., rope_dim:]
        else:
            x_rot = x
            x_pass = None
        d_rot = x_rot.shape[-1]
        pair_dim = d_rot // 2  # number of (even, odd) pairs

        batch_size, seq_len = x_rot.shape[0], x_rot.shape[1]
        pos = np.asarray(positions, dtype=np.int32)
        if pos.ndim == 1 and pos.shape[0] == seq_len:
            pos = np.broadcast_to(pos, (batch_size, seq_len))  # (B, S)

        # theta_k = 10000 ** (-2k / d_rot) for k = 0..pair_dim-1  — (pair_dim,)
        freqs = 1.0 / (10000.0 ** (np.arange(pair_dim, dtype=np.float32) * 2.0 / d_rot))
        angles = pos[:, :, np.newaxis] * freqs[np.newaxis, np.newaxis, :]  # (B, S, pair_dim)

        cos = np.cos(angles)  # (B, S, pair_dim)
        sin = np.sin(angles)  # (B, S, pair_dim)

        # Pair up the last dimension: (B, S, H, pair_dim, 2)
        x_flat = x_rot.reshape(x_rot.shape[:-1] + (pair_dim, 2))
        x_even = x_flat[..., 0]  # (B, S, H, pair_dim) — the "x_m" element
        x_odd = x_flat[..., 1]  # (B, S, H, pair_dim) — the "x_{m+1}" element

        # Broadcast cos/sin over the head axis: (B, S, 1, pair_dim)
        cos_b = cos[:, :, np.newaxis, :]
        sin_b = sin[:, :, np.newaxis, :]

        # 2D rotation of each pair (the formula in the docstring).
        y_even = x_even * cos_b - x_odd * sin_b  # (B, S, H, pair_dim)
        y_odd = x_even * sin_b + x_odd * cos_b  # (B, S, H, pair_dim)

        rotated = np.stack([y_even, y_odd], axis=-1).reshape(x_rot.shape)  # (B, S, H, d_rot)

        if x_pass is not None:
            rotated = np.concatenate([rotated, x_pass], axis=-1)  # (B, S, H, D)
        state = {"freqs": freqs, "angles": angles, "cos": cos, "sin": sin}
        return rotated, state

    def backward(self, dout: np.ndarray, x: np.ndarray, positions: np.ndarray, rope_dim: int = 0) -> np.ndarray:
        """Analytic backward (inverse rotations).

        dout: (B, S, H, D) upstream gradient (same shape as forward output).
        x: (B, S, H, D) the forward input (needed only for the split sizes).
        positions, rope_dim: the same arguments forward used.

        Returns: dx (B, S, H, D).
        """
        d = x.shape[-1]
        if rope_dim > 0 and rope_dim < d:
            dy_rot = dout[..., :rope_dim]
            dy_pass = dout[..., rope_dim:]
            x_rot = x[..., :rope_dim]
        else:
            dy_rot = dout
            dy_pass = None
            x_rot = x
        d_rot = x_rot.shape[-1]
        pair_dim = d_rot // 2

        batch_size, seq_len = x_rot.shape[0], x_rot.shape[1]
        pos = np.asarray(positions, dtype=np.int32)
        if pos.ndim == 1 and pos.shape[0] == seq_len:
            pos = np.broadcast_to(pos, (batch_size, seq_len))  # (B, S)

        # Same angles as forward (recomputed — RoPE is stateless).
        freqs = 1.0 / (10000.0 ** (np.arange(pair_dim, dtype=np.float32) * 2.0 / d_rot))
        angles = pos[:, :, np.newaxis] * freqs[np.newaxis, np.newaxis, :]  # (B, S, pair_dim)
        cos_b = np.cos(angles)[:, :, np.newaxis, :]  # (B, S, 1, pair_dim)
        sin_b = np.sin(angles)[:, :, np.newaxis, :]  # (B, S, 1, pair_dim)

        # Inverse (transposed) rotation of each pair — the R^T in the docstring.
        y_flat = dy_rot.reshape(dy_rot.shape[:-1] + (pair_dim, 2))
        y_even = y_flat[..., 0]  # (B, S, H, pair_dim)
        y_odd = y_flat[..., 1]  # (B, S, H, pair_dim)
        x_even = y_even * cos_b + y_odd * sin_b  # (B, S, H, pair_dim)
        x_odd = -y_even * sin_b + y_odd * cos_b  # (B, S, H, pair_dim)

        dx_rot = np.stack([x_even, x_odd], axis=-1).reshape(dy_rot.shape)  # (B, S, H, d_rot)

        if dy_pass is not None:
            return np.concatenate([dx_rot, dy_pass], axis=-1)  # (B, S, H, D)
        return dx_rot
