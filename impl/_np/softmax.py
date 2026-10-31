# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Small shared numeric helpers for the NumPy reference track.

One stable softmax, used by every operator that turns scores into a
probability distribution (attention weights, the MoE router, the
LM-head distribution on the learning page, the spec engine's samplers).
Keeping it in ONE place means the "subtract the row max before exp"
trick is written (and taught) once.
"""

from __future__ import annotations

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``.

    p = exp(x - max(x)) / sum(exp(x - max(x))) — the max subtraction is
    mathematically a no-op (softmax is shift-invariant) but keeps exp()
    from overflowing on large logits. ``-inf`` entries (causal mask)
    map to exactly 0.

    x: any shape → same shape, rows summing to 1 along ``axis``.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
