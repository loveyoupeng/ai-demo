# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Shared numeric helpers for the torch-family tracks (torch / triton / cuda).

The torch equivalent of ``impl/_np/softmax.py``: one stable softmax,
used by every torch-family site that turns scores into a probability
distribution (the display-path attention weights, the MoE router in the
torch/triton/cuda models and learning records, the drafters' head
distribution, the learning page's samplers). NumPy's reference lives in
``impl/_np/softmax.py``; the two are the same formula, so cross-backend
records stay bit-comparable.
"""

from __future__ import annotations

import numpy as np
import torch


def stable_softmax_t(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax along ``dim`` (torch tensors).

    p = exp(x - max(x)) / sum(exp(x - max(x))) — the max subtraction is
    shift-invariant but keeps exp() from overflowing. ``-inf`` entries
    (causal masks) map to exactly 0.
    """
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def stable_softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis`` (NumPy arrays) — the
    torch-family learning adapters sample over NumPy copies, so they need
    the NumPy twin of :func:`stable_softmax_t`."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
