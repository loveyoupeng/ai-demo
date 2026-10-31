# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""NumPy training-loop helpers: one train step + gradient clipping.

The single-batch step (forward → loss → analytic backward → clip →
optimizer step) is what every NumPy entry point runs per batch
(``scripts/train.py``, ``scripts/train_real_tinystories.py``,
``scripts/verify_equivalence.py``); GRPO (``impl/_np/rl.py``) reuses
``clip_gradients`` on its dlogits-driven backward.
"""

from __future__ import annotations

import logging

import numpy as np

from impl._np.cross_entropy import CrossEntropyLoss
from impl._np.model import NumPyModel
from impl._np.optimizer import AdamW

logger = logging.getLogger(__name__)


def compute_gradient_norm(grads: dict[str, np.ndarray]) -> float:
    """Global L2 norm of all gradient tensors concatenated (scalar float)."""
    total = 0.0
    for grad in grads.values():
        total += float(np.sum(grad.astype(np.float64) ** 2))
    return float(np.sqrt(total))


def clip_gradients(grads: dict[str, np.ndarray], max_norm: float) -> None:
    """Clip gradients by global L2 norm, modifying the dict in-place.

    When the global norm exceeds ``max_norm``, every gradient tensor is
    scaled by the same factor ``max_norm / global_norm``. ``max_norm <= 0``
    disables clipping.
    """
    if max_norm <= 0.0:
        return
    global_norm = compute_gradient_norm(grads)
    if global_norm <= max_norm:
        return
    scaling_factor = max_norm / global_norm
    for grad in grads.values():
        grad *= scaling_factor  # in-place scalar multiplication


def train_step(
    model: NumPyModel,
    batch_input: np.ndarray,
    batch_target: np.ndarray,
    loss_fn: CrossEntropyLoss,
    optimizer: AdamW,
    max_norm: float = 1.0,
) -> float:
    """Execute one training step on a single batch; return the scalar loss.

    Forward → loss → analytic backward → clip → optimizer step.
    """
    logits = model.forward(batch_input)  # (B, S, V)
    loss = loss_fn.forward(logits, batch_target)  # scalar float
    grads = model.backward(batch_input, batch_target)
    clip_gradients(grads, max_norm=max_norm)
    optimizer.step(model.get_all_parameters(), grads)
    return float(loss)
