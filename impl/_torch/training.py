"""Training loop utilities for PyTorch implementation.

Provides train_step, compute_gradient_norm, and clip_gradients for
use with the PyTorch decoder-only transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def clip_gradients(grads: dict[str, torch.Tensor], max_norm: float) -> None:
    """Clip gradients by global L2 norm, modifying the dict in-place.

    Parameters
    ----------
    grads : dict[str, torch.Tensor]
        Gradient dictionary — values are mutated in-place.
    max_norm : float
        Maximum allowed L2 norm.  If 0.0, no clipping is performed.
    """
    if max_norm <= 0.0:
        return
    global_norm = float(compute_gradient_norm(grads))
    if global_norm <= max_norm:
        return
    scaling_factor = max_norm / global_norm
    for grad in grads.values():
        grad *= scaling_factor  # in-place scalar multiplication


def compute_gradient_norm(grads: dict[str, torch.Tensor]) -> float:
    """Compute the global L2 norm of all gradient tensors.

    Returns
    -------
    norm : float
        Scalar global L2 norm.  Returns 0.0 if all gradients are zero.
    """
    total_sq_norm = 0.0
    for grad in grads.values():
        total_sq_norm += float(torch.sum(grad**2))
    return float(torch.sqrt(torch.tensor(total_sq_norm, dtype=torch.float64)))


def train_step(
    model: nn.Module,
    batch_input: torch.Tensor,
    batch_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Execute one training step with PyTorch autograd.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    batch_input : torch.Tensor
        Input tokens, shape (B, S).
    batch_target : torch.Tensor
        Target tokens, shape (B, S).
    optimizer : torch.optim.Optimizer
        An optimizer with a `.step()` method.
    loss_fn : nn.Module or callable
        A callable that takes (logits, targets) and returns scalar loss.

    Returns
    -------
    loss : float
        The scalar loss value for this batch.
    """
    # 1. Forward pass
    logits = model(batch_input)

    # 2. Compute loss
    loss = loss_fn(logits, batch_target)

    # 3. Backward pass: autograd computes all gradients
    loss.backward()

    # 4. Optimizer step: apply gradients
    optimizer.step()

    # 5. Clear gradients for next step
    optimizer.zero_grad()

    return loss.item()
