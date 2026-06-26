"""Training loop utilities for PyTorch implementation.

Provides train_step, compute_gradient_norm, and clip_gradients for
use with the PyTorch decoder-only transformer.
"""

from __future__ import annotations

import logging
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
        logger.debug("clip_gradients() max_norm=0.0 skipping")
        return
    global_norm = float(compute_gradient_norm(grads))
    logger.debug("clip_gradients() pre_clip_norm=%.6f max_norm=%.4f", global_norm, max_norm)
    if global_norm <= max_norm:
        logger.debug("clip_gradients() norm %.6f <= max_norm %.4f skipping", global_norm, max_norm)
        return
    logger.info("clip_gradients() clipping global_norm=%.6f → max_norm=%.4f factor=%.6f", global_norm, max_norm, max_norm / global_norm)
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


def _log_grad_stats(grads: dict[str, torch.Tensor]) -> None:
    """Log per-layer gradient L2 norms for debugging vanishing/exploding gradients.

    Extracts layer index from keys matching the ``*.layers.<N>.*`` pattern
    (e.g. ``stack.layers.0.mha.Wq.weight``).

    Parameters
    ----------
    grads : dict[str, torch.Tensor]
        Gradient dictionary from ``model.named_parameters()``.
    """
    pattern = re.compile(r"\.layers\.(\d+)\.")
    layer_norms: dict[int, float] = {}
    for name, grad in grads.items():
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx not in layer_norms:
                layer_norms[layer_idx] = 0.0
            layer_norms[layer_idx] += float(torch.sum(grad ** 2))
    for layer_idx in layer_norms:
        layer_norms[layer_idx] = float(torch.sqrt(torch.tensor(layer_norms[layer_idx], dtype=torch.float64)))
    if not layer_norms:
        return
    max_layer = max(layer_norms)
    parts = [f"layer{i}={layer_norms[i]:.4f}" for i in range(max_layer + 1)]
    avg = float(torch.mean(torch.tensor(list(layer_norms.values()), dtype=torch.float64)).item())
    logger.debug("grad_stats() %s avg=%.4f", " ".join(parts), avg)


def train_step(
    model: nn.Module,
    batch_input: torch.Tensor,
    batch_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    max_norm: float = 1.0,
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
    max_norm : float, default 1.0
        Maximum allowed L2 norm for gradient clipping.  Pass 0.0 to
        disable clipping entirely.

    Returns
    -------
    loss : float
        The scalar loss value for this batch.

    """
    # 1. Forward pass
    logger.debug("train_step() forward batch_input=%s", list(batch_input.shape))
    logits = model(batch_input)
    logger.debug("train_step() forward complete logits=%s", list(logits.shape))

    # 2. Compute loss — reshape logits (B, S, V) → (B*S, V) and
    #    targets (B, S) → (B*S) for CrossEntropyLoss compatibility
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = batch_target.reshape(-1)
    loss = loss_fn(flat_logits, flat_target)
    logger.info("train_step() loss=%.6f", float(loss))

    # 3. Backward pass: autograd computes all gradients
    loss.backward()
    logger.debug("train_step() backward complete")

    # 4. Clip gradients to stabilize training (especially with Post-Norm)
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad
    logger.debug("train_step() gradient_clip n_params=%d max_norm=%.4f", len(grads), max_norm)
    clip_gradients(grads, max_norm=max_norm)
    grad_norm = compute_gradient_norm(grads)
    logger.debug("train_step() post_clip_grad_norm=%.6f", grad_norm)
    _log_grad_stats(grads)

    # 5. Optimizer step: apply gradients
    logger.debug("train_step() optimizer_step n_params=%d", len(grads))
    optimizer.step()

    # 6. Clear gradients for next step
    optimizer.zero_grad()

    return loss.item()
