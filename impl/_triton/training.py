"""Training loop utilities for Triton implementation.

Provides train_step, compute_gradient_norm, and clip_gradients for
use with the Triton decoder-only transformer. Since TritonModel uses
PyTorch tensors and autograd (via autograd.Function wrappers in the
Triton kernels), the training loop is identical to the PyTorch backend.

Architecture
-------------
Training loop:
    logits     = model(batch_input)        # (B, S, V)
    loss       = loss_fn(logits, tgt)      # scalar
    loss.backward()                        # autograd through Triton kernels
    clip_gradients(grads, max_norm)        # gradient clipping
    optimizer.step(params, grads)          # updates params in-place
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def clip_gradients(grads: dict[str, torch.Tensor], max_norm: float) -> None:
    """Clip gradients by global L2 norm, modifying the dict in-place.

    Logs the pre-clip norm and whether clipping was triggered.

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
    logger.info("clip_gradients() clipping global_norm=%.6f -> max_norm=%.4f factor=%.6f", global_norm, max_norm, max_norm / global_norm)
    scaling_factor = max_norm / global_norm
    for grad in grads.values():
        grad *= scaling_factor  # in-place scalar multiplication


def compute_gradient_norm(grads: dict[str, torch.Tensor]) -> float:
    """Compute the global L2 norm of all gradient tensors.

    Logs the per-tensor contribution when DEBUG level is enabled.

    Returns
    -------
    norm : float
        Scalar global L2 norm.  Returns 0.0 if all gradients are zero.
    """
    total_sq_norm = 0.0
    for i, grad in enumerate(grads.values()):
        sq = float(torch.sum(grad**2))
        total_sq_norm += sq
        logger.debug("compute_gradient_norm() tensor[%d] sq_norm=%.6f running_total=%.6f", i, sq, total_sq_norm)
    return float(torch.sqrt(torch.tensor(total_sq_norm, dtype=torch.float64)))


def train_step(
    model: nn.Module,
    batch_input: torch.Tensor,
    batch_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    max_norm: float = 1.0,
) -> float:
    """Execute one training step with PyTorch autograd through Triton kernels.

    Full forward -> backward -> clip -> optimizer cycle:

        1. Forward pass: model(batch_input) -> logits (B, S, V)
        2. Loss computation: loss_fn(logits, batch_target) -> scalar float
        3. Backward pass: loss.backward() -> autograd through Triton
        4. Gradient clipping: clip_gradients(grads, max_norm)
        5. Optimizer step: optimizer.step() -> modifies params in-place
        6. Clear gradients: optimizer.zero_grad()
        7. Return loss value

    Parameters
    ----------
    model : nn.Module
        The model to train (works with TorchModel or TritonModel).
    batch_input : torch.Tensor, shape (B, S)
        Input tokens.
    batch_target : torch.Tensor, shape (B, S)
        Target tokens for cross-entropy.
    optimizer : torch.optim.Optimizer
        An optimizer with a .step() method.
    loss_fn : nn.Module
        A callable that takes (flat_logits, flat_targets) and returns scalar loss.
    max_norm : float, default 1.0
        Maximum allowed L2 norm for gradient clipping.  Pass 0.0 to disable.

    Returns
    -------
    loss : float
        The scalar loss value for this batch.
    """
    # 1. Forward pass — output shape (B, S, V) for vocab prediction
    logger.debug("train_step() forward batch_input=%s", list(batch_input.shape))
    logits = model(batch_input)
    logger.debug("train_step() forward complete logits=%s", list(logits.shape))

    # 2. Reshape logits/targets for cross-entropy: (B*S, V) & (B*S,)
    #    Each token position is treated independently as a classification
    #    problem over the vocabulary.
    logits_flat = logits.reshape(-1, logits.shape[-1])  # (B*S, V)
    target_flat = batch_target.reshape(-1)  # (B*S,)
    logger.debug("train_step() reshape logits_flat=%s target_flat=%s", list(logits_flat.shape), list(target_flat.shape))

    # 3. Compute loss — cross-entropy over all token positions
    loss = loss_fn(logits_flat, target_flat)
    logger.info("train_step() loss=%.6f", float(loss))

    # 3. Backward pass: autograd computes all gradients through Triton kernels
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

    # 5. Optimizer step: apply gradients
    logger.debug("train_step() optimizer_step n_params=%d", len(grads))
    optimizer.step()

    # 6. Clear gradients for next step
    optimizer.zero_grad()

    return loss.item()
