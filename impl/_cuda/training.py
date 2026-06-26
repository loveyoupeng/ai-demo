"""Training loop utilities for CUDA-based decoder-only transformer.

Uses PyTorch autograd — CUDA model weights have requires_grad_(True) set,
so loss.backward() computes all gradients through CuTransformerBlock internals.

Architecture
-------------
Training loop:
    for epoch in range(epochs):
        for batch in dataset:
            logits = model.forward(batch_input)       # (B, S, V)
            loss = loss_fn(logits, batch_target)       # scalar
            loss.backward()                            # accumulates gradients
            clip_gradients(grads, max_norm)            # in-place
            optimizer.step()                           # modifies params in-place
            optimizer.zero_grad()                      # clear for next step
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def compute_gradient_norm(grads: dict[str, torch.Tensor]) -> float:
    """Compute the global L2 norm of all gradient tensors.

    Logs the total parameter count and whether clipping was triggered.

    Parameters
    ----------
    grads : dict[str, torch.Tensor]
        Gradient dictionary — keys are parameter names, values are tensors.
        Each tensor is on CUDA with shape matching the corresponding parameter.

    Returns
    -------
    norm : float
        Scalar global L2 norm. Returns 0.0 if all gradients are zero.
    """
    logger.debug("compute_gradient_norm() n_params=%d", len(grads))
    total_sq_norm = 0.0
    for _i, grad in enumerate(grads.values()):
        sq = float(torch.sum(grad ** 2))
        total_sq_norm += sq
    result = float(torch.sqrt(torch.tensor(total_sq_norm, dtype=torch.float64)))
    logger.debug("compute_gradient_norm() total_sq_norm=%.6f final_norm=%.6f", total_sq_norm, result)
    return result


def _log_grad_stats(grads: dict[str, torch.Tensor]) -> None:
    """Log per-layer gradient L2 norms for debugging vanishing/exploding gradients.

    Extracts layer index from keys like ``blocks.0.Wq`` where the second
    dotted segment is the layer number.

    Parameters
    ----------
    grads : dict[str, torch.Tensor]
        Gradient dictionary.
    """
    layer_norms: dict[int, float] = {}
    for key, grad in grads.items():
        parts = key.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            layer_idx = int(parts[1])
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


def clip_gradients(grads: dict[str, torch.Tensor], max_norm: float) -> None:
    """Clip gradients by global L2 norm, modifying the dict in-place.

    Logs the pre-clip norm and whether clipping was triggered.

    Parameters
    ----------
    grads : dict[str, torch.Tensor]
        Gradient dictionary — values are mutated in-place.
    max_norm : float
        Maximum allowed L2 norm. If 0.0, no clipping is performed.
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
        grad *= scaling_factor


def train_step(
    model: Any,
    batch_input: torch.Tensor,
    batch_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    max_norm: float = 1.0,
) -> float:
    r"""Execute one training step with PyTorch autograd on CUDA.

    Full forward -> backward -> optimizer cycle:

        1. Forward pass: model(batch_input) -> logits (B, S, V)
        2. Loss computation: loss_fn(logits, batch_target) -> scalar float
        3. Backward pass: loss.backward() -> accumulates gradients in all params
        4. Clip gradients: clip_gradients(grads, max_norm) — in-place
        5. Optimizer step: optimizer.step() — modifies params in-place
        6. Clear gradients: optimizer.zero_grad() — prepare for next step

    Parameters
    ----------
    model : Any
        The model to train. Must support:
        - ``model(batch_input) -> logits`` (forward pass, B,S -> B,S,V)
        - ``model.parameters()`` or ``named_parameters()`` (param access)
        Works with nn.Module (PyTorch) or plain-tensor models (CUDA) that
        have ``requires_grad_(True)`` on all weight tensors.
    batch_input : torch.Tensor, shape (B, S)
        Input tokens or features, shape (B, S). On CUDA device.
    batch_target : torch.Tensor, shape (B, S)
        Target tokens, shape (B, S). On CUDA device, dtype long for CE.
    optimizer : torch.optim.Optimizer
        An optimizer with .step() and .zero_grad() methods.
        e.g. torch.optim.AdamW, torch.optim.SGD
    loss_fn : Any
        A callable that takes (logits, targets) and returns scalar loss.
        e.g. torch.nn.CrossEntropyLoss()
    max_norm : float, default 1.0
        Maximum allowed L2 norm for gradient clipping. Pass 0.0 to disable.

    Returns
    -------
    loss : float
        The scalar loss value for this batch.

    Shape flow for token-based training:
        Input: tokens (B, S) — int64 token IDs
        Embedding: (B, S, D)
        Stack: (B, S, D)
        Output proj: (B, S, V)
        Reshape for CE: (B*S, V) -> (B*S,)
    """
    # 1. Forward pass — model produces logits
    logger.debug("train_step() forward batch_input=%s device=%s", list(batch_input.shape), batch_input.device)
    logits = model(batch_input)
    logger.debug("train_step() forward complete logits=%s device=%s", list(logits.shape), logits.device)

    # 2. Compute loss — CrossEntropyLoss expects: logits (N, C), targets (N,)
    #    Reshape: (B, S, V) -> (B*S, V) and (B, S) -> (B*S)
    flat_logits = logits.reshape(-1, logits.shape[-1])  # (B*S, V)
    flat_target = batch_target.reshape(-1)  # (B*S)
    logger.debug("train_step() reshape flat_logits=%s flat_target=%s", list(flat_logits.shape), list(flat_target.shape))
    loss = loss_fn(flat_logits, flat_target)
    logger.info("train_step() loss=%.6f", float(loss))

    # 3. Backward pass — autograd computes all gradients
    # For CUDA model: gradients accumulate in model.stacking.blocks[i].Wq.grad etc.
    loss.backward()
    logger.debug("train_step() backward complete")

    # 4. Collect gradients — works for both nn.Module and plain-tensor models
    grads: dict[str, torch.Tensor] = {}
    if hasattr(model, "named_parameters"):
        # PyTorch nn.Module (standard path)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad
    else:
        # CUDA model: iterate through stacking.blocks explicitly
        model_name = type(model).__name__
        logger.debug("train_step() collecting_grads model_type=%s has_stacking=%s", model_name, hasattr(model, "stacking"))
        if hasattr(model, "stacking") and hasattr(model.stacking, "blocks"):
            for i, block in enumerate(model.stacking.blocks):
                for attr_name in ["Wq", "Wk", "Wv", "Wo", "ln1_gamma", "ln2_gamma", "gate1", "gate2", "expert_weights", "expert_bias", "routing_weights"]:
                    attr = getattr(block, attr_name, None)
                    if attr is not None and attr.grad is not None:
                        grads[f"blocks.{i}.{attr_name}"] = attr.grad
        # Also handle model-level parameters (embedding, output_proj, etc.)
        for attr_name in ["embedding_weights", "final_ln_gamma", "output_proj_weights", "output_proj_bias", "output_W1", "output_W2", "output_W3"]:
            attr = getattr(model, attr_name, None)
            if attr is not None and attr.grad is not None:
                grads[attr_name] = attr.grad
    logger.info("train_step() gradient_collect params_with_grad=%d", len(grads))

    # 5. Clip gradients to stabilize training
    logger.debug("train_step() clip n_grads=%d max_norm=%.4f", len(grads), max_norm)
    clip_gradients(grads, max_norm=max_norm)
    grad_norm = compute_gradient_norm(grads)
    logger.debug("train_step() post_clip_grad_norm=%.6f", grad_norm)
    _log_grad_stats(grads)

    # 6. Optimizer step and clear for next batch
    logger.debug("train_step() optimizer_step n_grads=%d", len(grads))
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
