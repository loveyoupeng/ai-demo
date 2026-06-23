"""Training loop utilities for CUDA-based decoder-only transformer.

Uses PyTorch autograd — CUDA model weights have requires_grad_(True) set,
so loss.backward() computes all gradients through CuTransformerBlock internals.

Architecture
------------
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

from typing import Any

import torch


def compute_gradient_norm(grads: dict[str, torch.Tensor]) -> float:
    """Compute the global L2 norm of all gradient tensors.

    Returns the Euclidean norm of the concatenated gradient vectors
    from all parameters: ||g||_2 = sqrt(sum_k ||g_k||_2^2)

    Parameters
    ----------
    grads : dict[str, torch.Tensor]
        Gradient dictionary — keys are parameter names, values are tensors.
        Each tensor is on CUDA with shape matching the corresponding parameter.

    Returns
    -------
    norm : float
        Scalar global L2 norm. Returns 0.0 if all gradients are zero.

    Examples
    --------
    >>> grads = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    >>> norm = compute_gradient_norm(grads)
    >>> abs(norm - 5.4772) < 1e-4
    True
    """
    total_sq_norm = 0.0
    for grad in grads.values():
        total_sq_norm += float(torch.sum(grad ** 2))
    return float(torch.sqrt(torch.tensor(total_sq_norm, dtype=torch.float64)))


def clip_gradients(grads: dict[str, torch.Tensor], max_norm: float) -> None:
    """Clip gradients by global L2 norm, modifying the dict in-place.

    When the global gradient norm exceeds `max_norm`, every gradient tensor
    is scaled by the same factor `max_norm / global_norm` so that the
    overall norm equals `max_norm` (or stays unchanged if already below).

    Parameters
    ----------
    grads : dict[str, torch.Tensor]
        Gradient dictionary — values are mutated in-place.
    max_norm : float
        Maximum allowed L2 norm. If 0.0, no clipping is performed.

    Examples
    --------
    >>> from impl._cuda.training import clip_gradients
    >>> grads = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    >>> clip_gradients(grads, max_norm=1.0)
    >>> norm = compute_gradient_norm(grads)
    >>> abs(norm - 1.0) < 1e-6
    True
    """
    if max_norm <= 0.0:
        return
    global_norm = float(compute_gradient_norm(grads))
    if global_norm <= max_norm:
        return
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
    r"""Execute one training step with PyTorch autograd.

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
    batch_input : torch.Tensor
        Input tokens or features, shape (B, S). On CUDA device.
    batch_target : torch.Tensor
        Target tokens, shape (B, S). On CUDA device, dtype long for CE.
    optimizer : torch.optim.Optimizer
        An optimizer with .step() and .zero_grad() methods.
        e.g. torch.optim.AdamW, torch.optim.SGD
    loss_fn : Any
        A callable that takes (logits, targets) and returns scalar loss.
        e.g. torch.nn.CrossEntropyLoss()
    max_norm : float, default 1.0
        Maximum allowed L2 norm for gradient clipping. Pass 0.0 to
        disable clipping entirely.

    Returns
    -------
    loss : float
        The scalar loss value for this batch.

    Examples
    --------
    >>> import torch, torch.nn as nn
    >>> model = nn.Linear(8, 8)
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    >>> loss_fn = nn.CrossEntropyLoss()
    >>> x = torch.randn(4, 8)
    >>> t = torch.zeros(4, 8, dtype=torch.long)
    >>> loss = train_step(model, x, t, optimizer, loss_fn)
    >>> isinstance(loss, float)
    True

    Notes
    -----
    For CUDA-specific models (e.g. CUDAModel / CuDecoderStack):
    - Weights should be moved to CUDA before training
    - Parameters have requires_grad_(True) set explicitly
    - Gradients accumulate via PyTorch autograd through block.py internals

    Shape flow for token-based training:
        Input: tokens (B, S) — int64 token IDs
        Embedding: (B, S, D)
        Stack: (B, S, D)
        Output proj: (B, S, V)
        Reshape for CE: (B*S, V) -> (B*S,)
    """
    # 1. Forward pass — model produces logits
    # logits shape: (B, S, V) where V = vocab_size
    logits = model(batch_input)

    # 2. Compute loss
    # CrossEntropyLoss expects: logits (N, C), targets (N,)
    # Reshape: (B, S, V) -> (B*S, V) and (B, S) -> (B*S)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = batch_target.reshape(-1)
    loss = loss_fn(flat_logits, flat_target)

    # 3. Backward pass — autograd computes all gradients
    # For CUDA model: gradients accumulate in model.stacking.blocks[i].Wq.grad etc.
    loss.backward()

    # 4. Collect gradients — works for both nn.Module and plain-tensor models
    grads: dict[str, torch.Tensor] = {}
    if hasattr(model, "named_parameters"):
        # PyTorch nn.Module (standard path)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad
    else:
        # CUDA model: iterate through stacking.blocks explicitly
        # model.stacking is a CuDecoderStack with model.stacking.blocks as list
        if hasattr(model, "stacking") and hasattr(model.stacking, "blocks"):
            for i, block in enumerate(model.stacking.blocks):
                for attr_name in [
                    "Wq",
                    "Wk",
                    "Wv",
                    "Wo",
                    "ln1_gamma",
                    "ln2_gamma",
                    "gate1",
                    "gate2",
                    "expert_weights",
                    "expert_bias",
                    "routing_weights",
                ]:
                    attr = getattr(block, attr_name, None)
                    if attr is not None and attr.grad is not None:
                        grads[f"blocks.{i}.{attr_name}"] = attr.grad
        # Also handle model-level parameters (embedding, output_proj, etc.)
        for attr_name in [
            "embedding_weights",
            "final_ln_gamma",
            "output_proj_weights",
            "output_proj_bias",
            "output_W1",
            "output_W2",
            "output_W3",
        ]:
            attr = getattr(model, attr_name, None)
            if attr is not None and attr.grad is not None:
                grads[attr_name] = attr.grad

    # 5. Clip gradients to stabilize training
    clip_gradients(grads, max_norm=max_norm)

    # 6. Optimizer step and clear for next batch
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()