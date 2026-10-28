# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""GRPO RL for the Triton track — kernels as the rollout/forward backbone.

Same GRPO math and loop shape as ``impl/_torch/rl.py`` (which documents
the objective; ``impl/_np/rl.py`` holds the derivations). The difference
is where the matmuls land: ``TritonModel`` runs attention/FFN through
Triton kernels, the backward flows through the kernels' autograd
wrappers, and rollouts use ``shared.generator.TextGenerator`` over the
Triton model's own forward_step path.

Usage::

    trainer = TritonGRPOTrainer(model, reference, lr=1e-4)
    stats = trainer.train_step(batch_ids, response_mask, advantages)
    # batch_ids (B, S) int64; response_mask (B, S) int64; advantages (B,)
"""

from __future__ import annotations

import logging

import torch

from impl._torch.rl import TorchGRPOTrainer

logger = logging.getLogger(__name__)


class TritonGRPOTrainer(TorchGRPOTrainer):
    """Triton-backed GRPO: Triton kernels for forward, autograd backward.

    Inherits the whole GRPO loop from the torch trainer (the SFT
    convention — one trainer family, model injected); only the model
    type and default device differ.
    """

    IGNORE_INDEX = -100

    def __init__(self, model: torch.nn.Module, reference: torch.nn.Module, device: str = "cuda", **kwargs) -> None:
        # The torch trainer's __init__ moves models and builds the
        # optimizer/sampler; TritonModel is an nn.Module, so everything
        # applies as-is. Explicit override only for the type signature.
        super().__init__(model, reference, device=device, **kwargs)
