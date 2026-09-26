"""SFT trainer for the Triton track — kernels as the compute backbone.

Same training skeleton as the torch track; the difference is where the
matmuls land. ``TritonModel`` runs its attention/FFN through Triton kernels;
the backward recomputes through the autograd-friendly PyTorch reference
(each Triton kernel's ``autograd.Function`` wraps a ``backward`` that calls
``F.scaled_dot_product_attention``), so one ``loss.backward()`` flows
through them.

Usage::

    trainer = TritonSFTTrainer(model, lr=1e-3)
    loss = trainer.train_step(batch)  # batch: input_ids, target_ids, mask
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from impl._triton.model import TritonModel


class TritonSFTTrainer:
    """Triton-backed SFT. The ``forward`` runs on Triton kernels,
    the backward through the autograd-friendly PyTorch reference.
    """

    IGNORE_INDEX = -100

    def __init__(self, model: TritonModel, device: str = "cuda", lr: float = 1e-3, weight_decay: float = 0.01) -> None:
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor, response_mask: torch.Tensor) -> float:
        """One step on (B, S) tokens: masked CE on the response."""
        self.model.train()
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=self.IGNORE_INDEX,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return float(loss.item())
