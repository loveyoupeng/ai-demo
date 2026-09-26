"""SFT trainer for the PyTorch track — the production idiom.

The NumPy track's `impl/_np/sft.py` is the teaching baseline (same formula,
plain math). This class is the framework implementation: no hand-rolled loss
masking arithmetic (``F.cross_entropy(..., ignore_index=-100)`` handles it),
no explicit backward graph (``loss.backward()`` does it), no hand-rolled
optimizer (``torch.optim.AdamW`` handles the moments), no manual parameter
walk (autograd accumulates on Parameters).

Same masking contract as the NumPy track: `targets[b, s] == -100` at prompt
positions, the true next-token ID at response positions, and the loop is
benchmark-identical to Pre-Training (same model, same optimizer) — the only
difference is the data.

Usage::

    trainer = TorchSFTTrainer(model, device="cpu", lr=1e-3)
    loss = trainer.train_step(batch)          # one optimizer step
    # batch: {"input_ids": Tensor, "target_ids": Tensor, "mask": Tensor}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from impl._torch.layers import TorchModel


class TorchSFTTrainer:
    """The PyTorch SFT trainer.

    ``step`` is a standard PyTorch training step — zero_grad, forward,
    masked cross-entropy, backward, AdamW step — but the *loss masking* is
    handled by the framework (``ignore_index``), not the model. The NumPy
    track's training loop makes the masking explicit in math; this class
    makes the contract explicit in code.
    """

    # The dataset starts with the prompt; response costs are masked. The
    # "baseline" (pre-training) treats every position as a label. SFT sets
    # them to this constant.
    IGNORE_INDEX = -100

    def __init__(self, model: TorchModel, device: str = "cpu", lr: float = 1e-3, weight_decay: float = 0.01) -> None:
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor, response_mask: torch.Tensor) -> float:
        """One SFT training step on one batch.

        input_ids  (B, S) int64   — prompt + response
        target_ids (B, S) int64   — next-token targets (IGNORE_INDEX at prompt)
        response_mask (B, S) bool — True at response positions
        """
        self.model.train()
        device = self.device
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        # (B, S, V) logit stream
        logits = self.model(input_ids)

        # The mask *is* the target: ignore_index kills the log rows where
        # target_ids == IGNORE_INDEX, everywhere else Computes & grads apply.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),  # (B*S, V)
            target_ids.reshape(-1),  # (B*S,)
            ignore_index=self.IGNORE_INDEX,
        )

        loss.backward()
        # standard gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        """Average loss over a held-out set (no parameter updates)."""
        self.model.eval()
        losses = []
        for x, y, _mask in batches:
            logits = self.model(x.to(self.device))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.to(self.device).reshape(-1),
                ignore_index=self.IGNORE_INDEX,
            )
            losses.append(float(loss.item()))
        return sum(losses) / len(losses) if losses else 0.0
