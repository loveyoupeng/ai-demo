from __future__ import annotations

from typing import Protocol

import numpy as np
from core.base_backend import BaseTransformerBackend
from loss import CrossEntropyLoss


class Optimizer(Protocol):
    """Protocol for optimizer classes with a step method."""

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        pass


class Trainer:
    """
    Trainer class to handle the training loop, loss calculation, and weight updates.
    Works with any backend implementing BaseTransformerBackend interface.
    """

    def __init__(
        self,
        backend: BaseTransformerBackend,
        optimizer: Optimizer,
        loss_fn: CrossEntropyLoss,
    ):
        self.backend = backend
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history: dict[str, list[float]] = {"loss": []}

    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray) -> float:
        """
        Performs a single training step.

        Args:
            input_ids: [Batch, Seq_Len] integer token IDs
            target_ids: [Batch, Seq_Len] integer token IDs (shifted version of input)

        Returns:
            loss: The calculated loss for this step
        """
        logits, cache = self.backend.forward(input_ids)
        loss, grad_logits = self.loss_fn.forward(logits, target_ids)
        grads = self.backend.backward(grad_logits, cache)

        # Get parameters, train, then push changes back to the backend
        params = self.backend.get_params()
        self.optimizer.step(params, grads)
        self.backend.set_params(params)

        return loss

    def fit(self, data_loader: object, epochs: int) -> None:
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (input_ids, target_ids) in enumerate(data_loader):  # type: ignore[operator]
                loss = self.train_step(input_ids, target_ids)
                total_loss += loss
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")

            avg_loss = total_loss / len(data_loader)  # type: ignore[arg-type]
            self.history["loss"].append(avg_loss)
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
