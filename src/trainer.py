import numpy as np
from typing import Any
from model.transformer import Transformer
from loss import CrossEntropyLoss


class Trainer:
    """
    Trainer class to handle the training loop, loss calculation, and weight updates.
    """

    def __init__(
        self,
        model: Transformer,
        optimizer: Any,
        loss_fn: CrossEntropyLoss,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {"loss": []}

    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray) -> float:
        """
        Performs a single training step.

        Args:
            input_ids: [Batch, Seq_Len] integer token IDs
            target_ids: [Batch, Seq_Len] integer token IDs (shifted version of input)

        Returns:
            loss: The calculated loss for this step
        """
        # 1. Forward pass
        logits, cache = self.model.forward(input_ids)

        # 2. Calculate loss (Cross-Entropy)
        loss, grad_logits = self.loss_fn.forward(logits, target_ids)

        # 3. Backward pass
        # This is where we propagate gradients through the whole model.
        grads = self.model.backward(grad_logits, cache)

        # 4. Update weights
        self.optimizer.step(self.model.get_params(), grads)

        return loss

    def fit(self, data_loader: Any, epochs: int):
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (input_ids, target_ids) in enumerate(data_loader):
                loss = self.train_step(input_ids, target_ids)
                total_loss += loss
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")

            avg_loss = total_loss / len(data_loader)
            self.history["loss"].append(avg_loss)
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
