import numpy as np
from typing import Tuple

class CrossEntropyLoss:
    """
    Cross-Entropy Loss for language modeling.
    """

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Args:
            logits: [Batch, Seq_Len, Vocab_Size]
            targets: [Batch, Seq_Len] integer token IDs
        Returns:
            loss: scalar float
            grad_logits: [Batch, Seq_Len, Vocab_Size] gradient of loss w.r.t. logits
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 1. Softmax for probabilities
        # Subtract max for numerical stability
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # 2. Calculate loss
        # We need the probability of the target tokens
        # targets is [Batch, Seq_Len]
        # We use advanced indexing to pick the right probs
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        
        # target_probs shape: [Batch, Seq_Len]
        target_probs = probs[batch_indices, seq_indices, targets]
        
        # Avoid log(0)
        loss = -np.mean(np.log(target_probs + 1e-12))
        
        # 3. Calculate gradient w.r.t. logits
        # dL/dz_j = (P_j - 1(j == target)) / (Batch * Seq_Len)
        grad_logits = probs.copy()
        
        # Subtract 1 from the target positions
        # Create a mask for targets
        target_mask = np.zeros_like(probs)
        target_mask[batch_indices, seq_indices, targets] = 1.0
        
        grad_logits = (grad_logits - target_mask) / (batch_size * seq_len)
        
        return float(loss), grad_logits
