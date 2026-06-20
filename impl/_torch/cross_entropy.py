"""Cross-entropy loss for PyTorch implementation.

Wraps torch.nn.functional.cross_entropy with shift/mask support for
causal language modeling.  This mirrors impl/_np/cross_entropy.py
and is tested separately in tests/unit/_torch/test_cross_entropy.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CrossEntropyLoss:
    """Cross-entropy loss with optional shift and padding mask support.

    Parameters
    ----------
    shift : bool, default True
        Whether to shift logits/targets so that prediction at time step t
        predicts the token at t+1.  This is standard for causal language
        modeling where the model autoregressively predicts the next token.
    ignore_index : int, default -100
        Target index whose loss contribution is ignored (e.g. padding).

    Forward
    -------
    logits : torch.Tensor, shape (batch_size, seq_len, vocab_size)
        Model output logits (unnormalized).
    targets : torch.Tensor, shape (batch_size, seq_len)
        Ground-truth token IDs (integers).
    mask : torch.Tensor, shape (batch_size, seq_len) or None, optional
        Binary mask tensor (1 = include in loss, 0 = ignore).  If None,
        every position contributes equally to the final mean loss.

    """

    def __init__(self, shift: bool = True, ignore_index: int = -100) -> None:
        self.shift = shift
        self.ignore_index = ignore_index

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss over a batch of sequences.

        Parameters
        ----------
        logits : torch.Tensor, shape (B, S, V)
            Predicted logits from the model.  V = vocab_size.
        targets : torch.Tensor, shape (B, S)
            Ground-truth token IDs.
        mask : torch.Tensor, shape (B, S) or None, optional
            Binary mask (1 = included, 0 = excluded from the mean).

        Returns
        -------
        loss : torch.Tensor
            Scalar mean loss, dtype float64, over all non-masked and
            non-ignored positions.  Returns 0.0 if no positions remain
            (all masked or ignored).

        """
        # --- 1. Optional autoregressive shift ---
        # In causal LM the model at position t predicts the token at t+1.
        # After shifting:
        #   logits: (B, S-1, V)  -> logits[:, :-1]
        #   targets: (B, S-1)     -> targets[:, 1:]
        if self.shift:
            logits = logits[:, :-1]
            targets = targets[:, 1:]
            if mask is not None:
                mask = mask[:, 1:]

        # --- 2. Cast to float64 for numerical stability and parity ---
        logits = logits.to(dtype=torch.float64)
        targets = targets.to(dtype=torch.int64)

        # --- 3. Per-position loss via F.cross_entropy ---
        # Reorder dimensions: (B, S, V) -> (B*S, V), (B, S) -> (B*S,)
        B, S, V = logits.shape
        flat_logits = logits.reshape(B * S, V)
        flat_targets = targets.reshape(B * S)

        # F.cross_entropy expects (N, C) logits and (N,) targets
        # ignore_index skips target positions with the given label
        per_pos_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.ignore_index, reduction="none")

        # Reshape back to (B, S)
        loss_2d = per_pos_loss.reshape(B, S)

        # --- 4. Apply mask ---
        # valid = masked AND not-ignored (F.cross_entropy already zeroed ignored)
        # --- 5. Build valid_count mask (non-ignored AND optionally masked) ---
        if mask is not None:
            mask = mask.to(dtype=torch.float64)
            valid_count_mask = mask
        else:
            valid_count_mask = (targets != self.ignore_index).to(dtype=torch.float64)

        valid_count = valid_count_mask.sum().item()

        if valid_count == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=logits.device)

        masked_loss = loss_2d * valid_count_mask
        total_loss = masked_loss.sum().item()

        return torch.tensor(total_loss / valid_count, dtype=torch.float64, device=logits.device)
