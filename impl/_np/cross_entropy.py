"""Cross-entropy loss and auxiliary training utilities for NumPy implementation.

All forward passes accept NumPy arrays and return NumPy scalars or arrays.
Matrix dimensions are annotated in docstrings for clarity.
"""

from __future__ import annotations

import numpy as np


class CrossEntropyLoss:
    """Cross-entropy loss with optional shift and padding mask support.

    Parameters
    ----------
    shift : bool, default True
        Whether to shift logits/targets so that prediction at time step t
        predicts the token at t+1. This is standard for causal language
        modeling where the model autoregressively predicts the next token.
    ignore_index : int, default -100
        Target index whose loss contribution is ignored (e.g. padding).

    Forward
    -------
    logits : np.ndarray, shape (batch_size, seq_len, vocab_size)
        Model output logits (unnormalized).
    targets : np.ndarray, shape (batch_size, seq_len)
        Target token IDs (integers).
    mask : np.ndarray, shape (batch_size, seq_len) or None, optional
        Binary mask array (1 = include in loss, 0 = ignore). If None,
        every position contributes equally to the final mean loss.

    """

    def __init__(self, shift: bool = True, ignore_index: int = -100) -> None:
        self.shift = shift
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute cross-entropy loss over a batch of sequences.

        The numerically stable formulation uses the log-sum-exp trick:

            log_softmax(x) = x - logsumexp(x)          element-wise
            loss           = -log_softmax(x)[target]    per-position loss

        which simplifies to:

            loss = logsumexp(logits) - logits[target]   scalar per position

        Parameters
        ----------
        logits : np.ndarray, shape (B, S, V)
            Predicted logits from the model. V = vocab_size.
        targets : np.ndarray, shape (B, S)
            Ground-truth token IDs.
        mask : np.ndarray, shape (B, S) or None, optional
            Binary mask (1 = included, 0 = excluded from the mean).

        Returns
        -------
        loss : np.ndarray
            Scalar mean loss, dtype float64, over all non-masked and
            non-ignored positions. Returns 0.0 if no positions remain
            (all masked or ignored).

        """
        # --- 1. Optional autoregressive shift  ---
        # In causal LM the model at position t predicts the token at t+1.
        # After shifting:
        #   logits: (B, S-1, V)  -> logits[:, :-1]
        #   targets: (B, S-1)     -> targets[:, 1:]
        if self.shift:
            logits = logits[:, :-1]  # (B, S-1, V)
            targets = targets[:, 1:]  # (B, S-1)
            if mask is not None:
                mask = mask[:, 1:]  # (B, S-1)

        # --- 2. Per-position log-sum-exp (numerical stability)  ---
        # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))  — per row
        # This prevents overflow when computing softmax via exponentials.
        max_logit = np.max(logits, axis=-1, keepdims=True)  # (B, S, 1)
        shift_logits = logits - max_logit  # (B, S, V)  numerically shifted, max is 0 in exp

        logsumexp = np.log(np.sum(np.exp(shift_logits), axis=-1, keepdims=True))  # (B, S, 1)

        # log_softmax(logits)[target] for each (batch, position)
        # Gather shift_logits[target] where shift_logits = logits - max  (numerically stable)
        # loss = logsumexp(shift) - shift_logits[target]  equals  logsumexp(logits) - logits[target]
        # Guard the gather: ignored targets may hold out-of-range sentinel
        # values (e.g. -100); those positions are zeroed in step 3 anyway.
        in_range = (targets >= 0) & (targets < logits.shape[-1])
        target_index = np.where(in_range[..., np.newaxis], targets[..., np.newaxis], 0)  # (B, S, 1)
        target_logit = np.take_along_axis(shift_logits, target_index, axis=-1)  # (B, S, 1)

        # Per-position loss: logsumexp - logit[target]  >= 0
        loss = logsumexp - target_logit  # (B, S, 1)  float64

        # --- 3. Apply ignore_index (skip padding / control tokens)  ---
        ignore_mask = targets == self.ignore_index  # (B, S) bool
        ignore_mask_3d = ignore_mask[..., np.newaxis]  # (B, S, 1)  broadcast-compatible
        loss = np.where(ignore_mask_3d, 0.0, loss)  # (B, S, 1)  zeros at ignored positions

        # --- 4. Apply mask ---
        # valid = masked AND not-ignored.  When mask=None, everything (non-ignored) is valid.
        valid_mask = (mask == 1.0) & ~ignore_mask if mask is not None else ~ignore_mask  # (B, S)

        # --- 5. Mean over valid positions  ---
        valid_count = np.sum(valid_mask)  # scalar > 0
        if valid_count == 0:
            return np.array(0.0, dtype=np.float64)

        if mask is not None:
            # When mask is provided: sum only at masked positions
            m = mask[:, :, np.newaxis]  # (B, S, 1)
            total_loss = float(np.sum(loss * m.astype(np.float64)))  # scalar
        else:
            # No mask: sum all (non-ignored) positions
            total_loss = float(np.sum(loss))  # (B, S, 1) -> scalar

        mean_loss = total_loss / valid_count  # float64 scalar
        return mean_loss

    def backward(self, logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Gradient of the loss w.r.t. the logits (mirrors ``forward`` exactly).

        Per-position derivation: loss_t = logsumexp(L_t) − L_t[T_t], and

            d loss_t / d L_t = softmax(L_t) − onehot(T_t)

        (the logsumexp term cancels: d logsumexp/dL_t = softmax). The mean
        over valid positions divides every position by ``valid_count``;
        ignored (``target == ignore_index``) and masked-out positions get
        exactly zero gradient, matching the zeroed losses in ``forward``.

        With ``shift=True`` the forward used ``logits[:, :-1]`` vs
        ``targets[:, 1:]``; the returned gradient is un-shifted back to the
        original (B, S, V) layout, with the last position at zero.

        Returns: dlogits, same shape and (float) dtype as ``logits``.
        """
        if self.shift:
            shifted_mask = mask[:, 1:] if mask is not None else None
            d_shifted = self._backward_unshifted(logits[:, :-1], targets[:, 1:], shifted_mask)
            full = np.zeros_like(logits, dtype=d_shifted.dtype)
            full[:, :-1] = d_shifted
            return full
        return self._backward_unshifted(logits, targets, mask)

    def _backward_unshifted(self, logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Backward for already-aligned logits/targets (no shift)."""
        # d logsumexp / d logits = softmax — reuse the stable form of forward.
        max_logit = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logit)
        softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (B, S, V)

        # one-hot of the target per position.
        onehot = np.zeros_like(softmax)
        # Guard: ignored positions may carry out-of-range indices (e.g. -100);
        # their gradient is zeroed below anyway, so place them at index 0.
        in_range = (targets >= 0) & (targets < logits.shape[-1])
        np.put_along_axis(onehot, np.where(in_range[..., np.newaxis], targets[..., np.newaxis], 0), 1.0, axis=-1)

        dlogits = softmax - onehot  # (B, S, V)

        # Zero the positions that forward excluded from the mean.
        ignore = targets == self.ignore_index
        valid = (~ignore) & (np.ones_like(ignore, dtype=bool) if mask is None else (mask == 1.0))
        dlogits = np.where(valid[..., np.newaxis], dlogits, 0.0)

        valid_count = float(np.sum(valid))
        if valid_count == 0:
            return np.zeros_like(logits, dtype=dlogits.dtype)
        return dlogits / valid_count
