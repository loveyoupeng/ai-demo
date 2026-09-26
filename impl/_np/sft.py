"""SFT trainer for the NumPy reference implementation — the teaching baseline.

One small function: masked-loss fine-tuning. Everything beside the mask is
ordinary language model training (the same forward, the same analytic
backward, the same AdamW) — the only thing SFT adds is *which positions the
loss sees*.

A batch row is:
    input_ids   x[i]     — the whole sentence (prompt + response)
    target_ids  t[i]     — the token the model should predict at x[i]
    They are pre-shifted: t[i] = x[i+1] (next-token). Prompt positions are
    set to -100 => the cross-entropy loss treats them as ignore-index.

The NumPy track's whole value here is that this — mask the labels, same
loss — is *all* the math of SFT. Every other track mirrors this shape.

Typical calls:
    >>> model = NumPyModel(cfg)
    >>> params = model.get_all_parameters()  # the flat Keys dict
    >>> nw, _, loss = sft_epoch(model, batches, lr=1e-3, params=params)

    # batches comes from scripts/train_dataset_to_batches() which uses
    # SFTLoader from shared/sft_data.py.
"""

from __future__ import annotations

import numpy as np

from impl._np.model import NumPyModel
from impl._np.optimizer import AdamW

# The CrossEntropyLoss ignore_index convention in this repo (=-100) — the
# mask shows up as "-100 at target position", not as a separate array.
_SFT_MASK = -100


def sft_step(
    model: NumPyModel,
    inputs: np.ndarray,
    targets: np.ndarray,
    optimizer: AdamW,
) -> float:
    """One step: masked forward → analytic backward → AdamW update.

    Parameters
    ----------
    model : NumPyModel
        The model being fine-tuned; its parameters update in-place.
    inputs : (B, S) int
        The full token stream (prompt + response, ``max_len``).
    targets : (B, S) int
        Next-token labels where prompt positions are _SFT_MASK (=-100);
        SFT only grades the response side. `ignore_index` is baked into
        ``CrossEntropyLoss``, so the existing pre-training pipeline
        (``backward`` + ``AdamW.step``) runs *unchanged* post-masking.

    Returns
    -------
    float
        The (masked) loss for this step, for logging.
    """
    # 1. Forward: full sequence, no difference from pre-training
    logits = model.forward(inputs)  # (B, S, V)
    # 2. Loss: CE respects the mask (targets at -100 contribute nothing).
    from impl._np.cross_entropy import CrossEntropyLoss

    loss = CrossEntropyLoss(shift=False, ignore_index=_SFT_MASK).forward(logits, targets)
    # 3. Backward: the same analytic chain as pre-training
    grads = model.backward(inputs, targets)
    # 4. Update parameters. AdamW is the same optimizer used in pre-training
    # — masked loss only affects which parameters see a nonzero gradient.
    params = model.get_all_parameters()
    optimizer.step(params, grads)
    model.load_from_numpy_dict(params)  # push updated weights back into the model
    return float(loss)


def sft_epoch(
    model: NumPyModel,
    batches: list[tuple[np.ndarray, np.ndarray]],
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> tuple[int, int, float]:
    """Run one full pass over every SFT batch and return progress counters.

    Param updates happen in-place; we return two counters so the caller can
    log a real number of optimizer step iterations:

    Returns
    -------
    (n_tokens_trained, n_valid_positions, mean_loss_this_epoch)
    """
    opt = AdamW(lr=lr, weight_decay=weight_decay)
    total_tokens = 0
    total_positions = 0
    losses = []
    for inputs, targets in batches:
        losses.append(sft_step(model, inputs, targets, opt))
        total_positions += int((targets != _SFT_MASK).sum())
        total_tokens += int(inputs.size)
    return total_tokens, total_positions, float(np.mean(losses)) if losses else 0.0
