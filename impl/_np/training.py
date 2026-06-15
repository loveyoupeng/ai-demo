"""Training loop utilities for NumPy-based decoder-only transformer.

Provides TrainingConfig, TrainingState, train_step, and gradient norm
computation / clipping helpers for use with the NumPyModel.

Architecture
------------
Training loop:
    for epoch in range(epochs):
        for batch in dataset:
            logits     = model.forward(input)        # (B, S, V)
            loss       = loss_fn.forward(logits, tgt) # scalar
            grads      = model.backward(logits, tgt, inp)     # dict of params
            optimizer.step(params, grads)             # modifies params in-place
"""

from __future__ import annotations

import numpy as np

from impl._np.cross_entropy import CrossEntropyLoss
from impl._np.model import NumPyModel
from impl._np.optimizer import AdamW


class TrainingConfig:
    """Configuration for training a decoder-only transformer.

    Parameters
    ----------
    lr : float
        Learning rate for the optimizer (default 3e-4).
    epochs : int
        Number of training epochs (default 10).
    batch_size : int
        Number of sequences per batch (default 16).
    max_seq_len : int
        Maximum sequence length (default 512).
    device : str
        Computed device string (default "cpu").
    grad_accum_steps : int
        Number of sub-batches to accumulate gradients across before
        optimizer.step() (default 1).
    max_grad_norm : float
        Clip gradients by global L2 norm. 1.0 (default).
    log_every : int
        Log training progress every N batches (default 10).
    """

    lr: float
    epochs: int
    batch_size: int
    max_seq_len: int
    device: str
    grad_accum_steps: int
    max_grad_norm: float
    log_every: int

    def __init__(
        self,
        lr: float = 3e-4,
        epochs: int = 10,
        batch_size: int = 16,
        max_seq_len: int = 512,
        device: str = "cpu",
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_every: int = 10,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every


class TrainingState:
    """Mutable state updated by the training loop.

    Track training progress metrics that are accessed by both inner-loop
    (per-step) and outer-loop (per-epoch) code.
    """

    # Global optimizer step counter across all epochs
    step: int = 0

    # Cumulative loss (sum of all batch losses computed so far)
    total_loss: float = 0.0

    # Mean loss = total_loss / number_of_batches_processed
    mean_loss: float = 0.0


def train_step(
    model: NumPyModel,
    batch_input: np.ndarray,
    batch_target: np.ndarray,
    loss_fn: CrossEntropyLoss,
    optimizer: AdamW,
) -> float:
    """Execute one training step on a single batch.

    Full forward → backward → optimizer cycle:

        1. Forward pass: model(batch_input) → logits (B, S, V)
        2. Loss computation: loss_fn(logits, batch_target) → scalar float
        3. Backward pass: model.backward(logits, targets, input) → grads dict
        4. Optimizer step: optimizer.step(params, grads) — modifies params in-place
        5. Return loss value

    Parameters
    ----------
    model : NumPyModel
        The decoder-only transformer model with forward/backward/parameter methods.
    batch_input : np.ndarray, shape (batch_size, seq_len), dtype int32
        Token IDs for the input sequences.
    batch_target : np.ndarray, shape (batch_size, seq_len), dtype int32
        Ground-truth token IDs for computing cross-entropy loss.
    loss_fn : CrossEntropyLoss
        Loss function that computes cross-entropy over logits and targets.
    optimizer : AdamW
        Optimizer that updates model parameters in-place.

    Returns
    -------
    loss : float
        The scalar cross-entropy loss for this batch.

    Examples
    --------
    >>> import numpy as np
    >>> from impl._np.model import NumPyModel
    >>> from impl._np.cross_entropy import CrossEntropyLoss
    >>> from impl._np.optimizer import AdamW
    >>> from impl._np.training import train_step
    >>> model = NumPyModel(vocab_size=16, embed_dim=32, n_layers=1,
    ...                    n_heads=2, n_experts=2, ff_dim=16, k=1)
    >>> x = np.random.randint(0, 16, (2, 4), dtype=np.int32)
    >>> t = np.random.randint(0, 16, (2, 4), dtype=np.int32)
    >>> loss = train_step(model, x, t, CrossEntropyLoss(), AdamW(lr=0.01))
    >>> isinstance(loss, float)
    True
    """
    # --- 1. Forward pass --------------------------------------------------
    # batch_input:  (B, S)     — token IDs, dtype int32
    # model.forward maps token IDs through embedding → transformer stack →
    # output projection to produce logits over the vocabulary.
    #
    # Output shape:   (B, S, V)  where V = vocab_size
    # Each element logits[b, s, v] is the unnormalized log-probability
    # that the model assigns to vocabulary token v at position s of batch b.
    logits = model.forward(batch_input)  # (B, S, V)

    # --- 2. Loss computation -------------------------------------------------
    # loss_fn.forward computes cross-entropy over (logits, targets):
    #   log_softmax(logits[target]) averaged over all (batch, seq) positions.
    #
    # Shape of intermediate computation: (B, S, 1) — per-position loss
    # Output: scalar float (mean over all positions)
    loss = loss_fn.forward(logits, batch_target)  # scalar float

    # --- 3. Backward pass (numerical gradients) ------------------------------
    # model.backward recomputes the forward pass internally and uses
    # finite-difference to compute gradient of loss w.r.t. every parameter.
    #
    # grads[k] has the same shape as params[k], i.e. the gradient of the
    # scalar loss with respect to each element of the parameter tensor.
    grads = model.backward(logits, batch_target, batch_input)  # dict[str, ndarray]

    # --- 4. Optimizer step ---------------------------------------------------
    # Gather the current parameter dictionary from the model.  The optimizer
    # will modify these arrays in-place.
    #
    # params:  dict[str, ndarray] — key = parameter name, value = tensor
    # grads:   dict[str, ndarray] — same keys, same shapes as params
    #
    # AdamW updates rule (per element):
    #   m = β1 · m + (1−β1) · g
    #   v = β2 · v + (1−β2) · g²
    #   θ ← θ − lr · (m̂ / (v̂ + ε)¹ᐟ² + wd · θ)
    # where m̂, v̂ are bias-corrected first/second moment estimates.
    params = model.get_all_parameters()
    optimizer.step(params, grads)  # in-place modification

    # --- 5. Return the scalar loss value for logging -------------------------
    return float(loss)  # type: ignore[return-value]


def compute_gradient_norm(grads: dict[str, np.ndarray]) -> float:
    """Compute the global L2 norm of all gradient tensors.

    The global gradient norm is the Euclidean norm of the concatenated
    gradient vectors from all parameters:

        ||g||₂ = √( Σₖ ||gₖ||₂² )

    which is the norm of the full flattened gradient vector.

    Parameters
    ----------
    grads : dict[str, np.ndarray]
        Gradient dictionary with same keys as model.get_all_parameters().
        Each value is a numpy array of the same shape as the corresponding
        parameter.

    Returns
    -------
    norm : float
        Scalar global L2 norm.  Returns 0.0 if all gradients are zero.

    Examples
    --------
    >>> grads = {"w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)}
    >>> norm = compute_gradient_norm(grads)
    >>> abs(norm - 5.4772) < 1e-4
    True
    """
    # Compute the sum of squared norms across ALL gradient tensors.
    # For each grad tensor g: ||g||₂² = Σᵢⱼ gᵢⱼ²
    # Global norm: √( Σₖ ||gₖ||₂² )

    total_sq_norm = 0.0  # scalar accumulator

    for grad in grads.values():
        # grad: shape (d₁, d₂, ..., dₙ) — any shape matching a parameter
        # grad**2 : element-wise square, same shape
        # np.sum(grad**2) : sum of all squared elements → scalar
        #
        # This is the squared L2 norm of the flattened gradient vector:
        #   ||grad||₂² = Σᵢ grad[i]²
        total_sq_norm += float(np.sum(grad**2))  # scalar float

    # Global L2 norm = sqrt(sum of per-parameter squared norms)
    return float(np.sqrt(total_sq_norm))


def clip_gradients(grads: dict[str, np.ndarray], max_norm: float) -> None:
    """Clip gradients by global L2 norm, modifying the dict in-place.

    When the global gradient norm exceeds `max_norm`, every gradient tensor
    is scaled by the same factor `max_norm / global_norm` so that the
    overall norm equals `max_norm` (or stays unchanged if already below).

    This is the standard "global norm clipping" strategy used in
    sequence-to-sequence models and LLM training to prevent exploding
    gradients.

    Parameters
    ----------
    grads : dict[str, np.ndarray]
        Gradient dictionary — values are mutated in-place.
    max_norm : float
        Maximum allowed L2 norm.  If 0.0, no clipping is performed.

    Examples
    --------
    >>> from impl._np.training import clip_gradients, compute_gradient_norm
    >>> import numpy as np
    >>> grads = {"w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)}
    >>> clip_gradients(grads, max_norm=1.0)
    >>> norm = compute_gradient_norm(grads)
    >>> abs(norm - 1.0) < 1e-6
    True
    """
    # Guard against zero max_norm — no clipping needed
    if max_norm <= 0.0:
        return

    # Step 1: Compute current global gradient norm
    # global_norm is the L2 norm of all gradient elements concatenated
    global_norm = compute_gradient_norm(grads)  # scalar float

    # If norm is already below max_norm, nothing to do
    if global_norm <= max_norm:
        return

    # Step 2: Scale all gradients uniformly toward max_norm
    # scaling_factor = max_norm / global_norm  (always < 1.0 here)
    # After scaling: global_norm' = global_norm * scaling_factor = max_norm
    #
    # For each gradient tensor g with shape (d₁, ..., dₙ):
    #   g ← g * scaling_factor  → new shape (d₁, ..., dₙ) with same shape
    scaling_factor = max_norm / global_norm

    for grad in grads.values():
        # grad: ndarray of any shape — element-wise multiply by scalar
        grad *= scaling_factor  # in-place scalar multiplication
