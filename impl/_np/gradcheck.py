"""Finite-difference gradient checker for the NumPy track.

``NumPyModel.backward`` is analytic (and O(forward)); this module keeps the
old finite-difference machinery as a *test reference* only:

- ``finite_difference_gradient`` — central difference of the loss w.r.t. one
  flat element of one parameter array.
- ``check_model_gradients`` — runs the analytic backward once and compares it
  against finite differences on a small random sample of elements per
  parameter, reporting the worst relative error per parameter.

The relative error uses ``|a-n| / max(1, |a|, |n|)`` so that (near-)zero
gradients — e.g. the top-k router weights a.e. (see ``impl._np.moe``) —
don't amplify float dust into false failures.

Kink handling: the MoE top-k selection is a discrete function of the
router scores, so the loss is piecewise smooth. At elements whose ±eps
perturbation flips the selection, the numeric derivative is *not* a
gradient and the element is skipped (``check_model_gradients`` detects
flips by recomputing every MoE support at the base point and at ±eps).
"""

from __future__ import annotations

import numpy as np

from impl._np.cross_entropy import CrossEntropyLoss
from impl._np.model import NumPyModel
from impl._np.moe import MixtureOfExperts

__all__ = ["finite_difference_gradient", "check_model_gradients"]


def _loss(model: NumPyModel, input_ids: np.ndarray, targets: np.ndarray) -> float:
    """Scalar loss the model's backward differentiates (forward + CE).

    Targets are pre-shifted next-token labels (see ``NumPyModel.backward``),
    so the CE must use ``shift=False`` to match the analytic gradient.
    """
    logits = model.forward(input_ids)
    return float(CrossEntropyLoss(shift=False).forward(logits, targets))


def finite_difference_gradient(
    model: NumPyModel,
    input_ids: np.ndarray,
    targets: np.ndarray,
    param: np.ndarray,
    flat_idx: int,
    eps: float = 1e-6,
) -> float:
    """Central finite difference of the loss w.r.t. ``param.reshape(-1)[flat_idx]``.

    ``param`` must be a *live* view of the model storage (e.g. from
    ``model._param_arrays()``) so the perturbation is seen by ``forward``.
    """
    flat = param.reshape(-1)
    original = flat[flat_idx]
    flat[flat_idx] = original + eps
    loss_plus = _loss(model, input_ids, targets)
    flat[flat_idx] = original - eps
    loss_minus = _loss(model, input_ids, targets)
    flat[flat_idx] = original
    return (loss_plus - loss_minus) / (2.0 * eps)


def check_model_gradients(
    model: NumPyModel,
    input_ids: np.ndarray,
    targets: np.ndarray,
    samples_per_param: int = 5,
    seed: int = 0,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compare ``model.backward`` against finite differences.

    Returns a dict mapping parameter key → worst relative error over the
    sampled elements (``|a-n| / max(1, |a|, |n|)``). Elements whose ±eps
    perturbation flips any MoE top-k support are skipped (the numeric
    derivative is undefined at the kink).
    """
    grads = model.backward(input_ids, targets)
    params = model._param_arrays()
    rng = np.random.default_rng(seed)

    # Selections at the base point (computed once; None → no MoE layers).
    base_selections = _moe_selections(model, input_ids)

    result: dict[str, float] = {}
    for key, param in params.items():
        flat = param.reshape(-1)
        n_samples = min(samples_per_param, flat.size)
        worst = 0.0
        for j in rng.integers(0, flat.size, n_samples):
            if base_selections is not None:
                # Perturb and check every MoE support stays unchanged.
                original = flat[j]
                flips = False
                for delta in (eps, -eps):
                    flat[j] = original + delta
                    if _any_support_differs(model, input_ids, base_selections):
                        flips = True
                        break
                flat[j] = original
                if flips:
                    continue
            analytic = grads[key].reshape(-1)[j]
            numeric = finite_difference_gradient(model, input_ids, targets, flat, j, eps)
            denom = max(1.0, abs(analytic), abs(numeric))
            worst = float(max(worst, abs(analytic - numeric) / denom))
        result[key] = worst
    return result


def _moe_selections(model: NumPyModel, input_ids: np.ndarray) -> list[np.ndarray] | None:
    """Per-MoE-layer top-k support (bool (B, S, E)) under the current params.

    Recomputes the forward stack and each router with the exact tie rule of
    ``MixtureOfExperts.forward`` (``probs >= k-threshold``). Returns None if
    the model has no MoE layers (then no flip is possible and callers skip
    the check).
    """
    x_in0 = model.embedding.forward(input_ids)
    positions = np.arange(input_ids.shape[1], dtype=np.int32)
    selections: list[np.ndarray] = []
    x = x_in0
    for block in model.stack.layers:
        mlp = block.mlp
        if not isinstance(mlp, MixtureOfExperts):
            continue
        # Recompute the FFN input exactly as the block forward does.
        h = block.input_layernorm.forward(x)
        h = h + block.self_attn.forward(h, positions)
        h = block.post_attention_layernorm.forward(h)
        scores = h @ mlp.gate  # (B, S, E)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        E = mlp.n_experts
        if mlp.top_k < E:
            order = np.argsort(probs, axis=-1)[:, :, ::-1]  # (B, S, E) descending
            kth_idx = order[:, :, mlp.top_k - 1 : mlp.top_k]  # (B, S, 1)
            threshold = np.take_along_axis(probs, kth_idx, axis=-1)  # (B, S, 1)
            selections.append(probs >= threshold)  # (B, S, E)
        x = block.forward(x, positions)
    return selections or None


def _any_support_differs(
    model: NumPyModel,
    input_ids: np.ndarray,
    base_selections: list[np.ndarray],
) -> bool:
    """True if any MoE support differs from ``base_selections`` right now."""
    current = _moe_selections(model, input_ids)
    if current is None:
        return False
    return any(not np.array_equal(base, cur) for base, cur in zip(base_selections, current, strict=True))
