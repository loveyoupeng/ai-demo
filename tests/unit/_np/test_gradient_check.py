"""Gradient checks: analytic backward vs finite differences.

Every NumPy operator's ``backward`` is verified against a central finite
difference of a linear functional ``sum(dout * forward(x))`` (per operator)
or of the full CE loss (full model). All arithmetic runs in float64
(param arrays are cast in-place) so the finite-difference noise stays far
below the 1e-5 tolerance.

Kink handling: the MoE top-k selection is discrete, so the loss is
piecewise smooth. Elements whose ±eps perturbation flips the selection have
no valid numeric derivative and are skipped (the analytic gradient is
correct a.e.). The full-model checker
(``impl._np.gradcheck.check_model_gradients``) does this automatically.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from impl._np.attention import MultiHeadAttention
from impl._np.cross_entropy import CrossEntropyLoss
from impl._np.embedding import Embedding
from impl._np.ffn import SwiGLUFFN
from impl._np.gradcheck import check_model_gradients
from impl._np.layernorm import RMSNorm
from impl._np.model import NumPyModel
from impl._np.moe import MixtureOfExperts
from impl._np.rope import RoPE
from shared.config import TransformerConfig

EPS = 1e-6


def _cast64(arr: np.ndarray) -> np.ndarray:
    """Float64 copy — call sites must *reassign* (in-place keeps float32 dtype)."""
    return arr.astype(np.float64)


def _fd_param(get_flat: Callable[[], np.ndarray], idx: int, value_fn: Callable[[], float], eps: float = EPS) -> float:
    """Central finite difference of ``value_fn()`` w.r.t. ``get_flat()[idx]``.

    ``get_flat`` must return a *live* view so the perturbation is visible
    to ``value_fn``.
    """
    flat = get_flat()
    original = flat[idx]
    flat[idx] = original + eps
    v_plus = value_fn()
    flat[idx] = original - eps
    v_minus = value_fn()
    flat[idx] = original
    return (v_plus - v_minus) / (2.0 * eps)


def _rel(a: float, n: float) -> float:
    """Relative error with a floor of 1 (avoids amplifying float dust at zeros)."""
    return abs(a - n) / max(1.0, abs(a), abs(n))


def _worst_rel(analytic: np.ndarray, numeric_fn: Callable[[int], float], samples: int, seed: int) -> float:
    """Worst ``_rel`` over ``samples`` random elements of ``analytic``."""
    rng = np.random.default_rng(seed)
    flat = analytic.reshape(-1)
    worst = 0.0
    for j in rng.integers(0, flat.size, min(samples, flat.size)):
        worst = max(worst, _rel(float(flat[j]), numeric_fn(int(j))))
    return worst


class TestRMSNormGradient:
    """RMSNorm backward vs finite difference."""

    def test_gamma_and_dx(self) -> None:
        D = 8
        op = RMSNorm(D)
        op.gamma = _cast64(op.gamma)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 4, D))
        dout = rng.normal(size=(2, 4, D))

        dx_an, dg_an = op.backward(dout, x)

        def val_x(j: int) -> float:
            return _fd_param(lambda: x.reshape(-1), j, lambda: float(np.sum(dout * op.forward(x))))

        def val_g(j: int) -> float:
            return _fd_param(lambda: op.gamma, j, lambda: float(np.sum(dout * op.forward(x))))

        assert _worst_rel(dx_an, val_x, samples=8, seed=1) < 1e-5
        assert _worst_rel(dg_an, val_g, samples=D, seed=2) < 1e-5


class TestSwiGLUGradient:
    """SwiGLUFFN backward vs finite difference."""

    def test_dx_and_weights(self) -> None:
        rng = np.random.default_rng(0)
        op = SwiGLUFFN(embed_dim=6, ff_dim=10, seed=1)
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(op, name, _cast64(getattr(op, name)))
        x = rng.normal(size=(2, 5, 6))
        dout = rng.normal(size=(2, 5, 6))

        dx_an, w_an = op.backward(dout, x)

        def val_x(j: int) -> float:
            return _fd_param(lambda: x.reshape(-1), j, lambda: float(np.sum(dout * op.forward(x))))

        assert _worst_rel(dx_an, val_x, samples=8, seed=1) < 1e-5
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight = getattr(op, name)  # bind the array per iteration

            def val_w(j: int, w: np.ndarray = weight) -> float:
                return _fd_param(lambda: w.reshape(-1), j, lambda: float(np.sum(dout * op.forward(x))))

            assert _worst_rel(w_an[name], val_w, samples=6, seed=2) < 1e-5


class TestRoPEGradient:
    """RoPE backward vs finite difference (full and partial rotation)."""

    @pytest.mark.parametrize("rope_dim", [0, 4])
    def test_dx(self, rope_dim: int) -> None:
        rng = np.random.default_rng(0)
        D, H = 12, 2
        op = RoPE()
        x = rng.normal(size=(2, 6, H, D))
        dout = rng.normal(size=(2, 6, H, D))
        positions = np.arange(6, dtype=np.int32)

        dx_an = op.backward(dout, x, positions, rope_dim)

        def val(j: int) -> float:
            return _fd_param(lambda: x.reshape(-1), j, lambda: float(np.sum(dout * op.forward(x, positions, rope_dim))))

        assert _worst_rel(dx_an, val, samples=10, seed=1) < 1e-5


class TestMHAGradient:
    """MultiHeadAttention backward vs finite difference (dense and GQA)."""

    @pytest.mark.parametrize("n_heads,n_groups", [(4, 4), (4, 2)])
    def test_dx_and_projections(self, n_heads: int, n_groups: int) -> None:
        rng = np.random.default_rng(0)
        D = 16
        op = MultiHeadAttention(embed_dim=D, n_heads=n_heads, n_groups=n_groups, rope_dim=0, seed=3)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(op, name, _cast64(getattr(op, name)))
        x = rng.normal(size=(2, 5, D))
        dout = rng.normal(size=(2, 5, D))
        positions = np.arange(5, dtype=np.int32)

        dx_an, w_an = op.backward(dout, x, positions)

        def val_x(j: int) -> float:
            return _fd_param(lambda: x.reshape(-1), j, lambda: float(np.sum(dout * op.forward(x, positions))))

        assert _worst_rel(dx_an, val_x, samples=8, seed=1) < 1e-5
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weight = getattr(op, name)  # bind the array per iteration

            def val_w(j: int, w: np.ndarray = weight) -> float:
                return _fd_param(lambda: w.reshape(-1), j, lambda: float(np.sum(dout * op.forward(x, positions))))

            assert _worst_rel(w_an[name], val_w, samples=6, seed=2) < 1e-5


def _moe_support(x: np.ndarray, gate: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """The exact top-k support mask the MoE forward/backward use."""
    scores = x @ gate
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    if top_k < n_experts:
        order = np.argsort(probs, axis=-1)[:, :, ::-1]
        kth = order[:, :, top_k - 1 : top_k]
        threshold = np.take_along_axis(probs, kth, axis=-1)
        return probs >= threshold
    return np.ones_like(probs, dtype=bool)


def _fd_with_flip_guard(
    param_flat: np.ndarray,
    j: int,
    value_fn: Callable[[], float],
    support_fn: Callable[[], np.ndarray],
) -> float | None:
    """Finite difference, or None if the perturbation flips the MoE support."""
    original = param_flat[j]
    base = support_fn()
    param_flat[j] = original + EPS
    flips = not np.array_equal(base, support_fn())
    param_flat[j] = original - EPS
    flips = flips or (not np.array_equal(base, support_fn()))
    param_flat[j] = original
    if flips:
        return None
    return _fd_param(lambda: param_flat, j, value_fn)


def _check_weight_flip_aware(
    flat: np.ndarray,
    an_flat: np.ndarray,
    value_fn: Callable[[], float],
    support_fn: Callable[[], np.ndarray],
    seed: int,
    need: int = 3,
    tries: int = 40,
) -> float:
    """Worst ``_rel`` over up to ``need`` non-flip elements (flips skipped)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    checked = 0
    for _ in range(tries):
        j = int(rng.integers(0, flat.size))
        num = _fd_with_flip_guard(flat, j, value_fn, support_fn)
        if num is None:
            continue
        worst = max(worst, _rel(an_flat[j], num))
        checked += 1
        if checked >= need:
            break
    assert checked >= 1, "all sampled elements flip the selection"
    return worst


class TestMoEGradient:
    """MixtureOfExperts backward vs finite difference (flip-aware)."""

    @pytest.mark.parametrize("n_experts,top_k", [(2, 1), (4, 3), (3, 2)])
    def test_dx_and_weights(self, n_experts: int, top_k: int) -> None:
        rng = np.random.default_rng(0)
        D = 8
        op = MixtureOfExperts(embed_dim=D, n_experts=n_experts, ff_dim=12, top_k=top_k, seed=4)
        op.gate = _cast64(op.gate)
        for expert in op.experts:
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(expert, name, _cast64(getattr(expert, name)))
        x = rng.normal(size=(2, 5, D))
        dout = rng.normal(size=(2, 5, D))

        def value_fn() -> float:
            return float(np.sum(dout * op.forward(x)))

        def support_fn() -> np.ndarray:
            return _moe_support(x, op.gate, n_experts, top_k)

        dx_an, w_an = op.backward(dout, x)

        # dx: the input x also affects the router (scores = x @ gate), so
        # flip-guard this too.
        def val_x(j: int) -> float:
            res = _fd_with_flip_guard(x.reshape(-1), j, value_fn, support_fn)
            assert res is not None
            return res

        assert _worst_rel(dx_an, val_x, samples=12, seed=1) < 1e-5

        # Router gate: flips skip the element (a.e. correctness is the claim).
        assert (
            _check_weight_flip_aware(op.gate.reshape(-1), w_an["gate"].reshape(-1), value_fn, support_fn, seed=7) < 1e-5
        )

        # Expert weights (same flip guard for uniformity).
        for e_idx, expert in enumerate(op.experts):
            for name in ("gate_proj", "up_proj", "down_proj"):
                worst_e = _check_weight_flip_aware(
                    getattr(expert, name).reshape(-1),
                    w_an["experts"][e_idx][name].reshape(-1),
                    value_fn,
                    support_fn,
                    seed=8 + e_idx,
                )
                assert worst_e < 1e-5


class TestEmbeddingGradient:
    """Embedding backward vs finite difference."""

    def test_weight(self) -> None:
        rng = np.random.default_rng(0)
        V, D = 16, 8
        op = Embedding(vocab_size=V, embed_dim=D)
        op.weight = _cast64(op.weight)
        ids = rng.integers(0, V, (2, 6)).astype(np.int32)
        dout = rng.normal(size=(2, 6, D))

        dW_an = op.backward(dout, ids)

        def val(j: int) -> float:
            return _fd_param(lambda: op.weight.reshape(-1), j, lambda: float(np.sum(dout * op.forward(ids))))

        assert _worst_rel(dW_an, val, samples=10, seed=1) < 1e-5


class TestCrossEntropyGradient:
    """CrossEntropyLoss.backward vs finite difference (all shift/mask combos)."""

    @pytest.mark.parametrize("shift", [True, False])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_backward(self, shift: bool, use_mask: bool) -> None:
        rng = np.random.default_rng(0)
        m = CrossEntropyLoss(shift=shift, ignore_index=-100)
        logits = rng.normal(size=(2, 5, 16))
        targets = rng.integers(0, 16, (2, 5)).astype(np.int32)
        targets[1, 4] = -100
        mask_arr = None if not use_mask else np.array([[1, 1, 1, 0, 0], [1, 1, 0, 1, 1]], dtype=float)

        dlogits_an = m.backward(logits, targets, mask_arr)

        def val(j: int) -> float:
            return _fd_param(lambda: logits.reshape(-1), j, lambda: float(m.forward(logits, targets, mask_arr)))

        assert _worst_rel(dlogits_an, val, samples=10, seed=1) < 1e-5


# ---------------------------------------------------------------------------
# Full-model checks
# ---------------------------------------------------------------------------


def _model64(cfg: TransformerConfig) -> NumPyModel:
    """Build a model whose parameter arrays are all float64 (via reload)."""
    m = NumPyModel(cfg)
    params = {key: _cast64(arr) for key, arr in m._param_arrays().items()}
    m.load_from_numpy_dict(params)
    return m


_X = np.array([[3, 7, 1, 9, 2, 5, 8, 4]], dtype=np.int32)
_T = np.array([[7, 1, 9, 2, 5, 8, 4, 0]], dtype=np.int32)


@pytest.mark.timeout(120)
class TestModelGradients:
    """Full NumPyModel analytic backward vs finite difference."""

    def test_dense(self) -> None:
        cfg = TransformerConfig.from_dict(
            {
                "vocab_size": 16,
                "embed_dim": 8,
                "n_layers": 1,
                "n_heads": 2,
                "n_experts": 1,
                "top_k": 1,
                "expert_dim": 12,
                "seed": 0,
            }
        )
        errors = check_model_gradients(_model64(cfg), _X, _T, samples_per_param=4, seed=0)
        assert max(errors.values()) < 1e-5, f"dense model gradient check failed: {errors}"

    def test_moe_flip_aware(self) -> None:
        cfg = TransformerConfig.from_dict(
            {
                "vocab_size": 16,
                "embed_dim": 8,
                "n_layers": 2,
                "n_heads": 2,
                "n_experts": 2,
                "top_k": 1,
                "expert_dim": 12,
                "seed": 0,
            }
        )
        errors = check_model_gradients(_model64(cfg), _X, _T, samples_per_param=4, seed=0)
        assert max(errors.values()) < 1e-5, f"MoE model gradient check failed: {errors}"

    def test_gqa(self) -> None:
        cfg = TransformerConfig.from_dict(
            {
                "vocab_size": 16,
                "embed_dim": 8,
                "n_layers": 2,
                "n_heads": 4,
                "n_groups": 2,
                "n_experts": 1,
                "top_k": 1,
                "expert_dim": 12,
                "seed": 7,
            }
        )
        errors = check_model_gradients(_model64(cfg), _X, _T, samples_per_param=4, seed=0)
        assert max(errors.values()) < 1e-5, f"GQA model gradient check failed: {errors}"
