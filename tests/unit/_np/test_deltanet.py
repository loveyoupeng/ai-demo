# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Unit tests for the NumPy Gated DeltaNet (ADR 0006): forward, step/chunk
parity, and the analytic backward against finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from impl._np.deltanet import GatedDeltaNet


def _model(seed: int = 1, D: int = 16, H: int = 4, S: int = 6, B: int = 2):
    rng = np.random.default_rng(0)
    m = GatedDeltaNet(D, H, seed=seed)
    x = rng.normal(0, 1, size=(B, S, D)).astype(np.float64)
    return m, x


class TestGatedDeltaNetForward:
    def test_output_shape_and_finite(self):
        m, x = _model()
        out = m.forward(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()

    def test_deterministic(self):
        m, x = _model()
        assert np.array_equal(m.forward(x), m.forward(x))

    def test_step_parity(self):
        """prefill-free: stepping token-by-token == one full forward."""
        m, x = _model()
        full = m.forward(x)
        B, S = x.shape[:2]
        hd = m.head_dim
        cache = {"s": np.zeros((B, m.n_heads, hd, hd)), "s_hist": []}
        outs = [m.forward_step(x[:, t : t + 1], t, cache) for t in range(S)]
        assert np.allclose(np.concatenate(outs, axis=1), full, atol=1e-12)

    def test_chunk_parity(self):
        """two chunks (3+3) == one full forward."""
        m, x = _model()
        full = m.forward(x)
        B, S = x.shape[:2]
        hd = m.head_dim
        cache = {"s": np.zeros((B, m.n_heads, hd, hd)), "s_hist": []}
        o1 = m.forward_chunk(x[:, :3], 0, cache)
        o2 = m.forward_chunk(x[:, 3:], 3, cache)
        assert np.allclose(np.concatenate([o1, o2], axis=1), full, atol=1e-12)

    def test_state_grows(self):
        m, x = _model()
        B, S = x.shape[:2]
        hd = m.head_dim
        cache = {"s": np.zeros((B, m.n_heads, hd, hd)), "s_hist": []}
        m.forward_chunk(x, 0, cache)
        assert cache["s"].shape == (B, m.n_heads, hd, hd)
        assert len(cache["s_hist"]) == S


class TestGatedDeltaNetBackward:
    @pytest.mark.parametrize(
        "param", ["q_proj", "k_proj", "v_proj", "o_proj", "a_proj", "dt_bias", "gate_proj", "norm"]
    )
    def test_param_gradient_matches_finite_diff(self, param: str):
        rng = np.random.default_rng(1)
        m = GatedDeltaNet(8, 2, seed=3)
        x = rng.normal(0, 0.5, size=(1, 4, 8)).astype(np.float64)
        out = m.forward(x)
        dout = 2.0 * out
        _dx, dparams = m.backward(dout, x)

        def loss() -> float:
            return float(np.sum(m.forward(x) ** 2))

        p = m.norm.gamma if param == "norm" else getattr(m, param)
        g = dparams[param]
        flat = p.reshape(-1)
        gf = g.reshape(-1)
        for j in rng.choice(flat.size, size=min(3, flat.size), replace=False):
            h = 1e-6 * max(1.0, abs(flat[j]))
            orig = flat[j]
            flat[j] = orig + h
            lp = loss()
            flat[j] = orig - h
            lm = loss()
            flat[j] = orig
            num = (lp - lm) / (2 * h)
            rel = abs(num - gf[j]) / max(1.0, abs(num), abs(gf[j]))
            assert rel < 5e-2, f"{param}[{j}]: num={num:.6f} ana={gf[j]:.6f} rel={rel:.2e}"

    def test_dx_matches_finite_diff(self):
        rng = np.random.default_rng(2)
        m = GatedDeltaNet(8, 2, seed=5)
        x = rng.normal(0, 0.5, size=(1, 4, 8)).astype(np.float64)
        out = m.forward(x)
        dx, _ = m.backward(2.0 * out, x)

        def loss() -> float:
            return float(np.sum(m.forward(x) ** 2))

        xf = x.reshape(-1)
        dxf = dx.reshape(-1)
        for j in rng.choice(xf.size, size=5, replace=False):
            h = 1e-6 * max(1.0, abs(xf[j]))
            orig = xf[j]
            xf[j] = orig + h
            lp = loss()
            xf[j] = orig - h
            lm = loss()
            xf[j] = orig
            num = (lp - lm) / (2 * h)
            rel = abs(num - dxf[j]) / max(1.0, abs(num), abs(dxf[j]))
            assert rel < 1e-6, f"dx[{j}]: num={num:.8f} ana={dxf[j]:.8f}"
