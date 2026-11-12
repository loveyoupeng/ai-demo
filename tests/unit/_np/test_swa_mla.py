# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""SWA (ADR 0008) + MLA (ADR 0009) unit tests for the NumPy reference track.

Covers:
- SWA banded mask equals full attention on the first w+1 rows and differs
  beyond the window.
- SWA decode slices the last w+1 cache rows (step == full forward).
- MLA latent cache: step/chunk parity with the dense forward, and the
  analytic backward through the low-rank factorization (finite differences).
- Registry: MLA configs emit kv_down/k_up/v_up instead of k/v_proj.
"""

from __future__ import annotations

import numpy as np
import pytest

from impl._np.attention import MultiHeadAttention
from shared.config import TransformerConfig
from shared.constants import Attn, Keys
from shared.registry import ParameterRegistry


def _attn(swa: int = 0, mla: int = 0, seed: int = 7) -> MultiHeadAttention:
    return MultiHeadAttention(16, 2, 2, 0, seed=seed, swa_window=swa, mla_latent_dim=mla)


def _cache_for(a: MultiHeadAttention, dtype) -> dict:
    if a.mla_latent_dim > 0:
        return {"c": np.zeros((1, 1, 0, a.mla_latent_dim), dtype=dtype)}
    return {
        "k": np.zeros((1, 2, 0, 8), dtype=dtype),
        "v": np.zeros((1, 2, 0, 8), dtype=dtype),
    }


def test_swa_first_window_rows_match_full_attention() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 10, 16))
    full = _attn(swa=0)
    win = _attn(swa=3)
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(win, attr)[...] = getattr(full, attr)
    o_full, o_win = full.forward(x), win.forward(x)
    assert np.allclose(o_full[:, :4], o_win[:, :4], atol=1e-12)
    assert not np.allclose(o_full[:, 4:], o_win[:, 4:], atol=1e-9)


def test_swa_mask_is_banded() -> None:
    a = _attn(swa=2)
    x = np.zeros((1, 6, 16))
    st = a.forward_state(x, 0).ctx.merged()
    mask = st["causal_mask"]  # (S, S), True = masked
    # row 5: masked cols are j > 5 OR 5 - j > 2 → j < 3
    assert mask[5].tolist() == [True, True, True, False, False, False]


@pytest.mark.parametrize("swa,mla", [(0, 0), (3, 0), (0, 4), (3, 4)])
def test_step_and_chunk_parity(swa: int, mla: int) -> None:
    a = _attn(swa=swa, mla=mla)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 6, 16))
    full = a.forward(x)
    cache = _cache_for(a, x.dtype)
    step = np.concatenate([a.forward_step(x[:, t : t + 1], t, cache) for t in range(6)], axis=1)
    assert np.abs(full - step).max() < 1e-10
    cache2 = _cache_for(a, x.dtype)
    chunk = a.forward_chunk(x, 0, cache2)
    assert np.abs(full - chunk).max() < 1e-10


def test_mla_cache_stores_latent_only() -> None:
    a = _attn(mla=4)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 5, 16))
    cache = _cache_for(a, x.dtype)
    for t in range(5):
        a.forward_step(x[:, t : t + 1], t, cache)
    assert cache["c"].shape == (1, 1, 5, 4)  # (B, 1, t, L) — 4 floats/token, not 2*G*hd=32
    assert "k" not in cache and "v" not in cache


def test_mla_backward_matches_finite_differences() -> None:
    a = _attn(mla=4)
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "kv_down_proj", "k_up_proj", "v_up_proj"):
        setattr(a, attr, getattr(a, attr).astype(np.float64))
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 5, 16)).astype(np.float64)
    dout = rng.normal(size=(1, 5, 16)).astype(np.float64)
    dx, dp = a.backward(dout, x)

    def loss() -> float:
        return float(np.sum(a.forward(x) * dout))

    def numgrad(w: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        g = np.zeros_like(w)
        it = np.nditer(w, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            old = w[i]
            w[i] = old + eps
            fp = loss()
            w[i] = old - eps
            fm = loss()
            w[i] = old
            g[i] = (fp - fm) / (2 * eps)
            it.iternext()
        return g

    for name in ("q_proj", "o_proj", "kv_down_proj", "k_up_proj", "v_up_proj"):
        ng = numgrad(getattr(a, name))
        rel = np.abs(ng - dp[name]).max() / (np.abs(ng).max() + 1e-12)
        assert rel < 1e-5, f"{name} rel err {rel}"
    ng_dx = numgrad(x)
    assert np.abs(ng_dx - dx).max() / (np.abs(ng_dx).max() + 1e-12) < 1e-5


def test_registry_mla_keys() -> None:
    cfg = TransformerConfig(
        vocab_size=64,
        embed_dim=16,
        n_layers=1,
        n_heads=2,
        n_deltanet_sublayers=1,
        context_length=32,
        mla_latent_dim=4,
    )
    keys = [e.key for e in ParameterRegistry(cfg).entries if "self_attn" in e.key]
    assert Keys.attn(0, 1, Attn.KV_DOWN_PROJ) in keys
    assert Keys.attn(0, 1, Attn.K_UP_PROJ) in keys
    assert Keys.attn(0, 1, Attn.V_UP_PROJ) in keys
    assert Keys.attn(0, 1, Attn.K_PROJ) not in keys
    assert Keys.attn(0, 1, Attn.V_PROJ) not in keys


def test_registry_classic_keys_unchanged() -> None:
    cfg = TransformerConfig(
        vocab_size=64, embed_dim=16, n_layers=1, n_heads=2, n_deltanet_sublayers=1, context_length=32
    )
    keys = [e.key for e in ParameterRegistry(cfg).entries if "self_attn" in e.key]
    assert Keys.attn(0, 1, Attn.K_PROJ) in keys
    assert Keys.attn(0, 1, Attn.KV_DOWN_PROJ) not in keys


def test_registry_alternating_mla_mha_per_layer() -> None:
    """ADR 0011: group-of-2 alternation — even blocks get MLA keys, odd get MHA."""
    cfg = TransformerConfig(
        vocab_size=64,
        embed_dim=16,
        n_layers=4,
        n_heads=2,
        n_deltanet_sublayers=1,
        context_length=32,
        mla_latent_dim=4,
        mla_group_size=2,
    )
    assert [cfg.has_mla_at(i) for i in range(4)] == [True, False, True, False]
    keys = [e.key for e in ParameterRegistry(cfg).entries]
    for i in (0, 2):
        assert Keys.attn(i, 1, Attn.KV_DOWN_PROJ) in keys
        assert Keys.attn(i, 1, Attn.K_PROJ) not in keys
    for i in (1, 3):
        assert Keys.attn(i, 1, Attn.K_PROJ) in keys
        assert Keys.attn(i, 1, Attn.KV_DOWN_PROJ) not in keys


def test_mixed_mla_mha_model_parity() -> None:
    """A group-of-2 alternating model: caches alternate {'c'} / {'k','v'},
    step+chunk parity holds, save/load round-trips."""
    from impl._np.model import NumPyModel

    cfg = TransformerConfig(
        vocab_size=64,
        embed_dim=16,
        n_layers=4,
        n_heads=2,
        n_deltanet_sublayers=1,
        context_length=32,
        swa_window=8,
        mla_latent_dim=4,
        mla_group_size=2,
        seed=1,
    )
    m = NumPyModel(cfg)
    ids = np.arange(10, dtype=np.int32)[None, :]
    full = m.forward(ids)
    cache = m.make_cache(1)
    m.forward_prefill(ids, cache=cache)
    kinds = ["c" if "c" in c[1] else "kv" for c in cache]
    assert kinds == ["c", "kv", "c", "kv"]
    step = m.forward_step(np.array([[5]], dtype=np.int32), 10, cache)
    full2 = m.forward(np.concatenate([ids, [[5]]], axis=1))
    assert np.abs(full2[:, -1] - step[:, -1]).max() < 1e-10
    ch = m.forward_chunk(np.array([[6, 7]], dtype=np.int32), 11, cache)
    ch = ch[0] if isinstance(ch, tuple) else ch
    full3 = m.forward(np.concatenate([ids, [[5, 6, 7]]], axis=1))
    assert np.abs(full3[:, -2:] - ch).max() < 1e-10
    m2 = NumPyModel(cfg, m.get_all_parameters())
    assert np.allclose(m2.forward(ids), full)
