"""Speculative decoding tests — the lossless contract, chunk parity, and
sidecar round-trips (ADR 0003; docs/specs/speculative-decoding.md).

The seams (highest first, per the spec's Testing Decisions):

1. *Generation seam* — greedy speculative output must be token-identical
   to plain greedy, per drafter, per k. The lossless oracle.
2. *KV-step interface* — ``forward_chunk`` must equal repeated
   ``forward_step`` (and prefill+chunk must equal one full forward) on
   every track. The fast verification path is exact.
3. *Sidecar seam* — a sidecar saved from the NumPy drafter loads into the
   torch drafter (same keys, same shapes, same draft tokens); corruption
   (missing/extra/stale-shape keys) fails fast.
4. *Schedule* — the DSpark survival schedule caps the verified length and
   its state snapshot is well-formed.
5. *Rejection sampling* — the NumPy theorem path runs and produces a
   sequence of the requested shape (distributional identity needs many
   samples; the smoke here pins the mechanics, not the distribution).
"""

from __future__ import annotations

import numpy as np
import pytest

from impl._np.drafters import DSparkDrafter, MTPDrafter, make_drafter_meta
from impl._np.inference import TextGenerator
from impl._np.model import NumPyModel
from impl._np.spec import SCHEDULE_TARGET, SpeculativeGenerator, _SurvivalSchedule
from shared.config import TransformerConfig
from shared.draft import DSHARK, MTP, expected_drafter_params, load_drafter, save_drafter


def _make_model(seed: int = 3, **overrides) -> NumPyModel:
    """A small dense model for the speculative-decoding tests."""
    cfg = {
        "vocab_size": 64,
        "embed_dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "seed": seed,
    }
    cfg.update(overrides)
    return NumPyModel(TransformerConfig.from_dict(cfg))


def _make_drafter(model: NumPyModel, family: str, k: int, ff_dim: int = 48, seed: int = 5):
    meta = make_drafter_meta(family, model.config, block_size=k, ff_dim=ff_dim)
    if seed:
        meta = meta  # seed flows through the constructor below
    cls = MTPDrafter if family == MTP else DSparkDrafter
    return cls(meta, model.embedding.weight, model.lm_head_weight, seed=seed)


PROMPT = np.array([[5, 12, 33, 7]], dtype=np.int32)


class TestChunkParity:
    """forward_chunk == stepwise == full forward (the verification exactness)."""

    @pytest.mark.parametrize("overrides", [{}, {"n_groups": 2}, {"n_experts": 3, "top_k": 2}])
    def test_prefill_chunk_equals_forward(self, overrides: dict) -> None:
        model = _make_model(seed=7, **overrides)
        rng = np.random.default_rng(1)
        ids = rng.integers(0, 64, size=(2, 5)).astype(np.int32)

        ref = model.forward(ids)

        cache = model.make_cache(2)
        model.forward_prefill(ids[:, :2], cache)
        chunk_logits, _hidden = model.forward_chunk(ids[:, 2:], 2, cache)
        np.testing.assert_allclose(chunk_logits, ref[:, 2:, :], rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("overrides", [{}, {"n_groups": 2}])
    def test_chunk_equals_repeated_steps(self, overrides: dict) -> None:
        model = _make_model(seed=11, **overrides)
        rng = np.random.default_rng(2)
        ids = rng.integers(0, 64, size=(1, 6)).astype(np.int32)

        cache_c = model.make_cache(1)
        model.forward_prefill(ids[:, :3], cache_c)
        chunk_logits, _ = model.forward_chunk(ids[:, 3:], 3, cache_c)

        cache_s = model.make_cache(1)
        model.forward_prefill(ids[:, :3], cache_s)
        step_logits = [model.forward_step(ids[:, [i]], i, cache_s) for i in range(3, 6)]
        np.testing.assert_allclose(chunk_logits, np.concatenate(step_logits, axis=1), rtol=1e-6, atol=1e-6)


class TestLosslessGreedy:
    """Spec-greedy ≡ plain-greedy — the acceptance-defining oracle."""

    @pytest.mark.parametrize("family,k", [(MTP, 4), (DSHARK, 8)])
    @pytest.mark.parametrize("seed", [3, 11])
    def test_greedy_tokens_identical(self, family: str, k: int, seed: int) -> None:
        model = _make_model(seed=seed)
        plain = TextGenerator(model, max_new_tokens=30, temperature=0.0).generate_greedy(PROMPT)
        drafter = _make_drafter(model, family, k)
        seq, stats = SpeculativeGenerator(model, drafter).generate_greedy(PROMPT, 30, k=k)
        np.testing.assert_array_equal(seq, plain)
        assert seq.shape == (1, 4 + 30)
        assert stats["n_rounds"] >= 1
        assert stats["spec"] == family

    @pytest.mark.parametrize("family", [MTP, DSHARK])
    def test_smaller_k_still_lossless(self, family: str) -> None:
        """The k slider: truncation must not break the contract."""
        model = _make_model(seed=42)
        plain = TextGenerator(model, max_new_tokens=20, temperature=0.0).generate_greedy(PROMPT)
        drafter = _make_drafter(model, family, 8)
        for kk in (1, 2, 3, 8):
            seq, _ = SpeculativeGenerator(model, drafter).generate_greedy(PROMPT, 20, k=kk)
            np.testing.assert_array_equal(seq, plain)

    @pytest.mark.parametrize("family", [MTP, DSHARK])
    def test_max_new_tokens_respected_mid_round(self, family: str) -> None:
        """A round that would overshoot n must be truncated exactly at n."""
        model = _make_model(seed=9)
        drafter = _make_drafter(model, family, 8)
        seq, _stats = SpeculativeGenerator(model, drafter).generate_greedy(PROMPT, 5, k=8)
        assert seq.shape == (1, 4 + 5)


class TestRejectionSampling:
    """The NumPy theorem path — mechanics (distribution identity needs many
    samples; this pins the shapes, the accept/reject bookkeeping, and that
    temperature flows through)."""

    @pytest.mark.parametrize("family", [MTP, DSHARK])
    def test_sampled_runs_and_shapes(self, family: str) -> None:
        model = _make_model(seed=13)
        drafter = _make_drafter(model, family, 4)
        seq, stats = SpeculativeGenerator(model, drafter).generate_sampled(PROMPT, 12, temperature=0.8, seed=1)
        assert seq.shape == (1, 4 + 12)
        assert stats["mode"] == "sampled"
        assert stats["temperature"] == 0.8
        for r in stats["rounds"]:
            assert len(r["accept_mask"]) == r["k_eff"]
            assert r["accepted"] <= r["k_eff"]


class TestSurvivalSchedule:
    """DSpark's confidence schedule (the second teaching point)."""

    def test_schedule_caps_and_updates(self) -> None:
        sched = _SurvivalSchedule(4)
        assert sched.verify_len() == 4  # no data → full block
        # Every round accepts exactly 2 of 4 → p(1)=p(2)=1.0, p(3)=p(4)=0.0.
        for _ in range(10):
            sched.update(2)
        assert sched.verify_len() == 2
        # Improve: now 3 survive → the cap grows to 3.
        for _ in range(50):
            sched.update(3)
        assert sched.verify_len() == 3
        state = sched.state()
        assert state["target"] == SCHEDULE_TARGET
        assert state["rounds"] == 60
        assert len(state["survival"]) == 5  # index 0..4

    @pytest.mark.parametrize("family,expect_scheduled", [(MTP, False), (DSHARK, True)])
    def test_engine_schedule_presence(self, family: str, expect_scheduled: bool) -> None:
        """DSpark runs confidence-scheduled by default; MTP does not."""
        model = _make_model(seed=21)
        drafter = _make_drafter(model, family, 8)
        _seq, stats = SpeculativeGenerator(model, drafter).generate_greedy(PROMPT, 10, k=8)
        assert (stats["schedule"] is not None) == expect_scheduled


class TestSidecarRoundTrip:
    """The checkpoint seam: save → load → identical draft; corruption fails fast."""

    @pytest.mark.parametrize("family", [MTP, DSHARK])
    def test_save_load_round_trip(self, tmp_path, family: str) -> None:
        model = _make_model(seed=3)
        meta = make_drafter_meta(family, model.config, block_size=4 if family == MTP else 8, ff_dim=48)
        cls = MTPDrafter if family == MTP else DSparkDrafter
        drafter = cls(meta, model.embedding.weight, model.lm_head_weight, seed=5)
        params = drafter.get_all_parameters()

        # params match the documented key scheme exactly
        expected = {key: shape for key, shape, _t in expected_drafter_params(meta)}
        assert set(params) == set(expected)
        for key, shape in expected.items():
            assert tuple(np.asarray(params[key]).shape) == shape

        save_drafter(tmp_path, meta, params)
        meta2, params2 = load_drafter(tmp_path)
        drafter2 = cls(meta2, model.embedding.weight, model.lm_head_weight, seed=5)
        drafter2.load_from_numpy_dict(params2)

        hidden = np.random.default_rng(0).standard_normal((1, model.config.embed_dim)).astype(np.float32)
        tok = np.array([7], dtype=np.int32)
        drafter.reset()
        drafter2.reset()
        t1, p1 = drafter.draft(tok, hidden, min(4, meta.block_size))
        t2, p2 = drafter2.draft(tok, hidden, min(4, meta.block_size))
        np.testing.assert_array_equal(t1, t2)
        for a, b in zip(p1, p2, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_corrupt_sidecar_fails_fast(self, tmp_path) -> None:
        model = _make_model(seed=3)
        meta = make_drafter_meta(MTP, model.config, block_size=4, ff_dim=48)
        params = MTPDrafter(meta, model.embedding.weight, model.lm_head_weight).get_all_parameters()
        # Drop a key → validation must raise, not load garbage.
        save_drafter(tmp_path, meta, params)
        with np.load(str(tmp_path / "draft.npz")) as f:
            loaded = {k: f[k] for k in f.files if k != "mtp.norm.weight"}
        with pytest.raises(ValueError, match="missing"):
            save_drafter(tmp_path, meta, loaded)
        # Extra key → also rejected.
        loaded["mtp.bogus.weight"] = np.zeros(3)
        with pytest.raises(ValueError, match="unexpected"):
            save_drafter(tmp_path, meta, loaded)
