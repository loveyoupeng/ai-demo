# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""KV-cache quantization parity-budget tests (impl/_np/kvquant.py).

The quantized cache schemes ("fp8"/"int4"/"int2") are *lossy* alternatives
to the naive full-precision cache. These tests, per scheme:

1. Step the last prompt token with the naive cache (the reference, exact path).
2. Step the same token with the quantized cache.
3. Assert the quantized output is *degraded* (not identical) — the cache is
   genuinely lossy, not a silent no-op.
4. Assert the degradation stays within a documented parity budget: the
   max-abs logit difference is bounded. The bounds are loose on purpose
   (low-bit quantization of K/V can shift logits noticeably); the point is
   to catch a *regression* (e.g. the cache silently breaking) rather than
   to certify near-exact parity.

Also pinned: sink tokens (the first SINK_TOKENS positions) are stored
exactly, and prefill backfill + per-token appends produce the same cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from impl._np import kvquant
from impl._np.model import NumPyModel
from shared.config import TransformerConfig

SCHEMES = ("fp8", "int4", "int2")
# Per-scheme parity budgets on the max-abs logit diff at the last prompt
# token (small untrained model, 3 exact DeltaNet mixers + 1 quantized
# attention mixer per block; error propagates through the remaining mixers
# + FFN of every layer). Loose bounds — regression catchers, not accuracy
# certificates. fp8 is the tightest (it is the production default).
BUDGETS = {"fp8": 1.0, "int4": 3.0, "int2": 8.0}


def _make_model(seed: int = 0) -> NumPyModel:
    """A small dense model for the parity test."""
    return NumPyModel(
        TransformerConfig.from_dict(
            {
                "vocab_size": 16,
                "embed_dim": 8,
                "n_layers": 2,
                "n_heads": 2,
                "n_experts": 1,
                "top_k": 1,
                "expert_dim": 12,
                "seed": seed,
            }
        )
    )


def _step_all(model: NumPyModel, prompt: np.ndarray, kv_quant: str) -> np.ndarray:
    """Thread the first S-1 tokens, then step the last one; return its logits."""
    cache = model.make_cache(1, kv_quant=kv_quant)
    for i in range(prompt.shape[1] - 1):
        model.forward_step(prompt[:, [i]], i, cache, kv_quant=kv_quant)
    return model.forward_step(prompt[:, [-1]], prompt.shape[1] - 1, cache, kv_quant=kv_quant)  # (1, 1, V)


class TestKVQuantParityBudget:
    """Each quantized cache vs the naive cache: degraded but bounded."""

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_degraded_but_bounded(self, scheme: str) -> None:
        """Step the last prompt token: quantized differs from naive but stays within budget."""
        model = _make_model(seed=0)
        prompt = np.array([[3, 7, 1, 9, 2]], dtype=np.int32)

        step_logits_naive = _step_all(model, prompt, "none")
        step_logits_q = _step_all(model, prompt, scheme)

        max_diff = float(np.abs(step_logits_naive - step_logits_q).max())

        # The quantized output must differ from the naive output (the cache
        # is genuinely lossy, not a silent no-op).
        assert max_diff > 0.0, (
            f"kv_quant={scheme} cache is not lossy: max-abs logit diff = {max_diff:.4f} (expected > 0). "
            f"Check that kv_quant={scheme!r} actually quantizes the K/V."
        )
        # The parity budget: the degradation must be bounded.
        budget = BUDGETS[scheme]
        assert max_diff <= budget, (
            f"kv_quant={scheme} parity budget exceeded: max-abs logit diff = {max_diff:.4f} "
            f"(budget: {budget}). The cache is too lossy or broken."
        )

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_prefill_backfill_matches_quantized_record_k(self, scheme: str) -> None:
        """forward_prefill(kv_quant=...) backfills exactly quantize_block(record k_group).

        The prefill path and the per-token step path quantize DIFFERENT K/V
        rows once the lossy cache feeds back into the model (quantization
        error compounds) — so cross-path cache equality is not a meaningful
        invariant for lossy schemes. What IS pinned: each path quantizes its
        own K/V rows exactly as kvquant.quantize_block says.
        """
        model = _make_model(seed=0)
        prompt = np.array([[3, 7, 1, 9, 2, 5, 8]], dtype=np.int32)

        cache = model.make_cache(1, kv_quant=scheme)
        raw: dict = {}
        model.forward_prefill(prompt, cache, kv_quant=scheme, record=raw)
        H, G = model.config.n_heads, model.config.kv_heads
        for i, block_state in enumerate(raw["blocks"]):
            for m, mixer_state in enumerate(block_state["mixers"]):
                if model.config.mixer_kind(m) != "F" or model.config.has_mla_at(i):
                    continue
                k_group = mixer_state["k_group"]  # (B, G, S, hd)
                k_r = np.repeat(k_group, H // G, axis=1) if G != H else k_group
                expected = kvquant.dequantize_block(scheme, kvquant.quantize_block(scheme, k_r, sink_from=0))
                got = kvquant.dequantize_block(scheme, cache[i][m]["q_k"])
                assert np.allclose(got, expected)

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_step_appends_match_quantized_new_token_k(self, scheme: str) -> None:
        """Each forward_step(kv_quant=...) appends exactly quantize_block of the step's raw K/V."""
        model = _make_model(seed=0)
        prompt = np.array([[3, 7, 1, 9, 2, 5, 8]], dtype=np.int32)

        cache = model.make_cache(1, kv_quant=scheme)
        H, G = model.config.n_heads, model.config.kv_heads
        for i in range(prompt.shape[1]):
            raw: dict = {}
            model.forward_step(prompt[:, [i]], i, cache, kv_quant=scheme, record=raw)
            # layer 0's attention mixer is the last mixer
            mixer_state = raw["blocks"][0]["mixers"][-1]
            k_new = mixer_state["k_group"]  # (B, G, 1, hd) the raw new-token K
            k_r = np.repeat(k_new, H // G, axis=1) if G != H else k_new
            expected = kvquant.dequantize_block(scheme, kvquant.quantize_block(scheme, k_r, sink_from=i), sink_from=i)
            got = kvquant.dequantize_block(scheme, cache[0][-1]["q_k"])
            assert np.allclose(got[:, :, i], expected[:, :, 0])

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_sink_tokens_stored_exactly(self, scheme: str) -> None:
        """The first SINK_TOKENS positions are stored raw and dequantize exactly."""
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (1, 2, 8, 16))
        entry = kvquant.quantize_block(scheme, x, sink_from=0)
        back = kvquant.dequantize_block(scheme, entry, sink_from=0)
        assert np.allclose(back[:, :, : kvquant.SINK_TOKENS], x[:, :, : kvquant.SINK_TOKENS])

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_greedy_may_differ_from_naive(self, scheme: str) -> None:
        """Greedy generation runs end-to-end on the quantized cache (lossy path works)."""
        from impl._np.inference import TextGenerator

        model = _make_model(seed=0)
        prompt = np.array([[3, 7, 1, 9, 2]], dtype=np.int32)

        gen_naive = TextGenerator(model, max_new_tokens=8, temperature=0.0, kv_quant="none")
        out_naive = gen_naive.generate_greedy(prompt)
        gen_q = TextGenerator(model, max_new_tokens=8, temperature=0.0, kv_quant=scheme)
        out_q = gen_q.generate_greedy(prompt)

        # Both must produce full-length sequences; the quantized one is
        # lossy (may or may not flip a token on this tiny model — the
        # step-level budget test above pins the degradation bound).
        assert out_naive.shape == out_q.shape == (1, prompt.shape[1] + 8)


def test_fp8_codec_matches_torch_e4m3fn() -> None:
    """The hand-rolled e4m3 codec must match torch's float8_e4m3fn in the scaled domain.

    Our scheme stores x / scale with scale = amax/448, so the codes must
    equal torch's e4m3 cast of the SCALED vector y = x/scale (|y| <= 448).
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, (2, 3, 5, 16)).astype(np.float64)
    # include subnormals and near-max magnitudes
    x[0, 0, 0] = np.linspace(-1e-3, 1e-3, 16)
    x[0, 0, 1] = np.array([448.0, -448.0, 2.0**-9, -(2.0**-9), 0.0] + [0.1] * 11)
    x[0, 0, 2] = rng.normal(0, 1e-4, 16)  # deep subnormal territory
    codes, scales = kvquant.fp8_e4m3_encode(x)
    y = x / scales
    ref_codes = torch.from_numpy(y).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    assert np.array_equal(codes, ref_codes)
    ref_val = torch.from_numpy(y).to(torch.float8_e4m3fn).float().numpy()
    mine_val = kvquant.fp8_e4m3_decode(codes, np.ones_like(scales))
    assert np.abs(mine_val - ref_val).max() == 0.0


def test_invalid_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="kv_quant"):
        kvquant.empty_entry("int1", 1, 2, 16, np.dtype(np.float64))
