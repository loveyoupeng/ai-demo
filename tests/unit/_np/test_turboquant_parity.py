"""TurboQuant KV-cache parity-budget test.

The TurboQuant cache (1-bit sign + per-channel scale) is a *lossy* alternative
to the naive full-precision cache. This test:

1. Steps the last prompt token with the naive cache (the reference, exact path).
2. Steps the same token with the TurboQuant cache.
3. Asserts the TurboQuant output is *degraded* (not identical) — the cache is
   genuinely lossy, not a silent no-op.
4. Asserts the degradation stays within a documented parity budget: the
   max-abs logit difference is bounded. The bound is loose on purpose (1-bit
   quantization of K/V can shift logits noticeably); the point is to catch a
   *regression* (e.g. the cache silently breaking) rather than to certify
   near-exact parity.
"""

from __future__ import annotations

import numpy as np

from impl._np.model import NumPyModel
from shared.config import TransformerConfig


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


class TestTurboQuantParityBudget:
    """TurboQuant cache vs naive cache: degraded but bounded."""

    def test_turboquant_degraded_but_bounded(self) -> None:
        """Step the last prompt token: TurboQuant differs from naive but stays within budget.

        The parity budget: the max-abs logit difference at the last prompt
        token (where both caches have attended to the same 5 prompt tokens)
        must be <= 2.0. This is a loose bound — 1-bit quantization of K/V in
        a small untrained model can shift logits by ~0.5-1.0 — but a broken
        cache (wrong dequantize, missing GQA repeat, wrong position) would
        exceed it by a wide margin.
        """
        model = _make_model(seed=0)
        prompt = np.array([[3, 7, 1, 9, 2]], dtype=np.int32)

        # Naive cache (reference): thread the first 4 tokens, then step the
        # last token (position 4) to get the reference logits.
        cache_naive = model.make_cache(1, quantize=False)
        for i in range(4):
            model.forward_step(prompt[:, [i]], i, cache_naive, quantize=False)
        step_logits_naive = model.forward_step(prompt[:, [-1]], 4, cache_naive, quantize=False)  # (1, 1, V)

        # TurboQuant cache (lossy): same threading, but the K/V are 1-bit
        # quantized at each step.
        cache_turbo = model.make_cache(1, quantize=True)
        for i in range(4):
            model.forward_step(prompt[:, [i]], i, cache_turbo, quantize=True)
        step_logits_turbo = model.forward_step(prompt[:, [-1]], 4, cache_turbo, quantize=True)  # (1, 1, V)

        max_diff = float(np.abs(step_logits_naive - step_logits_turbo).max())

        # The TurboQuant output must differ from the naive output (the cache
        # is genuinely lossy, not a silent no-op).
        assert max_diff > 0.0, (
            f"TurboQuant cache is not lossy: max-abs logit diff = {max_diff:.4f} (expected > 0). "
            f"Check that quantize=True actually 1-bit quantizes the K/V."
        )

        # The parity budget: the degradation must be bounded.
        assert max_diff <= 2.0, (
            f"TurboQuant cache parity budget exceeded: max-abs logit diff = {max_diff:.4f} "
            f"(budget: 2.0). The cache is too lossy or broken."
        )

    def test_turboquant_greedy_differs_from_naive(self) -> None:
        """Greedy generation: TurboQuant output differs from naive (the cache is lossy)."""
        from impl._np.inference import TextGenerator

        model = _make_model(seed=0)
        prompt = np.array([[3, 7, 1, 9, 2]], dtype=np.int32)

        gen_naive = TextGenerator(model, max_new_tokens=8, temperature=0.0, quantize=False)
        out_naive = gen_naive.generate_greedy(prompt)

        gen_turbo = TextGenerator(model, max_new_tokens=8, temperature=0.0, quantize=True)
        out_turbo = gen_turbo.generate_greedy(prompt)

        # The TurboQuant output must differ from the naive output (the cache
        # is genuinely lossy, not a silent no-op).
        assert not np.array_equal(out_naive, out_turbo), (
            "TurboQuant output is identical to naive — the cache is not lossy. "
            "Check that quantize=True actually 1-bit quantizes the K/V."
        )
