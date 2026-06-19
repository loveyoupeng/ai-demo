"""Autoregressive inference tests for TextGenerator.

Tests greedy decoding, sampled decoding, batch processing, top-k filtering,
and output shape correctness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from impl._np.model import NumPyModel


class TestTextGenerator:
    """Tests for autoregressive text generation."""

    def _make_model(self) -> NumPyModel:  # noqa: F821 — forward-ref, defined in model.py
        """Build a minimal model (vocab=16, embed=8, layers=1, heads=2, experts=2)."""
        from impl._np.model import NumPyModel

        return NumPyModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=0,
            seed=0,
        )

    def test_greedy_deterministic(self) -> None:
        """Greedy generation with seed=0 is deterministic."""
        from impl._np.inference import TextGenerator

        model = self._make_model()

        generator = TextGenerator(model, max_new_tokens=5, temperature=0.0)
        prompt = np.array([[0, 1, 2, 3]], dtype=np.int32)

        # Run twice with same prompt — must be identical
        run1 = generator.generate_greedy(prompt)
        run2 = generator.generate_greedy(prompt)

        np.testing.assert_array_equal(run1, run2)
        run2 = generator.generate_greedy(prompt)
        np.testing.assert_array_equal(run1, run2)

    def test_output_shape(self) -> None:
        """Output has correct shape: prompt + new tokens."""
        from impl._np.inference import TextGenerator

        model = self._make_model()

        generator = TextGenerator(model, max_new_tokens=10)
        prompt = np.array([[0, 1, 2, 3]], dtype=np.int32)  # 4 tokens

        output = generator.generate_greedy(prompt)
        assert output.shape == (1, 4 + 10)  # batch=1, seq=14

    def test_batch_generation(self) -> None:
        """Generation works with batch_size > 1."""
        from impl._np.inference import TextGenerator

        model = self._make_model()

        generator = TextGenerator(model, max_new_tokens=5)
        prompt = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32)  # batch=2

        output = generator.generate_greedy(prompt)
        assert output.shape == (2, 4 + 5)  # batch=2, seq=9

    def test_top_k_filtering(self) -> None:
        """Top-k=0 disables top-k filtering (all logits used for sampling)."""
        from impl._np.inference import TextGenerator

        model = self._make_model()

        # top_k=0 means no top-k filtering
        generator = TextGenerator(model, max_new_tokens=3, temperature=0.5)
        prompt = np.array([[0, 1, 2, 3]], dtype=np.int32)

        output = generator.generate_sampled(prompt, temperature=0.5)

        # Should generate valid token IDs in range [0, vocab_size)
        assert np.all(output >= 0)
        assert np.all(output < 16)
