"""B6.2: Full NumPyModel — forward + backward gradient computation.

Tests verify output shapes, gradient existence, and gradient shapes.
"""

import numpy as np

from impl._np.model import NumPyModel


class TestNumPyModelForward:
    """Test complete model forward pass."""

    def test_output_shape(self):
        """Input [B, S] → output [B, S, V]."""
        input_ids = np.random.default_rng(0).integers(0, 16, (2, 4), dtype=np.int32)

        model = NumPyModel(
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
        logits = model.forward(input_ids)

        assert logits.shape == (2, 4, 16), f"Expected (2, 4, 16), got {logits.shape}"

    def test_with_embedding(self):
        """Different inputs → different outputs via embedding."""
        input_ids = np.array([[0, 1, 2, 3]], dtype=np.int32)
        alt_ids = np.array([[10, 11, 12, 13]], dtype=np.int32)

        model = NumPyModel(
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

        logits = model.forward(input_ids)
        alt_logits = model.forward(alt_ids)

        assert not np.allclose(logits, alt_logits, atol=1e-3), (
            "Different inputs should produce different outputs"
        )

    def test_gradient_existence(self):
        """All parameters have gradients after backward."""
        input_ids = np.random.default_rng(5).integers(0, 16, (1, 2), dtype=np.int32)
        targets = np.random.default_rng(6).integers(0, 16, (1, 2), dtype=np.int32)

        model = NumPyModel(
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

        logits = model.forward(input_ids)
        grads = model.backward(logits, targets, input_ids)

        params = model.get_all_parameters()
        for name, param in params.items():
            assert name in grads, f"Missing gradient for {name}"
            assert grads[name].shape == param.shape, (
                f"Gradient shape mismatch for {name}: {grads[name].shape} vs {param.shape}"
            )
        # All gradients should be finite (some may be zero due to numerical precision)
        for name, grad in grads.items():
            assert np.all(np.isfinite(grad)), f"Non-finite gradient for {name}"

    def test_small_model(self):
        """Works with minimal config (vocab=16, D=32, layers=1, heads=2)."""
        input_ids = np.random.default_rng(10).integers(0, 16, (1, 3), dtype=np.int32)

        model = NumPyModel(
            vocab_size=16,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=0,
            seed=42,
        )

        logits = model.forward(input_ids)
        assert logits.shape == (1, 3, 16), f"Expected (1, 3, 16), got {logits.shape}"
        assert np.all(np.isfinite(logits)), "Logits should be finite"


class TestNumPyModelBackward:
    """Test backward pass gradient computation."""

    def test_gradient_shapes(self):
        """All gradient shapes match parameter shapes."""
        input_ids = np.random.default_rng(20).integers(0, 16, (1, 2), dtype=np.int32)
        targets = np.random.default_rng(21).integers(0, 16, (1, 2), dtype=np.int32)

        model = NumPyModel(
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

        logits = model.forward(input_ids)
        grads = model.backward(logits, targets, input_ids)

        params = model.get_all_parameters()
        for name, param in params.items():
            assert name in grads, f"Missing {name} in grads"
            assert grads[name].shape == param.shape, (
                f"Shape mismatch: {name} grads {grads[name].shape} != param {param.shape}"
            )
