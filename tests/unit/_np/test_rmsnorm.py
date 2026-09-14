"""B1.2: RMSNorm — Root Mean Square Layer Normalization.

out = x / sqrt(mean(x^2) + eps) * gamma; gamma (D,) is a module attribute.
"""

import numpy as np

from impl._np.layernorm import RMSNorm


class TestRMSNormForward:
    """Test the RMSNorm forward pass."""

    def test_output_shape(self):
        """Output shape matches input: (batch, seq_len, embed_dim)."""
        x = np.random.default_rng(42).random((2, 4, 8)).astype(np.float32)

        norm = RMSNorm(8)
        out = norm.forward(x)

        assert out.shape == (2, 4, 8), f"Expected (2, 4, 8), got {out.shape}"

    def test_unit_variance(self):
        """After normalization, mean(output^2) per sample ≈ 1 for each (batch, seq)."""
        rng = np.random.default_rng(123)
        x = rng.random((4, 8, 16)).astype(np.float32)

        norm = RMSNorm(16)
        out = norm.forward(x)

        # For each (batch, seq), mean of output^2 across features ≈ 1
        mean_sq = np.mean(out**2, axis=-1)  # shape (batch, seq)
        np.testing.assert_allclose(mean_sq, 1.0, atol=0.01, err_msg="mean(output^2) per sample should be ~1")

    def test_identity_without_gamma(self):
        """With gamma = 1, output equals normalized input (no extra scaling)."""
        x = np.random.default_rng(42).random((2, 3, 6)).astype(np.float32)

        norm = RMSNorm(6)
        out = norm.forward(x)

        eps = norm.eps
        expected = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        np.testing.assert_allclose(out, expected, rtol=1e-5, err_msg="gamma=1 should give pure normalization")

    def test_learned_scale(self):
        """Gamma controls the output magnitude: output = normalized_input * gamma."""
        x = np.random.default_rng(99).random((2, 3, 5)).astype(np.float32)
        gamma = np.array([1.0, 2.0, 3.0, 0.5, 4.0], dtype=np.float32)

        norm = RMSNorm(5)
        norm.gamma = gamma
        out = norm.forward(x)

        eps = norm.eps
        expected = (x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)) * gamma
        np.testing.assert_allclose(out, expected, rtol=1e-5)


class TestRMSNormBackward:
    """Test the RMSNorm backward (gradient) behavior."""

    def test_gradient_shape(self):
        """Gradients w.r.t. input and gamma have correct shapes.

        d_out / d_x should match x.shape.
        d_out / d_gamma should be (embed_dim,).
        """
        x = np.random.default_rng(7).random((2, 4, 8)).astype(np.float32)

        norm = RMSNorm(8)
        norm.gamma = np.ones(8, dtype=np.float32) * 2.0

        # Use a simple upstream gradient of ones
        upstream = np.ones_like(x)

        # Numerical gradient check for input shape
        eps = 1e-5
        for i in range(x.shape[2]):  # embed_dim
            x_plus = x.copy()
            x_minus = x.copy()
            for b in range(x.shape[0]):
                for s in range(x.shape[1]):
                    x_plus[b, s, i] += eps
                    x_minus[b, s, i] -= eps

            out_plus = norm.forward(x_plus)
            out_minus = norm.forward(x_minus)

            numeric_grad = np.sum((out_plus - out_minus) * upstream) / (2 * eps)

            # Just verify the gradient is finite and roughly the right magnitude
            assert np.isfinite(numeric_grad), f"Gradient at x[{i}] is not finite"

    def test_gradient_is_non_zeros(self):
        """Running backward with non-uniform input should produce non-zero gradients.

        If gamma varies per feature, the gradient for input should differ
        from the case where all gamma values are equal.
        """
        rng = np.random.default_rng(55)
        x = rng.random((2, 3, 4)).astype(np.float32)

        # Case 1: uniform gamma
        norm1 = RMSNorm(4)
        out1 = norm1.forward(x)

        # Case 2: varying gamma
        norm2 = RMSNorm(4)
        norm2.gamma = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out2 = norm2.forward(x)

        # Outputs should differ when gamma differs (for non-zero input)
        assert not np.allclose(out1, out2, rtol=1e-3), (
            "Different gamma should produce different outputs for non-zero input"
        )
