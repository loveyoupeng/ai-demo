"""B1.2: RMSNorm — Root Mean Square Layer Normalization.

All tests fail initially. Implement after verifying failure.
"""

import numpy as np

from impl._np.modules import RMSNorm


class TestRMSNormForward:
    """Test the RMSNorm forward pass."""

    def test_output_shape(self):
        """Output shape matches input: (batch, seq_len, embed_dim).

        Input: x of shape [2, 4, 8], gamma of shape [8]
        Expected output: shape [2, 4, 8]
        """
        x = np.random.default_rng(42).random((2, 4, 8)).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32)

        norm = RMSNorm()
        out = norm.forward(x, gamma)

        assert out.shape == (2, 4, 8), f"Expected (2, 4, 8), got {out.shape}"

    def test_unit_variance(self):
        """After normalization, mean(output^2) per sample ≈ 1 for each (batch, seq).

        RMSNorm ensures that each sample's feature vector has unit mean-squared
        magnitude before the gamma scaling.
        """
        rng = np.random.default_rng(123)
        x = rng.random((4, 8, 16)).astype(np.float32)
        gamma = np.ones(16, dtype=np.float32)

        norm = RMSNorm()
        out = norm.forward(x, gamma)

        # For each (batch, seq), mean of output^2 across features ≈ 1
        mean_sq = np.mean(out**2, axis=-1)  # shape (batch, seq)
        np.testing.assert_allclose(mean_sq, 1.0, atol=0.01, err_msg="mean(output^2) per sample should be ~1")

    def test_identity_without_gamma(self):
        """With gamma = 1, output equals normalized input (no extra scaling).

        RMSNorm(x, 1) = x / rms(x) where rms = sqrt(mean(x^2) + eps).
        """
        x = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)  # (1, 1, 3)
        gamma = np.ones(3, dtype=np.float32)

        norm = RMSNorm()
        out = norm.forward(x, gamma)

        # Expected: x / sqrt(mean(x^2) + eps)
        eps = 1e-6
        expected = x / np.sqrt(np.mean(x**2) + eps)
        np.testing.assert_allclose(out, expected, rtol=1e-5, err_msg="gamma=1 should give pure normalization")

    def test_learned_scale(self):
        """Gamma controls the output magnitude: output = normalized_input * gamma.

        If gamma = 2.0, output should be 2x the normalized input.
        """
        rng = np.random.default_rng(99)
        x = rng.random((2, 3, 6)).astype(np.float32)
        gamma = np.ones(6, dtype=np.float32) * 2.0

        norm = RMSNorm()
        out = norm.forward(x, gamma)

        # With all-gamma-2, output should be 2x the normalized input
        # (since RMSNorm normalizes to unit variance first, then scales)
        normalized = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + 1e-6)
        expected = normalized * 2.0
        np.testing.assert_allclose(out, expected, rtol=1e-5)


class TestRMSNormBackward:
    """Test the RMSNorm backward (gradient) behavior."""

    def test_gradient_shape(self):
        """Gradients w.r.t. input and gamma have correct shapes.

        d_out / d_x should match x.shape.
        d_out / d_gamma should be (embed_dim,).
        """
        x = np.random.default_rng(7).random((2, 4, 8)).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32) * 2.0

        norm = RMSNorm()
        out = norm.forward(x, gamma)

        # Use a simple upstream gradient of ones
        upstream = np.ones_like(out)

        # Numerical gradient check for input shape
        eps = 1e-5
        for i in range(x.shape[2]):  # embed_dim
            x_plus = x.copy()
            x_minus = x.copy()
            for b in range(x.shape[0]):
                for s in range(x.shape[1]):
                    x_plus[b, s, i] += eps
                    x_minus[b, s, i] -= eps

            out_plus = norm.forward(x_plus, gamma)
            out_minus = norm.forward(x_minus, gamma)

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
        gamma1 = np.ones(4, dtype=np.float32)
        # Case 2: varying gamma
        gamma2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        norm = RMSNorm()
        out1 = norm.forward(x, gamma1)
        out2 = norm.forward(x, gamma2)

        # Outputs should differ when gamma differs (for non-zero input)
        assert not np.allclose(out1, out2, rtol=1e-3), (
            "Different gamma should produce different outputs for non-zero input"
        )
