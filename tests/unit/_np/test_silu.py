"""B1.3: SiLU (Swish) activation — element-wise x * sigmoid(x).

All tests fail initially. Implement after verifying failure.
"""

import numpy as np

from impl._np.modules import SiLULayer


class TestSiLULayer:
    """Test the SiLU layer forward pass."""

    def test_output_shape(self):
        """Output shape matches input exactly."""
        x = np.random.default_rng(42).random((2, 4, 8)).astype(np.float32)

        layer = SiLULayer()
        out = layer.forward(x)

        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    def test_output_at_zero(self):
        """SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0."""
        x = np.zeros((3, 3, 3), dtype=np.float32)

        layer = SiLULayer()
        out = layer.forward(x)

        np.testing.assert_array_equal(out, 0.0, err_msg="SiLU(0) should be exactly 0")

    def test_output_range_large_positive(self):
        """For large positive x, SiLU(x) ≈ x (approaches identity)."""
        x = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

        layer = SiLULayer()
        out = layer.forward(x)

        # For large x, sigmoid(x) ≈ 1, so SiLU(x) ≈ x
        np.testing.assert_allclose(out, x, rtol=0.01, err_msg="SiLU(x) should closely follow x for large positive x")

    def test_output_range_negative(self):
        """For large negative x, SiLU(x) ≈ 0 (suppressed)."""
        x = np.array([[[-10.0, -5.0, -2.0]]], dtype=np.float32)

        layer = SiLULayer()
        out = layer.forward(x)

        # For negative x, sigmoid(x) is near 0, so SiLU(x) ≈ 0 (slightly negative)
        assert np.all(out <= 0.0), f"SiLU of negative input should be <= 0, got {out}"
        # For x <= -2, SiLU should be very close to 0
        np.testing.assert_allclose(out, 0.0, atol=0.25, err_msg="SiLU(-large) should be near zero")
