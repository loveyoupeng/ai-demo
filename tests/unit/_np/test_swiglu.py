"""B1.4: SwiGLU Feedforward — SiLU(w1 @ x) * (w3 @ x) @ w2.

All tests fail initially. Implement after verifying failure.
"""

import numpy as np

from impl._np.modules import SwiGLUFFN


class TestSwiGLUFFNForward:
    """Test the SwiGLU feedforward forward pass."""

    def test_output_shape(self):
        """[batch, seq_len, embed_dim] → [batch, seq_len, embed_dim].

        Input x: (2, 4, 8), output must match (2, 4, 8) regardless of ff_dim.
        """
        x = np.random.default_rng(42).random((2, 4, 8)).astype(np.float32)
        ff_dim = 16  # 2x embed_dim

        layer = SwiGLUFFN(x.shape[-1], ff_dim, seed=0)
        out = layer.forward(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_gating_behavior(self):
        """The feedforward combines w1 and w3 projections through SiLU gating.

        If w1 produces zeros, the gating gate=SiLU(zero)=zero → whole output is zero.
        """
        x = np.ones((1, 1, 4), dtype=np.float32)
        ff_dim = 8

        layer = SwiGLUFFN(4, ff_dim, seed=0)
        out = layer.forward(x)

        # Output should not be all zeros (weights are randomly initialized, not zeroed)
        assert not np.allclose(out, 0.0), (
            "SwiGLU output should not be all zeros with random weights"
        )

    def test_ff_dim_independence(self):
        """Output shape is always [batch, seq_len, embed_dim] regardless of ff_dim."""
        x = np.random.default_rng(10).random((3, 5, 12)).astype(np.float32)

        for ff_dim in [4, 12, 24, 48]:
            layer = SwiGLUFFN(x.shape[-1], ff_dim, seed=0)
            out = layer.forward(x)
            assert out.shape == x.shape, f"ff_dim={ff_dim}: expected {x.shape}, got {out.shape}"

    def test_gradient_existence_w1(self):
        """SwiGLU computes through w1 — changing w1 should change output."""
        x = np.ones((1, 1, 4), dtype=np.float32)

        layer = SwiGLUFFN(4, 8, seed=0)
        out_base = layer.forward(x.copy())

        # Perturb the first weight matrix slightly
        w1_orig = layer.W1.copy()
        layer.W1 = w1_orig + 0.1

        out_perturbed = layer.forward(x.copy())

        assert not np.allclose(out_base, out_perturbed, rtol=1e-3), (
            "Perturbing W1 should change output"
        )

    def test_gradient_existence_w3(self):
        """SwiGLU computes through w3 — changing w3 should change output."""
        x = np.ones((1, 1, 4), dtype=np.float32)

        layer = SwiGLUFFN(4, 8, seed=0)
        out_base = layer.forward(x.copy())

        w3_orig = layer.W3.copy()
        layer.W3 = w3_orig + 0.1

        out_perturbed = layer.forward(x.copy())

        assert not np.allclose(out_base, out_perturbed, rtol=1e-3), (
            "Perturbing W3 should change output"
        )
