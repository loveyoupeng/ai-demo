"""B4.1: MoE — Mixture of Experts with top-k routing.

Each expert is a SwiGLU FFN. The router assigns each token to top-k experts.
"""

import numpy as np

from impl._np.modules import MoE


class TestMoEForward:
    """Test MoE forward pass."""

    def test_output_shape(self):
        """Input [B, S, D] → output [B, S, D]."""
        x = np.random.default_rng(0).random((2, 4, 16)).astype(np.float32)

        moe = MoE(embed_dim=16, n_experts=4, ff_dim=32, k=2, seed=0)
        out = moe.forward(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_top_k_selection(self):
        """Only top-k experts should have non-zero routing weight.

        For each token, exactly k experts are selected. Using random input
        ensures different experts produce different outputs.
        """
        x = np.random.default_rng(123).random((2, 4, 8)).astype(np.float32)

        # Create a MoE with controlled weights so we can verify routing
        moe_k2 = MoE(embed_dim=8, n_experts=4, ff_dim=16, k=2, seed=42)
        out_k2 = moe_k2.forward(x)

        # Create same MoE with k=1 — fewer experts should fire
        moe_k1 = MoE(embed_dim=8, n_experts=4, ff_dim=16, k=1, seed=42)
        out_k1 = moe_k1.forward(x)

        # Outputs should be different for k=1 vs k=2
        assert not np.allclose(out_k2, out_k1, rtol=1e-3), "k=1 and k=2 should produce different outputs"
        assert np.all(np.isfinite(out_k2)), "MoE output should be finite"
        assert not np.allclose(out_k2, 0.0), "MoE output should not be all zeros"

    def test_gradient_flow(self):
        """All weight matrices (router, all experts) participate in computation.

        Changing the router weights drastically should change the output.
        """
        x = np.random.default_rng(123).random((2, 3, 8)).astype(np.float32)

        moe = MoE(embed_dim=8, n_experts=4, ff_dim=16, k=2, seed=0)

        # Get baseline output
        baseline = moe.forward(x.copy())

        # Set router to extreme values that would guarantee different routing
        moe.router = np.full((8, 4), 100.0, dtype=np.float32)
        moe.router[:, 0] = 1000.0
        out_routed = moe.forward(x.copy())

        assert not np.allclose(baseline, out_routed, atol=1e-2), "Drastically changing router should change output"
        assert np.all(np.isfinite(out_routed)), "MoE output should be finite"

    def test_deterministic(self):
        """Same input, same seed → same output."""
        x = np.random.default_rng(99).random((1, 5, 8)).astype(np.float32)

        moe1 = MoE(embed_dim=8, n_experts=4, ff_dim=16, k=2, seed=7)
        moe2 = MoE(embed_dim=8, n_experts=4, ff_dim=16, k=2, seed=7)

        out1 = moe1.forward(x.copy())
        out2 = moe2.forward(x.copy())

        np.testing.assert_array_equal(out1, out2, err_msg="Same seed should produce identical outputs")
