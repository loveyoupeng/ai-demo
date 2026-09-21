"""B4.1: MoE — Mixture of Experts with top-k routing.

Each expert is a SwiGLU FFN. The router assigns each token to top-k experts.
"""

import numpy as np

from impl._np.moe import MixtureOfExperts


class TestMoEForward:
    """Test MoE forward pass."""

    def test_output_shape(self):
        """Input [B, S, D] → output [B, S, D]."""
        x = np.random.default_rng(0).random((2, 4, 16)).astype(np.float32)

        moe = MixtureOfExperts(embed_dim=16, n_experts=4, ff_dim=32, top_k=2, seed=0)
        out = moe.forward(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_top_k_selection(self):
        """Only top-k experts should have non-zero routing weight.

        For each token, exactly k experts are selected. Using random input
        ensures different experts produce different outputs.
        """
        x = np.random.default_rng(123).random((2, 4, 8)).astype(np.float32)

        # Create a MoE with controlled weights so we can verify routing
        moe_k2 = MixtureOfExperts(embed_dim=8, n_experts=4, ff_dim=16, top_k=2, seed=42)
        out_k2 = moe_k2.forward(x)

        # Create same MoE with top_k=1 — fewer experts should fire
        moe_k1 = MixtureOfExperts(embed_dim=8, n_experts=4, ff_dim=16, top_k=1, seed=42)
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

        moe = MixtureOfExperts(embed_dim=8, n_experts=4, ff_dim=16, top_k=2, seed=0)

        # Get baseline output
        baseline = moe.forward(x.copy())

        # Set router to extreme values that would guarantee different routing
        moe.gate = np.full((8, 4), 100.0, dtype=np.float32)
        moe.gate[:, 0] = 1000.0
        out_routed = moe.forward(x.copy())

        assert not np.allclose(baseline, out_routed, atol=1e-2), "Drastically changing router should change output"
        assert np.all(np.isfinite(out_routed)), "MoE output should be finite"

    def test_deterministic(self):
        """Same input, same seed → same output."""
        x = np.random.default_rng(99).random((1, 5, 8)).astype(np.float32)

        moe1 = MixtureOfExperts(embed_dim=8, n_experts=4, ff_dim=16, top_k=2, seed=7)
        moe2 = MixtureOfExperts(embed_dim=8, n_experts=4, ff_dim=16, top_k=2, seed=7)

        out1 = moe1.forward(x.copy())
        out2 = moe2.forward(x.copy())

        np.testing.assert_array_equal(out1, out2, err_msg="Same seed should produce identical outputs")


class TestSharedExperts:
    """DeepSeek-style shared experts (ADR 0002): ungated, averaged, additive."""

    def test_zeroed_shared_experts_match_plain_moe(self):
        """Zero-weight shared experts contribute exactly nothing."""
        x = np.random.default_rng(1).normal(size=(2, 4, 8)).astype(np.float32)
        moe = MixtureOfExperts(embed_dim=8, n_experts=3, ff_dim=16, top_k=1, seed=5, n_shared_experts=2)
        ref = MixtureOfExperts(embed_dim=8, n_experts=3, ff_dim=16, top_k=1, seed=5)
        for shared in moe.shared_experts:
            shared.gate_proj[...] = 0.0
            shared.up_proj[...] = 0.0
            shared.down_proj[...] = 0.0
        np.testing.assert_array_equal(moe.forward(x.copy()), ref.forward(x.copy()))

    def test_shared_branch_is_additive_and_averaged(self):
        """out = routed + mean(E_shared(x)) — independent of the router."""
        x = np.random.default_rng(1).normal(size=(2, 4, 8)).astype(np.float32)
        moe = MixtureOfExperts(embed_dim=8, n_experts=3, ff_dim=16, top_k=1, seed=5, n_shared_experts=2)
        ref = MixtureOfExperts(embed_dim=8, n_experts=3, ff_dim=16, top_k=1, seed=5)
        routed = ref.forward(x.copy())
        shared_sum = moe.shared_experts[0].forward(x) + moe.shared_experts[1].forward(x)
        expected = routed + shared_sum / 2
        np.testing.assert_allclose(moe.forward(x.copy()), expected, rtol=1e-5, atol=1e-6)

    def test_shared_experts_change_output(self):
        """A trained (non-zero) shared expert must actually affect the output."""
        x = np.random.default_rng(1).normal(size=(2, 4, 8)).astype(np.float32)
        moe = MixtureOfExperts(embed_dim=8, n_experts=3, ff_dim=16, top_k=1, seed=5, n_shared_experts=1)
        ref = MixtureOfExperts(embed_dim=8, n_experts=3, ff_dim=16, top_k=1, seed=5)
        assert not np.allclose(moe.forward(x.copy()), ref.forward(x.copy()), atol=1e-4)
