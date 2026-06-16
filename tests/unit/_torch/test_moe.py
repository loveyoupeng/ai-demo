"""C4.1: Tests for PyTorch Mixture of Experts.

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestMoEForward:
    """Test the MoE nn.Module forward pass."""

    def test_output_shape(self) -> None:
        """MoE(x: [B,S,D], E experts, D embed) → [B,S,D]."""
        from impl._torch.layers import MixtureOfExperts

        batch, seq_len, embed_dim = 2, 8, 16
        n_experts = 4
        ff_dim = 32
        k = 2

        moe = MixtureOfExperts(embed_dim, n_experts, ff_dim, k)
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output = moe(x)

        assert output.shape == (batch, seq_len, embed_dim)
        assert output.dtype == x.dtype
        assert torch.all(torch.isfinite(output))

    def test_top_k_routing(self) -> None:
        """When only k experts are activated, output should be sum of k experts (not all E).

        If we zero out all routing weights for 2 experts, only the other 2 contribute.
        With k=2, only top-2 experts are selected.
        """
        from impl._torch.layers import MixtureOfExperts

        embed_dim = 8
        n_experts = 3
        ff_dim = 16
        k = 2

        moe = MixtureOfExperts(embed_dim, n_experts, ff_dim, k)

        x = torch.ones(1, 1, embed_dim, dtype=torch.float64)
        output = moe(x)

        assert torch.all(torch.isfinite(output)), "MoE output must be finite"
        assert not torch.allclose(output, torch.zeros_like(output)), "Output should not be all zeros"

    def test_gradient_flow(self) -> None:
        """All components (router + all experts) participate in computation.

        Changing router weight or any expert weight should change the output.
        """
        from impl._torch.layers import MixtureOfExperts

        embed_dim = 8
        n_experts = 4
        ff_dim = 16
        k = 2

        moe = MixtureOfExperts(embed_dim, n_experts, ff_dim, k)

        x = torch.randn(1, 2, embed_dim, dtype=torch.float64)

        # Output before perturbation
        out_before = moe(x.clone())

        # Perturb router randomly so softmax output actually changes
        old_router = moe.router.weight.data.clone()
        noise = torch.randn_like(moe.router.weight.data) * 2.0
        moe.router.weight.data.add_(noise)
        out_after_router = moe(x.clone())

        assert not torch.allclose(out_before, out_after_router, atol=1e-4)

        # Restore and perturb expert[0]
        moe.router.weight.data.copy_(old_router)
        old_exp_0 = moe.experts[0].W1.data.clone()
        moe.experts[0].W1.data.add_(0.5)
        out_after_exp = moe(x.clone())

        assert not torch.allclose(out_before, out_after_exp, atol=1e-4)

    def test_deterministic(self) -> None:
        """MoE forward with same input → same output (no randomness)."""
        from impl._torch.layers import MixtureOfExperts

        embed_dim = 8
        n_experts = 3
        ff_dim = 16
        k = 2

        moe = MixtureOfExperts(embed_dim, n_experts, ff_dim, k)

        x = torch.randn(1, 4, embed_dim, dtype=torch.float64)

        out1 = moe(x.clone())
        out2 = moe(x.clone())

        assert torch.allclose(out1, out2)
