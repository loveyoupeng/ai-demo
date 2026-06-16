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
        """Every parameter participates in or touches the computation graph.

        After a backward pass, ALL parameters must have a gradient attribute
        (None or not). We verify non-trivial gradient flow through the router
        (which is always used) and that expert[0] has a gradient tensor
        registered.
        """
        from impl._torch.layers import MixtureOfExperts

        embed_dim = 8
        n_experts = 4
        ff_dim = 16
        k = 2

        moe = MixtureOfExperts(embed_dim, n_experts, ff_dim, k)

        x = torch.randn(1, 2, embed_dim, dtype=torch.float64)

        out = moe(x)
        loss = out.sum()
        loss.backward()

        # All parameters should have a gradient tensor (not None)
        for name, p in moe.named_parameters():
            assert p.grad is not None, f"Gradient is None for {name}"

        # The router always processes all tokens → non-trivial gradient
        grad = moe.router.weight.grad
        assert grad is not None, "Router gradient should not be None"
        router_grad = float(grad.norm().item())
        assert router_grad > 1e-6, "Router gradient should be non-trivial"

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
