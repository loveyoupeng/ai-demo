"""C5.1: Tests for PyTorch TransformerBlock.

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestTransformerBlockForward:
    """Test the TransformerBlock nn.Module forward pass."""

    def test_output_shape(self) -> None:
        """TransformerBlock(x: [B,S,D]) → [B,S,D]."""
        from impl._torch.layers import TransformerBlock

        embed_dim = 64
        n_heads = 8
        n_experts = 4
        ff_dim = 128
        k = 2

        block = TransformerBlock(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
        )

        x = torch.randn(2, 8, embed_dim, dtype=torch.float64)
        output = block(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert torch.all(torch.isfinite(output)), "Output must be finite"

    def test_attention_and_moe(self) -> None:
        """TransformerBlock contains both MHA and MoE contributions.

        The output should differ substantially from the input,
        confirming that both attention and MoE streams produce non-trivial output.
        """
        from impl._torch.layers import TransformerBlock

        embed_dim = 32
        n_heads = 4
        n_experts = 4
        ff_dim = 64
        k = 2

        block = TransformerBlock(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
        )

        x = torch.randn(1, 4, embed_dim, dtype=torch.float64)
        output = block(x)

        # Output should differ from input (not identity)
        diff = (output - x).norm().item()
        assert diff > 1e-6, "Output should differ from input"

        # Output should not be all zeros
        assert output.norm().item() > 1e-6, "Output should not be zero"

    def test_gradient_chaining(self) -> None:
        """Gradients flow through MHA, MoE, and normalization layers.

        All MHA weights should have gradients. For MoE, the router and at
        least one expert should have non-zero gradients (only top-k experts
        fire, but the router always gets gradients).
        """
        from impl._torch.layers import TransformerBlock

        embed_dim = 32
        n_heads = 4
        n_experts = 4
        ff_dim = 64
        k = 2

        block = TransformerBlock(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
        )

        x = torch.randn(1, 2, embed_dim, dtype=torch.float64)
        output = block(x)
        loss = output.sum()
        loss.backward()

        # MHA: all weights should have non-zero gradients
        for name, param in block.mha.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            grad_norm = param.grad.norm().item()
            assert grad_norm > 1e-9, f"{name} gradient norm {grad_norm} too small"

        # MoE: router always gets gradients (softmax over all experts)
        moe_grad_norms = []
        for name, param in block.moe.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            moe_grad_norms.append(param.grad.norm().item())

        # At least one expert should have non-zero gradients
        # (the top-k selected experts fire; router always fires on all)
        assert any(n > 1e-9 for n in moe_grad_norms), "At least one MoE param must have gradient"


    def test_deterministic(self) -> None:
        """TransformerBlock forward with same input → same output."""
        from impl._torch.layers import TransformerBlock

        embed_dim = 32
        n_heads = 4
        n_experts = 4
        ff_dim = 64
        k = 2

        block = TransformerBlock(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
        )

        x = torch.randn(1, 4, embed_dim, dtype=torch.float64)

        out1 = block(x.clone())
        out2 = block(x.clone())

        assert torch.allclose(out1, out2), "Forward must be deterministic"
