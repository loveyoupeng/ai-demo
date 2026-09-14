"""C6.1: Tests for PyTorch DecoderStack.

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch

from shared.config import TransformerConfig


class TestDecoderStackForward:
    """Test the DecoderStack nn.Module forward pass."""

    def test_output_shape(self) -> None:
        """DecoderStack(x: [B,S,D]) → [B,S,D]."""
        from impl._torch.layers import DecoderStack

        embed_dim = 64
        n_heads = 8
        n_experts = 4
        ff_dim = 128
        k = 2
        n_layers = 3

        stack = DecoderStack(
            TransformerConfig(
                n_layers=n_layers,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_groups=n_heads,
                n_experts=n_experts,
                expert_dim=ff_dim,
                top_k=k,
                rope_dim=0,
            ),
        ).double()

        x = torch.randn(2, 8, embed_dim, dtype=torch.float64)
        output = stack(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert torch.all(torch.isfinite(output)), "Output must be finite"

    def test_gradient_chaining(self) -> None:
        """Gradients flow through all stacked layers."""
        from impl._torch.layers import DecoderStack

        embed_dim = 32
        n_heads = 4
        n_experts = 4
        ff_dim = 64
        k = 2
        n_layers = 3

        stack = DecoderStack(
            TransformerConfig(
                n_layers=n_layers,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_groups=n_heads,
                n_experts=n_experts,
                expert_dim=ff_dim,
                top_k=k,
                rope_dim=0,
            ),
        ).double()

        x = torch.randn(1, 4, embed_dim, dtype=torch.float64)
        output = stack(x)
        loss = output.sum()
        loss.backward()

        # Check that every layer has non-zero gradients
        block_module: list[torch.nn.Module] = list(stack.layers)  # noqa: TID251
        for layer_idx in range(len(block_module)):
            block: torch.nn.Module = block_module[layer_idx]
            for name, param in block.self_attn.named_parameters():  # pyright: ignore
                assert param.grad is not None, f"layer {layer_idx} attn {name} has no gradient"
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-9, f"layer {layer_idx} attn {name} grad norm {grad_norm} too small"

            # At least some FFN/MoE params must have gradients
            for name, param in block.mlp.named_parameters():  # pyright: ignore
                assert param.grad is not None, f"layer {layer_idx} mlp {name} has no gradient"

    def test_single_layer(self) -> None:
        """Works with n_layers=1 — no special casing."""
        from impl._torch.layers import DecoderStack

        embed_dim = 32
        n_heads = 4
        n_experts = 4
        ff_dim = 64
        k = 2

        stack = DecoderStack(
            TransformerConfig(
                n_layers=1,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_groups=n_heads,
                n_experts=n_experts,
                expert_dim=ff_dim,
                top_k=k,
                rope_dim=0,
            ),
        ).double()

        x = torch.randn(1, 2, embed_dim, dtype=torch.float64)
        output = stack(x)

        # Output must match input shape
        assert output.shape == x.shape
        # Output must be finite
        assert torch.all(torch.isfinite(output))
        # Output must differ from input (single block produces non-trivial output)
        diff = (output - x).norm().item()
        assert diff > 1e-6, "Single block output must differ from input"
