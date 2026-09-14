"""E8: DecoderStack — chain n_layers of TritonTransformerBlock."""

import pytest
import torch

from shared.config import TransformerConfig


def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestDecoderStackWiring:
    """Tests for TritonDecoderStack Python wiring."""

    @pytest.mark.timeout(10)
    def test_output_shape(self):
        """DecoderStack output has same shape as input."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonDecoderStack

        B, S, D, n_layers = 2, 4, 16, 3

        stack = TritonDecoderStack(
            TransformerConfig(
                n_layers=n_layers,
                embed_dim=D,
                n_heads=4,
                n_groups=4,
                n_experts=4,
                expert_dim=32,
                top_k=2,
            ),
        )

        x = torch.randn(B, S, D, dtype=torch.float32, device="cuda")
        for layer in stack.layers:
            layer.eval()
        out = stack(x)
        assert out.shape == (B, S, D)

    @pytest.mark.timeout(10)
    def test_gradient_chaining(self):
        """Gradients flow through all stacked layers."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonDecoderStack

        B, S, D = 2, 4, 16
        n_layers = 3

        stack = TritonDecoderStack(
            TransformerConfig(
                n_layers=n_layers,
                embed_dim=D,
                n_heads=4,
                n_groups=4,
                n_experts=4,
                expert_dim=32,
                top_k=2,
            ),
        )

        x = torch.randn(B, S, D, dtype=torch.float32, device="cuda", requires_grad=True)
        for layer in stack.layers:
            layer.eval()
        out = stack(x)
        out.sum().backward()

        assert x.grad is not None, "Input gradient should not be None"
        assert x.grad.shape == x.shape

    @pytest.mark.timeout(30)
    def test_parity_with_torch(self):
        """Same weights → same output as PyTorch DecoderStack (rtol=1e-2)."""
        import impl._torch.layers as torch_layers
        from impl._triton.transformer import TritonDecoderStack

        skip_if_no_gpu()

        B, S, D, n_heads, n_experts, ff_dim, k = 2, 4, 16, 4, 4, 32, 2
        n_layers = 2

        # Create PyTorch stack
        torch_stack = torch_layers.DecoderStack(
            TransformerConfig(
                n_layers=n_layers,
                embed_dim=D,
                n_heads=n_heads,
                n_groups=n_heads,
                n_experts=n_experts,
                expert_dim=ff_dim,
                top_k=k,
                rope_dim=0,
            ),
        ).cuda()

        # Create Triton stack
        triton_stack = TritonDecoderStack(
            TransformerConfig(
                n_layers=n_layers,
                embed_dim=D,
                n_heads=n_heads,
                n_groups=n_heads,
                n_experts=n_experts,
                expert_dim=ff_dim,
                top_k=k,
            ),
        )

        # Copy weights attribute by attribute (FFN is raw (in,out) in triton).
        from impl._torch.layers import MixtureOfExperts as TorchMoE
        from impl._triton.transformer import TritonMixtureOfExperts

        for i in range(n_layers):
            tb = triton_stack.blocks[i]
            tblock = torch_stack.layers[i]

            tmlp = tblock.mlp
            rmlp = tb.mlp
            assert isinstance(tmlp, TorchMoE)
            assert isinstance(rmlp, TritonMixtureOfExperts)

            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                getattr(tb.self_attn, name).weight.data.copy_(getattr(tblock.self_attn, name).weight.data)

            # MoE router and expert weights
            rmlp.gate.weight.data.copy_(tmlp.gate.weight.data)
            for j in range(n_experts):
                rmlp.expert_list[j].gate_proj.data.copy_(tmlp.expert_list[j].gate_proj.data)
                rmlp.expert_list[j].up_proj.data.copy_(tmlp.expert_list[j].up_proj.data)
                rmlp.expert_list[j].down_proj.data.copy_(tmlp.expert_list[j].down_proj.data)

            # Normalization weights
            tb.input_layernorm.weight.data.copy_(tblock.input_layernorm.weight.data)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            tb.post_attention_layernorm.weight.data.copy_(tblock.post_attention_layernorm.weight.data)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

        def sync_to_cuda(block):
            for p in block.parameters():
                if not p.is_cuda:
                    p.data = p.data.cuda()

        sync_to_cuda(triton_stack)

        torch_stack.eval()
        triton_stack.eval()

        x = torch.randn(B, S, D, dtype=torch.float32, device="cuda")

        # Forward through PyTorch stack
        y_torch = torch_stack(x)

        # Forward through Triton stack
        y_triton = triton_stack(x)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-2)
