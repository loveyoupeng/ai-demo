"""E8: DecoderStack — chain n_layers of TritonTransformerBlock."""

import pytest
import torch


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
            n_layers=n_layers,
            embed_dim=D,
            n_heads=4,
            n_experts=4,
            ff_dim=32,
            k=2,
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
            n_layers=n_layers,
            embed_dim=D,
            n_heads=4,
            n_experts=4,
            ff_dim=32,
            k=2,
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
            n_layers=n_layers,
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
        ).cuda()

        # Create Triton stack
        triton_stack = TritonDecoderStack(
            n_layers=n_layers,
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        # Copy weights: MHA projections and biases (transpose for nn.Linear convention [out,in])
        for i in range(n_layers):
            tb = triton_stack.layers[i]
            tblock = torch_stack.layers[i]

            tb.mha.Wq.data.copy_(tblock.mha.Wq.weight.data.t())
            tb.mha.bq.data.copy_(tblock.mha.Wq.bias.data)
            tb.mha.Wk.data.copy_(tblock.mha.Wk.weight.data.t())
            tb.mha.bk.data.copy_(tblock.mha.Wk.bias.data)
            tb.mha.Wv.data.copy_(tblock.mha.Wv.weight.data.t())
            tb.mha.bv.data.copy_(tblock.mha.Wv.bias.data)
            tb.mha.Wo.data.copy_(tblock.mha.Wo.weight.data.t())
            tb.mha.bo.data.copy_(tblock.mha.Wo.bias.data)

            # MoE router and expert weights
            tb.moe.W_router.data.copy_(tblock.moe.router.weight.data.t())
            tb.moe.b_router.data.copy_(tblock.moe.router.bias.data)
            for j in range(n_experts):
                tb.moe.experts[j].W1.data.copy_(tblock.moe.experts[j].W1.data)
                tb.moe.experts[j].W2.data.copy_(tblock.moe.experts[j].W2.data)
                tb.moe.experts[j].W3.data.copy_(tblock.moe.experts[j].W3.data)

            # Normalization and gate weights
            tb.ln1_gamma.data.copy_(tblock.ln1.gamma.data)
            tb.ln2_gamma.data.copy_(tblock.ln2.gamma.data)
            tb.gate1.data.copy_(tblock.gate1.data)
            tb.gate2.data.copy_(tblock.gate2.data)

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
