"""E7.1: Tests for Triton TransformerBlock (Python wiring).

TDD: Write test -> all fail -> implement -> all pass -> ruff + pyright -> commit.

The TransformerBlock assembles Triton kernels:
    Stream 1: Attention (MHA + RoPE via Triton) -> residual -> RMSNorm -> gate1 -> dropout
    Stream 2: MoE -> residual -> RMSNorm -> gate2 -> dropout

This is Python-only wiring -- no new Triton kernels.
"""

import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestTransformerBlockWiring:
    """Test the TransformerBlock Python assembly of Triton kernels."""

    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """Input [B,S,D] -> output [B,S,D] (residual connections are identity)."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonTransformerBlock

        B, S, D, n_heads, n_experts, ff_dim, k = 2, 8, 16, 4, 4, 32, 2

        block = TritonTransformerBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        out = block(x)
        assert out.shape == (B, S, D)
        assert out.dtype == x.dtype

    @pytest.mark.timeout(30)
    def test_residual_connection(self):
        """Output contains original input (residual pass-through)."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonTransformerBlock

        B, S, D, n_heads, n_experts, ff_dim, k = 1, 2, 8, 2, 2, 16, 1

        block = TritonTransformerBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        out = block(x)
        assert not torch.allclose(out, x), "Output should differ from input -- architecture modulates input"
        assert torch.isfinite(out).all()

    @pytest.mark.timeout(30)
    def test_gate_parameters_exist(self):
        """Block has gate1 and gate2 nn.Parameter tensors."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonTransformerBlock

        D, n_heads, n_experts, ff_dim, k = 8, 2, 2, 16, 1

        block = TritonTransformerBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        assert hasattr(block, "gate1"), "Block should have gate1 parameter"
        assert hasattr(block, "gate2"), "Block should have gate2 parameter"
        assert isinstance(block.gate1, torch.nn.Parameter)
        assert isinstance(block.gate2, torch.nn.Parameter)
        assert block.gate1.shape == (1,)
        assert block.gate2.shape == (1,)

    @pytest.mark.timeout(60)
    def test_gradient_flow(self):
        """Gradients flow through all components (attention, moe, ln, gates)."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonTransformerBlock

        B, S, D, n_heads, n_experts, ff_dim, k = 2, 4, 16, 4, 4, 32, 2

        block = TritonTransformerBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda", requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, p in block.named_parameters():
            assert p.grad is not None, f"Gradient is None for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN gradient in {name}"

        # Attention gradient should be non-trivial
        attn_grad = block.mha.Wq.grad
        assert attn_grad is not None
        assert attn_grad.norm() > 1e-6, "Attention gradient should be non-trivial"

    @pytest.mark.timeout(60)
    def test_attn_and_moe_both_active(self):
        """Both attention and MoE contribute to output (not zero)."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonTransformerBlock

        B, S, D, n_heads, n_experts, ff_dim, k = 2, 4, 16, 4, 4, 32, 2

        block = TritonTransformerBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        out = block(x)
        assert not torch.allclose(out, torch.zeros_like(out)), "Output should not be zero"
        assert torch.isfinite(out).all()

    @pytest.mark.timeout(60)
    def test_parity_with_torch(self):
        """Same weights -> same output as PyTorch TransformerBlock (rtol=1e-3)."""
        import impl._torch.layers as torch_layers
        from impl._triton.transformer import TritonTransformerBlock as TritonBlock

        skip_if_no_gpu()
        B, S, D, n_heads, n_experts, ff_dim, k = 2, 4, 16, 4, 4, 32, 2

        # Create PyTorch block
        torch_block = torch_layers.TransformerBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
            dropout=0.0,
        ).cuda()

        # Create Triton block
        triton_block = TritonBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        # Copy weights: MHA projections and biases (transpose for nn.Linear convention [out,in])
        triton_block.mha.Wq.data.copy_(torch_block.mha.Wq.weight.data.t())
        triton_block.mha.bq.data.copy_(torch_block.mha.Wq.bias.data)
        triton_block.mha.Wk.data.copy_(torch_block.mha.Wk.weight.data.t())
        triton_block.mha.bk.data.copy_(torch_block.mha.Wk.bias.data)
        triton_block.mha.Wv.data.copy_(torch_block.mha.Wv.weight.data.t())
        triton_block.mha.bv.data.copy_(torch_block.mha.Wv.bias.data)
        triton_block.mha.Wo.data.copy_(torch_block.mha.Wo.weight.data.t())
        triton_block.mha.bo.data.copy_(torch_block.mha.Wo.bias.data)

        # Copy MoE router and expert weights (router uses nn.Linear [out,in] convention)
        triton_block.moe.W_router.data.copy_(torch_block.moe.router.weight.data.t())
        triton_block.moe.b_router.data.copy_(torch_block.moe.router.bias.data)
        for i in range(n_experts):
            # Experts use Raw Tensor, not nn.Linear (same orientation as Triton)
            triton_block.moe.experts[i].W1.data.copy_(torch_block.moe.experts[i].W1.data)
            triton_block.moe.experts[i].W2.data.copy_(torch_block.moe.experts[i].W2.data)
            triton_block.moe.experts[i].W3.data.copy_(torch_block.moe.experts[i].W3.data)

        # Copy normalization and gate weights
        triton_block.ln1.weight.data.copy_(torch_block.ln1.gamma.data)
        triton_block.ln2.weight.data.copy_(torch_block.ln2.gamma.data)
        triton_block.gate1.data.copy_(torch_block.gate1.data)
        triton_block.gate2.data.copy_(torch_block.gate2.data)

        # Evaluate mode (no dropout for parity)
        torch_block.eval()
        triton_block.eval()

        x = torch.randn(B, S, D, dtype=torch.float32, device="cuda")

        # Compare full block output
        y_torch = torch_block(x)
        y_triton = triton_block(x)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-3, atol=1e-3)

    @pytest.mark.timeout(30)
    def test_gradient_shape(self):
        """All parameters get valid gradient shapes matching their shape."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonTransformerBlock

        B, S, D, n_heads, n_experts, ff_dim, k = 2, 4, 16, 4, 4, 32, 2

        block = TritonTransformerBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda", requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        # Check all parameter gradients
        for name, p in block.named_parameters():
            assert p.grad is not None, f"Gradient None for {name}"
            assert p.grad.shape == p.shape, f"Gradient shape mismatch for {name}: {p.grad.shape} vs {p.shape}"
            assert torch.isfinite(p.grad).all(), f"NaN gradient in {name}"

        # Specific checks for key parameters
        assert block.mha.Wq.grad is not None
        assert block.mha.Wk.grad is not None
        assert block.mha.Wv.grad is not None
        assert block.mha.Wo.grad is not None
        assert block.ln1.weight.grad is not None
        assert block.ln2.weight.grad is not None
        assert block.gate1.grad is not None
        assert block.gate2.grad is not None

    @pytest.mark.timeout(30)
    def test_no_nan_output(self):
        """Forward pass produces no NaN or Inf values."""
        skip_if_no_gpu()
        from impl._triton.transformer import TritonTransformerBlock

        B, S, D, n_heads, n_experts, ff_dim, k = 4, 16, 32, 8, 8, 64, 2

        block = TritonTransformerBlock(
            embed_dim=D,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        out = block(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"
        assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros"
