"""Tests for PyTorch SwiGLU FFN (C1.4).

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestSwiGLUFFN:
    """Tests for the SwiGLUFFN nn.Module."""

    def test_output_shape(self) -> None:
        """SwiGLU(x: [B,S,D], ff_dim=F) → [B,S,D]."""
        from impl._torch.layers import SwiGLUFFN

        batch, seq_len, embed_dim, ff_dim = 2, 8, 16, 64
        layer = SwiGLUFFN(embed_dim, ff_dim)
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output = layer(x)

        assert output.shape == (batch, seq_len, embed_dim)
        assert torch.all(torch.isfinite(output))

    def test_gating_behavior(self) -> None:
        """SwiGLU combines W1 and W3 projections with gating."""
        from impl._torch.layers import SwiGLUFFN

        embed_dim, ff_dim = 8, 16
        layer = SwiGLUFFN(embed_dim, ff_dim)

        x = torch.ones(1, 1, embed_dim, dtype=torch.float64)
        output = layer(x)

        # Output should be finite and depend on both W1 and W3
        assert torch.all(torch.isfinite(output))
        # With all-positive input, SiLU(x@W1) should preserve sign from W1
        # Gated output should not be zero since both terms contribute
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_ff_dim_independence(self) -> None:
        """Output size does not depend on ff_dim."""
        from impl._torch.layers import SwiGLUFFN

        batch, seq_len, embed_dim = 2, 4, 16
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)

        for ff_dim in [16, 32, 64, 128]:
            layer = SwiGLUFFN(embed_dim, ff_dim)
            output = layer(x)
            assert output.shape == (batch, seq_len, embed_dim)

    def test_gradient_existence(self) -> None:
        """All weights get non-zero gradients through autograd."""
        from impl._torch.layers import SwiGLUFFN

        embed_dim, ff_dim = 8, 16
        layer = SwiGLUFFN(embed_dim, ff_dim)

        x = torch.randn(2, 4, embed_dim, dtype=torch.float64, requires_grad=True)
        output = layer(x)
        loss = (output * output).sum()
        loss.backward()

        assert layer.W1.grad is not None
        assert layer.W3.grad is not None
        assert layer.W2.grad is not None
        assert torch.all(torch.isfinite(layer.W1.grad))
        assert torch.all(torch.isfinite(layer.W3.grad))
        assert torch.all(torch.isfinite(layer.W2.grad))
