"""Tests for PyTorch RMSNorm (C1.2).

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestRMSNormForward:
    """Tests for the RMSNorm nn.Module forward pass."""

    def test_output_shape(self) -> None:
        """RMSNorm(x: [B,S,D], g: [D]) → [B,S,D]."""
        from impl._torch.layers import RMSNorm

        batch, seq_len, embed_dim = 2, 8, 16
        layer = RMSNorm(embed_dim)
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output = layer(x)

        assert output.shape == (batch, seq_len, embed_dim)
        assert output.dtype == x.dtype
        assert torch.all(torch.isfinite(output))

    def test_variance_normalized_per_sample(self) -> None:
        """Each sample's per-feature variance is exactly 1 (when gamma=1)."""
        from impl._torch.layers import RMSNorm

        batch, seq_len, embed_dim = 2, 8, 32
        layer = RMSNorm(embed_dim)

        with torch.no_grad():
            layer.gamma.fill_(1.0)

        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output = layer(x)

        # Each (batch, seq) sample independently has unit variance
        # because RMSNorm normalizes per-sample
        for i in range(batch):
            for j in range(seq_len):
                sample = output[i, j]
                sample_variance = (sample * sample).mean()
                assert torch.allclose(sample_variance, torch.tensor(1.0, dtype=torch.float64), atol=1e-5)

    def test_learned_scale(self) -> None:
        """Setting gamma=2 should approximately double the output."""
        from impl._torch.layers import RMSNorm

        batch, seq_len, embed_dim = 1, 4, 8
        layer = RMSNorm(embed_dim)

        # Set gamma to 2
        with torch.no_grad():
            layer.gamma.fill_(2.0)

        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output = layer(x)

        # Normalize x manually
        eps = 1e-6
        normalized = x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + eps)
        expected = normalized * 2.0

        assert torch.allclose(output, expected, atol=1e-5)

    def test_broadcasts_across_batch(self) -> None:
        """Same gamma applied to all batch elements."""
        from impl._torch.layers import RMSNorm

        embed_dim = 4
        layer = RMSNorm(embed_dim)

        with torch.no_grad():
            layer.gamma.fill_(1.0)

        batch_x = torch.randn(3, 6, embed_dim, dtype=torch.float64)
        output = layer(batch_x)

        # Each batch element should have the same normalization
        for b in range(3):
            eps = 1e-6
            expected = batch_x[b] / (torch.sqrt(torch.mean(batch_x[b] ** 2, dim=-1, keepdim=True)) + eps)
            assert torch.allclose(output[b], expected, atol=1e-5)


class TestRMSNormBackward:
    """Tests for RMSNorm gradient flow."""

    def test_gradient_shape(self) -> None:
        """Gradient w.r.t. input has same shape as input."""
        from impl._torch.layers import RMSNorm

        batch, seq_len, embed_dim = 2, 8, 16
        layer = RMSNorm(embed_dim)
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64, requires_grad=True)
        output = layer(x)
        output.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))

    def test_gradient_correct(self) -> None:
        """Gradients flow through RMSNorm and affect weight gamma too."""
        from impl._torch.layers import RMSNorm

        batch, seq_len, embed_dim = 4, 16, 32
        layer = RMSNorm(embed_dim)
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert layer.gamma.grad is not None
        assert layer.gamma.grad.shape == layer.gamma.shape
        assert torch.all(torch.isfinite(layer.gamma.grad))
