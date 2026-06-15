"""Tests for PyTorch SiLU activation (C1.3).

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestSiLULayer:
    """Tests for the SiLULayer nn.Module."""

    def test_output_shape(self) -> None:
        """SiLU preserves input shape."""
        from impl._torch.layers import SiLULayer

        layer = SiLULayer()
        for shape in [(4, 8), (2, 4, 16), (1, 1, 1, 32)]:
            x = torch.randn(*shape, dtype=torch.float64)
            output = layer(x)
            assert output.shape == shape
            assert torch.all(torch.isfinite(output))

    def test_output_at_zero(self) -> None:
        """SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0."""
        from impl._torch.layers import SiLULayer

        layer = SiLULayer()
        x = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        output = layer(x)
        assert torch.allclose(output, torch.zeros_like(x))

    def test_output_range_large_positive(self) -> None:
        """SiLU(x) ≈ x for large positive x (sigmoid ≈ 1)."""
        from impl._torch.layers import SiLULayer

        layer = SiLULayer()
        x = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float64)
        output = layer(x)
        # For large x, sigmoid(x) ≈ 1, so SiLU(x) ≈ x
        assert torch.allclose(output, x, atol=1e-3)

    def test_output_range_negative(self) -> None:
        """SiLU(x) ≈ 0 for large negative x (sigmoid ≈ 0)."""
        from impl._torch.layers import SiLULayer

        layer = SiLULayer()
        x = torch.tensor([[-10.0, -20.0, -30.0]], dtype=torch.float64)
        output = layer(x)
        # For large negative x, sigmoid(x) ≈ 0, so SiLU(x) ≈ 0
        assert torch.allclose(output, torch.zeros_like(x), atol=1e-3)

    def test_gradient_flow(self) -> None:
        """Gradients flow through SiLU."""
        from impl._torch.layers import SiLULayer

        layer = SiLULayer()
        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        output = layer(x)
        output.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))
