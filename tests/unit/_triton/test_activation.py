import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestSiLUKernel:
    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """Input [B, S, D] → output [B, S, D]."""
        skip_if_no_gpu()
        from impl._triton.activation import silu

        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        y = silu(x)
        assert y.shape == (B, S, D), f"Expected {(B, S, D)}, got {y.shape}"

    @pytest.mark.timeout(30)
    def test_output_at_zero(self):
        """SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0."""
        skip_if_no_gpu()
        from impl._triton.activation import silu

        x = torch.tensor([0.0], dtype=torch.float64, device="cuda")
        y = silu(x)
        assert torch.abs(y) < 1e-10, f"SiLU(0) should be 0, got {y}"

    @pytest.mark.timeout(30)
    def test_output_range_large_positive(self):
        """SiLU(10) ≈ 10 (near-identity)."""
        skip_if_no_gpu()
        from impl._triton.activation import silu

        x = torch.tensor([10.0], dtype=torch.float64, device="cuda")
        y = silu(x)
        # SiLU(10) = 10 * sigmoid(10) ≈ 10 * 0.99995 ≈ 9.9995
        assert 9.9 < y < 10.0, f"SiLU(10) should be ≈10, got {y}"

    @pytest.mark.timeout(30)
    def test_output_range_negative(self):
        """SiLU(-10) ≈ -10 * exp(-10) ≈ 0 (suppressed)."""
        skip_if_no_gpu()
        from impl._triton.activation import silu

        x = torch.tensor([-10.0], dtype=torch.float64, device="cuda")
        y = silu(x)
        # SiLU(-10) = -10 * sigmoid(-10) ≈ -10 * 4.5e-5 ≈ -0.00045
        assert -0.01 < y < 0.01, f"SiLU(-10) should be ≈0, got {y}"

    @pytest.mark.timeout(30)
    def test_gradient_correct(self):
        """Autograd produces correct gradients through the kernel."""
        skip_if_no_gpu()
        from impl._triton.activation import silu

        x = torch.randn(2, 4, 8, dtype=torch.float64, device="cuda", requires_grad=True)
        y = silu(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should not be None"
        assert x.grad.shape == x.shape, f"Grad shape {x.grad.shape} != input shape {x.shape}"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"

    @pytest.mark.timeout(30)
    def test_parity_with_numpy(self):
        """Same float64 input → same output as NumPy (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        import numpy as np

        from impl._triton.activation import silu

        B, S, D = 2, 4, 8
        x_np = np.random.randn(B, S, D).astype(np.float64)
        x_torch = torch.from_numpy(x_np).cuda()

        y_triton = silu(x_torch).cpu().numpy()

        # NumPy reference: 1 / (1 + exp(-x)) * x
        sigmoid_x = 1.0 / (1.0 + np.exp(-x_np))
        y_numpy = x_np * sigmoid_x

        np.testing.assert_allclose(
            y_numpy, y_triton, rtol=1e-4, atol=1e-4,
            err_msg="Triton SiLU != NumPy reference"
        )

    @pytest.mark.timeout(30)
    def test_parity_with_torch(self):
        """Same float64 input → same output as torch.nn.SiLU()."""
        skip_if_no_gpu()
        from impl._triton.activation import silu

        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")

        y_triton = silu(x)
        y_torch = torch.nn.functional.silu(x)

        torch.testing.assert_close(
            y_triton, y_torch, rtol=1e-4, atol=1e-4,
            msg="Triton SiLU != torch.nn.SiLU"
        )
