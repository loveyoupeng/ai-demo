import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestSiLUCUDA:
    @pytest.mark.timeout(30)
    def test_silu_matches_torch_float32(self):
        """Same float32 input → same output as torch.nn.functional.silu (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        from impl._cuda.activation import silu

        B, S, D = 2, 4, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, D, dtype=torch.float32, device="cuda")

        y_cuda = silu(x)
        y_torch = torch.nn.functional.silu(x)

        torch.testing.assert_close(
            y_cuda, y_torch, rtol=1e-4, atol=1e-4, msg="CUDA SiLU != torch.nn.functional.silu float32"
        )

    @pytest.mark.timeout(30)
    def test_silu_matches_torch_float64(self):
        """Same float64 input → same output as torch.nn.functional.silu (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        from impl._cuda.activation import silu

        B, S, D = 2, 4, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")

        y_cuda = silu(x)
        y_torch = torch.nn.functional.silu(x)

        torch.testing.assert_close(
            y_cuda, y_torch, rtol=1e-4, atol=1e-4, msg="CUDA SiLU != torch.nn.functional.silu float64"
        )

    @pytest.mark.timeout(30)
    def test_silu_input_gradient(self):
        """Gradient through CUDA SiLU matches torch.nn.functional.silu gradient."""
        skip_if_no_gpu()
        from impl._cuda.activation import silu

        B, S, D = 2, 4, 8
        torch.manual_seed(42)
        x_cuda = torch.randn(B, S, D, dtype=torch.float64, device="cuda", requires_grad=True)
        x_torch = x_cuda.clone().detach().requires_grad_(True)

        y_cuda = silu(x_cuda)
        y_torch = torch.nn.functional.silu(x_torch)

        loss_cuda = y_cuda.sum()
        loss_torch = y_torch.sum()

        loss_cuda.backward()
        loss_torch.backward()

        torch.testing.assert_close(
            x_cuda.grad, x_torch.grad, rtol=1e-4, atol=1e-4, msg="CUDA SiLU gradient != torch gradient"
        )

    @pytest.mark.timeout(30)
    def test_silu_shapes(self):
        """SiLU preserves shape for 1D, 2D, 3D inputs."""
        skip_if_no_gpu()
        from impl._cuda.activation import silu

        torch.manual_seed(42)
        shapes = [(8,), (4, 8), (2, 4, 8)]

        for shape in shapes:
            x = torch.randn(*shape, dtype=torch.float64, device="cuda")
            y = silu(x)
            assert y.shape == shape, f"Expected shape {shape}, got {y.shape}"
            assert torch.isfinite(y).all(), f"NaN/Inf detected in shape {shape}"
