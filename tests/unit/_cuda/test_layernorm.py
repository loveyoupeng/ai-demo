import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestRMSNormCUDA:
    @pytest.mark.timeout(30)
    def test_rmsnorm_matches_torch_float32(self):
        """Same float32 input → same output as torch.nn.functional.rms_norm (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        import torch.nn.functional as F

        from impl._cuda.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, D, dtype=torch.float32, device="cuda")
        gamma = torch.ones(D, dtype=torch.float32, device="cuda")

        y_cuda = rmsnorm(x, gamma)
        y_torch = F.rms_norm(x, [D], weight=gamma, eps=1e-6)

        torch.testing.assert_close(
            y_cuda, y_torch, rtol=1e-4, atol=1e-4, msg="CUDA RMSNorm != torch.nn.functional.rms_norm float32"
        )

    @pytest.mark.timeout(30)
    def test_rmsnorm_matches_torch_float64(self):
        """Same float64 input → same output as torch.nn.functional.rms_norm (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        import torch.nn.functional as F

        from impl._cuda.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        gamma = torch.ones(D, dtype=torch.float64, device="cuda")

        y_cuda = rmsnorm(x, gamma)
        y_torch = F.rms_norm(x, [D], weight=gamma, eps=1e-6)

        torch.testing.assert_close(
            y_cuda, y_torch, rtol=1e-4, atol=1e-4, msg="CUDA RMSNorm != torch.nn.functional.rms_norm float64"
        )

    @pytest.mark.timeout(30)
    def test_rmsnorm_shapes(self):
        """RMSNorm preserves shape for 1D, 2D, 3D inputs."""
        skip_if_no_gpu()
        from impl._cuda.layernorm import rmsnorm

        torch.manual_seed(42)
        shapes = [(8,), (4, 8), (2, 4, 8)]

        for shape in shapes:
            D = shape[-1]
            x = torch.randn(*shape, dtype=torch.float64, device="cuda")
            gamma = torch.ones(D, dtype=torch.float64, device="cuda")
            y = rmsnorm(x, gamma)
            assert y.shape == shape, f"Expected shape {shape}, got {y.shape}"
            assert torch.isfinite(y).all(), f"NaN/Inf detected in shape {shape}"

    @pytest.mark.timeout(30)
    def test_rmsnorm_unit_variance(self):
        """After normalization with gamma=1, per-row mean of y^2 ≈ 1."""
        skip_if_no_gpu()
        from impl._cuda.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        gamma = torch.ones(D, dtype=torch.float64, device="cuda")
        y = rmsnorm(x, gamma)
        # For each row, mean of squared output over features should be ≈ 1
        mean_sq_per_row = torch.mean(y**2, dim=-1)  # (B, S)
        assert torch.allclose(mean_sq_per_row, torch.ones_like(mean_sq_per_row), rtol=1e-4, atol=1e-4)
