import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestRMSNormKernel:
    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """Input [B, S, D], gamma [D] → output [B, S, D]."""
        skip_if_no_gpu()
        from impl._triton.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        gamma = torch.ones(D, dtype=torch.float64, device="cuda")
        y = rmsnorm(x, gamma)
        assert y.shape == (B, S, D), f"Expected {(B, S, D)}, got {y.shape}"

    @pytest.mark.timeout(30)
    def test_unit_variance(self):
        """After normalization, per-row mean of y^2 ≈ 1 (with gamma=1)."""
        skip_if_no_gpu()
        from impl._triton.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        gamma = torch.ones(D, dtype=torch.float64, device="cuda")
        y = rmsnorm(x, gamma)
        # For each row, mean of squared output over features should be ≈ 1
        mean_sq_per_row = torch.mean(y ** 2, dim=-1)  # (B, S)
        assert torch.allclose(mean_sq_per_row, torch.ones_like(mean_sq_per_row), rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_identity_without_gamma(self):
        """With gamma=1, output ≈ normalized input."""
        skip_if_no_gpu()
        from impl._triton.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        gamma = torch.ones(D, dtype=torch.float64, device="cuda")
        y = rmsnorm(x, gamma)
        # Should be x / rms(x)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        expected = x / rms
        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_learned_scale(self):
        """gamma controls output magnitude."""
        skip_if_no_gpu()
        from impl._triton.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        gamma = torch.full((D,), 2.0, dtype=torch.float64, device="cuda")
        y = rmsnorm(x, gamma)
        # With gamma=2, output magnitude should be ≈ 2x gamma=1 output
        gamma1 = torch.ones(D, dtype=torch.float64, device="cuda")
        y1 = rmsnorm(x, gamma1)
        torch.testing.assert_close(y, y1 * 2.0, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_gradient_shape(self):
        """Gradient w.r.t. input has same shape as input."""
        skip_if_no_gpu()
        from impl._triton.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda", requires_grad=True)
        gamma = torch.ones(D, dtype=torch.float64, device="cuda")
        y = rmsnorm(x, gamma)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should not be None"
        assert x.grad.shape == x.shape, f"Grad shape {x.grad.shape} != input shape {x.shape}"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"

    @pytest.mark.timeout(30)
    def test_gradient_correct(self):
        """Autograd gradient check against finite difference."""
        skip_if_no_gpu()
        from impl._triton.layernorm import rmsnorm

        D = 8
        x = torch.randn(1, 1, D, dtype=torch.float64, device="cuda", requires_grad=True)
        gamma = torch.ones(D, dtype=torch.float64, device="cuda")
        epsilon = 1e-5

        y = rmsnorm(x, gamma)
        loss = y.sum()
        loss.backward()
        grad_numerical = x.grad.clone() if x.grad is not None else torch.zeros_like(x)

        # Numerical gradient via finite difference
        x_num = x.detach().clone().requires_grad_(False)
        for i in range(D):
            x_perturbed = x_num.clone()
            x_perturbed[0, 0, i] += epsilon
            y_p = rmsnorm(x_perturbed, gamma)
            num_grad = y_p.sum().item()

            x_perturbed = x_num.clone()
            x_perturbed[0, 0, i] -= epsilon
            y_m = rmsnorm(x_perturbed, gamma)
            num_grad = (num_grad - y_m.sum().item()) / (2 * epsilon)

            numerical_grad = torch.tensor(num_grad, dtype=torch.float64, device="cuda")
            assert torch.isclose(grad_numerical[0, 0, i], numerical_grad, rtol=1e-2, atol=1e-2), (
                f"Gradient mismatch at index {i}: "
                f"numerical={numerical_grad}, autograd={grad_numerical[0, 0, i]}"
            )

    @pytest.mark.timeout(30)
    def test_parity_with_numpy(self):
        """Same float64 input → same output as NumPy RMSNorm (rtol=1e-4)."""
        skip_if_no_gpu()
        import numpy as np

        from impl._triton.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        x_np = np.random.randn(B, S, D).astype(np.float64)
        gamma_np = np.random.randn(D).astype(np.float64) + 1.0  # Avoid zeros
        x_torch = torch.from_numpy(x_np).cuda()
        gamma_torch = torch.from_numpy(gamma_np).cuda()

        y_triton = rmsnorm(x_torch, gamma_torch).cpu().numpy()

        # NumPy reference
        eps = 1e-6
        rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True)) + eps
        y_numpy = (x_np / rms) * gamma_np

        np.testing.assert_allclose(y_numpy, y_triton, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_parity_with_torch(self):
        """Same float64 input → same output as PyTorch RMSNorm (rtol=1e-4)."""
        skip_if_no_gpu()
        import torch.nn.functional as F

        from impl._triton.layernorm import rmsnorm

        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        gamma = torch.ones(D, dtype=torch.float64, device="cuda")

        y_triton = rmsnorm(x, gamma)
        y_torch = F.rms_norm(x, [D], weight=gamma, eps=1e-6)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-4, atol=1e-4)
