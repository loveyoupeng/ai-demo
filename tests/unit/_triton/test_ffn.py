import numpy as np
import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestSwiGLUFFNKernel:
    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """Input [B, S, D], weights [D, ff_dim] -> output [B, S, D]."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 2, 4, 8, 16
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        w1 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w3 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w2 = torch.randn(ff_dim, D, dtype=torch.float64, device="cuda")
        y = swiglu_ffn(x, w1, w3, w2)
        assert y.shape == (B, S, D), f"Expected {(B, S, D)}, got {y.shape}"

    @pytest.mark.timeout(30)
    def test_no_bias(self):
        """Zero input produces zero output (no bias)."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 1, 1, 8, 16
        x = torch.zeros(B, S, D, dtype=torch.float64, device="cuda")
        w1 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w3 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w2 = torch.randn(ff_dim, D, dtype=torch.float64, device="cuda")
        y = swiglu_ffn(x, w1, w3, w2)
        torch.testing.assert_close(y, torch.zeros_like(y), rtol=1e-14, atol=1e-14)

    @pytest.mark.timeout(30)
    def test_positive_activation(self):
        """Positive inputs through w1 can produce positive gate."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 2, 4, 8, 16
        x = torch.full((B, S, D), 1.0, dtype=torch.float64, device="cuda")
        w1 = torch.full((D, ff_dim), 1.0, dtype=torch.float64, device="cuda")
        w3 = torch.full((D, ff_dim), 1.0, dtype=torch.float64, device="cuda")
        w2 = torch.full((ff_dim, D), 1.0, dtype=torch.float64, device="cuda")
        y = swiglu_ffn(x, w1, w3, w2)
        # gate = SiLU(sum(w1) * ones) — should be positive
        # gated @ w2 — should be non-zero
        assert not torch.allclose(y, torch.zeros_like(y))
        # Verify all values are finite
        assert torch.isfinite(y).all(), "Output contains NaN or Inf"

    @pytest.mark.timeout(30)
    def test_gradient_shape(self):
        """Gradients w.r.t. all weights have correct shapes."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 2, 4, 8, 16
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda", requires_grad=True)
        w1 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda", requires_grad=True)
        w3 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda", requires_grad=True)
        w2 = torch.randn(ff_dim, D, dtype=torch.float64, device="cuda", requires_grad=True)
        y = swiglu_ffn(x, w1, w3, w2)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None and x.grad.shape == x.shape
        assert w1.grad is not None and w1.grad.shape == w1.shape
        assert w3.grad is not None and w3.grad.shape == w3.shape
        assert w2.grad is not None and w2.grad.shape == w2.shape
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(w1.grad).all()

    @pytest.mark.timeout(30)
    def test_parity_with_numpy(self):
        """Same float64 input → same output as NumPy SwiGLU (rtol=1e-4)."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 2, 4, 8, 16
        x_np = np.random.randn(B, S, D).astype(np.float64)
        w1_np = np.random.randn(D, ff_dim).astype(np.float64)
        w3_np = np.random.randn(D, ff_dim).astype(np.float64)
        w2_np = np.random.randn(ff_dim, D).astype(np.float64)

        x_t = torch.from_numpy(x_np).cuda()
        w1_t = torch.from_numpy(w1_np).cuda()
        w3_t = torch.from_numpy(w3_np).cuda()
        w2_t = torch.from_numpy(w2_np).cuda()

        y_triton = swiglu_ffn(x_t, w1_t, w3_t, w2_t).cpu().numpy()

        # NumPy reference
        gate = np.sinh(x_np) / (1.0 + np.exp(-x_np)) @ w1_np  # (B, S, ff_dim) — SiLU(x @ w1)
        # Correction: SiLU(x @ W1) not SiLU(x) @ W1
        gate = self._silu_np(x_np @ w1_np)  # (B, S, ff_dim)
        proj = x_np @ w3_np  # (B, S, ff_dim)
        gated = gate * proj  # (B, S, ff_dim)
        y_numpy_ref = gated @ w2_np  # (B, S, D)

        np.testing.assert_allclose(y_numpy_ref, y_triton, rtol=1e-4, atol=1e-4)

    @staticmethod
    def _silu_np(x):
        """NumPy SiLU: x * sigmoid(x)."""
        return x / (1.0 + np.exp(-x))

    @pytest.mark.timeout(30)
    def test_parity_with_torch(self):
        """Same float64 input → same output as PyTorch SwiGLU (rtol=1e-4)."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 2, 4, 8, 16
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        w1 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w3 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w2 = torch.randn(ff_dim, D, dtype=torch.float64, device="cuda")

        y_triton = swiglu_ffn(x, w1, w3, w2)
        y_torch = torch.nn.functional.silu(x @ w1) * (x @ w3) @ w2

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-4, atol=1e-4)


class TestSwiGLUAdvanced:
    @pytest.mark.timeout(30)
    def test_batched_sequence(self):
        """Different batches can have different outputs."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 3, 5, 12, 24
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        w1 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w3 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w2 = torch.randn(ff_dim, D, dtype=torch.float64, device="cuda")
        y = swiglu_ffn(x, w1, w3, w2)
        assert y.shape == (B, S, D)
        assert torch.isfinite(y).all()

    @pytest.mark.timeout(30)
    def test_large_ff_dim(self):
        """ff_dim much larger than embed_dim (up-scaled FFN)."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 1, 2, 8, 64  # 8x up-scaling
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        w1 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w3 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w2 = torch.randn(ff_dim, D, dtype=torch.float64, device="cuda")
        y = swiglu_ffn(x, w1, w3, w2)
        assert y.shape == (B, S, D)
        torch.testing.assert_close(y, torch.nn.functional.silu(x @ w1) * (x @ w3) @ w2, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_small_ff_dim(self):
        """ff_dim smaller than embed_dim (down-scaled FFN)."""
        skip_if_no_gpu()
        from impl._triton.ffn import swiglu_ffn

        B, S, D, ff_dim = 1, 2, 64, 8  # 8x down-scaling
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        w1 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w3 = torch.randn(D, ff_dim, dtype=torch.float64, device="cuda")
        w2 = torch.randn(ff_dim, D, dtype=torch.float64, device="cuda")
        y = swiglu_ffn(x, w1, w3, w2)
        assert y.shape == (B, S, D)
        torch.testing.assert_close(y, torch.nn.functional.silu(x @ w1) * (x @ w3) @ w2, rtol=1e-4, atol=1e-4)
