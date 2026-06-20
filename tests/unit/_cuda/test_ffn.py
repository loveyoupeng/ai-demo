"""SwiGLU Feed-Forward Network — CUDA SiLU + PyTorch matmul.

Tests the hybrid approach: CUDA C kernel for SiLU activation,
PyTorch matmul for matrix multiplication.

Learning objectives:
- When to use CUDA kernels (element-wise ops) vs PyTorch (matmul)
- CUDA C SiLU kernel + PyTorch dispatcher integration
- Gradients through element-wise × matmul combinations
"""

from __future__ import annotations

import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestSwiGLU:
    @pytest.mark.timeout(30)
    def test_swiglu_matches_torch_float32(self):
        """SwiGLU output matches torch reference (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        from impl._cuda.ffn import swiglu_ffn

        B, S, D, FF = 2, 4, 8, 16
        torch.manual_seed(42)
        x = torch.randn(B, S, D, dtype=torch.float32, device="cuda")
        w1 = torch.randn(D, FF, dtype=torch.float32, device="cuda")
        w3 = torch.randn(D, FF, dtype=torch.float32, device="cuda")
        w2 = torch.randn(FF, D, dtype=torch.float32, device="cuda")

        out_cuda = swiglu_ffn(x, w1, w3, w2)

        # Reference: SiLU(x @ W1) * (x @ W3) @ W2
        gate = torch.nn.functional.silu(x @ w1)
        proj = x @ w3
        out_ref = (gate * proj) @ w2

        torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4, msg="CUDA SwiGLU != torch reference (f32)")

    @pytest.mark.timeout(30)
    def test_swiglu_matches_torch_float64(self):
        """SwiGLU output matches torch reference in float64 (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        from impl._cuda.ffn import swiglu_ffn

        B, S, D, FF = 2, 4, 8, 16
        torch.manual_seed(42)
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        w1 = torch.randn(D, FF, dtype=torch.float64, device="cuda")
        w3 = torch.randn(D, FF, dtype=torch.float64, device="cuda")
        w2 = torch.randn(FF, D, dtype=torch.float64, device="cuda")

        out_cuda = swiglu_ffn(x, w1, w3, w2)

        gate = torch.nn.functional.silu(x @ w1)
        proj = x @ w3
        out_ref = (gate * proj) @ w2

        torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4, msg="CUDA SwiGLU != torch reference (f64)")

    @pytest.mark.timeout(30)
    def test_swiglu_shapes(self):
        """SwiGLU handles various (B, S, D, FF) shapes."""
        skip_if_no_gpu()
        from impl._cuda.ffn import swiglu_ffn

        torch.manual_seed(42)
        config = [(1, 1, 64, 128), (2, 8, 64, 256), (4, 2, 32, 64)]

        for B, S, D, FF in config:
            x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
            w1 = torch.randn(D, FF, dtype=torch.float64, device="cuda")
            w3 = torch.randn(D, FF, dtype=torch.float64, device="cuda")
            w2 = torch.randn(FF, D, dtype=torch.float64, device="cuda")

            out = swiglu_ffn(x, w1, w3, w2)
            assert out.shape == (B, S, D), f"Expected ({B}, {S}, {D}), got {out.shape}"
            assert torch.isfinite(out).all(), f"NaN/Inf found in shape {(B, S, D, FF)}"
