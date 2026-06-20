"""Scaled dot-product attention — stable softmax kernel + weighted sum.

Tests the hybrid approach: CUDA C kernel for stable softmax and weighted
sum, PyTorch for QKV projections and output projection.

Learning objectives:
- Stable softmax (max-subtract-then-exp) in CUDA C
- Warp-level reduction for max and sum
- Weighted sum kernel for attention @ V
- Integration with PyTorch matmuls for full SDPA
"""

from __future__ import annotations

import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestScaledAttention:
    @pytest.mark.timeout(30)
    def test_attention_matches_torch_float32(self):
        """SDPA output matches torch reference (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        from impl._cuda.attention import scaled_dot_product_attention

        B, H, S, D = 2, 4, 8, 16
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, dtype=torch.float32, device="cuda")
        k = torch.randn(B, H, S, D, dtype=torch.float32, device="cuda")
        v = torch.randn(B, H, S, D, dtype=torch.float32, device="cuda")

        out_cuda = scaled_dot_product_attention(q, k, v)
        out_torch = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )

        torch.testing.assert_close(
            out_cuda, out_torch, rtol=1e-4, atol=1e-4, msg="CUDA SDPA != torch SDPA (f32)"
        )

    @pytest.mark.timeout(30)
    def test_attention_matches_torch_float64(self):
        """SDPA output matches torch reference in float64 (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        from impl._cuda.attention import scaled_dot_product_attention

        B, H, S, D = 2, 4, 8, 16
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, dtype=torch.float64, device="cuda")
        k = torch.randn(B, H, S, D, dtype=torch.float64, device="cuda")
        v = torch.randn(B, H, S, D, dtype=torch.float64, device="cuda")

        out_cuda = scaled_dot_product_attention(q, k, v)
        out_torch = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )

        torch.testing.assert_close(
            out_cuda, out_torch, rtol=1e-4, atol=1e-4, msg="CUDA SDPA != torch SDPA (f64)"
        )

    @pytest.mark.timeout(30)
    def test_attention_shapes(self):
        """SDPA handles various (B, H, S, D) shapes."""
        skip_if_no_gpu()
        from impl._cuda.attention import scaled_dot_product_attention

        torch.manual_seed(42)
        config = [
            (1, 1, 4, 8),    # minimal
            (2, 4, 8, 16),   # medium
            (2, 8, 16, 64),  # larger
        ]

        for B, H, S, D in config:
            q = torch.randn(B, H, S, D, dtype=torch.float64, device="cuda")
            k = torch.randn(B, H, S, D, dtype=torch.float64, device="cuda")
            v = torch.randn(B, H, S, D, dtype=torch.float64, device="cuda")

            out = scaled_dot_product_attention(q, k, v)
            assert out.shape == (B, H, S, D), f"Expected {(B, H, S, D)}, got {out.shape}"
            assert torch.isfinite(out).all(), f"NaN/Inf detected in shape {(B, H, S, D)}"

    @pytest.mark.timeout(30)
    def test_attention_weights_sum_to_one(self):
        """Attention weights sum to 1 over keys (valid softmax)."""
        skip_if_no_gpu()
        from impl._cuda.attention import scaled_dot_product_attention

        B, H, S, D = 2, 4, 8, 16
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, dtype=torch.float64, device="cuda")
        k = torch.randn(B, H, S, D, dtype=torch.float64, device="cuda")
        v = torch.ones(B, H, S, D, dtype=torch.float64, device="cuda")  # all ones → output = mean

        out = scaled_dot_product_attention(q, k, v)
        # With V=1, attention output should be average of rows (which all equal 1)
        expected = torch.ones_like(out)
        torch.testing.assert_close(
            out, expected, rtol=1e-4, atol=1e-4, msg="Attention output ≠ 1 when V=1"
        )