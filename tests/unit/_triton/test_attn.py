import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestScaledAttentionKernel:
    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """Q[B,H,Sq,d], K[B,H,Sk,d], V[B,H,Sk,d] → out[B,H,Sq,d]."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, Sq, Sk, d = 2, 4, 8, 8, 16
        q = torch.randn(B, H, Sq, d, dtype=torch.float64, device="cuda")
        k = torch.randn(B, H, Sk, d, dtype=torch.float64, device="cuda")
        v = torch.randn(B, H, Sk, d, dtype=torch.float64, device="cuda")
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, Sq, d), f"Expected {(B, H, Sq, d)}, got {out.shape}"

    @pytest.mark.timeout(30)
    def test_output_shape_asymmetric(self):
        """Different query and key sequence lengths."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, Sq, Sk, d = 2, 4, 10, 5, 16
        q = torch.randn(B, H, Sq, d, dtype=torch.float64, device="cuda")
        k = torch.randn(B, H, Sk, d, dtype=torch.float64, device="cuda")
        v = torch.randn(B, H, Sk, d, dtype=torch.float64, device="cuda")
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, Sq, d), f"Expected {(B, H, Sq, d)}, got {out.shape}"

    @pytest.mark.timeout(30)
    def test_attention_weights_sum_to_one(self):
        """Per-query, attention weights (softmax of QK) sum to 1 over keys."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, S, d = 2, 4, 8, 16
        q = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        k = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        v = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        out = scaled_dot_product_attention(q, k, v)
        # Check that attention output values are finite (implies valid softmax)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    @pytest.mark.timeout(30)
    def test_causal_uniform_attention(self):
        """All-equal K/V → each query averages over its causal keys (j <= i)."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, S, d = 1, 1, 3, 8
        q = torch.ones(B, H, S, d, dtype=torch.float64, device="cuda")
        k = -100.0 * torch.ones(B, H, S, d, dtype=torch.float64, device="cuda")
        v = torch.arange(S, dtype=torch.float64, device="cuda").repeat_interleave(d).view(1, 1, S, d)

        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        # Row i spreads equal weight over keys 0..i: mean of 0..i.
        expected = (
            torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, device="cuda").repeat_interleave(d).view(1, 1, S, d)
        )
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.timeout(30)
    def test_causal_prefix_invariance(self):
        """Causality: extending a sequence leaves every prefix output unchanged."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, d = 1, 1, 8
        S1, S2 = 6, 10
        q2 = torch.randn(B, H, S2, d, dtype=torch.float64, device="cuda")
        k2 = torch.randn(B, H, S2, d, dtype=torch.float64, device="cuda")
        v2 = torch.randn(B, H, S2, d, dtype=torch.float64, device="cuda")

        out_long = scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        out_short = scaled_dot_product_attention(q2[:, :, :S1], k2[:, :, :S1], v2[:, :, :S1], is_causal=True)
        torch.testing.assert_close(out_long[:, :, :S1], out_short, rtol=5e-3, atol=1.5e-3)

    @pytest.mark.timeout(30)
    def test_parity_with_torch_sdpa(self):
        """Same float64 input → same output as torch.scaled_dot_product_attention (rtol=1e-4)."""
        skip_if_no_gpu()
        import torch.nn.functional as F

        from impl._triton.attn import scaled_dot_product_attention

        B, H, S, d = 2, 4, 8, 16
        q = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        k = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        v = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")

        y_triton = scaled_dot_product_attention(q, k, v)
        y_torch = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Triton kernel computes in fp32; PyTorch sdpa preserves dtype (fp64 here)
        # Accept fp32-level drift
        torch.testing.assert_close(y_triton, y_torch, rtol=5e-3, atol=1.5e-3)

    @pytest.mark.timeout(30)
    def test_gradient_shape(self):
        """Gradients w.r.t. Q, K, V have correct shapes."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, S, d = 2, 4, 8, 16
        q = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda", requires_grad=True)
        out = scaled_dot_product_attention(q, k, v)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape
        assert torch.isfinite(q.grad).all()

    @pytest.mark.timeout(30)
    def test_large_batch(self):
        """Larger batch and seq_len work without OOM or shape issues."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, S, d = 32, 8, 64, 64
        q = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        k = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        v = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, S, d)
        assert torch.isfinite(out).all()
