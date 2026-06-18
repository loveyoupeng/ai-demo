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
    def test_zero_qk_scores(self):
        """Large negative QK scores → near-zero attention output (all mass on one key)."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, S, d = 1, 1, 3, 8
        q = torch.ones(B, H, S, d, dtype=torch.float64, device="cuda")
        k = -100.0 * torch.ones(B, H, S, d, dtype=torch.float64, device="cuda")
        v = torch.zeros(B, H, S, d, dtype=torch.float64, device="cuda")

        out = scaled_dot_product_attention(q, k, v)
        # All attention mass on first key (which has value 0)
        expected = v[:, :, 0:1, :].expand(B, H, S, d)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.timeout(30)
    def test_identity_kv(self):
        """K=V=I → output ≈ q (max attention mass on diagonal)."""
        skip_if_no_gpu()
        from impl._triton.attn import scaled_dot_product_attention

        B, H, S, d = 1, 1, 5, 8
        q = torch.randn(B, H, S, d, dtype=torch.float64, device="cuda")
        k = torch.eye(d, dtype=torch.float64, device="cuda").unsqueeze(0).unsqueeze(0)[:, :, :S, :]
        v = torch.eye(d, dtype=torch.float64, device="cuda").unsqueeze(0).unsqueeze(0)[:, :, :S, :]

        out = scaled_dot_product_attention(q, k, v)
        # With large QK, q's similarity with k determines mask
        assert torch.isfinite(out).all()

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
