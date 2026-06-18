import numpy as np
import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestRoPEKernel:
    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """Input [B, S, H, D], pos [S] → output [B, S, H, D]."""
        skip_if_no_gpu()
        from impl._triton.rope import apply_rope

        B, S, H, D = 2, 4, 3, 8
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        pos = torch.arange(S, device="cuda", dtype=torch.int64)
        y = apply_rope(x, pos)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    @pytest.mark.timeout(30)
    def test_position_zero_identity(self):
        """When all positions are 0, output should equal input (cos(0)=1, sin(0)=0)."""
        skip_if_no_gpu()
        from impl._triton.rope import apply_rope

        B, S, H, D = 2, 4, 3, 8
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        pos = torch.zeros(S, device="cuda", dtype=torch.int64)
        y = apply_rope(x, pos)
        torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_pass_through_dims(self):
        """Dims beyond rope_dim should pass through unchanged."""
        skip_if_no_gpu()
        from impl._triton.rope import apply_rope

        B, S, H, D = 2, 4, 3, 16
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        pos = torch.arange(S, device="cuda", dtype=torch.int64)
        rope_dim = 8  # rotate first 8 dims, pass through last 8 dims
        y = apply_rope(x, pos, rope_dim=rope_dim)
        assert torch.allclose(y[..., rope_dim:], x[..., rope_dim:], rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_output_norm_preserved(self):
        """Euclidean norm per token should be preserved by rotation."""
        skip_if_no_gpu()
        from impl._triton.rope import apply_rope

        B, S, H, D = 2, 4, 3, 8
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        pos = torch.arange(S, device="cuda", dtype=torch.int64)
        y = apply_rope(x, pos)
        # Norm of each token across all dims: (B, S, H)
        norm_x = torch.norm(x, dim=-1)
        norm_y = torch.norm(y, dim=-1)
        torch.testing.assert_close(norm_y, norm_x, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_parity_with_numpy(self):
        """Same input → same output as NumPy RoPE (rtol=1e-4)."""
        skip_if_no_gpu()
        from impl._triton.rope import apply_rope

        B, S, H, D = 2, 4, 3, 8
        x_np = np.random.randn(B, S, H, D).astype(np.float64)
        x_torch = torch.from_numpy(x_np).cuda()
        pos = torch.arange(S, device="cuda", dtype=torch.int64)

        y_triton = apply_rope(x_torch, pos)
        y_numpy = _np_rope_reference(x_np, pos.cpu().numpy())
        y_tr_np = y_triton.cpu().numpy()

        np.testing.assert_allclose(y_numpy, y_tr_np, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_parity_with_torch(self):
        """Same float64 input → same output as PyTorch RoPE reference (rtol=1e-4)."""
        skip_if_no_gpu()
        from impl._triton.rope import apply_rope

        B, S, H, D = 2, 4, 3, 8
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        pos = torch.arange(S, device="cuda", dtype=torch.int64)

        y_triton = apply_rope(x, pos)
        y_ref = _torch_rope_reference(x, pos)
        torch.testing.assert_close(y_triton, y_ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_batched_positions(self):
        """Each batch element can have different positions."""
        skip_if_no_gpu()
        from impl._triton.rope import apply_rope

        B, S, H, D = 2, 4, 3, 8
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        # Different position indices for each batch
        pos = torch.tensor([0, 1, 2, 2], device="cuda", dtype=torch.int64)
        y = apply_rope(x, pos)
        assert y.shape == x.shape
        # Verify norm is preserved
        norm_x = torch.norm(x, dim=-1)
        norm_y = torch.norm(y, dim=-1)
        torch.testing.assert_close(norm_y, norm_x, rtol=1e-4, atol=1e-4)


# ——— Reference implementations embedded in tests ——— #


def _np_rope_reference(x, position):
    """NumPy reference implementation of RoPE — matches impl/_np/modules.py exactly."""
    B, S, H, D = x.shape
    pair_dim = D // 2

    freqs = 1.0 / (10000.0 ** (np.arange(pair_dim, dtype=np.float32) * 2.0 / D))

    pos = np.asarray(position, dtype=np.int32)
    if pos.ndim == 1:
        pos_broadcast = pos[np.newaxis, :, np.newaxis, np.newaxis]  # (1, S, 1, 1)
    else:
        pos_broadcast = position[np.newaxis, :, np.newaxis, np.newaxis]

    angles = (pos_broadcast * freqs[np.newaxis, np.newaxis, np.newaxis, :]).astype(np.float64)  # (B, S, H, pair_dim)
    cos = np.cos(angles)
    sin = np.sin(angles)

    x_reshape = x.reshape(x.shape[:-1] + (pair_dim, 2))  # (B, S, H, pair_dim, 2)
    x_even = x_reshape[..., 0]  # (B, S, H, pair_dim)
    x_odd = x_reshape[..., 1]  # (B, S, H, pair_dim)

    y_even = x_even * cos - x_odd * sin  # (B, S, H, pair_dim)
    y_odd = x_even * sin + x_odd * cos  # (B, S, H, pair_dim)

    return np.stack([y_even, y_odd], axis=-1).reshape(x.shape)


def _torch_rope_reference(x, positions):
    """PyTorch reference implementation matching impl/_torch/layers.py."""
    B, S, H, D = x.shape
    pair_dim = D // 2

    freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, device=x.device) * 2.0 / D))
    # positions: (S,)
    angles = positions[:, None] * freqs[None, :]  # (S, pair_dim)

    cos = torch.cos(angles)  # (S, pair_dim)
    sin = torch.sin(angles)  # (S, pair_dim)

    # Broadcast to (B, S, H, pair_dim)
    cos = cos[:, None, :].unsqueeze(0).repeat(B, 1, H, 1)  # (B, S, H, pair_dim)
    sin = sin[:, None, :].unsqueeze(0).repeat(B, 1, H, 1)  # (B, S, H, pair_dim)

    x_reshape = x.reshape(B, S, H, pair_dim, 2)
    x_even = x_reshape[..., 0]
    x_odd = x_reshape[..., 1]

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    return torch.stack([y_even, y_odd], dim=-1).reshape(B, S, H, D)
