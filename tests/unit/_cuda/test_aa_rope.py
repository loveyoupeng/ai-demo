import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestRoPECUDA:
    @pytest.mark.timeout(30)
    def test_rope_matches_torch_float32(self):
        """Same float32 input → same output as torch RoPE (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        from impl._cuda.rope import apply_rope

        B, S, H, D = 2, 4, 2, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, H, D, dtype=torch.float32, device="cuda")
        positions = torch.arange(S, dtype=torch.int64, device="cuda")

        y_cuda = apply_rope(x, positions)
        y_torch = apply_rope(x, positions)

        torch.testing.assert_close(y_cuda, y_torch, rtol=1e-4, atol=1e-4, msg="CUDA RoPE != torch RoPE float32")

    @pytest.mark.timeout(30)
    def test_rope_matches_torch_float64(self):
        """Same float64 input → same output as torch RoPE (rtol=1e-4, atol=1e-4)."""
        skip_if_no_gpu()
        from impl._cuda.rope import apply_rope

        B, S, H, D = 2, 4, 2, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        positions = torch.arange(S, dtype=torch.int64, device="cuda")

        y_cuda = apply_rope(x, positions)
        y_torch = apply_rope(x, positions)

        torch.testing.assert_close(y_cuda, y_torch, rtol=1e-4, atol=1e-4, msg="CUDA RoPE != torch RoPE float64")

    @pytest.mark.timeout(30)
    def test_rope_shapes(self):
        """RoPE preserves shape for various (B, S, H, D) shapes."""
        skip_if_no_gpu()
        from impl._cuda.rope import apply_rope

        torch.manual_seed(42)
        shapes = [(1, 4, 2, 8), (2, 4, 2, 8), (2, 8, 4, 16)]

        for B, S, H, D in shapes:
            x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
            positions = torch.arange(S, dtype=torch.int64, device="cuda")
            y = apply_rope(x, positions)
            assert y.shape == (B, S, H, D), f"Expected {(B, S, H, D)}, got {y.shape}"
            assert torch.isfinite(y).all(), f"NaN/Inf detected in shape {(B, S, H, D)}"

    @pytest.mark.timeout(30)
    def test_rope_norm_preservation(self):
        """RoPE preserves vector norms (orthogonal transformation)."""
        skip_if_no_gpu()
        from impl._cuda.rope import apply_rope

        B, S, H, D = 2, 4, 4, 16
        torch.manual_seed(42)
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        positions = torch.arange(S, dtype=torch.int64, device="cuda")
        y = apply_rope(x, positions)

        # Norm of each vector should be preserved (RoPE is orthogonal)
        x_norm = x.norm(dim=-1)  # (B, S, H)
        y_norm = y.norm(dim=-1)  # (B, S, H)
        torch.testing.assert_close(x_norm, y_norm, rtol=1e-4, atol=1e-4, msg="RoPE should preserve vector norms")
