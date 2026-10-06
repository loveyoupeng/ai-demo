"""Elementwise CUDA kernel tests — SiLU, RMSNorm, RoPE, SwiGLU.

These kernels compile independently (no cross-compilation dependencies).
Each runs in its own subprocess via the conftest batching strategy.
"""

from __future__ import annotations

import pytest
import torch


def skip_if_no_gpu() -> None:
    """Skip test if no CUDA GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")


# ================================================================
# SECTION: Activation Kernels — SiLU
# ================================================================


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


# ================================================================
# SECTION: Normalization Kernels — RMSNorm
# ================================================================


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
    def test_rmsnorm_multi_warp_float32(self):
        """Embed dims wider than one warp (D > 32) must normalize EVERY element.

        Regression: the block reduction used to publish the block total only
        to warp 0, so elements 32..D-1 were scaled by a zero/garbage inv_rms.
        """
        skip_if_no_gpu()
        import torch.nn.functional as F

        from impl._cuda.layernorm import rmsnorm

        for D in (64, 96, 128):
            torch.manual_seed(42)
            x = torch.randn(2, 4, D, dtype=torch.float32, device="cuda")
            gamma = torch.randn(D, dtype=torch.float32, device="cuda")
            y_cuda = rmsnorm(x, gamma)
            y_torch = F.rms_norm(x, [D], weight=gamma, eps=1e-6)
            torch.testing.assert_close(
                y_cuda, y_torch, rtol=1e-4, atol=1e-4, msg=f"CUDA RMSNorm != torch RMSNorm at D={D} (multi-warp block)"
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


# ================================================================
# SECTION: Positional Encoding — RoPE
# ================================================================


class TestRoPECUDA:
    @pytest.mark.timeout(30)
    def test_rope_matches_numpy_reference_float32(self):
        """Full rotation: CUDA float32 matches the NumPy reference (rtol=1e-4)."""
        skip_if_no_gpu()
        import numpy as np

        from impl._cuda.rope import apply_rope
        from impl._np.rope import RoPE

        B, S, H, D = 2, 4, 2, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, H, D, dtype=torch.float32, device="cuda")
        positions = torch.arange(S, dtype=torch.int64, device="cuda")

        y_cuda = apply_rope(x, positions)
        y_ref = RoPE().forward(x.cpu().numpy(), positions.cpu().numpy(), rope_dim=0)

        torch.testing.assert_close(
            y_cuda,
            torch.tensor(np.asarray(y_ref), dtype=torch.float32, device="cuda"),
            rtol=1e-4,
            atol=1e-4,
            msg="CUDA RoPE (full) != NumPy reference float32",
        )

    @pytest.mark.timeout(30)
    def test_rope_matches_numpy_reference_float64(self):
        """Full rotation: CUDA float64 matches the NumPy reference tightly."""
        skip_if_no_gpu()
        import numpy as np

        from impl._cuda.rope import apply_rope
        from impl._np.rope import RoPE

        B, S, H, D = 2, 4, 2, 8
        torch.manual_seed(42)
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        positions = torch.arange(S, dtype=torch.int64, device="cuda")

        y_cuda = apply_rope(x, positions)
        y_ref = RoPE().forward(x.cpu().numpy(), positions.cpu().numpy(), rope_dim=0)

        torch.testing.assert_close(
            y_cuda,
            torch.tensor(np.asarray(y_ref), dtype=torch.float64, device="cuda"),
            rtol=1e-6,
            atol=1e-6,
            msg="CUDA RoPE (full) != NumPy reference float64",
        )

    @pytest.mark.timeout(30)
    def test_rope_partial_rotation_matches_numpy(self):
        """Partial RoPE (rope_dim=D//2): prefix rotated like the NumPy reference,
        suffix dims pass through unchanged."""
        skip_if_no_gpu()
        import numpy as np

        from impl._cuda.rope import apply_rope
        from impl._np.rope import RoPE

        B, S, H, D = 2, 5, 2, 16
        rope_dim = 8  # true partial: 8 < head_dim 16
        torch.manual_seed(7)
        x = torch.randn(B, S, H, D, dtype=torch.float64, device="cuda")
        positions = torch.arange(S, dtype=torch.int64, device="cuda")

        y_cuda = apply_rope(x, positions, rope_dim=rope_dim)
        y_ref = RoPE().forward(x.cpu().numpy(), positions.cpu().numpy(), rope_dim=rope_dim)

        # Suffix dims must pass through untouched (identity, not rotation)
        torch.testing.assert_close(
            y_cuda[..., rope_dim:], x[..., rope_dim:], rtol=0, atol=0, msg="partial RoPE rotated the suffix dims"
        )
        # Full result matches the NumPy reference (rotated prefix included)
        torch.testing.assert_close(
            y_cuda,
            torch.tensor(np.asarray(y_ref), dtype=torch.float64, device="cuda"),
            rtol=1e-6,
            atol=1e-6,
            msg="CUDA partial RoPE != NumPy reference — rope_dim must be wired through",
        )

    @pytest.mark.timeout(30)
    def test_rope_partial_backward_matches_numpy(self):
        """Partial RoPE backward: CUDA autograd gradient matches the NumPy
        analytic backward; suffix gradient passes through unchanged."""
        skip_if_no_gpu()
        import numpy as np

        from impl._cuda.rope import apply_rope
        from impl._np.rope import RoPE

        B, S, H, D = 2, 5, 2, 16
        rope_dim = 8
        torch.manual_seed(11)
        x_np = np.random.RandomState(11).randn(B, S, H, D)
        dout_np = np.random.RandomState(13).randn(B, S, H, D)
        pos_np = np.arange(S, dtype=np.int64)

        # CUDA autograd path
        x = torch.tensor(x_np, dtype=torch.float64, device="cuda", requires_grad=True)
        positions = torch.tensor(pos_np, device="cuda")
        y = apply_rope(x, positions, rope_dim=rope_dim)
        y.backward(torch.tensor(dout_np, dtype=torch.float64, device="cuda"))
        dx_cuda = x.grad

        # NumPy analytic backward (verified reference)
        dx_ref = RoPE().backward(dout_np, x_np, pos_np.astype(np.int32), rope_dim=rope_dim)

        torch.testing.assert_close(
            dx_cuda,
            torch.tensor(dx_ref, dtype=torch.float64, device="cuda"),
            rtol=1e-6,
            atol=1e-6,
            msg="CUDA partial RoPE backward != NumPy reference",
        )
        # Suffix gradient is the identity path
        torch.testing.assert_close(
            dx_cuda[..., rope_dim:],
            torch.tensor(dout_np, dtype=torch.float64, device="cuda")[..., rope_dim:],
            rtol=1e-9,
            atol=1e-12,
            msg="suffix gradient must pass through unchanged",
        )

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


# ================================================================
# SECTION: Feed-Forward Networks — SwiGLU
# ================================================================


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
