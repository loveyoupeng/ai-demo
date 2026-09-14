"""Tests for CUDA TransformerBlock — assembly of all CUDA primitives.

Tests cover:
  - Shape correctness (all tensors maintain proper dimensions)
  - Weight initialization (norms ones, projections bounded)
  - Attention computation (CUDA SDPA integration)
  - MoE integration (torch routing + per-expert CUDA SwiGLU)
  - Parameter availability for cross-backend parity
"""

from __future__ import annotations

import pytest
import torch

from impl._cuda.block import (
    CuTransformerBlock,
    _init_weight,
    _init_zeros,
)
from shared.config import TransformerConfig


def _block_cfg(**overrides) -> TransformerConfig:
    defaults: dict[str, object] = dict(
        vocab_size=64, embed_dim=32, n_layers=1, n_heads=4, n_experts=2, top_k=2, expert_dim=64, rope_dim=0, seed=42
    )
    defaults.update(overrides)
    if "n_groups" not in overrides and "n_heads" in overrides:
        defaults["n_groups"] = defaults["n_heads"]
    return TransformerConfig.from_dict(defaults)


class TestBlockInit:
    """Test TransformerBlock parameter initialization."""

    def test_block_creates_without_error(self) -> None:
        """Block instantiation must not raise."""
        block = CuTransformerBlock(_block_cfg(), seed=42)
        assert block is not None

    def test_block_attributes_present(self) -> None:
        """All expected attributes must be present."""
        block = CuTransformerBlock(_block_cfg(), seed=42)
        required = [
            "input_layernorm_gamma",
            "post_attention_layernorm_gamma",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "router",
            "expert_gate_proj",
            "expert_up_proj",
            "expert_down_proj",
        ]
        for attr in required:
            assert hasattr(block, attr), f"Missing attribute: {attr}"
            assert isinstance(getattr(block, attr), torch.Tensor), f"{attr} is not Tensor"

    def test_ln_gamma_shape(self) -> None:
        """RMSNorm gamma: (D,)."""
        D = 32
        block = CuTransformerBlock(_block_cfg(embed_dim=D), seed=42)
        assert block.input_layernorm_gamma.shape == (D,)
        assert block.post_attention_layernorm_gamma.shape == (D,)
        assert torch.allclose(block.input_layernorm_gamma, torch.ones(D))
        assert torch.allclose(block.post_attention_layernorm_gamma, torch.ones(D))

    def test_mha_weight_shapes(self) -> None:
        """Q/K/V/O projections: (D, H*hd) / (D, G*hd) / (H*hd, D)."""
        D, H, G, hd = 32, 4, 4, 8
        block = CuTransformerBlock(_block_cfg(embed_dim=D, n_heads=H, n_groups=G), seed=42)
        assert block.q_proj.shape == (D, H * hd)
        assert block.k_proj.shape == (D, G * hd)
        assert block.v_proj.shape == (D, G * hd)
        assert block.o_proj.shape == (H * hd, D)

    def test_gqa_k_v_shapes(self) -> None:
        """GQA: k/v have n_groups heads, q has n_heads heads."""
        D, H, G = 32, 6, 3
        block = CuTransformerBlock(_block_cfg(embed_dim=D, n_heads=H, n_groups=G), seed=42)
        assert block.q_proj.shape == (D, 6 * (D // 6))
        assert block.k_proj.shape == (D, 3 * (D // 6))
        assert block.v_proj.shape == (D, 3 * (D // 6))

    def test_moe_weight_shapes(self) -> None:
        """MoE: router (D, E); expert gate/up (E, D, FF); expert down (E, FF, D)."""
        E, D, FF = 2, 32, 64
        block = CuTransformerBlock(_block_cfg(embed_dim=D, n_experts=E, expert_dim=FF), seed=42)
        assert block.router.shape == (D, E)
        assert block.expert_gate_proj.shape == (E, D, FF)
        assert block.expert_up_proj.shape == (E, D, FF)
        assert block.expert_down_proj.shape == (E, FF, D)

    def test_head_dim_divisibility(self) -> None:
        """head_dim = embed_dim // n_heads."""
        block = CuTransformerBlock(_block_cfg(embed_dim=64, n_heads=8), seed=42)
        assert block.head_dim == 8


class TestInitHelpers:
    """Test weight initialization helper functions."""

    def test_init_weight_output_shape(self) -> None:
        """_init_weight returns (rows, cols) on CPU."""
        w = _init_weight(10, 20, seed=0)
        assert w.shape == (10, 20)
        assert w.device.type == "cpu"

    def test_init_zeros_output_shape(self) -> None:
        """_init_zeros returns all zeros on CPU."""
        z = _init_zeros((5, 10))
        assert z.shape == (5, 10)
        assert torch.allclose(z, torch.zeros(5, 10))
        assert z.device.type == "cpu"

    def test_init_weight_no_nan_or_inf(self) -> None:
        """_init_weight must produce finite values — no NaN or Inf."""
        w = _init_weight(64, 64, seed=42)
        assert not torch.isnan(w).any(), "Initialized weights contain NaN"
        assert not torch.isinf(w).any(), "Initialized weights contain Inf"

    def test_init_weight_finite_range(self) -> None:
        """_init_weight produces values within [min, max] bounds."""
        w = _init_weight(64, 64, seed=42)
        bound = (6.0 / (64 + 64)) ** 0.5
        assert w.min() >= -bound, f" Weight min {w.min().item():.4f} below lower bound {-bound:.4f}"
        assert w.max() <= bound, f" Weight max {w.max().item():.4f} above upper bound {bound:.4f}"

    def test_init_weight_reproducible(self) -> None:
        """_init_weight with same seed produces identical output."""
        w1 = _init_weight(64, 64, seed=99)
        w2 = _init_weight(64, 64, seed=99)
        assert torch.equal(w1, w2), "Same seed must produce identical weights"
        assert not torch.equal(_init_weight(64, 64, seed=99), _init_weight(64, 64, seed=100)), (
            "Different seeds must differ"
        )

    def test_init_weight_forward_no_nan(self) -> None:
        """Block with _init_weight weights produces valid forward output (no NaN)."""
        w = _init_weight(64, 64, seed=42)
        x = torch.randn(2, 4, 64, device="cuda")
        result = x @ w.cuda()
        assert not torch.isnan(result).any(), "Forward through initialized weight produces NaN"
        assert not torch.isinf(result).any(), "Forward through initialized weight produces Inf"


class TestBlockForward:
    """Test TransformerBlock forward pass on CUDA."""

    @pytest.fixture()
    def block_on_cuda(self) -> CuTransformerBlock:
        """Create block with MoE (forward moves weights to the input device)."""
        return CuTransformerBlock(
            _block_cfg(embed_dim=64, n_heads=4, n_experts=4, expert_dim=128, top_k=2, rope_dim=16, seed=42), seed=42
        )

    @pytest.fixture()
    def positions(self) -> torch.Tensor:
        return torch.arange(8, device="cuda")

    def test_forward_shape(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward must preserve (B, S, D) shape."""
        B, S, D = 2, 8, 64
        x = torch.randn(B, S, D, device="cuda")
        out = block_on_cuda.forward(x, positions)
        assert out.shape == (B, S, D), f"Expected ({B}, {S}, {D}), got {out.shape}"

    def test_forward_fp32(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward must work with fp32 input."""
        B, S, D = 1, 4, 64
        x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
        out = block_on_cuda.forward(x, torch.arange(S, device="cuda"))
        assert out.dtype == torch.float32
        assert out.shape == (B, S, D)

    def test_forward_no_rope(self) -> None:
        """Forward with rope_dim=0 must work."""
        block = CuTransformerBlock(
            _block_cfg(embed_dim=64, n_heads=4, n_experts=1, top_k=1, expert_dim=128, rope_dim=0, seed=42), seed=42
        )
        B, S, D = 2, 4, 64
        x = torch.randn(B, S, D, device="cuda")
        out = block.forward(x, torch.arange(S, device="cuda"))
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()

    def test_forward_with_positions(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward with explicit positions must work."""
        B, S, D = 1, 8, 64
        x = torch.randn(B, S, D, device="cuda")
        out = block_on_cuda.forward(x, positions)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()

    def test_forward_with_rope(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward with rope_dim > 0 must produce finite output."""
        B, S, D = 1, 8, 64
        x = torch.randn(B, S, D, device="cuda")
        out = block_on_cuda.forward(x, positions)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()

    def test_forward_all_zero_block(self) -> None:
        """Block with all-zero weights → attention/FFN contribute nothing (residual only)."""
        block = CuTransformerBlock(
            _block_cfg(embed_dim=16, n_heads=2, n_experts=1, top_k=1, expert_dim=32, rope_dim=0, seed=0), seed=0
        )
        with torch.no_grad():
            block.q_proj.zero_()
            block.k_proj.zero_()
            block.v_proj.zero_()
            block.o_proj.zero_()
            block.gate_proj.zero_()
            block.up_proj.zero_()
            block.down_proj.zero_()
        d = 16
        s = 2
        x = torch.zeros(1, s, d, device="cuda")
        x[0, 0, :8] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        x[0, 1, 8:] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        out = block.forward(x, torch.zeros(2, device="cuda"))
        assert out.shape == (1, s, d)
        # Zero weights: out = x + 0 + 0 = x
        assert torch.allclose(out, x, atol=1e-4)

    def test_forward_large_batch(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward must handle larger batch sizes."""
        B, S, D = 4, 16, 64
        x = torch.randn(B, S, D, device="cuda")
        pos = torch.arange(S, device="cuda")
        out = block_on_cuda.forward(x, pos)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()


class TestBlockMoEIntegration:
    """Test MoE integration within the block."""

    @pytest.fixture()
    def block_on_cuda(self) -> CuTransformerBlock:
        return CuTransformerBlock(
            _block_cfg(embed_dim=32, n_heads=2, n_experts=3, expert_dim=64, top_k=2, rope_dim=0, seed=7), seed=7
        )

    def test_moe_output_via_block(self, block_on_cuda: CuTransformerBlock) -> None:
        """MoE output must be (B, S, D)."""
        B, S, D = 1, 4, 32
        x = torch.randn(B, S, D, device="cuda")
        out = block_on_cuda.forward(x, torch.zeros(S, device="cuda"))
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()

    def test_moe_output_changes_with_weights(self, block_on_cuda: CuTransformerBlock) -> None:
        """Different expert weights must produce different outputs."""
        B, S, D = 1, 2, 32
        x = torch.randn(B, S, D, device="cuda")

        out1 = block_on_cuda.forward(x, torch.zeros(S, device="cuda"))

        # Perturb the expert weights
        with torch.no_grad():
            block_on_cuda.expert_gate_proj.add_(torch.randn_like(block_on_cuda.expert_gate_proj) * 0.1)
            block_on_cuda.expert_up_proj.add_(torch.randn_like(block_on_cuda.expert_up_proj) * 0.1)
            block_on_cuda.expert_down_proj.add_(torch.randn_like(block_on_cuda.expert_down_proj) * 0.1)
        out2 = block_on_cuda.forward(x, torch.zeros(S, device="cuda"))

        assert not torch.allclose(out1, out2, atol=1e-4)
        assert torch.isfinite(out2).all()
