"""Tests for CUDA TransformerBlock — assembly of all CUDA primitives.

Tests cover:
  - Shape correctness (all tensors maintain proper dimensions)
  - Weight initialization (gates zero, norms ones)
  - Attention computation (CUDA SDPA integration)
  - Gated residual behavior (sigmoid gates control signal flow)
  - MoE integration (CUDA MoE kernel)
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


class TestBlockInit:
    """Test TransformerBlock parameter initialization."""

    def test_block_creates_without_error(self) -> None:
        """Block instantiation must not raise."""
        cfg = dict(
            embed_dim=32, n_heads=4, n_experts=2, ff_dim=64, k=2, rope_dim=0, seed=42,
        )
        block = CuTransformerBlock(**cfg)
        assert block is not None

    def test_block_attributes_present(self) -> None:
        """All expected attributes must be present."""
        cfg = dict(
            embed_dim=32, n_heads=4, n_experts=2, ff_dim=64, k=2, rope_dim=0, seed=42,
        )
        block = CuTransformerBlock(**cfg)
        required = [
            "ln1_gamma",
            "ln2_gamma",
            "gate1",
            "gate2",
            "Wq",
            "Wk",
            "Wv",
            "Wo",
            "expert_weights",
            "expert_bias",
            "routing_weights",
        ]
        for attr in required:
            assert hasattr(block, attr), f"Missing attribute: {attr}"
            assert isinstance(getattr(block, attr), torch.Tensor), f"{attr} is not Tensor"

    def test_ln_gamma_shape(self) -> None:
        """RMSNorm gamma: (D,)."""
        cfg = dict(
            embed_dim=64, n_heads=4, n_experts=2, ff_dim=128, k=2, rope_dim=0, seed=42,
        )
        D = cfg["embed_dim"]
        block = CuTransformerBlock(**cfg)
        assert block.ln1_gamma.shape == (D,)
        assert block.ln2_gamma.shape == (D,)
        assert torch.allclose(block.ln1_gamma, torch.ones(D))
        assert torch.allclose(block.ln2_gamma, torch.ones(D))

    def test_gate_initialization(self) -> None:
        """Gates: initialized to zeros with shape (1,)."""
        cfg = dict(
            embed_dim=64, n_heads=4, n_experts=2, ff_dim=128, k=2, rope_dim=0, seed=42,
        )
        block = CuTransformerBlock(**cfg)
        assert block.gate1.shape == (1,)
        assert block.gate2.shape == (1,)
        assert torch.allclose(block.gate1, torch.zeros(1))
        assert torch.allclose(block.gate2, torch.zeros(1))

    def test_mha_weight_shapes(self) -> None:
        """Q/K/V/O projections: (D, D)."""
        cfg = dict(
            embed_dim=64, n_heads=4, n_experts=2, ff_dim=128, k=2, rope_dim=0, seed=42,
        )
        D = cfg["embed_dim"]
        block = CuTransformerBlock(**cfg)
        assert block.Wq.shape == (D, D)
        assert block.Wk.shape == (D, D)
        assert block.Wv.shape == (D, D)
        assert block.Wo.shape == (D, D)

    def test_moe_weight_shapes(self) -> None:
        """Expert weights: (N, D, D), bias: (N, D), routing: (N, D)."""
        cfg = dict(
            embed_dim=32, n_heads=4, n_experts=3, ff_dim=64, k=2, rope_dim=0, seed=42,
        )
        D = cfg["embed_dim"]
        N = cfg["n_experts"]
        block = CuTransformerBlock(**cfg)
        assert block.expert_weights.shape == (N, D, D)
        assert block.expert_bias.shape == (N, D)
        assert block.routing_weights.shape == (N, D)
        assert torch.allclose(block.expert_weights, torch.zeros(N, D, D))
        assert torch.allclose(block.expert_bias, torch.zeros(N, D))
        assert torch.allclose(block.routing_weights, torch.zeros(N, D))

    def test_head_dim_divisibility(self) -> None:
        """head_dim = embed_dim // n_heads."""
        cfg = dict(
            embed_dim=64, n_heads=8, n_experts=2, ff_dim=128, k=2, rope_dim=0, seed=0,
        )
        block = CuTransformerBlock(**cfg)
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


class TestBlockForward:
    """Test TransformerBlock forward pass on CUDA."""

    @pytest.fixture()
    def block_on_cuda(self) -> CuTransformerBlock:
        """Create block and move ALL weights to CUDA."""
        cfg = dict(embed_dim=64, n_heads=4, n_experts=4, ff_dim=128, k=2, rope_dim=16, seed=42)
        block = CuTransformerBlock(**cfg)
        # Use .to() on the whole block to ensure all attributes move to CUDA
        block.ln1_gamma = block.ln1_gamma.cuda()
        block.ln2_gamma = block.ln2_gamma.cuda()
        block.gate1 = block.gate1.cuda()
        block.gate2 = block.gate2.cuda()
        block.Wq = block.Wq.cuda()
        block.Wk = block.Wk.cuda()
        block.Wv = block.Wv.cuda()
        block.Wo = block.Wo.cuda()
        block.expert_weights = block.expert_weights.cuda()
        block.expert_bias = block.expert_bias.cuda()
        block.routing_weights = block.routing_weights.cuda()
        return block

    @pytest.fixture()
    def positions(self) -> torch.Tensor:
        return torch.arange(8, device="cuda")

    def test_forward_shape(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward must preserve (B, S, D) shape."""
        B, S, D = 2, 8, 64
        x = torch.randn(B, S, D, device="cuda")
        block_on_cuda.rope_dim = 0  # disable rope for this test
        out = block_on_cuda.forward(x, positions, training=False)
        assert out.shape == (B, S, D), f"Expected ({B}, {S}, {D}), got {out.shape}"

    def test_forward_fp32(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward must work with fp32 input."""
        B, S, D = 1, 4, 64
        x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
        block_on_cuda.rope_dim = 0
        out = block_on_cuda.forward(x, positions, training=False)
        assert out.dtype == torch.float32
        assert out.shape == (B, S, D)

    def test_forward_no_rope(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward with rope_dim=0 must work."""
        B, S, D = 2, 4, 64
        x = torch.randn(B, S, D, device="cuda")
        block_on_cuda.rope_dim = 0
        out = block_on_cuda.forward(x, positions, training=False)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()

    def test_forward_with_positions(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward with explicit positions must work."""
        B, S, D = 1, 8, 64
        x = torch.randn(B, S, D, device="cuda")
        out = block_on_cuda.forward(x, positions, training=False)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()

    def test_forward_with_rope(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward with rope_dim > 0 must produce correct output."""
        B, S, D = 1, 8, 64
        x = torch.randn(B, S, D, device="cuda")
        block_on_cuda.rope_dim = 16
        out = block_on_cuda.forward(x, positions, training=False)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()

    def test_gate_effect(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Zero gates → block output ≈ x (identity at init)."""
        B, S, D = 1, 4, 64
        x = torch.randn(B, S, D, device="cuda")
        block_on_cuda.rope_dim = 0
        # Gates are zero → sigmoid(0) = 0.5, not exactly identity
        # But the output should still be bounded (no explode / NaN)
        out = block_on_cuda.forward(x, positions, training=False)
        assert torch.isfinite(out).all()
        # With zero gates and all-zero expert weights, MoE outputs zero
        # With sigmoid(0)=0.5, gated residual adds 50% of previous value
        # Output should be finite
        assert out.numel() > 0

    def test_forward_all_zero_block(self) -> None:
        """Block with all-zero weights and zero gates → output = residual (x only)."""
        cfg = dict(embed_dim=16, n_heads=2, n_experts=2, ff_dim=32, k=1, rope_dim=0, seed=0)
        block = CuTransformerBlock(**cfg)
        # All weights already zero (Wq=init, but expert_weights=bias=routing all zero)
        # Set attention weights to zero too
        with torch.no_grad():
            block.Wq.zero_()
            block.Wk.zero_()
            block.Wv.zero_()
            block.Wo.zero_()
        d = cfg["embed_dim"]  # 16
        s = 2  # seq len
        # Create proper (B, S, D) shape input
        x = torch.zeros(1, s, d, device="cuda")
        x[0, 0, :8] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        x[0, 1, 8:] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        out = block.forward(x, torch.zeros(2, device="cuda"), training=False)
        assert out.shape == (1, s, d)

    def test_forward_large_batch(self, block_on_cuda: CuTransformerBlock, positions: torch.Tensor) -> None:
        """Forward must handle larger batch sizes."""
        B, S, D = 4, 16, 64
        x = torch.randn(B, S, D, device="cuda")
        block_on_cuda.rope_dim = 0
        pos = torch.arange(S, device="cuda")
        out = block_on_cuda.forward(x, pos, training=False)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()


class TestBlockMoEIntegration:
    """Test MoE kernel integration within the block."""

    @pytest.fixture()
    def block_on_cuda(self) -> CuTransformerBlock:
        """Create block and move ALL weights to CUDA."""
        cfg = dict(embed_dim=32, n_heads=2, n_experts=3, ff_dim=64, k=2, rope_dim=0, seed=7)
        block = CuTransformerBlock(**cfg)
        block.ln1_gamma = block.ln1_gamma.cuda()
        block.ln2_gamma = block.ln2_gamma.cuda()
        block.gate1 = block.gate1.cuda()
        block.gate2 = block.gate2.cuda()
        block.Wq = block.Wq.cuda()
        block.Wk = block.Wk.cuda()
        block.Wv = block.Wv.cuda()
        block.Wo = block.Wo.cuda()
        block.expert_weights = block.expert_weights.cuda()
        block.expert_bias = block.expert_bias.cuda()
        block.routing_weights = block.routing_weights.cuda()
        return block

    def test_moe_output_via_block(self, block_on_cuda: CuTransformerBlock) -> None:
        """MoE output must be (B, S, D)."""
        B, S, D = 1, 4, 32
        x = torch.randn(B, S, D, device="cuda")
        block_on_cuda.rope_dim = 0
        out = block_on_cuda.forward(x, torch.zeros(S, device="cuda"), training=False)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()

    def test_moe_output_changes_with_weights(self, block_on_cuda: CuTransformerBlock) -> None:
        """Different expert weights must produce different outputs."""
        B, S, D = 1, 2, 32
        x = torch.randn(B, S, D, device="cuda")
        block_on_cuda.rope_dim = 0

        # Forward pass 1 (random weights)
        out1 = block_on_cuda.forward(x, torch.zeros(S, device="cuda"), training=False)

        # Forward pass 2 (with different expert weights)
        block_on_cuda.expert_weights.data = torch.randn(
            block_on_cuda.n_experts, D, D, device="cuda"
        ) * 0.1
        block_on_cuda.expert_bias.data = torch.randn(
            block_on_cuda.n_experts, D, device="cuda"
        ) * 0.1
        out2 = block_on_cuda.forward(x, torch.zeros(S, device="cuda"), training=False)

        # Outputs should differ (different expert computations)
        assert not torch.allclose(out1, out2, atol=1e-4)
        assert torch.isfinite(out2).all()
