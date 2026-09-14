"""CUDAModel — full decoder-only transformer: embedding → stack → lm_head (F9).

Tests cover model creation, stack wiring/forward, and gradient flow.
All model weights start on CPU and move to the input's device in forward.
"""

from __future__ import annotations

import pytest
import torch

from impl._cuda.model import CUDAModel
from shared.config import TransformerConfig

# ── Test fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_config() -> TransformerConfig:
    """Minimal model config for fast tests."""
    return TransformerConfig(
        vocab_size=512,
        embed_dim=64,
        n_layers=1,
        n_heads=4,
        n_groups=4,
        n_experts=2,
        top_k=2,
        expert_dim=128,
        rope_dim=16,
        seed=42,
    )


@pytest.fixture
def base_config() -> TransformerConfig:
    """Base config shared by all tests."""
    return TransformerConfig(
        vocab_size=64,
        n_layers=2,
        embed_dim=128,
        n_heads=4,
        n_groups=4,
        n_experts=4,
        top_k=2,
        expert_dim=256,
        rope_dim=32,
        seed=42,
    )


@pytest.fixture
def decoder_stack(base_config):
    """Small CuDecoderStack: 2 layers, 128 dim, 4 heads, 4 experts."""
    from impl._cuda.stack import CuDecoderStack

    return CuDecoderStack(base_config)


@pytest.fixture
def sample_input():
    """Batch of size 2, sequence length 8, dim 128, on CUDA."""
    B, S, D = 2, 8, 128
    return torch.randn(B, S, D, device="cuda")


# ================================================================
# SECTION: Model Components — CUDAModel
# ================================================================


class TestCuModelInit:
    """CUDAModel creation tests."""

    def test_has_vocab_size(self, small_config) -> None:
        """Model has the correct vocabulary size."""
        model = CUDAModel(small_config)
        assert model.vocab_size == small_config.vocab_size

    def test_has_embed_dim(self, small_config) -> None:
        """Model has the correct embedding dimension."""
        model = CUDAModel(small_config)
        assert model.embed_dim == small_config.embed_dim

    def test_has_embedding(self, small_config) -> None:
        """Model has an (V, D) embedding table."""
        model = CUDAModel(small_config)
        assert model.embedding_weights.shape == (small_config.vocab_size, small_config.embed_dim)

    def test_has_final_ln(self, small_config) -> None:
        """Model has final_norm_gamma attribute."""
        model = CUDAModel(small_config)
        assert model.final_norm_gamma.shape == (small_config.embed_dim,)

    def test_has_lm_head(self, small_config) -> None:
        """Model has an (D, V) lm_head weight."""
        model = CUDAModel(small_config)
        assert model.lm_head_weight.shape == (small_config.embed_dim, small_config.vocab_size)

    def test_forward_output_shape(self, small_config) -> None:
        """Forward: tokens (B, S) → logits (B, S, V)."""
        model = CUDAModel(small_config)
        tokens = torch.randint(0, model.vocab_size, (2, 16), device="cuda", dtype=torch.int64)
        logits = model.forward(tokens)
        assert logits.shape == (2, 16, small_config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_get_all_parameters_roundtrip(self, small_config) -> None:
        """save/load round-trip preserves every parameter (Keys scheme)."""
        import numpy as np

        model = CUDAModel(small_config)
        params = model.get_all_parameters()
        model.load_from_numpy_dict(params)
        params2 = model.get_all_parameters()
        assert set(params2) == set(params)
        for key, value in params.items():
            np.testing.assert_allclose(params2[key], value, rtol=1e-6, atol=1e-6, err_msg=f"Parameter {key} changed")


# ================================================================
# SECTION: Model Components — CuDecoderStack
# ================================================================


class TestDecoderStackInit:
    """CuDecoderStack — chained transformer blocks (F8).

    Architecture:
        x [B, S, D] → block_0 → block_1 → ... → block_{n-1} → out [B, S, D]

        - No position embeddings (RoPE handles positional info inside attention)
        - No final RMSNorm (belongs to the parent model)
        - Pre-norm blocks with dense SwiGLU or MoE feed-forward
    """

    def test_creation(self, decoder_stack, base_config) -> None:
        """A DecoderStack can be created with the specified config."""
        assert decoder_stack.n_layers == base_config.n_layers
        assert decoder_stack.embed_dim == base_config.embed_dim
        assert decoder_stack.head_dim == base_config.embed_dim // base_config.n_heads
        assert len(decoder_stack.blocks) == base_config.n_layers

    def test_blocks_are_transformer_blocks(self, decoder_stack) -> None:
        """Every block in the stack is a CuTransformerBlock instance."""
        from impl._cuda.block import CuTransformerBlock

        for block in decoder_stack.blocks:
            assert isinstance(block, CuTransformerBlock)

    def test_blocks_have_correct_device(self, decoder_stack) -> None:
        """All block weights are on CPU (move to CUDA on forward)."""
        for block in decoder_stack.blocks:
            assert block.q_proj.device.type == "cpu"
            assert block.k_proj.device.type == "cpu"
            assert block.v_proj.device.type == "cpu"
            assert block.input_layernorm_gamma.device.type == "cpu"

    def test_rope_disabled(self, base_config) -> None:
        """DecoderStack with rope_dim=0 creates blocks without RoPE."""
        from impl._cuda.stack import CuDecoderStack

        cfg = TransformerConfig(n_layers=2, embed_dim=32, n_heads=4, rope_dim=0)
        stack = CuDecoderStack(cfg)
        for block in stack.blocks:
            assert block.rope_dim == 0

    def test_head_dim_divisibility(self, base_config) -> None:
        """Head dimension is embed_dim // n_heads."""
        from impl._cuda.stack import CuDecoderStack

        stack = CuDecoderStack(base_config)
        for block in stack.blocks:
            assert block.head_dim == base_config.embed_dim // base_config.n_heads


class TestDecoderStackForward:
    """CuDecoderStack forward pass tests."""

    def test_output_shape(self, decoder_stack, sample_input):
        """Forward output has same shape as input: (B, S, D)."""
        out = decoder_stack.forward(sample_input)
        assert out.shape == sample_input.shape

    def test_output_same_device(self, decoder_stack, sample_input):
        """Output is on the same device as input."""
        out = decoder_stack.forward(sample_input)
        assert out.device == sample_input.device

    def test_single_layer(self):
        """A 1-layer stack with default params produces valid output."""
        from impl._cuda.stack import CuDecoderStack

        cfg = TransformerConfig(n_layers=1, embed_dim=64, n_heads=4, n_experts=2, top_k=2, expert_dim=128, rope_dim=0)
        stack = CuDecoderStack(cfg)
        inp = torch.randn(1, 4, 64, device="cuda")
        out = stack.forward(inp)
        assert out.shape == (1, 4, 64)
        assert not torch.isnan(out).any(), "Single-layer output contains NaN"

    def test_multi_layer(self):
        """A 4-layer stack chains all layers correctly."""
        from impl._cuda.stack import CuDecoderStack

        cfg = TransformerConfig(n_layers=4, embed_dim=128, n_heads=8, n_experts=4, top_k=2, expert_dim=256, rope_dim=16)
        stack = CuDecoderStack(cfg)
        inp = torch.randn(2, 16, 128, device="cuda")
        out = stack.forward(inp)
        assert out.shape == inp.shape
        assert not torch.isnan(out).any(), "4-layer output contains NaN"

    def test_no_nan_with_rope(self):
        """Forward with RoPE produces no NaN or Inf values."""
        from impl._cuda.stack import CuDecoderStack

        cfg = TransformerConfig(n_layers=2, embed_dim=128, n_heads=4, n_experts=4, top_k=2, expert_dim=256, rope_dim=32)
        stack = CuDecoderStack(cfg)
        B, S, D = 2, 16, 128
        inp = torch.randn(B, S, D, device="cuda")
        positions = torch.arange(S, device="cuda")
        out = stack.forward(inp, positions=positions)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_batch(self):
        """Forward with larger batch size works correctly."""
        from impl._cuda.stack import CuDecoderStack

        cfg = TransformerConfig(n_layers=2, embed_dim=128, n_heads=4, n_experts=4, top_k=2, expert_dim=256, rope_dim=32)
        stack = CuDecoderStack(cfg)
        inp = torch.randn(8, 32, 128, device="cuda")
        out = stack.forward(inp)
        assert out.shape == inp.shape
        assert not torch.isnan(out).any()


class TestDecoderStackGradients:
    """Gradient flow through stacked blocks."""

    def test_gradient_flow(self, decoder_stack, sample_input):
        """Gradients flow through all stacked layers (input gradient)."""
        out = sample_input.clone()
        out.requires_grad = True
        result = decoder_stack.forward(out)
        loss = result.sum()
        loss.backward()
        assert out.grad is not None
        assert not torch.isnan(out.grad).any()
        assert not torch.isinf(out.grad).any()

    def test_gradient_no_nan_multi_layers(self):
        """4-layer stack produces valid input gradients (no NaN/Inf)."""
        from impl._cuda.stack import CuDecoderStack

        cfg = TransformerConfig(n_layers=4, embed_dim=128, n_heads=8, n_experts=4, top_k=2, expert_dim=256, rope_dim=16)
        stack = CuDecoderStack(cfg)
        inp = torch.randn(2, 16, 128, device="cuda", requires_grad=True)
        out = stack.forward(inp)
        loss = out.sum()
        loss.backward()
        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()
        assert not torch.isinf(inp.grad).any()

    def test_layernorm_gradients(self, decoder_stack, sample_input):
        """Layernorm gamma tensors receive valid gradients."""
        out = sample_input.clone()
        out.requires_grad = True
        result = decoder_stack.forward(out)
        loss = result.sum()
        loss.backward()
        for block in decoder_stack.blocks:
            assert block.input_layernorm_gamma.grad is not None
            assert not torch.isnan(block.input_layernorm_gamma.grad).any()
            assert block.post_attention_layernorm_gamma.grad is not None
            assert not torch.isnan(block.post_attention_layernorm_gamma.grad).any()
