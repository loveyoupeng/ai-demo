"""CUDAModel — full decoder-only transformer: embedding → stack → output (F9).

Tests for CuModel: creation, attributes, forward shape, output dimensions.

Architecture:
    Input:  tokens [B, S] (int64)
    │
    ├→ Embedding table lookup       [B, S, D]
    ├→ DecoderStack (n_layers)     [B, S, D]
    ├→ RMSNorm (final_ln)          [B, S, D]
    ├→ SwiGLU (output)             [B, S, D]
    └→ Linear (output_proj)        [B, S, V]
    │
    Output: logits [B, S, V]

Reference
---------
Vaswani et al. "Attention Is All You Need" (2017)
https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import pytest
import torch

from impl._cuda.model import CUDAModel

# ── Test fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_config():
    """Minimal model config for fast tests."""
    return dict(
        vocab_size=512,
        embed_dim=64,
        n_layers=1,
        n_heads=4,
        n_experts=2,
        ff_dim=128,
        k=2,
        rope_dim=16,
        seed=42,
    )


@pytest.fixture
def base_config():
    """Base config shared by all tests."""
    return dict(
        n_layers=2,
        embed_dim=128,
        n_heads=4,
        n_experts=4,
        ff_dim=256,
        k=2,
        rope_dim=64,
    )


@pytest.fixture
def decoder_stack(base_config):
    """Small CuDecoderStack: 2 layers, 128 dim, 4 heads, 4 experts."""
    from impl._cuda.stack import CuDecoderStack

    return CuDecoderStack(**base_config)


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

    def test_creation_fails_without_stack(self, small_config):
        """Forward should fail if model is not properly initialized."""

        model = CUDAModel(**small_config)
        # Clear weights to simulate incomplete initialization
        saved_weights = model.embedding_weights
        model.embedding_weights = None  # type: ignore[assignment]
        tokens = torch.randint(0, model.vocab_size, (2, 16), device="cuda", dtype=torch.int64)
        with pytest.raises((AttributeError, TypeError)):
            model.forward(tokens)
        model.embedding_weights = saved_weights  # Restore for other tests

    def test_has_vocab_size(self, small_config):
        """Model has the correct vocabulary size."""
        model = CUDAModel(**small_config)
        assert model.vocab_size == small_config["vocab_size"]

    def test_has_embed_dim(self, small_config):
        """Model has the correct embedding dimension."""
        model = CUDAModel(**small_config)
        assert model.embed_dim == small_config["embed_dim"]

    def test_has_n_layers(self, small_config):
        """Model stores n_layers attribute."""
        model = CUDAModel(**small_config)
        assert model.n_layers == small_config["n_layers"]

    def test_has_embedding(self, small_config):
        """Model has embedding_weight attribute."""
        model = CUDAModel(**small_config)
        assert hasattr(model, "embedding_weights")
        assert model.embedding_weights.shape == (small_config["vocab_size"], small_config["embed_dim"])

    def test_has_final_ln(self, small_config):
        """Model has final_ln_gamma attribute."""
        model = CUDAModel(**small_config)
        assert hasattr(model, "final_ln_gamma")
        assert model.final_ln_gamma.shape == (small_config["embed_dim"],)

    def test_has_output_proj(self, small_config):
        """Model has output_proj_weights and output_proj_bias."""
        model = CUDAModel(**small_config)
        assert hasattr(model, "output_proj_weights")
        assert model.output_proj_weights.shape == (small_config["embed_dim"], model.vocab_size)
        assert hasattr(model, "output_proj_bias")
        assert model.output_proj_bias.shape == (model.vocab_size,)


# ================================================================
# SECTION: Model Components — CuDecoderStack
# ================================================================


class TestDecoderStackInit:
    """CuDecoderStack — chained transformer blocks (F8).

    Tests for DecoderStack: basic wiring, forward shape,
    gradients through stacked layers, single-layer and multi-layer parity.

    Architecture:
        x [B, S, D] → block_0 → block_1 → ... → block_{n-1} → out [B, S, D]

        - No position embeddings (RoPE handles positional info inside attention)
        - No final RMSNorm (belongs to the parent model)
        - Post-norm gated residual with MoE

    Reference
    ---------
    Vaswani et al. "Attention Is All You Need" (2017)
    https://arxiv.org/abs/1706.03762
    """

    def test_creation(self, decoder_stack, base_config):
        """A DecoderStack can be created with the specified config."""
        assert decoder_stack.n_layers == base_config["n_layers"]
        assert decoder_stack.embed_dim == base_config["embed_dim"]
        assert decoder_stack.head_dim == base_config["embed_dim"] // base_config["n_heads"]
        assert len(decoder_stack.blocks) == base_config["n_layers"]

    def test_blocks_are_transformer_blocks(self, decoder_stack):
        """Every block in the stack is a CuTransformerBlock instance."""
        from impl._cuda.block import CuTransformerBlock

        for block in decoder_stack.blocks:
            assert isinstance(block, CuTransformerBlock)

    def test_blocks_have_correct_device(self, decoder_stack):
        """All block weights are on CPU (move to CUDA on forward)."""
        for block in decoder_stack.blocks:
            assert block.Wq.device.type == "cpu"
            assert block.Wk.device.type == "cpu"
            assert block.Wv.device.type == "cpu"
            assert block.ln1_gamma.device.type == "cpu"

    def test_rope_disabled(self, base_config):
        """DecoderStack with rope_dim=0 creates blocks without RoPE."""
        cfg = {**base_config, "rope_dim": 0}
        from impl._cuda.stack import CuDecoderStack

        stack = CuDecoderStack(**cfg)
        for block in stack.blocks:
            assert block.rope_dim == 0

    def test_head_dim_divisibility(self, base_config):
        """Head dimension is embed_dim // n_heads."""
        from impl._cuda.stack import CuDecoderStack

        stack = CuDecoderStack(**base_config)
        for block in stack.blocks:
            assert block.head_dim == base_config["embed_dim"] // base_config["n_heads"]


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

        cfg = dict(
            n_layers=1,
            embed_dim=64,
            n_heads=4,
            n_experts=2,
            ff_dim=128,
            k=2,
            rope_dim=0,
        )
        stack = CuDecoderStack(**cfg)
        inp = torch.randn(1, 4, 64, device="cuda")
        out = stack.forward(inp)
        assert out.shape == (1, 4, 64)
        assert not torch.isnan(out).any(), "Single-layer output contains NaN"

    def test_multi_layer(self):
        """A 4-layer stack chains all layers correctly."""
        from impl._cuda.stack import CuDecoderStack

        cfg = dict(
            n_layers=4,
            embed_dim=128,
            n_heads=8,
            n_experts=4,
            ff_dim=256,
            k=2,
            rope_dim=64,
        )
        stack = CuDecoderStack(**cfg)
        inp = torch.randn(2, 16, 128, device="cuda")
        out = stack.forward(inp)
        assert out.shape == inp.shape
        assert not torch.isnan(out).any(), "4-layer output contains NaN"

    def test_no_nan_with_rope(self):
        """Forward with RoPE produces no NaN or Inf values."""
        from impl._cuda.stack import CuDecoderStack

        cfg = dict(
            n_layers=2,
            embed_dim=128,
            n_heads=4,
            n_experts=4,
            ff_dim=256,
            k=2,
            rope_dim=64,
        )
        stack = CuDecoderStack(**cfg)
        B, S, D = 2, 16, 128
        inp = torch.randn(B, S, D, device="cuda")
        positions = torch.arange(S, device="cuda")
        out = stack.forward(inp, positions=positions)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_batch(self):
        """Forward with larger batch size works correctly."""
        from impl._cuda.stack import CuDecoderStack

        cfg = dict(
            n_layers=2,
            embed_dim=128,
            n_heads=4,
            n_experts=4,
            ff_dim=256,
            k=2,
            rope_dim=64,
        )
        stack = CuDecoderStack(**cfg)
        inp = torch.randn(8, 32, 128, device="cuda")
        out = stack.forward(inp)
        assert out.shape == inp.shape
        assert not torch.isnan(out).any()


class TestDecoderStackGradients:
    """Gradient flow through stacked blocks."""

    def test_gradient_flow(self, decoder_stack, sample_input):
        """Gradients flow through all stacked layers."""
        out = sample_input.clone()
        out.requires_grad = True
        result = decoder_stack.forward(out)
        loss = result.sum()
        loss.backward()
        assert out.grad is not None
        assert not torch.isnan(out.grad).any()
        assert not torch.isinf(out.grad).any()

    def test_gradient_no_nan_multi_layers(self):
        """4-layer stack produces valid gradients (no NaN/Inf)."""
        from impl._cuda.stack import CuDecoderStack

        cfg = dict(
            n_layers=4,
            embed_dim=128,
            n_heads=8,
            n_experts=4,
            ff_dim=256,
            k=2,
            rope_dim=64,
        )
        stack = CuDecoderStack(**cfg)
        inp = torch.randn(2, 16, 128, device="cuda", requires_grad=True)
        out = stack.forward(inp)
        loss = out.sum()
        loss.backward()
        for block in stack.blocks:
            assert block.Wq.grad is not None
            assert not torch.isnan(block.Wq.grad).any()
            assert not torch.isinf(block.Wq.grad).any()
            assert block.ln1_gamma.grad is not None
            assert not torch.isnan(block.ln1_gamma.grad).any()
            assert not torch.isinf(block.ln1_gamma.grad).any()

    def test_gated_gradients(self, decoder_stack, sample_input):
        """Gate parameters (gate1, gate2) produce valid gradients."""
        out = sample_input.clone()
        out.requires_grad = True
        result = decoder_stack.forward(out)
        loss = result.sum()
        loss.backward()
        for block in decoder_stack.blocks:
            assert block.gate1.grad is not None
            assert not torch.isnan(block.gate1.grad).any()
            assert block.gate2.grad is not None
            assert not torch.isnan(block.gate2.grad).any()