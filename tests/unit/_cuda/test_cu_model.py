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


# ── TestCuModelInit ────────────────────────────────────────────────────────────


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
