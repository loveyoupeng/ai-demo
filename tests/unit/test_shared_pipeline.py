"""Tests for shared pipeline — config + checkpoint save/load + roundtrip.

This module exercises the full data flow:
  config → save → load → config roundtrip → parameter roundtrip

No fixtures or network deps — minimal, fast tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestConfigSave:
    """Test TransformerConfig save → file exists and is valid JSON."""

    def test_save_config_creates_file(self):
        from shared.checkpoint import save_config
        from shared.config import TransformerConfig

        cfg = TransformerConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            assert path.exists()


class TestConfigLoad:
    """Test config load reconstructs config correctly."""

    def test_load_config_returns_transformer_config(self):
        from shared.checkpoint import load_config, save_config
        from shared.config import TransformerConfig

        cfg = TransformerConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            loaded = load_config(path)
            assert isinstance(loaded, TransformerConfig)


class TestCheckpointSaveLoadParams:
    """Test save_checkpoint / load_checkpoint with dummy ndarrays."""

    def test_save_checkpoint_creates_dir(self):
        from shared.checkpoint import save_checkpoint
        from shared.config import TransformerConfig

        cfg = TransformerConfig()
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(tmp, cfg, embed=np.zeros((10, 5)))
            assert (Path(tmp) / "model.npz").exists()


class TestConfigRoundTrip:
    """Test config save → load preserves all values."""

    def test_config_roundtrip_preserves_vocab(self):
        from shared.checkpoint import load_config, save_config
        from shared.config import TransformerConfig

        cfg = TransformerConfig(vocab_size=1024, embed_dim=256, n_layers=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            loaded = load_config(path)
            assert loaded.vocab_size == 1024
            assert loaded.embed_dim == 256
            assert loaded.n_layers == 4

    def test_config_roundtrip_preserves_seed(self):
        from shared.checkpoint import load_config, save_config
        from shared.config import TransformerConfig

        cfg = TransformerConfig(seed=123)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            loaded = load_config(path)
            assert loaded.seed == 123

    def test_config_roundtrip_preserves_n_heads(self):
        from shared.checkpoint import load_config, save_config
        from shared.config import TransformerConfig

        cfg = TransformerConfig(n_heads=16, n_groups=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            loaded = load_config(path)
            assert loaded.n_heads == 16
            assert loaded.n_groups == 8


class TestFullPipeline:
    """Test the full pipeline: config → save → load → verify params."""

    def test_pipeline_save_load_config_with_params(self):
        from shared.checkpoint import load_checkpoint, save_checkpoint
        from shared.config import TransformerConfig

        cfg = TransformerConfig(vocab_size=64, embed_dim=16, n_layers=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            embed = np.random.randn(64, 16)
            lm_head_w = np.random.randn(64, 16)
            lm_head_b = np.random.randn(1, 64)

            save_checkpoint(
                path,
                cfg,
                embed=embed,
                lm_head__weight=lm_head_w,
                lm_head__bias=lm_head_b,
            )

            params, loaded_cfg = load_checkpoint(path)

            assert loaded_cfg is not None
            assert loaded_cfg.vocab_size == 64
            assert loaded_cfg.embed_dim == 16
            assert len(params) == 3
            assert "embed" in params
            assert "lm_head__weight" in params
            assert "lm_head__bias" in params
            assert np.allclose(params["embed"], embed)
            assert np.allclose(params["lm_head__weight"], lm_head_w)
            assert np.allclose(params["lm_head__bias"], lm_head_b)


class TestMinimalConfig:
    """Test with minimum viable config — everything at minimum values."""

    def test_minimal_config_roundtrip(self):
        from shared.checkpoint import load_checkpoint, save_checkpoint
        from shared.config import TransformerConfig

        cfg = TransformerConfig(
            vocab_size=32,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_groups=1,
            n_experts=1,
            top_k=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            save_checkpoint(path, cfg, weight=np.ones((8, 8)))
            params, loaded_cfg = load_checkpoint(path)

            assert loaded_cfg is not None
            assert loaded_cfg.vocab_size == 32
            assert loaded_cfg.embed_dim == 8
            assert loaded_cfg.n_layers == 1
            assert "weight" in params


class TestCheckpointWithoutConfig:
    """Test checkpoint save/load when config is not provided."""

    def test_save_checkpoint_without_config(self):
        from shared.checkpoint import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            save_checkpoint(path, weight=np.zeros((10, 5)))
            params, loaded_cfg = load_checkpoint(path)
            assert params is not None
            assert loaded_cfg is None


class TestCheckpointDirectoryCreation:
    """Test that checkpoint dir is created automatically."""

    def test_nested_dir_created(self):
        from shared.checkpoint import save_checkpoint
        from shared.config import TransformerConfig

        cfg = TransformerConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b" / "c"
            save_checkpoint(path, cfg, weight=np.zeros((5, 5)))
            assert path.exists()


class TestCheckpointMissingFile:
    """Test error when loading non-existent checkpoint."""

    def test_load_missing_raises(self):
        from shared.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_such_dir"
            with pytest.raises(FileNotFoundError):
                load_checkpoint(path)
