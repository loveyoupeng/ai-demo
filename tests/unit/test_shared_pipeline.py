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
            save_checkpoint(tmp, cfg, params={"model.embed_tokens": np.zeros((cfg.vocab_size, cfg.embed_dim))})
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
        from shared.registry import ParameterRegistry

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            # A complete checkpoint: every registry key with its expected shape.
            reg = ParameterRegistry(cfg)
            params_in = {k: np.random.randn(*s) for k, s in reg.expected_shapes().items()}
            save_checkpoint(path, cfg, params=params_in)

            # Stale (pre-migration) keys must be rejected by the registry.
            on_disk = dict(np.load(path / "model.npz"))
            stale = dict(on_disk)
            stale["blocks.0.attn.q.weight"] = np.zeros(1)
            try:
                reg.validate(stale)
                raise AssertionError("stale checkpoint should have been rejected")
            except ValueError:
                pass

            params, loaded_cfg = load_checkpoint(path)

            assert loaded_cfg is not None
            assert loaded_cfg.vocab_size == 64
            assert loaded_cfg.embed_dim == 16
            assert len(params) == len(reg.keys())
            for k, v in params_in.items():
                assert k in params
                assert np.allclose(params[k], v), f"roundtrip mismatch for {k}"


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
        from shared.registry import ParameterRegistry

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            reg = ParameterRegistry(cfg)
            params_in = {k: np.ones(s) for k, s in reg.expected_shapes().items()}
            save_checkpoint(path, cfg, params=params_in)
            params, loaded_cfg = load_checkpoint(path)

            assert loaded_cfg is not None
            assert loaded_cfg.vocab_size == 32
            assert loaded_cfg.embed_dim == 8
            assert loaded_cfg.n_layers == 1
            assert set(params) == set(reg.keys())

    def test_save_checkpoint_without_config(self):
        """Without config.json the load skips registry validation (no config to derive keys from)."""
        from shared.checkpoint import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            params = {"model.embed_tokens": np.zeros((10, 5))}
            save_checkpoint(path, params=params)
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
            save_checkpoint(path, cfg, params={"model.embed_tokens": np.zeros((5, 5))})
            assert path.exists()


class TestCheckpointMissingFile:
    """Test error when loading non-existent checkpoint."""

    def test_load_missing_raises(self):
        from shared.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_such_dir"
            with pytest.raises(FileNotFoundError):
                load_checkpoint(path)
