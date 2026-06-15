"""Tests for shared.checkpoint — save/load .npz checkpoints.

No fixtures or network deps — pure JSON + numpy I/O.
Test one assertion at a time.
"""

import json
import tempfile
from pathlib import Path


class TestSaveConfig:
    """Test save_config writes valid JSON."""

    def test_save_config_creates_file(self):
        from shared.checkpoint import TransformerConfig, save_config

        cfg = TransformerConfig(vocab_size=128, embed_dim=64, n_layers=2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            assert path.exists()

    def test_save_config_json_parsable(self):
        from shared.checkpoint import TransformerConfig, save_config

        cfg = TransformerConfig(vocab_size=128, embed_dim=64, n_layers=2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert data["model_type"] == "decoder_transformer"

    def test_save_config_contains_vocab_size(self):
        from shared.checkpoint import TransformerConfig, save_config

        cfg = TransformerConfig(vocab_size=1024, embed_dim=256, n_layers=4)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            with open(path) as f:
                data = json.load(f)
            assert data["vocab_size"] == 1024
            assert data["embed_dim"] == 256
            assert data["n_layers"] == 4

    def test_save_config_excludes_derived(self):
        """Derived fields (head_dim, k_dim, v_dim) should not be in JSON."""
        from shared.checkpoint import TransformerConfig, save_config

        cfg = TransformerConfig(vocab_size=256, embed_dim=128, n_layers=2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            with open(path) as f:
                data = json.load(f)
            assert "head_dim" not in data
            assert "k_dim" not in data
            assert "v_dim" not in data

    def test_save_config_includes_seed(self):
        from shared.checkpoint import TransformerConfig, save_config

        cfg = TransformerConfig(vocab_size=256, embed_dim=128, n_layers=2, seed=123)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            with open(path) as f:
                data = json.load(f)
            assert data["seed"] == 123

    def test_save_config_includes_default_model_type(self):
        """Default model_type must be 'decoder_transformer'."""
        from shared.checkpoint import TransformerConfig, save_config

        cfg = TransformerConfig(vocab_size=256, embed_dim=128, n_layers=2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            with open(path) as f:
                data = json.load(f)
            assert data["model_type"] == "decoder_transformer"


class TestLoadConfig:
    """Test load_config reads JSON and reconstructs config."""

    def test_load_config_returns_correct_type(self):
        from shared.checkpoint import TransformerConfig, load_config, save_config

        cfg = TransformerConfig(vocab_size=128, embed_dim=64, n_layers=2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            loaded = load_config(path)
            assert isinstance(loaded, TransformerConfig)

    def test_load_config_matches_original(self):
        from shared.checkpoint import TransformerConfig, load_config, save_config

        cfg = TransformerConfig(
            vocab_size=512,
            embed_dim=128,
            n_layers=4,
            n_heads=8,
            n_groups=4,
            seed=99,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            loaded = load_config(path)
            assert loaded.vocab_size == cfg.vocab_size
            assert loaded.embed_dim == cfg.embed_dim
            assert loaded.n_layers == cfg.n_layers
            assert loaded.n_heads == cfg.n_heads
            assert loaded.n_groups == cfg.n_groups
            assert loaded.seed == cfg.seed

    def test_load_config_derived_fields_computed(self):
        """Derived fields (head_dim, k_dim, v_dim) should be computed."""
        from shared.checkpoint import TransformerConfig, load_config, save_config

        cfg = TransformerConfig(vocab_size=256, embed_dim=256, n_layers=1, n_heads=4, n_groups=4)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(cfg, path)
            loaded = load_config(path)
            assert loaded.head_dim == 64
            assert loaded.k_dim == 256  # n_groups=4 * head_dim=64
            assert loaded.v_dim == 256


class TestLoadSaveRoundTrip:
    """Test save→load roundtrip preserves all fields."""

    def test_roundtrip_small_config(self):
        from shared.checkpoint import TransformerConfig, load_config, save_config

        original = TransformerConfig(vocab_size=100, embed_dim=50, n_layers=1, seed=7)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(original, path)
            loaded = load_config(path)
            assert loaded.vocab_size == original.vocab_size
            assert loaded.head_dim == original.head_dim
            assert loaded.seed == original.seed

    def test_roundtrip_full_config(self):
        from shared.checkpoint import TransformerConfig, load_config, save_config

        original = TransformerConfig(
            vocab_size=4096,
            context_length=256,
            embed_dim=512,
            n_layers=8,
            n_heads=8,
            n_groups=8,
            rope_dim=0,
            n_experts=4,
            top_k=2,
            expert_dim=0,
            max_length=2048,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(original, path)
            loaded = load_config(path)
            for attr in [
                "vocab_size",
                "context_length",
                "embed_dim",
                "n_layers",
                "n_heads",
                "n_groups",
                "rope_dim",
                "n_experts",
                "top_k",
                "expert_dim",
                "max_length",
                "seed",
            ]:
                assert getattr(loaded, attr) == getattr(original, attr)
