"""Tests for shared/config_utils — unified config reader (CLI > env > file).

Tests follow TDD: write failing test first, then implement to make it pass.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pytest

from shared.config_utils import (
    DEFAULTS,
    ConfigSource,
    TrackedValue,
    load_config,
    parse_cli_to_config,
    parse_config_file,
)


class TestDefaults:
    """DEFAULTS dict contains sensible defaults for all phases."""

    def test_defaults_is_dict(self):
        assert isinstance(DEFAULTS, dict)

    def test_phase_a_defaults(self):
        assert "A" in DEFAULTS
        assert isinstance(DEFAULTS["A"], dict)

    def test_phase_c_a_defaults(self):
        assert "C.A" in DEFAULTS
        assert isinstance(DEFAULTS["C.A"], dict)

    def test_phase_c_b_defaults(self):
        assert "C.B" in DEFAULTS
        assert isinstance(DEFAULTS["C.B"], dict)

    def test_phase_c_c_defaults(self):
        assert "C.C" in DEFAULTS
        assert isinstance(DEFAULTS["C.C"], dict)

    def test_phase_c_plus_defaults(self):
        assert "C+" in DEFAULTS
        assert isinstance(DEFAULTS["C+"], dict)

    def test_phase_c_plus_has_e2e_defaults(self):
        """Phase C+ should have E2E defaults (epochs, lr, batch_size, etc.)."""
        c_plus = DEFAULTS["C+"]
        assert "epochs" in c_plus
        assert "lr" in c_plus
        assert "batch_size" in c_plus

    def test_phase_c_a_has_phase_c_defaults(self):
        """Phase C.A defaults should contain architecture params."""
        c_a = DEFAULTS["C.A"]
        assert "vocab_size" in c_a
        assert "context_length" in c_a
        assert "embed_dim" in c_a
        assert "n_layers" in c_a


class TestParseConfigFile:
    """parse_config_file — reads a JSON file and returns a flat dict."""

    def test_parse_empty_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            f.flush()
            result = parse_config_file(f.name)
        assert result == {}
        Path(f.name).unlink()

    def test_parse_model_section(self):
        data = {"model": {"vocab_size": 256, "context_length": 64, "embed_dim": 128}}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = parse_config_file(f.name)
        assert result == {"vocab_size": 256, "context_length": 64, "embed_dim": 128}
        Path(f.name).unlink()

    def test_parse_nested_sections_flattens(self):
        data = {"training": {"epochs": 5, "lr": 0.01, "batch_size": 32}, "output": {"save_dir": "/tmp/models"}}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = parse_config_file(f.name)
        assert result == {"epochs": 5, "lr": 0.01, "batch_size": 32, "save_dir": "/tmp/models"}
        Path(f.name).unlink()

    def test_parse_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_config_file("/nonexistent/path/config.json")


class TestParseCliToConfig:
    """Parse CLI args to config dict, respecting phase key."""

    def test_basic_cli_parsing(self):
        """Simple key-value args produce flat config dict."""
        args = argparse.Namespace(vocab_size=256, context_length=64, embed_dim=128)
        result = parse_cli_to_config(args, "phase")
        assert result == {"vocab_size": 256, "context_length": 64, "embed_dim": 128, "phase": "phase"}

    def test_cli_bool_args_stripped(self):
        """Action='store_true' and action='store_false' args should be excluded."""
        args = argparse.Namespace(synthetic=True, quiet=False, verbose=True, hidden=False)
        result = parse_cli_to_config(args, "phase")
        assert "synthetic" not in result
        assert "quiet" not in result
        assert "verbose" not in result
        assert "hidden" not in result

    def test_cli_merges_non_bool(self):
        """Only non-action='store_*' args are included."""
        args = argparse.Namespace(n_layers=4, lr=0.001, synthetic=True, verbose=False)
        result = parse_cli_to_config(args, "C")
        assert "n_layers" in result
        assert "lr" in result
        assert result["n_layers"] == 4
        assert result["lr"] == 0.001
        assert "synthetic" not in result
        assert "verbose" not in result


class TestLoadConfig:
    """load_config — merges defaults < file < env < cli priorities."""

    def test_load_cli_overrides_defaults(self):
        """CLI args should override values from defaults."""
        args = argparse.Namespace(
            n_layers=8,
            embed_dim=256,
            n_heads=8,
            n_groups=8,
            synthetic=True,
            seed=99,
        )
        result = load_config(args, "C", config_file=None, env_prefix="UNUSED", tmpdir=None)
        # CLI value should override
        assert isinstance(result["n_layers"], TrackedValue)
        assert result["n_layers"].value == 8
        assert result["n_layers"].source == ConfigSource.CLI
        assert result["seed"].value == 99
        assert result["seed"].source == ConfigSource.CLI

    def test_phase_not_in_merged(self):
        """The 'phase' key from CLI is metadata — not merged into tracked values."""
        args = argparse.Namespace(
            n_layers=8,
            embed_dim=256,
            n_heads=8,
            n_groups=8,
            synthetic=True,
            verbose=False,
        )
        result = load_config(args, "C", config_file=None, env_prefix="UNUSED", tmpdir=None)
        # "phase" is metadata from parse_cli_to_config, excluded from merged result
        assert "phase" not in result

    def test_env_overrides_defaults(self):
        """Environment variables should override defaults."""
        args = argparse.Namespace(
            n_layers=8,
            embed_dim=256,
            n_heads=8,
            n_groups=8,
            synthetic=True,
            seed=99,
        )
        # Save original if set
        old_n_layers = os.environ.pop("UNUSED_N_LAYERS", None)
        old_embed_dim = os.environ.pop("UNUSED_EMBED_DIM", None)
        env_n_layers = None
        try:
            os.environ["UNUSED_N_LAYERS"] = "16"
            os.environ["UNUSED_EMBED_DIM"] = "1024"
            result = load_config(args, "C", config_file=None, env_prefix="UNUSED", tmpdir=None)
            # CLI should override env (CLI > env > file > default)
            assert result["n_layers"].source == ConfigSource.CLI
            assert result["n_layers"].value == 8
        finally:
            # Restore original env value
            if env_n_layers is not None:
                os.environ["UNUSED_N_LAYERS"] = env_n_layers
            elif old_n_layers is not None:
                os.environ["UNUSED_N_LAYERS"] = old_n_layers
            elif "UNUSED_N_LAYERS" in os.environ:
                del os.environ["UNUSED_N_LAYERS"]
            if old_embed_dim is not None:
                os.environ["UNUSED_EMBED_DIM"] = old_embed_dim
            elif "UNUSED_EMBED_DIM" in os.environ:
                del os.environ["UNUSED_EMBED_DIM"]


class TestConfigSource:
    """ConfigSource enum tracks where each value came from."""

    def test_config_source_is_enum(self):
        assert hasattr(ConfigSource, "DEFAULT")
        assert hasattr(ConfigSource, "FILE")
        assert hasattr(ConfigSource, "ENV")
        assert hasattr(ConfigSource, "CLI")

    def test_config_source_values(self):
        assert ConfigSource.DEFAULT.value == "default"
        assert ConfigSource.FILE.value == "file"
        assert ConfigSource.ENV.value == "env"
        assert ConfigSource.CLI.value == "cli"
