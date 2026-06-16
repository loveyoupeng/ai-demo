"""Unified config reader with config-source tracking.

Supports three configuration sources with priority order:
    CLI args > Environment vars > Config file > Defaults

Each value in the merged result is annotated with its source via `ConfigSource`.

Example config file (JSON):
    {
        "training": {"epochs": 5, "lr": 0.001},
        "model": {"vocab_size": 256, "n_layers": 4},
        "output": {"save_dir": "resource/models/"}
    }

Section keys (e.g. "training", "model", "output") are flattened to top-level keys.
CLI args are filtered: action='store_true' / 'store_false' are excluded.

Phase keys in DEFAULTS:
    A       — NumPy Phase A (embedding + forward)
    C.A     — PyTorch Phase C.A (TorchModel)
    C.B     — PyTorch Phase C.B (tokenizer + checkpoint)
    C.C     — PyTorch Phase C.C (training)
    C+      — Phase C+ (E2E training scripts defaults)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = ["DEFAULTS", "ConfigSource", "load_config", "parse_config_file", "parse_cli_to_config"]


class ConfigSource(str, Enum):
    """Tracks where each config value came from."""

    DEFAULT = "default"
    FILE = "file"
    ENV = "env"
    CLI = "cli"


@dataclass(frozen=True)
class TrackedValue:
    """A config value with provenance tracking."""

    key: str
    value: Any
    source: ConfigSource

    @classmethod
    def from_default(cls, key: str, value: Any) -> TrackedValue:
        return cls(key, value, ConfigSource.DEFAULT)

    @classmethod
    def from_file(cls, key: str, value: Any) -> TrackedValue:
        return cls(key, value, ConfigSource.FILE)

    @classmethod
    def from_env(cls, key: str, value: Any) -> TrackedValue:
        return cls(key, value, ConfigSource.ENV)

    @classmethod
    def from_cli(cls, key: str, value: Any) -> TrackedValue:
        return cls(key, value, ConfigSource.CLI)

    def __gt__(self, other: object) -> bool:
        return True

    def __lt__(self, other: object) -> bool:
        return False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, dict[str, Any]] = {
    "A": {
        "vocab_size": 50,
        "context_length": 128,
        "embed_dim": 64,
        "n_layers": 6,
        "n_heads": 8,
        "n_experts": 4,
        "ff_dim": 128,
        "k": 2,
        "rope_dim": 8,
        "seed": 42,
    },
    "C.A": {
        "vocab_size": 4096,
        "context_length": 256,
        "embed_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "n_groups": 8,
        "n_experts": 4,
        "top_k": 2,
        "expert_dim": 0,
        "rope_dim": 0,
        "max_length": 2048,
        "quant_type": "none",
        "kvcache_type": "naive",
        "seed": 42,
        "ff_dim": 0,
    },
    "C.B": {
        "tokenizer_path": "shared/vocab.txt",
        "dataset_dir": "resource/tinystories/",
        "cache_dir": "resource/tinystories_cache/",
    },
    "C.C": {
        "dataset_path": "resource/tinystories/",
        "epochs": 5,
        "batch_size": 64,
        "lr": 0.001,
        "seed": 42,
        "save_steps": 50,
        "eval_steps": 25,
        "max_steps": None,
    },
    "C+": {
        "backend": "torch",
        "vocab_size": 256,
        "context_length": 128,
        "embed_dim": 256,
        "n_layers": 4,
        "n_heads": 8,
        "n_groups": 8,
        "n_experts": 4,
        "top_k": 2,
        "rope_dim": 0,
        "max_length": 512,
        "quant_type": "none",
        "kvcache_type": "naive",
        "dataset_path": "resource/tinystories/",
        "epochs": 5,
        "batch_size": 64,
        "lr": 0.001,
        "seed": 42,
        "save_steps": 100,
        "eval_steps": 50,
        "save_dir": "resource/models/",
    },
}


def _merge_nested(d: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested dict (like {"training": {"lr": 0.01}} -> {"lr": 0.01}).

    When the same key appears in multiple sections, the value from the
    LAST section wins (since the dict preserves insertion order).

    Args:
        d: A dict where values may be nested dicts of config key-value pairs.

    Returns:
        Flat dict: all nested values at the top level. Nested dicts that
        contain non-dict values (e.g. {"lr": 0.01}) are flattened.
        Non-dict values in top-level are kept as-is.
    """
    result: dict[str, Any] = {}

    for _, section_value in d.items():
        if isinstance(section_value, dict):
            for k, v in section_value.items():
                result[k] = v
        elif section_value is not None:
            result["_top"] = section_value

    return result


def parse_config_file(path: str | Path) -> dict[str, Any]:
    """Parse a JSON config file and return flattened key-value dict.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Flat dictionary: {"epochs": 5, "lr": 0.001, ...}

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return _merge_nested(data)


def parse_cli_to_config(args: Any, phase: str) -> dict[str, Any]:
    """Parse argparse.Namespace to a flat config dict, excluding boolean actions.

    Args:
        args: argparse.Namespace (the parsed command-line arguments).
        phase: Phase identifier to include in the output as a "phase" key.

    Returns:
        Flat dict with non-boolean CLI args, including a "phase" key.
    """
    result: dict[str, Any] = {"phase": phase}
    for key, value in vars(args).items():
        if isinstance(value, bool):
            # Action='store_true' / 'store_false' args are excluded
            continue
        if isinstance(value, str) and (value.startswith("--") or value.startswith("-")):
            # Skip values that look like flag strings (e.g. from misconfigured argparse)
            continue
        result[key] = value
    return result


def _get_env_value(prefix: str, key: str) -> str | None:
    """Look up environment variables for a given config key.

    Tries both direct ({prefix}_KEY) and subsystem-prefixed
    ({prefix}_TRAINING_KEY, {prefix}_MODEL_KEY) forms.

    Args:
        prefix: Environment variable prefix (e.g. "NPY" or "TORCH").
        key: Config key (e.g. "n_layers").

    Returns:
        The environment variable value, or None if not set.
    """
    # Try key directly
    env_key = f"{prefix}_{key.upper()}"
    val = os.environ.get(env_key)
    if val is not None:
        return val

    # Try with subsystem prefix
    for subsystem in ("training", "model", "output"):
        env_key = f"{prefix}_{subsystem.upper()}_{key.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            return val
    return None


def _convert_type(value: str, default_type: type) -> Any:
    """Convert a string value to the appropriate type based on the default.

    Args:
        value: The string value from the environment.
        default_type: The Python type to convert to.

    Returns:
        Converted value.
    """
    if default_type is int:
        return int(value)
    if default_type is float:
        return float(value)
    if default_type is bool:
        return value.lower() in ("true", "1", "yes")
    return value


def _apply_file(merged: dict[str, TrackedValue], config_file: str | None) -> None:
    """Merge a config file into the merged dict.

    Args:
        merged: The accumulated config dict.
        config_file: Path to the config file (None to skip).
    """
    if config_file is None:
        return
    try:
        file_data = parse_config_file(config_file)
        for key, value in file_data.items():
            merged[key] = TrackedValue.from_file(key, value)
    except FileNotFoundError:
        pass


def _apply_env(merged: dict[str, TrackedValue], env_prefix: str | None) -> None:
    """Merge environment variables into the merged dict.

    First checks for env overrides of existing keys, then checks for
    a raw JSON override in {prefix}_CONFIG.

    Args:
        merged: The accumulated config dict.
        env_prefix: Environment variable prefix (None to skip).
    """
    if env_prefix is None:
        return

    all_keys = list(merged.keys())
    for key in all_keys:
        val = _get_env_value(env_prefix, key)
        if val is not None:
            default_type = type(merged[key].value)
            converted = _convert_type(val, default_type)
            merged[key] = TrackedValue.from_env(key, converted)

    raw_env = os.environ.get(f"{env_prefix}_CONFIG")
    if raw_env is not None:
        try:
            raw_data = json.loads(raw_env)
            for k, v in _merge_nested(raw_data).items():
                merged[k] = TrackedValue.from_env(k, v)
        except (json.JSONDecodeError, ValueError):
            pass


def load_config(
    args: Any,
    phase: str,
    *,
    config_file: str | None = None,
    env_prefix: str | None = None,
    tmpdir: str | None = None,
) -> dict[str, TrackedValue]:
    """Load and merge configuration from all sources.

    Priority (high → low): CLI > env > file > defaults.

    Args:
        args: argparse.Namespace from parsed command-line arguments.
        phase: Phase identifier.
        config_file: Path to a JSON config file.
        env_prefix: Environment variable prefix (e.g. "NPY" or "TORCH").
        tmpdir: Temporary directory for intermediate files.

    Returns:
        Dict mapping each key to its TrackedValue (value + source).
    """
    # 1) Defaults (lowest priority)
    defaults_data = DEFAULTS.get(phase, {})
    merged: dict[str, TrackedValue] = {}
    for key, value in defaults_data.items():
        merged[key] = TrackedValue.from_default(key, value)

    # 2) Config file
    _apply_file(merged, config_file)

    # 3) Environment variables
    _apply_env(merged, env_prefix)

    # 4) CLI args (highest priority)
    cli_data = parse_cli_to_config(args, phase)
    for key, value in cli_data.items():
        if key == "phase":
            continue
        merged[key] = TrackedValue.from_cli(key, value)

    return merged
