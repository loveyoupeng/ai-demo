"""Checkpoint save/load for saving transformer parameters to disk.

Supports checkpoint format compatible across backends:
- config.json: Hyperparameters in JSON
- model.npz: NumPy binary arrays for every parameter

All backends can save to this format; NumPy uses `np.savez` directly,
PyTorch converts tensors to numpy via `.detach().cpu().numpy()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from shared.config import TransformerConfig
from shared.registry import ParameterRegistry


def save_config(config: TransformerConfig, config_path: Path) -> None:
    """Save TransformerConfig to a JSON file.

    Args:
        config: The configuration to serialize.
        config_path: Path to write the JSON file.

    Raises:
        OSError: If the file cannot be written.
    """
    # Write the JSON representation of the config
    data: dict[str, Any] = config.to_dict()
    data["model_type"] = "decoder_transformer"

    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(config_path: Path) -> TransformerConfig:
    """Load TransformerConfig from a JSON file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        TransformerConfig instance with all fields restored.

    Raises:
        FileNotFoundError: If config file does not exist.
        KeyError: If a required field is missing from JSON.
    """
    with open(config_path) as f:
        data = json.load(f)
    return TransformerConfig.from_dict(data)


def save_checkpoint(
    checkpoint_dir: str | Path,
    config: TransformerConfig | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint: config.json + a flat ``model.npz`` of all parameters.

    Args:
        checkpoint_dir: Directory to create/write (created if missing).
        config: The TransformerConfig (optional — a config-less npz is a
            weights-only checkpoint; loading it skips registry validation).
        params: Mapping of shared ``Keys`` names -> numpy arrays, i.e. the
            output of ``Model.get_all_parameters()``.

    The parameter keys are exactly the registry keys derived from ``config``
    (shared.registry.ParameterRegistry) — the single checkpoint format.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    if config is not None:
        save_config(config, path / "config.json")
    # np.savez needs numpy arrays; any torch tensors come from get_all_parameters().
    arrays = {
        name: (t if isinstance(t, np.ndarray) else t.detach().cpu().numpy()) for name, t in (params or {}).items()
    }
    np.savez(str(path / "model.npz"), **arrays)  # pyright: ignore[reportArgumentType]


def load_checkpoint(checkpoint_dir: str | Path) -> tuple[dict[str, Any], TransformerConfig | None]:
    """Load model parameters from disk as `.npz` file.

    Args:
        checkpoint_dir: Path to checkpoint directory.

    Returns:
        Tuple of (params dict, config or None).
        Keys are the normalized names matching the checkpoint format.
        If config.json exists, also returns the config object.

    Raises:
        FileNotFoundError: If model.npz does not exist.
    """
    checkpoint_dir = Path(checkpoint_dir)
    config = None
    config_path = checkpoint_dir / "config.json"

    if config_path.exists():
        config = load_config(config_path)

    model_path = checkpoint_dir / "model.npz"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_dir}/model.npz. "
            f"Run `uv run src/train.py train` first to produce training checkpoints."
        )

    # Load all arrays from npz
    params = dict(np.load(str(model_path)))

    # Validate against the registry so stale (pre-migration) checkpoints
    # fail fast with a clear message instead of half-loading.
    if config is not None:
        ParameterRegistry(config).validate(params)
    return params, config
