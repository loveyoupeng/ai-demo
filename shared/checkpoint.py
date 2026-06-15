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

# Keys in checkpoint that are NOT actual parameters.
# These are metadata that backends store in npz for self-documentation.
_CHECKPOINT_META_KEYS: set[str] = {
    "embed.weight",
    "lm_head.weight",
    "lm_head.bias",
    "transformer_layernorm.gamma",
    "transformer_layernorm.bias",
}


def _param_key_for_npz(key: str) -> str:
    """Normalize parameter name keys between backends.

    TransformerBlock naming:
        numpy:    blocks.0.attn.q.weight    ->  layers.0.attn.q
        torch:    blocks.0.attn.q.weight    ->  layers.0.attn.q

    The checkpoint format uses `layers.N` for block index, matching
    the design doc (Phase F cross-backend load).  This normalizer
    ensures NumPy and PyTorch produce identical `.npz` keys.

    Args:
        key: A parameter name string (backend-specific).

    Returns:
        Normalized key suitable for np.savez.
    """
    # Normalize blocks.N to layers.N
    if key.startswith("blocks."):
        parts = key.split(".")
        if len(parts) >= 3:
            parts[0] = "layers"
        return ".".join(parts)

    # Global parameters: embed.weight, lm_head.weight, lm_head.bias
    return key


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
    **params: Any,  # ndarray | torch.Tensor | ...
) -> None:
    """Save model parameters to disk as `.npz` file.

    Directory structure:
        checkpoint_dir/  (root for this checkpoint)
        ├── config.json  (Hyperparameters — written if config provided)
        └── model.npz    (NumPy binary for all parameter arrays)

    All tensors/arrays must be convertible with `.detach().cpu().numpy()`.
    This works for both NumPy arrays and PyTorch Tensors.

    Args:
        checkpoint_dir: Path to checkpoint directory.
        config: Optional transformer configuration.
        **params: Named parameter arrays to save. Keys become `.npz` keys.

    Raises:
        OSError: If directory cannot be created or file cannot be written.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.npz"

    save_config(config, config_path) if config else None

    # Convert numpy arrays to standard Python primitives for np.savez
    arrays: dict[str, Any] = {}

    for name, tensor in params.items():
        key = _param_key_for_npz(name)

        array = tensor if isinstance(tensor, np.ndarray) else tensor.detach().cpu().numpy()

        arrays[key] = array

    # Save the model checkpoint
    np.savez(str(model_path), **arrays)


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
    return params, config
