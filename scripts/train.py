#!/usr/bin/env python
"""Unified training script for decoder-only transformer models.

Supports both NumPy and PyTorch backends through a single entry point.

Usage:
    # Train with defaults
    uv run python scripts/train.py

    # Train PyTorch model with custom architecture
    uv run python scripts/train.py --backend torch --n_layers 2 --embed_dim 128 --n_experts 2

    # Train NumPy from config file
    uv run python scripts/train.py --config resource/models/config.json --backend numpy

    # Train with synthetic data (no dataset download)
    uv run python scripts/train.py --synthetic --backend torch --epochs 3

    # Save to custom directory
    uv run python scripts/train.py --backend torch --save_dir /tmp/my_model --seed 42

Environment variables:
    TORCH_N_LAYERS, TORCH_EMBED_DIM, etc. — override defaults for PyTorch
    NPY_N_LAYERS, NPY_EMBED_DIM, etc.      — override defaults for NumPy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402


def create_argparser() -> argparse.ArgumentParser:
    """Create the argument parser for the training script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a decoder-only transformer model.",
        epilog="""
examples:
  # Train with defaults (reads config.json or env vars)
  %(prog)s

  # Train PyTorch model, custom architecture
  %(prog)s --backend torch --n_layers 2 --embed_dim 128 --n_experts 2

  # Train NumPy from config file + environment only
  %(prog)s --config resource/models/config.json --backend numpy

  # Train on synthetic data (no dataset download)
  %(prog)s --synthetic --backend torch --epochs 3

  # Save to custom directory
  %(prog)s --backend torch --save_dir /tmp/my_model --seed 123

  # Environment variable equivalent (no CLI args needed except --backend)
  export TORCH_EMBED_DIM=256; export TORCH_N_LAYERS=4
  %(prog)s --backend torch
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Backend (required) --
    parser.add_argument(
        "-b", "--backend",
        default="torch",
        choices=["numpy", "torch"],
        help="Backend implementation (default: torch)",
    )

    # -- Architecture --
    parser.add_argument("--vocab_size", default=256, type=int, help="Token vocabulary size (default: 256)")
    parser.add_argument("--ctx", "-c", default=128, type=int, help="Context length in tokens (default: 128)")
    parser.add_argument("--embed", "-e", default=256, type=int, help="Embedding dimension (default: 256)")
    parser.add_argument("--layers", "-l", default=4, type=int, help="Number of transformer blocks (default: 4)")
    parser.add_argument("--heads", "-H", default=8, type=int, help="Number of attention heads (default: 8)")
    parser.add_argument("--groups", "-g", default=8, type=int, help="KV query groups (default: 8 = self-attn)")
    parser.add_argument("--rope_dim", default=0, type=int, help="RoPE dimension — 0=full (default: 0)")
    parser.add_argument("--n_experts", default=4, type=int, help="Number of MoE experts (default: 4)")
    parser.add_argument("--top_k", default=2, type=int, help="Number of experts activated per token (default: 2)")
    parser.add_argument("--expert_dim", default=0, type=int, help="FFN inner dim — 0=4x embed (default: 0)")
    parser.add_argument("--max_length", default=512, type=int, help="Max generation length (default: 512)")

    # -- Training --
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size (default: 64)")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for AdamW (default: 0.001)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed (default: 42)")
    parser.add_argument("--save_steps", default=100, type=int, help="Save checkpoint every N steps (default: 100)")
    parser.add_argument("--eval_steps", default=50, type=int, help="Evaluate every N steps (default: 50)")
    parser.add_argument("--dataset_path", default="resource/tinystories/", help="Path to cached dataset")
    parser.add_argument("--synthetic", action="store_true", default=False, help="Use synthetic data instead of TinyStories")
    parser.add_argument("--save_dir", default="resource/models/", help="Directory to save checkpoints")

    return parser


def build_config(args: argparse.Namespace, backend: str) -> dict:
    """Build a configuration dict from CLI args and defaults.

    Args:
        args: Parsed argparse.Namespace.
        backend: Either 'numpy' or 'torch'.

    Returns:
        Flat config dict ready for downstream use.
    """
    config: dict = {}

    # Always include backend
    config["backend"] = backend

    # Architecture
    for key in [
        "vocab_size", "ctx", "context_length", "embed", "embed_dim", "layers",
        "n_layers", "heads", "n_heads", "groups", "n_groups", "rope_dim",
        "n_experts", "top_k", "expert_dim", "max_length", "seed",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Training
    for key in ["epochs", "batch_size", "lr", "save_steps", "eval_steps", "dataset_path", "synthetic", "save_dir"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Normalize common aliases
    if "ctx" in config and "context_length" not in config:
        config["context_length"] = config.pop("ctx")
    if "embed" in config and "embed_dim" not in config:
        config["embed_dim"] = config.pop("embed")
    if "layers" in config and "n_layers" not in config:
        config["n_layers"] = config.pop("layers")
    if "heads" in config and "n_heads" not in config:
        config["n_heads"] = config.pop("heads")
    if "groups" in config and "n_groups" not in config:
        config["n_groups"] = config.pop("groups")

    return config


def build_model(backend: str, config: dict) -> tuple:
    """Build a model instance from config.

    Args:
        backend: 'numpy' or 'torch'.
        config: Configuration dict.

    Returns:
        Tuple of (model_instance, config_for_logging).
    """
    if backend == "numpy":
        from impl._np.model import NumPyModel
        from shared.config import TransformerConfig

        cfg = TransformerConfig(
            vocab_size=config.get("vocab_size", 256),
            context_length=config.get("context_length", 128),
            embed_dim=config.get("embed_dim", 256),
            n_layers=config.get("n_layers", 4),
            n_heads=config.get("n_heads", 8),
            n_groups=config.get("n_groups", 8),
            rope_dim=config.get("rope_dim", 0),
            n_experts=config.get("n_experts", 4),
            top_k=config.get("top_k", 2),
            expert_dim=config.get("expert_dim", 0),
            max_length=config.get("max_length", 512),
            seed=config.get("seed", 42),
        )
        model = NumPyModel(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.embed_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_experts=cfg.n_experts,
            ff_dim=cfg.expert_dim or (cfg.embed_dim * 4),
            k=cfg.top_k,
            rope_dim=cfg.rope_dim,
            seed=cfg.seed,
        )
        return model, cfg

    # PyTorch
    from impl._torch.layers import TorchModel
    from shared.config import TransformerConfig

    cfg = TransformerConfig(
        vocab_size=config.get("vocab_size", 256),
        context_length=config.get("context_length", 128),
        embed_dim=config.get("embed_dim", 256),
        n_layers=config.get("n_layers", 4),
        n_heads=config.get("n_heads", 8),
        n_groups=config.get("n_groups", 8),
        rope_dim=config.get("rope_dim", 0),
        n_experts=config.get("n_experts", 4),
        top_k=config.get("top_k", 2),
        expert_dim=config.get("expert_dim", 0),
        max_length=config.get("max_length", 512),
        seed=config.get("seed", 42),
    )
    model = TorchModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_experts=cfg.n_experts,
        ff_dim=cfg.expert_dim or (cfg.embed_dim * 4),
        k=cfg.top_k,
        rope_dim=cfg.rope_dim,
        seed=cfg.seed,
    )
    return model, cfg


def get_dataset(dataset_path: str, synthetic: bool, vocab_size: int, context_length: int):
    """Get training dataset.

    Args:
        dataset_path: Path to TinyStories dataset.
        synthetic: If True, generate random synthetic data.
        vocab_size: Vocabulary size.
        context_length: Sequence length.

    Returns:
        List of numpy arrays or torch tensors.
    """
    if synthetic:
        # Generate synthetic data
        rng = np.random.default_rng(42)
        num_samples = 1000
        data = []
        for _ in range(num_samples):
            length = rng.integers(10, context_length + 1)
            tokens = rng.integers(0, vocab_size, size=(length,))
            data.append(tokens)
        return data

    # Load from file
    dataset_files = sorted(Path(dataset_path).glob("*.npy"))
    if not dataset_files:
        print(f"Error: No dataset files found in {dataset_path}", file=sys.stderr)
        sys.exit(1)

    data = []
    for f in dataset_files:
        arr = np.load(f)
        for i in range(len(arr)):
            data.append(arr[i])
    return data


def run_training_numpy(model, optimizer, loss_fn, config: dict, dataset) -> list[float]:
    """Run the training loop for NumPy backend.

    Args:
        model: The NumPyModel instance.
        optimizer: The AdamW optimizer instance.
        loss_fn: The CrossEntropyLoss instance.
        config: Training configuration dict.
        dataset: List of numpy arrays.

    Returns:
        List of per-epoch average loss values.
    """
    from impl._np.training import train_step as numpy_train_step

    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 64)

    losses_per_epoch: list[float] = []

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses: list[float] = []
        total_batches = 0
        max_seq = config.get("context_length", 128)

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            batch_input = np.stack([b[:max_seq] for b in batch]).astype(np.int32)
            batch_target = np.stack([b[1 : max_seq + 1] for b in batch]).astype(np.int32)

            loss = float(numpy_train_step(model, batch_input, batch_target, loss_fn, optimizer))
            epoch_losses.append(loss)
            total_batches += 1

            # Log progress
            if total_batches % max(1, total_batches // 5) == 0 or total_batches == 1:  # noqa: SIM113
                current_loss = float(np.mean(epoch_losses))
                total_steps = epoch * total_batches + total_batches
                print(f"  Step {total_steps}: loss={current_loss:.4f}")

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_time = time.time() - epoch_start
        losses_per_epoch.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}: avg_loss={avg_loss:.4f} time={epoch_time:.2f}s")

    return losses_per_epoch


def run_training_torch(model, optimizer, loss_fn, config: dict, dataset) -> list[float]:
    """Run the training loop for PyTorch backend.

    Args:
        model: The TorchModel instance.
        optimizer: The torch optimizer instance.
        loss_fn: The nn.CrossEntropyLoss instance.
        config: Training configuration dict.
        dataset: List of numpy arrays (to be converted to torch tensors).

    Returns:
        List of per-epoch average loss values.
    """
    import torch

    from impl._torch.training import train_step as torch_train_step

    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 64)

    losses_per_epoch: list[float] = []

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses: list[float] = []
        total_batches = 0
        max_seq = config.get("context_length", 128)

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            batch_input = torch.stack([torch.tensor(b[:max_seq]) for b in batch])
            batch_target = torch.stack([torch.tensor(b[1 : max_seq + 1]) for b in batch])


            # Use detach to prevent gradient through targets
            batch_target = batch_target.long()
            loss = float(
                torch_train_step(
                    model, batch_input, batch_target, optimizer, loss_fn
                )
            )
            epoch_losses.append(loss)
            total_batches += 1

            # Log progress
            if total_batches % max(1, total_batches // 5) == 0 or total_batches == 1:
                current_loss = float(torch.tensor(epoch_losses).mean().item())
                total_steps = epoch * total_batches + total_batches
                print(f"  Step {total_steps}: loss={current_loss:.4f}")

        avg_loss = float(torch.tensor(epoch_losses).mean().item()) if epoch_losses else 0.0
        epoch_time = time.time() - epoch_start
        losses_per_epoch.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}: avg_loss={avg_loss:.4f} time={epoch_time:.2f}s")

    return losses_per_epoch


def run_training(model, optimizer, loss_fn, config: dict, dataset, backend: str) -> list[float]:
    """Run the training loop for either backend.

    Args:
        model: The model instance (NumPyModel or TorchModel).
        optimizer: The optimizer instance.
        loss_fn: The loss function instance.
        config: Training configuration dict.
        dataset: List of numpy arrays or torch tensors.
        backend: 'numpy' or 'torch'.

    Returns:
        List of per-epoch average loss values.
    """
    if backend == "numpy":
        return run_training_numpy(model, optimizer, loss_fn, config, dataset)
    else:
        return run_training_torch(model, optimizer, loss_fn, config, dataset)


def save_checkpoint(model, config: dict, cfg, save_dir: str, backend: str) -> str:
    """Save model checkpoint.

    Args:
        model: The trained model.
        config: Training config dict.
        cfg: TransformerConfig instance.
        save_dir: Directory to save checkpoints.
        backend: 'numpy' or 'torch'.

    Returns:
        Path to the saved checkpoint directory.
    """
    from shared.checkpoint import save_checkpoint as save_ckpoint

    seed = config.get("seed", 42)
    checkpoint_dir = Path(save_dir) / f"{backend}_{seed}"

    # Save model parameters
    if backend == "numpy":
        params = model.get_all_parameters()
        save_ckpoint(checkpoint_dir, config=cfg, **params)
    else:
        # PyTorch: save state dict
        params: dict = {}
        for name, param in model.named_parameters():
            params[name] = param.detach().cpu()

        save_ckpoint(checkpoint_dir, config=cfg, **params)

    print(f"Checkpoint saved to {checkpoint_dir}")
    return str(checkpoint_dir)


def main() -> int:
    """Entry point for the training script.

    Returns:
        Exit code (0 for success, 1 for user error, 2 for runtime error).
    """
    try:
        parser = create_argparser()
        args = parser.parse_args()
        backend = args.backend

        # Build config
        config = build_config(args, backend)

        print(f"Backend: {backend}")
        print(f"Vocab: {config.get('vocab_size', 256)}, Context: {config.get('context_length', 128)}")
        print(f"Embed: {config.get('embed_dim', 256)}, Layers: {config.get('n_layers', 4)}")
        print(f"Heads: {config.get('n_heads', 8)}, Experts: {config.get('n_experts', 4)}")
        print(f"Epochs: {config.get('epochs', 5)}, Batch: {config.get('batch_size', 64)}")
        print(f"LR: {config.get('lr', 0.001)}, Seed: {config.get('seed', 42)}")
        print()

        # Build model
        model, cfg = build_model(backend, config)
        print("Model built successfully")

        # Get dataset
        dataset_path = config.get("dataset_path", "resource/tinystories/")
        synthetic = config.get("synthetic", False)
        dataset = get_dataset(dataset_path, synthetic, config.get("vocab_size", 256), config.get("context_length", 128))
        print(f"Dataset: {len(dataset)} samples")

        # Initialize loss and optimizer
        if backend == "numpy":
            from impl._np.cross_entropy import CrossEntropyLoss
            from impl._np.optimizer import AdamW

            loss_fn = CrossEntropyLoss()
            optimizer = AdamW(lr=config.get("lr", 0.001))
        else:
            import torch.nn as nn
            import torch.optim as optim

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=config.get("lr", 0.001))

        # Run training
        losses = run_training(model, optimizer, loss_fn, config, dataset, backend)

        # Save checkpoint
        save_dir = config.get("save_dir", "resource/models/")
        save_checkpoint(model, config, cfg, save_dir, backend)

        print()
        print("Training complete!")
        print(f"Final losses per epoch: {[f'{v:.4f}' for v in losses]}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
