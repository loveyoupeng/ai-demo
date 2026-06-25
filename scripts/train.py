#!/usr/bin/env python
"""Unified training script for decoder-only transformer models.

Supports NumPy, PyTorch, Triton, and CUDA backends through a single entry point.

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
    TORCH_N_LAYERS, TORCH_EMBED_DIM, etc. — override defaults for PyTorch/Triton/CUDA
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
        "-b",
        "--backend",
        default="numpy",
        choices=["numpy", "torch", "triton", "cuda"],
        help="Backend implementation (default: numpy)",
    )

    # -- Architecture --
    parser.add_argument("--vocab_size", default=256, type=int, help="Token vocabulary size (default: 256)")
    parser.add_argument(
        "--ctx", "-c", "--context_length", default=128, type=int, help="Context length in tokens (default: 128)"
    )
    parser.add_argument(
        "--embed", "-e", "--embed_dim", default=256, type=int, help="Embedding dimension (default: 256)"
    )
    parser.add_argument(
        "--layers", "-l", "--n_layers", default=4, type=int, help="Number of transformer blocks (default: 4)"
    )
    parser.add_argument(
        "--heads", "-H", "--n_heads", default=8, type=int, help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--groups", "-g", "--n_groups", default=8, type=int, help="KV query groups (default: 8 = self-attn)"
    )
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
    parser.add_argument(
        "--synthetic", action="store_true", default=False, help="Use synthetic data instead of TinyStories"
    )
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
        "vocab_size",
        "ctx",
        "context_length",
        "embed",
        "embed_dim",
        "layers",
        "n_layers",
        "heads",
        "n_heads",
        "groups",
        "n_groups",
        "rope_dim",
        "n_experts",
        "top_k",
        "expert_dim",
        "max_length",
        "seed",
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
        backend: 'numpy', 'torch', 'triton', or 'cuda'.
        config: Configuration dict.

    Returns:
        Tuple of (model_instance, config_for_logging).
    """
    import torch

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

    if backend == "torch":
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

    if backend == "triton":
        from impl._triton.model import TritonModel
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
        D = cfg.embed_dim
        model = TritonModel(
            vocab_size=cfg.vocab_size,
            embed_dim=D,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_experts=cfg.n_experts,
            ff_dim=D * 2,
            k=cfg.top_k,
        )
        return model, cfg

    if backend == "cuda":
        from impl._cuda.model import CUDAModel
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

        D = cfg.embed_dim
        V = cfg.vocab_size
        model = CUDAModel(
            vocab_size=V,
            embed_dim=D,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_experts=cfg.n_experts,
            ff_dim=D * 2,
            k=cfg.top_k,
            rope_dim=D // cfg.n_heads if cfg.n_heads > 0 else 0,
            seed=cfg.seed,
        )
        for attr_name in [
            "Wq",
            "Wk",
            "Wv",
            "Wo",
            "gate1",
            "gate2",
            "expert_weights",
            "expert_bias",
            "routing_weights",
            "ln1_gamma",
            "ln2_gamma",
        ]:
            attr = getattr(model.stacking.blocks[0], attr_name, None)
            if attr is not None:
                attr.requires_grad_(True)
        for attr_name in [
            "embedding_weights",
            "final_ln_gamma",
            "output_proj_weights",
            "output_proj_bias",
            "output_W1",
            "output_W2",
            "output_W3",
        ]:
            attr = getattr(model, attr_name, None)
            if attr is not None:
                attr.requires_grad_(True)

        def _to_cuda(module, device: torch.device):
            for attr_name in [
                "embedding_weights",
                "final_ln_gamma",
                "output_proj_weights",
                "output_proj_bias",
                "output_W1",
                "output_W2",
                "output_W3",
            ]:
                attr = getattr(module, attr_name, None)
                if attr is not None:
                    setattr(module, attr_name, attr.to(device))
            for block in module.stacking.blocks:
                for sub_attr in block.__dict__:
                    sub = getattr(block, sub_attr)
                    if isinstance(sub, torch.Tensor):
                        setattr(block, sub_attr, sub.to(device))

        _to_cuda(model, torch.device("cuda:0"))
        return model, cfg

    raise ValueError(f"Unsupported backend: {backend}. Must be numpy, torch, triton, or cuda.")


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
        max_len = max(10, context_length)
        for _ in range(num_samples):
            length = rng.integers(5, min(20, max_len + 1))
            tokens = rng.integers(0, vocab_size, size=(length,))
            data.append(tokens)
        return data

    # Load from file — each .npy file is one training sample
    dataset_files = sorted(Path(dataset_path).glob("*.npy"))
    if not dataset_files:
        print(f"Error: No dataset files found in {dataset_path}", file=sys.stderr)
        sys.exit(1)

    data = []
    for f in dataset_files:
        arr = np.load(f)
        data.append(arr)
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
            # Pad all arrays to max_seq for stacking
            padded = []
            for b in batch:
                if len(b) < max_seq:
                    padded.append(np.pad(b, (0, max_seq - len(b)), constant_values=0))
                else:
                    padded.append(b[:max_seq])
            batch_input = np.stack(padded).astype(np.int32)
            # For targets: shift left by 1 and pad last position, also pad to max_seq if shorter
            batch_targets_padded = []
            for p in padded:
                if len(p) < max_seq:
                    target = np.append(p[1:], 0)
                    target = np.pad(target, (0, max_seq - len(target)), constant_values=0)
                else:
                    target = np.append(p[1:], 0)
                batch_targets_padded.append(target)
            batch_target = np.stack(batch_targets_padded).astype(np.int32)

            loss = float(numpy_train_step(model, batch_input, batch_target, loss_fn, optimizer))
            epoch_losses.append(loss)
            total_batches += 1  # noqa: SIM113

            # Log progress
            if total_batches % max(1, total_batches // 5) == 0 or total_batches == 1:
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
            # Pad all arrays to max_seq for stacking
            padded = []
            for b in batch:
                if len(b) < max_seq:
                    padded.append(np.pad(b, (0, max_seq - len(b)), constant_values=0))
                else:
                    padded.append(b[:max_seq])
            batch_input = torch.stack([torch.tensor(p) for p in padded])
            # For targets: shift left by 1 and pad last position
            targets = []
            for p in padded:
                t = np.append(p[1:], 0)
                if len(t) < max_seq:
                    t = np.pad(t, (0, max_seq - len(t)), constant_values=0)
                targets.append(t)
            batch_target = torch.stack([torch.tensor(t) for t in targets])

            # Use detach to prevent gradient through targets
            batch_target = batch_target.long()
            loss = float(torch_train_step(model, batch_input, batch_target, optimizer, loss_fn))
            epoch_losses.append(loss)
            total_batches += 1  # noqa: SIM113

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


def run_training_cuda(model, optimizer, loss_fn, config: dict, dataset, max_norm: float = 1.0) -> list[float]:
    """Run the training loop for CUDA backend.

    Args:
        model: The CUDAModel instance.
        optimizer: The torch optimizer instance.
        loss_fn: The nn.CrossEntropyLoss instance.
        config: Training configuration dict.
        dataset: List of numpy arrays (to be converted to CUDA torch tensors).
        max_norm: Gradient clipping norm (default: 1.0).

    Returns:
        List of per-epoch average loss values.
    """
    import torch

    from impl._cuda.training import train_step as cuda_train_step

    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 64)

    losses_per_epoch: list[float] = []

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses: list[float] = []
        total_batches = 0
        max_seq = config.get("context_length", 128)
        device = torch.device("cuda:0")

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            padded = []
            for b in batch:
                if len(b) < max_seq:
                    padded.append(np.pad(b, (0, max_seq - len(b)), constant_values=0))
                else:
                    padded.append(b[:max_seq])
            batch_input = torch.stack([torch.tensor(p) for p in padded]).to(device)
            targets = []
            for p in padded:
                t = np.append(p[1:], 0)
                if len(t) < max_seq:
                    t = np.pad(t, (0, max_seq - len(t)), constant_values=0)
                targets.append(t)
            batch_target = torch.stack([torch.tensor(t) for t in targets]).to(device)
            batch_target = batch_target.long()
            loss = float(cuda_train_step(model, batch_input, batch_target, optimizer, loss_fn, max_norm=max_norm))
            epoch_losses.append(loss)
            total_batches += 1

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
        model: The model instance (NumPyModel, TorchModel, TritonModel, or CUDAModel).
        optimizer: The optimizer instance.
        loss_fn: The loss function instance.
        config: Training configuration dict.
        dataset: List of numpy arrays or torch tensors.
        backend: 'numpy', 'torch', 'triton', or 'cuda'.

    Returns:
        List of per-epoch average loss values.
    """
    if backend == "numpy":
        return run_training_numpy(model, optimizer, loss_fn, config, dataset)
    elif backend == "cuda":
        return run_training_cuda(model, optimizer, loss_fn, config, dataset)
    else:
        return run_training_torch(model, optimizer, loss_fn, config, dataset)


def save_checkpoint(model, config: dict, cfg, save_dir: str, backend: str) -> str:
    """Save model checkpoint.

    Args:
        model: The trained model.
        config: Training config dict.
        cfg: TransformerConfig instance.
        save_dir: Directory to save checkpoints.
        backend: 'numpy', 'torch', 'triton', or 'cuda'.

    Returns:
        Path to the saved checkpoint directory.
    """
    from shared.checkpoint import save_checkpoint as save_ckpoint

    seed = config.get("seed", 42)
    checkpoint_dir = Path(save_dir) / f"{backend}_{seed}"

    if backend == "numpy":
        params = model.get_all_parameters()
        save_ckpoint(checkpoint_dir, config=cfg, **params)

    elif backend == "torch":
        params: dict = {}
        for name, param in model.named_parameters():
            params[name] = param.detach().cpu()
        save_ckpoint(checkpoint_dir, config=cfg, **params)

    elif backend == "triton":
        np_params = model.save_as_numpy()
        save_ckpoint(checkpoint_dir, config=cfg, **np_params)

    elif backend == "cuda":
        params: dict = {}

        def _save_tensor(module, prefix: str):
            for attr_name in [
                "embedding_weights",
                "final_ln_gamma",
                "output_proj_weights",
                "output_proj_bias",
                "output_W1",
                "output_W2",
                "output_W3",
            ]:
                attr = getattr(module, attr_name, None)
                if attr is not None:
                    params[attr_name] = attr.detach().cpu()
            for i, block in enumerate(module.stacking.blocks):
                for attr_name in [
                    "Wq",
                    "Wk",
                    "Wv",
                    "Wo",
                    "gate1",
                    "gate2",
                    "expert_weights",
                    "expert_bias",
                    "routing_weights",
                    "ln1_gamma",
                    "ln2_gamma",
                ]:
                    attr = getattr(block, attr_name, None)
                    if attr is not None:
                        params[f"{prefix}{i}.{attr_name}"] = attr.detach().cpu()

        _save_tensor(model, "blocks")
        save_ckpoint(checkpoint_dir, config=cfg, **params)

    else:
        raise ValueError(f"Unsupported backend for saving: {backend}")

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
            import torch
            import torch.nn as nn
            import torch.optim as optim

            loss_fn = nn.CrossEntropyLoss()
            if backend == "cuda":
                params = [p for p in model.stacking.blocks[0].__dict__.values() if isinstance(p, torch.Tensor)]
                for attr in [
                    "embedding_weights",
                    "final_ln_gamma",
                    "output_proj_weights",
                    "output_proj_bias",
                    "output_W1",
                    "output_W2",
                    "output_W3",
                ]:
                    attr_obj = getattr(model, attr, None)
                    if attr_obj is not None:
                        params.append(attr_obj)
                for block in model.stacking.blocks:
                    for sub_attr in block.__dict__:
                        sub = getattr(block, sub_attr)
                        if isinstance(sub, torch.Tensor):
                            params.append(sub)
                optimizer = optim.AdamW(params, lr=config.get("lr", 0.001))
            else:
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
