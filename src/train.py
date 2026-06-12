#!/usr/bin/env python3
"""
End-to-end training script for the MoE Transformer.

Run:
    uv run src/train.py train --epochs 10 --data-path datasets/tiny_shakespeare.txt
    uv run src/train.py train --epochs 5  # uses toy dataset if no file provided
    uv run src/train.py infer --checkpoint-name my_model
    uv run src/train.py generate --checkpoint-name my_model --prompt "to be" --num-tokens 100
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

from model.transformer import Transformer
from model.pytorch.transformer import PyTorchTransformer
from loss import CrossEntropyLoss
from optimizer import Adam
from training.data_loader import TextDataLoader
from tokenizer.char_tokenizer import CharTokenizer
from trainer import Trainer
from utils.checkpoint import ModelCheckpoint
from inference import AutoregressiveGenerator
from backends.numpy.numpy_backend import NumPyBackend
from backends.pytorch.pytorch_backend import PyTorchBackend


def get_backend(
    backend_name: str,
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    num_experts: int,
    max_seq_len: int,
):
    """Create a backend instance based on the specified backend name."""
    if backend_name == "numpy":
        return NumPyBackend(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_experts=num_experts,
            max_seq_len=max_seq_len,
        )
    elif backend_name == "torch":
        return PyTorchBackend(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_experts=num_experts,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(
            f"Unsupported backend: {backend_name}. Use 'numpy' or 'torch'."
        )


DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def download_tiny_shakespeare() -> str:
    """Download Tiny Shakespeare dataset if not present."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATASETS_DIR / "tiny_shakespeare.txt"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if data_path.exists():
        return str(data_path)

    print(f"Downloading dataset from {url}...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    data_path.write_text(text, encoding="utf-8")
    print(f"Saved {len(text)} characters to {data_path}")
    return str(data_path)


def run_train(args: argparse.Namespace) -> None:
    """Run training with optional dataset download."""
    # 1. Load data
    data_path = args.data
    if data_path is None:
        data_path = download_tiny_shakespeare()
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}, downloading Tiny Shakespeare...")
        data_path = download_tiny_shakespeare()

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters from {data_path}")

    # 2. Tokenize
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # 3. Build model
    backend = get_backend(
        args.backend,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        num_experts=args.experts,
        max_seq_len=args.max_context,
    )

    # 4. Setup training components
    optimizer = Adam(learning_rate=args.lr)
    loss_fn = CrossEntropyLoss()
    clip_value = args.clip if args.clip > 0 else None
    trainer = Trainer(backend, optimizer, loss_fn, clip_value=clip_value)

    # 5. Setup data loader
    data_loader = TextDataLoader(text, tokenizer, args.batch_size, args.seq_len)
    print(f"Data loader: {len(data_loader)} batches per epoch")

    # 6. Train
    print(f"\n{'=' * 60}")
    print(f"Training for {args.epochs} epochs...")
    print(f"{'=' * 60}")

    train_metrics = []
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch_num, (x_batch, y_batch) in enumerate(data_loader, 1):
            loss = trainer.train_step(x_batch, y_batch)
            epoch_losses.append(loss)
            if batch_num % 20 == 0 or batch_num == 1:
                avg = sum(epoch_losses) / len(epoch_losses)
                print(
                    f"  Epoch {epoch}, Batch {batch_num:3d}/{len(data_loader)}: avg_loss={avg:.4f}"
                )

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_metrics.append(float(avg_epoch_loss))
        print(
            f"Epoch {epoch:3d}/{args.epochs} complete — avg loss: {avg_epoch_loss:.4f}\n"
        )

    # 7. Save checkpoint
    checkpoint_path = args.checkpoint_name or "trained_model"
    checkpoint = ModelCheckpoint()
    checkpoint.save_checkpoint(backend.model, tokenizer, checkpoint_path)
    print(f"✓ Model and tokenizer saved to {checkpoint_path}.pkl")

    # 8. Save training metrics (simple text file)
    metrics_path = Path(__file__).parent.parent / f"{checkpoint_path}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("epoch,avg_loss\n")
        for i, loss in enumerate(train_metrics):
            f.write(f"{i + 1},{loss:.6f}\n")
    print(f"✓ Training metrics saved to {metrics_path}")

    # 9. Final evaluation on last batch
    last_loss = train_metrics[-1]
    print(f"\n{'=' * 60}")
    print(f"Training complete! Final avg loss: {last_loss:.4f}")
    print(f"{'=' * 60}")


def _generate_numpy(
    model: Transformer,
    tokenizer: CharTokenizer,
    prompt: str,
    num_new_tokens: int,
    temperature: float,
) -> np.ndarray:
    """Generate tokens using NumPy model with autoregressive generation."""
    generator = AutoregressiveGenerator(model, tokenizer, temperature=temperature)
    return generator.generate(prompt, num_new_tokens=num_new_tokens)


def _generate_torch(
    model: PyTorchTransformer,
    tokenizer: CharTokenizer,
    prompt: str,
    num_new_tokens: int,
    temperature: float,
) -> np.ndarray:
    """Generate tokens using PyTorch model with autoregressive generation and KV cache."""
    from model.pytorch.attention_kvcache import PyTorchTurboQuantCache

    current_ids = tokenizer.encode(prompt).reshape(1, -1)
    prompt_len = current_ids.shape[1]
    is_empty_prompt = prompt_len == 0
    if is_empty_prompt:
        current_ids = np.array([[0]], dtype=np.int32)

    # Access num_heads from first block's MHA to construct KV caches
    num_heads: int = model.blocks[0].mha.num_heads  # type: ignore[no-any-return]
    embed_dim: int = model.embed_dim
    num_layers: int = model.num_layers
    kv_caches = [
        PyTorchTurboQuantCache(
            embed_dim=embed_dim, num_heads=num_heads,
            max_seq_len=64, head_dim=embed_dim // num_heads,
        )
        for _ in range(num_layers)
    ]

    generated_ids: list[int] = []
    for step in range(num_new_tokens):
        input_tensor = torch.from_numpy(current_ids).to(torch.int64)
        logits_tensor, _ = model.forward(input_tensor, kv_caches=kv_caches)
        logits_np = logits_tensor.detach().float().cpu().numpy().astype(np.float64)

        next_token_logits = logits_np[0, -1, :] / max(temperature, 1e-8)
        max_val = np.max(next_token_logits)
        exp_logits = np.exp(next_token_logits - max_val)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        next_token_id = int(np.random.choice(len(probs), p=probs))
        next_token_id_arr = np.array([[next_token_id]], dtype=np.int32)
        current_ids = np.concatenate([current_ids, next_token_id_arr], axis=1)
        generated_ids.append(next_token_id)

    if is_empty_prompt:
        return np.array(generated_ids, dtype=np.int32)
    return np.array(generated_ids, dtype=np.int32)


def run_infer(args: argparse.Namespace) -> None:
    """Run inference on a trained model."""
    import pickle
    import numpy as np

    backend_name = args.backend

    with open(os.path.join("checkpoints", f"{args.checkpoint_name}.pkl"), "rb") as f:
        data = pickle.load(f)

    params = data["model_params"]

    import torch

    # Detect source format from value types in params dict
    source_is_torch = any(isinstance(v, torch.Tensor) for v in params.values() if v is not None)
    source_is_numpy = any(isinstance(v, np.ndarray) for v in params.values() if v is not None)

    if source_is_torch and backend_name == "numpy":
        # PT→NP: transpose lm_head [vocab, embed_dim] → [embed_dim, vocab]
        # then convert tensors to numpy arrays
        if "lm_head" in params and isinstance(params["lm_head"], torch.Tensor):
            params["lm_head"] = params["lm_head"].T
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                params[k] = v.detach().float().numpy()
    elif source_is_numpy and backend_name == "torch":
        # NP→PT: transpose lm_head [embed_dim, vocab] → [vocab, embed_dim]
        # and convert numpy arrays to torch tensors
        if "lm_head" in params and isinstance(params["lm_head"], np.ndarray):
            params["lm_head"] = torch.tensor(params["lm_head"]).T.contiguous()
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                params[k] = torch.tensor(v)

    tokenizer = CharTokenizer()
    tokenizer.chars = data["tokenizer"].chars
    tokenizer.vocab_size = data["tokenizer"].vocab_size
    tokenizer.char_to_int = data["tokenizer"].char_to_int
    tokenizer.int_to_char = data["tokenizer"].int_to_char

    backend_map = {
        "numpy": ("Transformer", Transformer, _generate_numpy),
        "torch": ("PyTorch", PyTorchTransformer, _generate_torch),
    }
    name, cls, gen_fn = backend_map[backend_name]

    model = cls(
        vocab_size=data["vocab_size"],
        embed_dim=data["embed_dim"],
        num_layers=data["num_layers"],
        num_heads=data["num_heads"],
        num_experts=data["num_experts"],
        max_seq_len=data["max_seq_len"],
    )
    model.set_params(params)

    prompt = args.prompt or "the"
    print(f"Running inference with backend: {backend_name}")
    print(f"Prompt: '{prompt}'")

    generated_ids = gen_fn(model, tokenizer, prompt, args.gen_len, args.temp)
    generated_text = tokenizer.decode(generated_ids)
    print(f"Generated: {generated_text}")


def run_generate(args: argparse.Namespace) -> None:
    """Alias for infer, generates more text by default."""
    if args.num_tokens is None:
        args.num_tokens = 200
    run_infer(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE Transformer Training & Inference")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--embed_dim", type=int, default=32, help="Embedding dimension"
    )
    train_parser.add_argument(
        "--layers", type=int, default=2, help="Number of transformer layers"
    )
    train_parser.add_argument(
        "--heads", type=int, default=4, help="Number of attention heads"
    )
    train_parser.add_argument(
        "--experts", type=int, default=4, help="Number of MoE experts"
    )
    train_parser.add_argument(
        "--max_context", type=int, default=64, help="Max sequence length"
    )
    train_parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument(
        "--clip",
        type=float,
        default=-1,
        help="Gradient clipping threshold (>0 to enable)",
    )
    train_parser.add_argument(
        "--checkpoint_name", type=str, default="trained_model", help="Checkpoint name"
    )
    train_parser.add_argument(
        "--data", type=str, default=None, help="Path to training data text file"
    )
    train_parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Backend to use for training (numpy or torch)",
    )

    # Inference subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference on trained model")
    infer_parser.add_argument(
        "--checkpoint_name", type=str, required=True, help="Checkpoint name"
    )
    infer_parser.add_argument("--prompt", type=str, default="the", help="Prompt text")
    infer_parser.add_argument(
        "--gen_len", type=int, default=50, help="Number of tokens to generate"
    )
    infer_parser.add_argument(
        "--temp", type=float, default=1.0, help="Sampling temperature"
    )
    infer_parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Backend to use for inference (numpy or torch)",
    )
    infer_parser.add_argument(
        "--num_new_tokens", type=int, default=50, help="Number of tokens to generate (alias for --gen_len)"
    )
    infer_parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (alias for --temp)"
    )

    # Generate subcommand
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text (alias for infer)"
    )
    gen_parser.add_argument(
        "--checkpoint_name", type=str, required=True, help="Checkpoint name"
    )
    gen_parser.add_argument("--prompt", type=str, default="the", help="Prompt text")
    gen_parser.add_argument(
        "--num_tokens", type=int, default=200, help="Number of tokens to generate"
    )
    gen_parser.add_argument(
        "--temp", type=float, default=1.0, help="Sampling temperature"
    )

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_infer(args)
    elif args.command == "generate":
        run_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
