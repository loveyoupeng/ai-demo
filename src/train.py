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
from typing import cast

from model.transformer import Transformer
from loss import CrossEntropyLoss
from optimizer import Adam
from training.data_loader import TextDataLoader
from tokenizer.char_tokenizer import CharTokenizer
from trainer import Trainer
from utils.checkpoint import ModelCheckpoint
from inference import AutoregressiveGenerator
from backends.numpy.numpy_backend import NumPyBackend


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
    backend = NumPyBackend(
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


def run_infer(args: argparse.Namespace) -> None:
    """Run inference on a trained model."""
    from model.transformer import Transformer

    checkpoint = ModelCheckpoint()
    loaded = checkpoint.load_checkpoint(
        args.checkpoint_name, Transformer, CharTokenizer
    )
    model: Transformer = cast(Transformer, loaded[0])
    tokenizer: CharTokenizer = cast(CharTokenizer, loaded[1])

    generator = AutoregressiveGenerator(model, tokenizer, temperature=args.temp)
    prompt = args.prompt or "the"
    print(f"Prompt: '{prompt}'")

    generated_ids = generator.generate(prompt, num_new_tokens=args.gen_len)
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
