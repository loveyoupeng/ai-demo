from __future__ import annotations

import argparse
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


def run_training(args):
    # 1. Setup Data
    if args.data_path:
        with open(args.data_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from {args.data_path}")
    else:
        # In a real scenario, we'd load from a file. For demo, we use a small text.
        text = "the quick brown fox jumps over the lazy dog. " * 50
        print("Using default toy dataset.")

    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    # 2. Initialize Model, Optimizer, Loss, Trainer
    backend = NumPyBackend(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        num_experts=args.experts,
        max_seq_len=args.max_context,
    )

    optimizer = Adam(learning_rate=args.lr)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(backend, optimizer, loss_fn)

    data_loader = TextDataLoader(text, tokenizer, args.batch_size, args.seq_len)

    # 3. Training
    print("Starting training...")
    trainer.fit(data_loader, epochs=args.epochs)
    print("Training completed.")

    # 4. Save Checkpoint
    checkpoint = ModelCheckpoint()
    checkpoint.save_checkpoint(backend.model, tokenizer, args.checkpoint_name)
    print(f"Model and tokenizer saved to {args.checkpoint_name}.pkl")


def run_inference(args):
    # 1. Load Checkpoint
    checkpoint = ModelCheckpoint()
    loaded = checkpoint.load_checkpoint(
        args.checkpoint_name, Transformer, CharTokenizer
    )
    model: Transformer = cast(Transformer, loaded[0])
    tokenizer: CharTokenizer = cast(CharTokenizer, loaded[1])

    # 2. Setup Generator
    generator = AutoregressiveGenerator(model, tokenizer, temperature=args.temp)

    # 3. Generate Text
    print(f"Prompt: {args.prompt}")
    generated_ids = generator.generate(args.prompt, num_new_tokens=args.gen_len)
    generated_text = tokenizer.decode(generated_ids)

    print(f"Generated Text: {generated_text}")


def main():
    parser = argparse.ArgumentParser(
        description="Transformer E2E Training and Inference"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Training Subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--embed_dim", type=int, default=32)
    train_parser.add_argument("--layers", type=int, default=2)
    train_parser.add_argument("--heads", type=int, default=4)
    train_parser.add_argument("--experts", type=int, default=4)
    train_parser.add_argument("--max_context", type=int, default=64)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--seq_len", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--checkpoint_name", type=str, default="demo_model")
    train_parser.add_argument(
        "--data_path", type=str, default=None, help="Path to a text file for training"
    )

    # Inference Subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--checkpoint_name",
        type=str,
        required=True,
        help="Name of checkpoint file (without .pkl)",
    )
    infer_parser.add_argument("--prompt", type=str, default="the", help="Prompt text")
    infer_parser.add_argument(
        "--gen_len", type=int, default=20, help="Number of tokens to generate"
    )
    infer_parser.add_argument(
        "--temp", type=float, default=1.0, help="Temperature for sampling"
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "infer":
        run_inference(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
