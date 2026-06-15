"""CLI for inference with NumPyModel in NumPy."""

from __future__ import annotations

import argparse

import numpy as np

from impl._np.inference import TextGenerator
from impl._np.model import NumPyModel


def main() -> None:
    """Entry point — parse arguments and generate text."""
    parser = argparse.ArgumentParser(description="Generate text with NumPy LLM")
    parser.add_argument("--prompt", type=str, default="hello", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens to generate")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Keep top-k logits (0 = off)",
    )
    parser.add_argument("--embed_dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of attention heads")
    args = parser.parse_args()

    model = NumPyModel(
        vocab_size=256,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_experts=2,
        ff_dim=args.embed_dim * 2,
        k=2,
        rope_dim=args.embed_dim // args.n_heads,
        seed=42,
    )

    generator = TextGenerator(
        model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    prompt_ids = np.array(
        [[ord(c) % 256 for c in args.prompt]],
        dtype=np.int32,
    )

    output = generator.generate(prompt_ids)
    generated_tokens = output[0].tolist()

    print(f"Prompt:     {args.prompt}")
    print(f"Generated:  {bytes(generated_tokens[len(args.prompt) :]).decode('utf-8', errors='replace')}")
    print(f"Full seq:   {bytes(generated_tokens).decode('utf-8', errors='replace')}")


def text_to_tokens(text: str) -> list[int]:
    """Convert text to token IDs using byte-level encoding.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    tokens : list[int]
        List of integer token IDs (one per UTF-8 byte).
    """
    return [b for b in text.encode("utf-8")]


def text_from_tokens(token_ids: list[int]) -> str:
    """Decode a list of token IDs back to text.

    Parameters
    ----------
    token_ids : list[int]
        List of integer token IDs (byte values 0-255).

    Returns
    -------
    text : str
        Decoded text string.
    """
    return bytes(token_ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    main()
