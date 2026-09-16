"""CLI for inference with NumPyModel in NumPy."""

from __future__ import annotations

import argparse

import numpy as np

from impl._np.inference import TextGenerator
from impl._np.model import NumPyModel
from shared.config import TransformerConfig


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
    parser.add_argument(
        "--learning",
        action="store_true",
        help="Learning mode: host a webpage for interactive inference, architecture/math visualization, and inference records",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for --learning (default 8000)")
    parser.add_argument(
        "--model",
        type=str,
        default="resource/models/learning_demo",
        help="Checkpoint dir for --learning (default: the demo model)",
    )
    args = parser.parse_args()
    if args.learning:
        # Opt-in learning mode — imported here so the flag-off path never touches it.
        from impl._np import learning_server

        lm, vocab, _cfg = learning_server.load_learning_model(args.model)
        server = learning_server.start_server(lm, vocab, port=args.port)
        print(f"Learning mode: http://127.0.0.1:{args.port}  (model: {args.model})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")
        return

    model = NumPyModel(
        TransformerConfig(
            vocab_size=256,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            seed=42,
        )
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
