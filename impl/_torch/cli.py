"""CLI for inference with PyTorch Model — auto-detects GPU."""

from __future__ import annotations

import argparse

import torch

from impl._torch.inference import TorchTextGenerator
from impl._torch.layers import TorchModel


def _get_device() -> torch.device:
    """Detect GPU → fall back to CPU.

    Returns
    -------
    device : torch.device
        ``cuda:0`` if ``torch.cuda.is_available()``, else ``cpu``.

    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def main() -> None:
    """Entry point — parse arguments and generate text."""
    parser = argparse.ArgumentParser(description="Generate text with PyTorch LLM")
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
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Device to run on (auto = detect GPU)",
    )
    args = parser.parse_args()

    # Resolve device: --device flag overrides auto-detection
    if args.device is None or args.device == "auto":
        device = _get_device()
    elif args.device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    device_label = f"{device.type}:{device.index if device.index is not None else 0}"
    print(f"Device: {device_label} ({'CUDA' if device.type == 'cuda' else 'CPU'})")

    model = TorchModel(
        vocab_size=256,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_experts=2,
        ff_dim=args.embed_dim * 2,
        k=2,
        rope_dim=args.embed_dim // args.n_heads,
        seed=42,
    ).to(device)

    generator = TorchTextGenerator(
        model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    prompt_ids = torch.tensor(
        [[ord(c) % 256 for c in args.prompt]],
        dtype=torch.int64,
        device=device,
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
