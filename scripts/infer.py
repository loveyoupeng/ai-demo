#!/usr/bin/env python
"""Unified inference script for decoder-only transformer models.

Supports both NumPy and PyTorch backends through a single entry point.

Usage:
    # Single prompt with default greedy decoding
    uv run python scripts/infer.py --model resource/models/torch_42/ --prompt "hello"

    # Interactive mode (reads from stdin)
    uv run python scripts/infer.py --model resource/models/torch_42/ --backend torch

    # Temperature sampling with top-k
    uv run python scripts/infer.py --model resource/models/torch_42/ --prompt "hello" \
        --temperature 0.8 --top_k 50

    # Specify max generation length
    uv run python scripts/infer.py --model resource/models/torch_42/ --prompt "hello" \
        --max_new_tokens 100 --greedy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402


def create_argparser() -> argparse.ArgumentParser:
    """Create the argument parser for the inference script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Generate text from a trained decoder-only transformer model.",
        epilog="""
examples:
  # Single prompt, greedy decoding (deterministic)
  %(prog)s --model resource/models/torch_42/ --prompt "hello"

  # Temperature sampling with top-k filtering
  %(prog)s --model resource/models/torch_42/ --prompt "hello" --temperature 0.8 --top_k 50

  # Interactive mode (reads prompts from stdin)
  %(prog)s --model resource/models/torch_42/ --backend torch

  # Specify max generation length
  %(prog)s --model resource/models/torch_42/ --prompt "hello" --max_new_tokens 100

  # NumPy backend with custom context length
  %(prog)s --model resource/models/numpy_42/ --backend numpy --context_length 256
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Backend --
    parser.add_argument(
        "-b",
        "--backend",
        default="torch",
        choices=["numpy", "torch"],
        help="Backend implementation (default: torch)",
    )

    # -- Model --
    parser.add_argument("--model", required=True, help="Path to saved checkpoint folder")
    parser.add_argument("--context_length", "-c", default=128, type=int, help="Context length (default: 128)")

    # -- Generation --
    parser.add_argument("--prompt", default=None, type=str, help="Single prompt string (skips interactive)")
    parser.add_argument("--max_new_tokens", default=50, type=int, help="Max tokens to generate (default: 50)")
    parser.add_argument(
        "--temperature", default=0.0, type=float, help="Sampling temperature — 0.0=greedy (default: 0.0)"
    )
    parser.add_argument("--top_k", default=0, type=int, help="Top-k sampling filter (default: 0 = off)")
    parser.add_argument(
        "--greedy", action="store_true", default=False, help="Use greedy decoding (argmax, deterministic)"
    )

    return parser


def load_model_from_checkpoint(model_path: str, backend: str):
    """Load a model from a checkpoint directory.

    Args:
        model_path: Path to the checkpoint directory.
        backend: Either 'numpy' or 'torch'.

    Returns:
        Tuple of (model_instance, config_dict).
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Import and use shared checkpoint loading
    from shared.checkpoint import load_checkpoint
    from shared.config import TransformerConfig

    # load_checkpoint returns (params_dict, config)
    params, cfg = load_checkpoint(str(model_dir))
    # Handle case where config might be None
    if cfg is None:
        raise ValueError(f"Config missing in checkpoint: {model_dir}")

    if backend == "numpy":
        from impl._np.model import NumPyModel

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
        # Load parameters into model by name
        loaded = model.get_all_parameters()
        for key, value in loaded.items():
            # Check if we have a matching checkpoint key (with or without _npz suffix)
            if key in params:
                value[:] = params[key]
            else:
                # Try with normalized key
                normalized = key.replace("blocks.", "layers.")
                if normalized in params:
                    value[:] = params[normalized]
        config_dict = {
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "embed_dim": cfg.embed_dim,
        }
        return model, config_dict

    # PyTorch
    import torch

    from impl._torch.layers import TorchModel

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
    # Load parameters into model
    params_torch = {k: torch.tensor(v) for k, v in params.items()}
    model.load_state_dict(params_torch)
    config_dict = {
        "vocab_size": cfg.vocab_size,
        "context_length": cfg.context_length,
        "embed_dim": cfg.embed_dim,
    }
    return model, config_dict


def encode_prompt(text: str, vocab_size: int = 256) -> list[int]:
    """Encode a text prompt into token IDs.

    Uses byte-level encoding: each character's Unicode code point mod vocab_size.

    Args:
        text: The prompt text to encode.
        vocab_size: Vocabulary size for encoding.

    Returns:
        List of token IDs.
    """
    return [ord(c) % vocab_size for c in text]


def decode_tokens(tokens: list[int], vocab_size: int = 256) -> str:
    """Decode token IDs back to text.

    Inverse of encode_prompt: each token mod 256 gives the original character.

    Args:
        tokens: List of token IDs.
        vocab_size: Vocabulary size (must be <= 256 for correct decoding).

    Returns:
        Decoded text string (UTF-8, errors replaced).
    """
    chars = []
    for t in tokens:
        ch = chr(t % 256)
        if ch.isprintable() or ch in ("\n", "\r", "\t"):
            chars.append(ch)
    return "".join(chars)


def generate_single(
    model, config: dict, prompt_text: str, max_new_tokens: int, temperature: float, top_k: int, backend: str
) -> dict:
    """Run inference on a single prompt.

    Args:
        model: The model instance (NumPyModel or TorchModel).
        config: Model config dict with vocab_size, context_length, etc.
        prompt_text: The input prompt string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        top_k: Top-k filter (0 = disabled).
        backend: 'numpy' or 'torch'.

    Returns:
        Dict with keys: input_tokens, generated_tokens, full_tokens, prompt_text, generated_text
    """
    vocab_size = config.get("vocab_size", 256)
    prompt_tokens = encode_prompt(prompt_text, vocab_size)

    if backend == "numpy":
        from impl._np.inference import TextGenerator as NpGen

        generator = NpGen(model, max_new_tokens, temperature, top_k)
        prompt_np = np.array([prompt_tokens], dtype=np.int32)  # (1, seq)
        all_tokens_np = generator.generate(prompt_np)  # (1, seq)
        all_tokens = all_tokens_np.flatten().tolist()
    else:
        import torch

        from impl._torch.inference import TorchTextGenerator as TorchGen

        generator = TorchGen(model, max_new_tokens, temperature, top_k)
        torch_prompt = torch.tensor([prompt_tokens])  # (1, seq)
        result = generator.generate(torch_prompt)  # (1, seq)
        if isinstance(result, torch.Tensor):
            all_tokens = result.flatten().detach().cpu().tolist()
        else:
            all_tokens = list(torch.tensor(result).flatten().tolist())

    generated = all_tokens[len(prompt_tokens) :] if len(all_tokens) > len(prompt_tokens) else []

    return {
        "input_tokens": prompt_tokens,
        "generated_tokens": generated,
        "full_tokens": all_tokens,
        "prompt_text": prompt_text,
        "generated_text": decode_tokens(generated, vocab_size),
        "full_text": decode_tokens(all_tokens, vocab_size),
    }


def main() -> int:
    """Entry point for the inference script.

    Returns:
        Exit code (0 for success, 1 for user error, 2 for runtime error).
    """
    try:
        parser = create_argparser()
        args = parser.parse_args()

        backend = args.backend
        model_path = args.model

        # Load model and config
        model, config = load_model_from_checkpoint(model_path, backend)

        # Decode strategy from args
        temp = args.temperature if not args.greedy else 0.0
        top_k_val = args.top_k

        print(f"Backend: {backend}")
        print(f"Model: {model_path}")
        print(f"Vocab: {config.get('vocab_size', 256)}")
        print(f"Context: {config.get('context_length', 128)}")
        print(f"Embed: {config.get('embed_dim', 256)}, Layers: {config.get('n_layers', 4)}")
        print()

        if args.prompt:
            # Single prompt mode
            result = generate_single(model, config, args.prompt, args.max_new_tokens, temp, top_k_val, backend)
            print(f"Prompt:     {result['prompt_text']}")
            print(f"Generated:  {result['generated_text']}")
            print(f"Full seq:   {result['full_text']}")
        else:
            # Interactive mode
            print("Interactive mode — type prompts (Ctrl+D to exit):")
            print()

            for line in sys.stdin:
                line = line.rstrip("\n")
                if not line:
                    continue
                result = generate_single(model, config, line, args.max_new_tokens, temp, top_k_val, backend)
                print(f"Prompt:     {result['prompt_text']}")
                print(f"Generated:  {result['generated_text']}")
                print(f"Full seq:   {result['full_text']}")
                print()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
