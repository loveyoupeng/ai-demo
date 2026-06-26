#!/usr/bin/env python
"""Unified inference script for decoder-only transformer models.

Supports NumPy, PyTorch, Triton, and CUDA backends through a single entry point.

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
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


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

  # CUDA backend inference
  %(prog)s --model resource/models/cuda_42/ --backend cuda --prompt "hello"

  # Triton backend inference
  %(prog)s --model resource/models/triton_42/ --backend triton --prompt "world"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Backend --
    parser.add_argument(
        "-b",
        "--backend",
        default="torch",
        choices=["numpy", "torch", "triton", "cuda"],
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
        backend: One of 'numpy', 'torch', 'triton', or 'cuda'.

    Returns:
        Tuple of (model_instance, config_dict).
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    from shared.checkpoint import load_checkpoint

    params, cfg = load_checkpoint(str(model_dir))
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
        loaded = model.get_all_parameters()
        for key, value in loaded.items():
            if key in params:
                value[:] = params[key]
            else:
                normalized = key.replace("blocks.", "layers.")
                if normalized in params:
                    value[:] = params[normalized]
        config_dict = {
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "embed_dim": cfg.embed_dim,
        }
        return model, config_dict

    import torch

    if backend == "torch":
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
        params_torch = {k: torch.tensor(v) for k, v in params.items()}
        model.load_state_dict(params_torch)
        config_dict = {
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "embed_dim": cfg.embed_dim,
        }
        return model, config_dict

    if backend == "triton":
        from impl._triton.model import TritonModel

        model = TritonModel(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.embed_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_experts=cfg.n_experts,
            ff_dim=cfg.expert_dim or (cfg.embed_dim * 2),
            k=cfg.top_k,
        )
        model.load_from_numpy_dict(params)
        config_dict = {
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "embed_dim": cfg.embed_dim,
        }
        return model, config_dict

    if backend == "cuda":
        from impl._cuda.model import CUDAModel

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
            seed=42,
        )

        def _load_tensor(module, src_dict, key):
            attr = getattr(module, key, None)
            if attr is not None and key in src_dict:
                tensor = torch.from_numpy(src_dict[key]).to(attr.device)
                attr.data.copy_(tensor)
            elif key in src_dict:
                setattr(module, key, torch.from_numpy(src_dict[key]).float())

        # Load model-level params
        for key in [
            "embedding_weights",
            "final_ln_gamma",
            "output_proj_weights",
            "output_proj_bias",
            "output_W1",
            "output_W2",
            "output_W3",
        ]:
            _load_tensor(model, params, key)

        # Load block-level params
        for i, block in enumerate(model.stacking.blocks):
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
                _load_tensor(block, params, f"blocks.{i}.{attr_name}")

        config_dict = {
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "embed_dim": cfg.embed_dim,
        }
        return model, config_dict

    raise ValueError(f"Unsupported backend: {backend}. Must be numpy, torch, triton, or cuda.")


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
        model: The model instance (NumPyModel, TorchModel, TritonModel, or CUDAModel).
        config: Model config dict with vocab_size, context_length, etc.
        prompt_text: The input prompt string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        top_k: Top-k filter (0 = disabled).
        backend: 'numpy', 'torch', 'triton', or 'cuda'.

    Returns:
        Dict with keys: input_tokens, generated_tokens, full_tokens, prompt_text, generated_text
    """
    import torch

    vocab_size = config.get("vocab_size", 256)
    prompt_tokens = encode_prompt(prompt_text, vocab_size)

    if backend == "numpy":
        from impl._np.inference import TextGenerator as NpGen

        generator = NpGen(model, max_new_tokens, temperature, top_k)
        prompt_np = np.array([prompt_tokens], dtype=np.int32)
        all_tokens_np = generator.generate(prompt_np)
        all_tokens = all_tokens_np.flatten().tolist()

    elif backend == "torch":
        from impl._torch.inference import TorchTextGenerator as TorchGen

        generator = TorchGen(model, max_new_tokens, temperature, top_k)
        torch_prompt = torch.tensor([prompt_tokens])
        result = generator.generate(torch_prompt)
        if isinstance(result, torch.Tensor):
            all_tokens = result.flatten().detach().cpu().tolist()
        else:
            all_tokens = list(torch.tensor(result).flatten().tolist())

    elif backend == "triton":
        from impl._triton.inference import TritonTextGenerator as TritonGen

        generator = TritonGen(model, max_new_tokens, temperature, top_k)
        torch_prompt = torch.tensor([prompt_tokens])
        result = generator.generate(torch_prompt)
        if isinstance(result, torch.Tensor):
            all_tokens = result.flatten().detach().cpu().tolist()
        else:
            all_tokens = list(torch.tensor(result).flatten().tolist())

    elif backend == "cuda":
        from impl._cuda.inference import CudaTextGenerator as CudaGen

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        generator = CudaGen(model, max_new_tokens, temperature, top_k)
        torch_prompt = torch.tensor([prompt_tokens], device=device)
        result = generator.generate(torch_prompt)
        if isinstance(result, torch.Tensor):
            all_tokens = result.flatten().tolist()
        else:
            all_tokens = list(torch.tensor(result).flatten().tolist())

    else:
        raise ValueError(f"Unsupported backend: {backend}")

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
        from shared.utils.logger_setup import setup_logging

        setup_logging()
        parser = create_argparser()
        args = parser.parse_args()

        backend = args.backend
        model_path = args.model
        logger.info("run_inference() starting backend=%s model=%s", backend, model_path)

        # Load model and config
        logger.info("run_inference() loading_model path=%s backend=%s", model_path, backend)
        model, config = load_model_from_checkpoint(model_path, backend)
        logger.info("run_inference() model_loaded vocab=%d context=%d", config.get("vocab_size", 256), config.get("context_length", 128))

        # Decode strategy from args
        temp = args.temperature if not args.greedy else 0.0
        top_k_val = args.top_k

        print(f"Backend: {backend}")
        print(f"Model: {model_path}")
        print(f"Vocab: {config.get('vocab_size', 256)}")
        print(f"Context: {config.get('context_length', 128)}")
        print(f"Embed: {config.get('embed_dim', 256)}, Layers: {config.get('n_layers', 4)}")
        logger.info("run_inference() config backend=%s vocab=%d context=%d layers=%d", backend, config.get("vocab_size", 256), config.get("context_length", 128), config.get("n_layers", 4))
        print()

        if args.prompt:
            # Single prompt mode
            logger.info("run_inference() single_prompt mode prompt=%r max_tokens=%d temp=%.2f top_k=%d", args.prompt, args.max_new_tokens, args.temperature if not args.greedy else 0.0, args.top_k)
            result = generate_single(model, config, args.prompt, args.max_new_tokens, temp, top_k_val, backend)
            logger.info(
                "run_inference() generation_complete prompt_len=%d gen_len=%d",
                len(result["input_tokens"]),
                len(result["generated_tokens"]),
            )
            print(f"Prompt:     {result['prompt_text']}")
            print(f"Generated:  {result['generated_text']}")
            print(f"Full seq:   {result['full_text']}")
        else:
            # Interactive mode
            logger.info("run_inference() interactive_mode")
            print("Interactive mode — type prompts (Ctrl+D to exit):")
            print()

            for line in sys.stdin:
                line = line.rstrip("\n")
                if not line:
                    continue
                logger.info("run_inference() interactive input prompt=%r", line)
                result = generate_single(model, config, line, args.max_new_tokens, temp, top_k_val, backend)
                logger.info(
                    "run_inference() interactive output prompt_len=%d gen_len=%d",
                    len(result["input_tokens"]),
                    len(result["generated_tokens"]),
                )
                print(f"Prompt:     {result['prompt_text']}")
                print(f"Generated:  {result['generated_text']}")
                print(f"Full seq:   {result['full_text']}")
                print()

        logger.info("run_inference() completed successfully")
        return 0

    except Exception as e:
        logger.error("run_inference() error exception=%s", str(e))
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
