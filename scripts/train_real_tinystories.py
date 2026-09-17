"""Recreate the four-backend TinyStories checkpoints (numpy/torch/triton/cuda).

The "real-data" checkpoints under ``resource/models/{backend}_real/`` are
git-ignored on purpose: each is ~38 MB of binary weights (the 50,257-wide GPT-2
embedding table + lm_head dominate — the transformer itself is only ~1 MB), so
the four of them are ~150 MB and would permanently bloat the git history. This
script regenerates them from scratch so anyone can rebuild them.

What it does:
    1. Loads the TinyStories training split and tokenizes it with the GPT-2 BPE
       tokenizer (vocab 50,257).
    2. Samples one shared set of random training windows (seeded) for every backend.
    3. Trains each backend for the same number of steps, from an IDENTICAL
       NumPy-initialized weight set, on the same data, with the same lr and
       gradient clipping. (Every backend loads the NumPy init via the shared
       checkpoint key scheme, so they all start from bit-identical weights.)
    4. Saves each backend to ``resource/models/{backend}_{suffix}/``.

The defaults reproduce the original run: 2 layers, D=96, H=4, ctx=96, 40 steps,
AdamW lr=3e-3, grad-clip 1.0, dense FFN (n_experts=1), seed 42.

Usage:
    # Full reproduction (all 4 backends, ~150 MB of checkpoints)
    uv run python -m scripts.train_real_tinystories

    # Quick small sanity run (fewer steps, two backends, throwaway suffix)
    uv run python -m scripts.train_real_tinystories --backends numpy,torch --num_batches 5 --suffix tmp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Add the project root to sys.path so `shared` / `impl` import when run as a script.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402

from shared.checkpoint import save_checkpoint  # noqa: E402
from shared.config import TransformerConfig  # noqa: E402
from shared.dataset import TextDataset, load_tinystories  # noqa: E402
from shared.tokenizer import create_tokenizer  # noqa: E402

BACKENDS = ("numpy", "torch", "triton", "cuda")


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments. Defaults reproduce the original 4-backend TinyStories run."""
    p = argparse.ArgumentParser(
        description="Recreate the 4-backend TinyStories checkpoints (git-ignored) from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run python -m scripts.train_real_tinystories\n"
            "  uv run python -m scripts.train_real_tinystories --backends numpy,torch --num_batches 5 --suffix tmp\n"
        ),
    )
    p.add_argument(
        "--num_stories", type=int, default=None, help="TinyStories train stories to use (default: all ~10,000)"
    )
    p.add_argument("--ctx", type=int, default=96, help="Context/window length in tokens (default 96)")
    p.add_argument("--embed", type=int, default=96, help="Embedding dimension D (default 96)")
    p.add_argument("--layers", type=int, default=2, help="Number of transformer blocks (default 2)")
    p.add_argument("--heads", type=int, default=4, help="Number of attention heads (default 4)")
    p.add_argument("--num_batches", type=int, default=40, help="Training steps = number of batches (default 40)")
    p.add_argument("--batch_size", type=int, default=8, help="Windows per batch (default 8)")
    p.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate (default 3e-3)")
    p.add_argument("--max_norm", type=float, default=1.0, help="Gradient L2 clip norm (default 1.0; 0 = off)")
    p.add_argument("--seed", type=int, default=42, help="Seed for data sampling + weight init (default 42)")
    p.add_argument("--out", type=str, default="resource/models", help="Output directory (default resource/models)")
    p.add_argument(
        "--suffix", type=str, default="real", help="Checkpoint dir suffix: {backend}_{suffix} (default 'real')"
    )
    p.add_argument(
        "--backends", type=str, default=",".join(BACKENDS), help="Comma-separated backends (default: all four)"
    )
    return p


def load_dataset(
    num_stories: int | None, ctx: int, batch_size: int, num_batches: int, seed: int
) -> tuple[Any, list[tuple[list[list[int]], list[list[int]]]], int]:
    """Load + tokenize TinyStories and sample the shared training windows.

    Returns (tokenizer, batches, n_tokens) where each batch is an
    (input_batch, target_batch) tuple: both are [batch_size][ctx] ints, and the
    target is the input shifted left by 1 (the pre-shifted next-token labels the
    unified cross-entropy contract expects).
    """
    stories = load_tinystories("train", num_stories)
    tokenizer = create_tokenizer()
    dataset = TextDataset(stories, tokenizer, context_length=ctx, seed=seed)  # type: ignore[reportArgumentType]
    batches = dataset.get_sequences(num_batches=num_batches, batch_size=batch_size)
    return tokenizer, batches, len(dataset.token_ids)


def _build_model(backend: str, cfg: TransformerConfig) -> Any:
    """Construct a fresh (CPU, seed-initialized) model for a backend."""
    if backend == "numpy":
        from impl._np.model import NumPyModel

        return NumPyModel(cfg)
    if backend == "torch":
        from impl._torch.layers import TorchModel

        return TorchModel(cfg)
    if backend == "triton":
        from impl._triton.model import TritonModel

        return TritonModel(cfg)
    from impl._cuda.model import CUDAModel

    return CUDAModel(cfg)


def _enable_cuda_grads(model: Any) -> None:
    """Set requires_grad on every CUDAModel tensor (blocks + the 3 top-level weights).

    ``load_from_numpy_dict`` reassigns the tensors to fresh CPU tensors, which lose
    requires_grad; the CUDA train_step needs gradients on all of them.
    """
    import torch

    for name in ("embedding_weights", "final_norm_gamma", "lm_head_weight"):
        getattr(model, name).requires_grad_(True)
    for block in model.stacking.blocks:
        for attr in block.__dict__.values():
            if isinstance(attr, torch.Tensor):
                attr.requires_grad_(True)


def _cuda_to_device(model: Any, device: Any) -> None:
    """Move every CUDAModel tensor to `device` by relocating its storage (``.data``).

    ``tensor.data = tensor.data.to(device)`` (not ``tensor = tensor.to(...)``) preserves
    leaf-ness and ``requires_grad``: a plain ``.to()`` on a grad-requiring tensor creates a
    non-leaf tensor that ``torch.optim.AdamW`` rejects.
    """
    import torch

    model.embedding_weights.data = model.embedding_weights.data.to(device)
    model.final_norm_gamma.data = model.final_norm_gamma.data.to(device)
    model.lm_head_weight.data = model.lm_head_weight.data.to(device)
    for block in model.stacking.blocks:
        for attr in block.__dict__.values():
            if isinstance(attr, torch.Tensor):
                attr.data = attr.data.to(device)


def _cuda_params(model: Any) -> list:
    """Collect every trainable torch tensor for the optimizer (CUDAModel has no .parameters())."""
    import torch

    params: list = [
        attr for block in model.stacking.blocks for attr in block.__dict__.values() if isinstance(attr, torch.Tensor)
    ]
    params += [model.embedding_weights, model.final_norm_gamma, model.lm_head_weight]
    return params


def _prepare(backend: str, model: Any, ref_params: dict, device: str | None) -> Any:
    """Load the identical NumPy init, then enable grads (cuda) / move to device."""
    model.load_from_numpy_dict(ref_params)
    if backend == "cuda":
        _enable_cuda_grads(model)
        if device is not None:
            _cuda_to_device(model, device)
    elif device is not None:  # torch / triton
        model = model.to(device)
    return model


def _export_params(backend: str, model: Any) -> dict[str, np.ndarray]:
    """All parameters under the shared key scheme (the format save_checkpoint writes)."""
    if backend in ("numpy", "cuda"):
        return model.get_all_parameters()
    return model.save_as_numpy()


def _train(
    backend: str, model: Any, batches: list, lr: float, max_norm: float, device: str | None
) -> tuple[float, float, float]:
    """Train `model` on `batches` (one train_step per batch); return (first, last, mean) loss."""
    if backend == "numpy":
        from impl._np.cross_entropy import CrossEntropyLoss
        from impl._np.optimizer import AdamW
        from impl._np.training import train_step

        # Targets are pre-shifted next-token labels, so no extra shift here.
        loss_fn = CrossEntropyLoss(shift=False)
        optimizer = AdamW(lr=lr)
        losses: list[float] = []
        for batch_input, batch_target in batches:
            x = np.asarray(batch_input, dtype=np.int32)  # (B, S) token IDs
            t = np.asarray(batch_target, dtype=np.int32)  # (B, S) pre-shifted labels
            losses.append(float(train_step(model, x, t, loss_fn, optimizer, max_norm)))
        return losses[0], losses[-1], float(np.mean(losses))

    # torch / triton / cuda share the (model, x, t, optimizer, loss_fn, max_norm) signature.
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if backend == "cuda":
        from impl._cuda.training import train_step

        params = _cuda_params(model)
    elif backend == "triton":
        from impl._triton.training import train_step

        params = list(model.parameters())
    else:
        from impl._torch.training import train_step

        params = list(model.parameters())

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params, lr=lr)
    losses = []
    for batch_input, batch_target in batches:
        x = torch.tensor(batch_input, dtype=torch.int32, device=device)  # (B, S)
        t = torch.tensor(batch_target, dtype=torch.long, device=device)  # (B, S)
        losses.append(float(train_step(model, x, t, optimizer, loss_fn, max_norm)))
    return losses[0], losses[-1], float(np.mean(losses))


def _cuda_device() -> str | None:
    """Return 'cuda:0' if a GPU is available, else None (numpy/torch fall back to CPU)."""
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - no torch installed
        return None


def main() -> int:
    """Run the 4-backend TinyStories training reproduction. Returns 0 on success."""
    args = build_parser().parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in backends if b not in BACKENDS]
    if unknown:
        print(f"Unknown backend(s): {unknown}. Valid: {list(BACKENDS)}", file=sys.stderr)
        return 2

    tokenizer, batches, n_tokens = load_dataset(
        args.num_stories, args.ctx, args.batch_size, args.num_batches, args.seed
    )
    if not batches:
        print("No training windows produced — check the dataset.", file=sys.stderr)
        return 1
    cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.ctx,
        embed_dim=args.embed,
        n_layers=args.layers,
        n_heads=args.heads,
        rope_dim=0,
        n_experts=1,  # dense SwiGLU FFN (the original checkpoints are dense)
        top_k=1,
        seed=args.seed,
    )
    device = _cuda_device()
    print(
        f"Dataset: {len(batches)} steps x {args.batch_size} windows x ctx {args.ctx} | "
        f"token stream {n_tokens:,} | vocab {cfg.vocab_size:,} | device {device or 'cpu'}"
    )

    # Reference weights: every backend trains from this exact (seed-initialized) NumPy set,
    # captured as a copy before any training mutates the arrays in place.
    ref_model = _build_model("numpy", cfg)
    ref_params = {k: np.asarray(v).copy() for k, v in ref_model.get_all_parameters().items()}

    out_root = Path(args.out)
    print()
    for backend in backends:
        if backend in ("triton", "cuda") and device is None:
            print(f"[skip] {backend}: requires a CUDA GPU (none available)")
            continue
        t0 = time.time()
        model = ref_model if backend == "numpy" else _prepare(backend, _build_model(backend, cfg), ref_params, device)
        first, last, mean = _train(backend, model, batches, args.lr, args.max_norm, device)
        out_dir = out_root / f"{backend}_{args.suffix}"
        save_checkpoint(out_dir, config=cfg, params=_export_params(backend, model))
        print(
            f"{backend:>7}: loss {first:8.4f} -> {last:8.4f} (mean {mean:8.4f})  {time.time() - t0:6.1f}s  -> {out_dir}"
        )

    print()
    print(
        f"Done. Checkpoints under {out_root}/  (each ~{cfg.vocab_size * cfg.embed_dim * 4 * 2 / 1e6:.0f} MB, dominated by embedding + lm_head)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
