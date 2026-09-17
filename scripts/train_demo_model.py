"""Train and export the learning-mode demo model.

The demo model is the default model for the learning-mode web page
(`impl/_np/cli.py --learning`). It is deliberately tiny (D=8, H=4, L=3,
E=3 MoE, char-level vocab V=20) so that every intermediate tensor is small
enough to display and inspect on the page.

Vocabulary: the top-19 most frequent letters of the TinyStories corpus plus
a space token. Out-of-vocabulary characters are dropped from the training
stream (the model therefore trains on the in-vocab sub-sequence).

Backends (all start from the SAME NumPy fp64 init and train on the SAME seeded
random windows, so the resulting weights agree across backends):
    numpy   (default)  pure NumPy, float64, analytic backward -- the learning-math
                       reference; runs on CPU.
    torch / triton / cuda  the GPU backends. This app targets Jetson AGX / GB10
                       (both NVIDIA + CUDA), so they run on cuda:0 by default;
                       override with --device (e.g. --device cpu) for testing.

Training: AdamW, lr=3e-3, gradient clipping 1.0, pre-shifted LM targets.

Export: the standard checkpoint format (config.json + model.npz via
shared.checkpoint) plus a ``vocab.json`` sidecar (the token string list) so
that the learning server can encode user input and decode output. Any backend
can load the result via `shared.load_checkpoint`.

Usage:
    uv run python -m scripts.train_demo_model                 # numpy, CPU
    uv run python -m scripts.train_demo_model --backend cuda  # GPU
    uv run python -m scripts.train_demo_model --backend torch --device cpu --steps 100
    uv run python -m scripts.train_demo_model --steps 3000 --out resource/models/learning_demo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from shared.checkpoint import save_checkpoint
from shared.config import TransformerConfig
from shared.dataset import load_tinystories

# --- Demo model hyperparameters (the "simple model" spec: 8 dim, 4 heads, 3 layers, 3 MoE experts) ---
DEFAULT_EMBED_DIM = 8
DEFAULT_N_HEADS = 4
DEFAULT_N_LAYERS = 3
DEFAULT_N_EXPERTS = 3
DEFAULT_VOCAB_SIZE = 20  # 19 letters + space
DEFAULT_CONTEXT = 32
DEFAULT_OUT = "resource/models/learning_demo"

# Backends that train on a torch device (cuda:0 by default); numpy is always CPU.
_GPU_BACKENDS = ("torch", "triton", "cuda")


def build_char_vocab(text: str, vocab_size: int) -> list[str]:
    """Top-(vocab_size-1) most frequent letters, alphabetically sorted, + space.

    The sorted letter order (with space last) keeps the vocab table stable
    across runs, so the exported model is reproducible.
    """
    freq: dict[str, int] = {}
    for c in text:
        if c.isalpha():
            freq[c] = freq.get(c, 0) + 1
    top = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])[: vocab_size - 1]]
    return sorted(top) + [" "]


def build_stream(stories: list[str], vocab: list[str]) -> list[int]:
    """Flatten the corpus to in-vocab token IDs (OOV characters dropped)."""
    idx = {c: i for i, c in enumerate(vocab)}
    stream: list[int] = []
    for story in stories:
        for c in story.lower():
            if c in idx:
                stream.append(idx[c])
    return stream


def random_windows(stream: list[int], ctx: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n (x, y) windows with y pre-shifted: y[t] = x[t+1] (the LM contract)."""
    idx = rng.integers(0, len(stream) - ctx - 1, size=n)
    xs = np.array([stream[i : i + ctx] for i in idx], dtype=np.int32)
    ys = np.array([stream[i + 1 : i + 1 + ctx] for i in idx], dtype=np.int32)
    return xs, ys


def generate_sampled(logits: np.ndarray, rng: np.random.Generator, temp: float, top_k: int | None) -> int:
    """Sample one token with temperature scaling and optional top-k filtering."""
    z = logits.astype(np.float64) / temp
    if top_k is not None:
        kth = np.partition(z, -top_k)[-1]
        z = np.where(z < kth, -np.inf, z)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def _numpy_init(cfg: TransformerConfig) -> dict[str, np.ndarray]:
    """The identical fp64 init every backend trains from (the NumPy reference weights)."""
    from impl._np.model import NumPyModel

    return {k: np.asarray(v) for k, v in NumPyModel(cfg).get_all_parameters().items()}


def _export_params(model: Any) -> dict[str, np.ndarray]:
    """The model's parameters as a flat dict of numpy arrays (the checkpoint format)."""
    import torch

    out: dict[str, np.ndarray] = {}
    for key, value in model.get_all_parameters().items():
        out[key] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return out


def _cuda_params(model: Any) -> list:
    """Every trainable tensor for the CUDA optimizer (CUDAModel has no ``.parameters()``)."""
    import torch

    params: list = [
        attr for block in model.stacking.blocks for attr in block.__dict__.values() if isinstance(attr, torch.Tensor)
    ]
    params += [model.embedding_weights, model.final_norm_gamma, model.lm_head_weight]
    return params


def _cuda_to_device(model: Any, device: Any) -> None:
    """Move every CUDAModel tensor to `device` by relocating its storage (``.data``).

    ``tensor.data = tensor.data.to(device)`` (not ``tensor = tensor.to(...)``) preserves
    leaf-ness and ``requires_grad``: a plain ``.to()`` on a grad-requiring tensor creates a
    non-leaf tensor that ``torch.optim.AdamW`` rejects.
    """
    model.embedding_weights.data = model.embedding_weights.data.to(device)
    model.final_norm_gamma.data = model.final_norm_gamma.data.to(device)
    model.lm_head_weight.data = model.lm_head_weight.data.to(device)
    for block in model.stacking.blocks:
        for attr in block.__dict__.values():
            if isinstance(attr, _cuda_tensor_type()):
                attr.data = attr.data.to(device)


def _cuda_tensor_type() -> type:
    """``torch.Tensor`` (imported lazily so the default numpy path never loads torch)."""
    import torch

    return torch.Tensor


def _enable_cuda_grads(model: Any) -> None:
    """Enable gradients on every CUDAModel tensor so autograd builds a graph through the kernels."""
    torch_tensor = _cuda_tensor_type()
    for block in model.stacking.blocks:
        for attr in block.__dict__.values():
            if isinstance(attr, torch_tensor):
                attr.requires_grad_(True)
    for name in ("embedding_weights", "final_norm_gamma", "lm_head_weight"):
        getattr(model, name).requires_grad_(True)


def _train_numpy(cfg: TransformerConfig, stream: list[int], args: argparse.Namespace) -> dict[str, np.ndarray]:
    """NumPy (CPU) training -- the learning-math reference (fp64, analytic backward)."""
    from impl._np.cross_entropy import CrossEntropyLoss
    from impl._np.model import NumPyModel
    from impl._np.optimizer import AdamW
    from impl._np.training import train_step

    model = NumPyModel(cfg)
    opt = AdamW(lr=args.lr)
    ce = CrossEntropyLoss(shift=False)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    loss = float("nan")
    for step in range(args.steps):
        xs, ys = random_windows(stream, args.ctx, args.batch, rng)
        loss = float(train_step(model, xs, ys, ce, opt, max_norm=args.max_norm))
        if (step + 1) % 250 == 0:
            print(f"  step {step + 1}/{args.steps}  loss={loss:.4f}")
    print(f"  done: {args.steps} steps in {time.time() - t0:.0f}s, final loss={loss:.4f}")
    return _export_params(model)


def _train_gpu(
    backend: str, cfg: TransformerConfig, stream: list[int], args: argparse.Namespace
) -> dict[str, np.ndarray]:
    """Torch/Triton/CUDA (GPU) training -- loads the NumPy init, trains with autograd + AdamW."""
    import torch
    import torch.nn.functional as F

    ref = _numpy_init(cfg)
    if backend == "torch":
        from impl._torch.layers import TorchModel

        model: Any = TorchModel(cfg)
    elif backend == "triton":
        from impl._triton.model import TritonModel

        model = TritonModel(cfg)
    else:  # cuda
        from impl._cuda.model import CUDAModel

        model = CUDAModel(cfg)

    model.load_from_numpy_dict(ref)
    if backend == "cuda":
        _enable_cuda_grads(model)
        _cuda_to_device(model, args.device)
    else:
        model = model.to(args.device)

    params: list = list(model.parameters()) if hasattr(model, "parameters") else _cuda_params(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    loss_val = float("nan")
    for step in range(args.steps):
        xs, ys = random_windows(stream, args.ctx, args.batch, rng)
        xs_t = torch.from_numpy(xs).to(args.device)  # int32 -> embedding index
        ys_t = torch.from_numpy(ys).long().to(args.device)  # int64 -> cross-entropy target
        logits = model(xs_t)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ys_t.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.max_norm)
        optimizer.step()
        loss_val = float(loss)
        if (step + 1) % 250 == 0:
            print(f"  step {step + 1}/{args.steps}  loss={loss_val:.4f}")
    print(f"  done: {args.steps} steps in {time.time() - t0:.0f}s, final loss={loss_val:.4f}")
    return _export_params(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + export the learning-mode demo model")
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch", "triton", "cuda"],
        help="training backend (default numpy, runs on CPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for torch/triton/cuda (default cuda:0); ignored for numpy",
    )
    parser.add_argument("--steps", type=int, default=1500, help="training steps (16 windows each)")
    parser.add_argument("--batch", type=int, default=16, help="windows per step")
    parser.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="gradient clip norm")
    parser.add_argument(
        "--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE, help="char vocab size (19 letters + space)"
    )
    parser.add_argument("--embed_dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--n_experts", type=int, default=DEFAULT_N_EXPERTS)
    parser.add_argument("--top_k", type=int, default=1, help="MoE top-k experts per token")
    parser.add_argument("--ctx", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="checkpoint output dir")
    args = parser.parse_args()

    print("=== 1. Corpus + vocab ===")
    stories = load_tinystories("train")
    text = "\n".join(stories)
    vocab = build_char_vocab(text, args.vocab_size)
    stream = build_stream(stories, vocab)
    print(f"stories: {len(stories)}  vocab={len(vocab)}: {vocab}")
    print(f"stream: {len(stream):,} in-vocab tokens")

    cfg = TransformerConfig(
        vocab_size=len(vocab),
        context_length=args.ctx,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        rope_dim=0,
        n_experts=args.n_experts,
        top_k=args.top_k,
        seed=args.seed,
    )

    device_desc = "cpu" if args.backend == "numpy" else args.device
    print(f"\n=== 2. Train (backend={args.backend}, device={device_desc}) ===")
    if args.backend in _GPU_BACKENDS:
        params = _train_gpu(args.backend, cfg, stream, args)
    else:
        params = _train_numpy(cfg, stream, args)

    print("\n=== 3. Export (checkpoint + vocab.json) ===")
    out = Path(args.out)
    save_checkpoint(str(out), cfg, params)
    out.joinpath("vocab.json").write_text(json.dumps(vocab))
    print(f"saved {out}/ (config.json, model.npz, vocab.json)")

    print("\n=== 4. Sample output (sanity: should be word-like, not random) ===")
    from impl._np.model import NumPyModel

    np_model = NumPyModel(cfg)
    np_model.load_from_numpy_dict(params)
    rng2 = np.random.default_rng(args.seed)
    prompt = [vocab.index(c) for c in "the she"]
    seq = list(prompt)
    for _ in range(150):
        logits = np_model.forward(np.array([seq[-args.ctx :]], dtype=np.int32))[0, -1]
        seq.append(generate_sampled(logits, rng2, temp=0.8, top_k=10))
    print(f"prompt {''.join(vocab[t] for t in prompt)!r}")
    print(f"sampled (T=0.8, top-k=10): {''.join(vocab[t] for t in seq[len(prompt) :])[:200]!r}")


if __name__ == "__main__":
    main()
