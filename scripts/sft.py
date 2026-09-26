"""Training pipeline entry point: pre → SFT (all guided by config + data).

The single user-facing command. ``--stage`` mirrors the real model-training
pipeline:

    uv run python -m scripts.sft --stage prepost --data resource/code_instructions.json

Stages
------
  pre    — pre-training on plain text (existing behavior via train.py)
  post   — SFT only, starting from a pre-trained checkpoint
  prepost — both: pre-train on tinystories, then SFT on the given JSON

SFT data to train on:

    uv run python -m scripts.sft --backend numpy --dataset resource/code_instructions.json
    uv run python -m scripts.sft --backend numpy --dataset resource/tool_calls.json
    uv run python -m scripts.sft --backend torch --dataset resource/tool_calls.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from shared.config import TransformerConfig
from shared.sft_data import SFTLoader

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resource"


def _load_numpy():
    from impl._np.model import NumPyModel
    from impl._np.sft import AdamW, sft_epoch

    return NumPyModel, AdamW, sft_epoch


def _load_torch():
    from impl._torch.layers import TorchModel
    from impl._torch.sft import TorchSFTTrainer

    return TorchModel, TorchSFTTrainer, None


def _load_triton():
    from impl._triton.model import TritonModel
    from impl._triton.sft import TritonSFTTrainer

    return TritonModel, TritonSFTTrainer, None


def _load_cuda():
    from impl._cuda.model import CUDAModel
    from impl._cuda.sft import CudaSFTTrainer

    return CUDAModel, CudaSFTTrainer, None


_BACKENDS = {
    "numpy": _load_numpy,
    "torch": _load_torch,
    "triton": _load_triton,
    "cuda": _load_cuda,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(_BACKENDS), required=True)
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="SFT dataset JSON (code_instructions.json or tool_calls.json)",
    )
    ap.add_argument("--model", type=Path, default=None, help="Checkpoint dir (input; skip pre-train)")
    ap.add_argument("--save-dir", type=Path, default=None, help="Save the post-trained model here")
    ap.add_argument("--stage", choices=["pre", "post", "prepost"], default="prepost")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    tokenizer = Tokenizer.from_file(str(RESOURCE_DIR / "bpe_tokenizer.json"))
    cfg = TransformerConfig.from_dict(
        {
            "vocab_size": tokenizer.get_vocab_size(),
            "embed_dim": 32,
            "n_layers": 2,
            "n_heads": 4,
            "context_length": args.max_len,
            "seed": 42,
        }
    )
    loader = SFTLoader(tokenizer, max_len=args.max_len)
    make_model, make_trainer, sft_epoch_fn = _BACKENDS[args.backend]()
    model = make_model(cfg)

    # Stage "pre": pre-training on the tinystories corpus (if none exists at
    # --model, run the regular pre-training pipeline to build one).
    if args.stage in ("pre", "prepost") and args.model is None:
        import subprocess

        save_dir = RESOURCE_DIR / "models" / f"{args.backend}_real"
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "scripts.train",
                "--backend",
                args.backend,
                "--synthetic",
                "--save_dir",
                str(save_dir),
            ],
            check=True,
        )
        args.model = save_dir
        print(f"[pre] pretrained → {save_dir}")

    if args.model is not None:
        params = dict(np.load(args.model / "model.npz"))
        model.load_from_numpy_dict(params)

    rows = loader.load_sft_data(args.dataset)
    b = loader.batch(rows)

    if args.backend == "numpy":
        # The Per-example batches stay NumPy/teaching shaped; each epoch
        # or run via sft_epoch considers all of them.
        np_rows = [(b.input_ids[i], b.target_ids[i]) for i in range(len(b.input_ids))]
        bs = 16
        for epoch in range(args.epochs):
            for j in range(0, len(np_rows), bs):
                loss = sft_epoch_fn(model, np_rows[j : j + bs], lr=args.lr)
                print(f"epoch {epoch + 1} batch {j // bs + 1} loss={loss:.4f}")
        _flush(model, args.save_dir, cfg)
    else:
        import torch

        trainer = make_trainer(model)
        x = torch.tensor(b.input_ids, dtype=torch.int64)
        y = torch.tensor(b.target_ids, dtype=torch.int64)
        m = torch.tensor(b.response_mask, dtype=torch.int64)
        bs = 16
        for epoch in range(args.epochs):
            losses = []
            for j in range(0, len(x), bs):
                losses.append(trainer.train_step(x[j : j + bs], y[j : j + bs], m[j : j + bs]))
            print(f"epoch {epoch + 1} mean_loss={float(np.mean(losses)):.4f}")
        _flush(model, args.save_dir, cfg)


def _flush(model, save_dir, cfg) -> None:
    if save_dir is None:
        return
    params = model.get_all_parameters() if hasattr(model, "get_all_parameters") else model.save_as_numpy()
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / "model.npz", **params)
    (save_dir / "config.json").write_text(json.dumps(cfg.to_dict()))


if __name__ == "__main__":
    main()
