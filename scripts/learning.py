#!/usr/bin/env python
# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Learning mode — host the interactive transformer visualization page.

Serves the learning page (architecture diagram with the actual numbers at
every step, click-to-inspect math, inference-record download) plus its JSON
API over a loaded checkpoint. The NumPy track serves records by default;
``--backend torch`` materializes the same checkpoint in PyTorch — the page
consumes either backend's records because their JSON shape is identical.

Usage:
    uv run python -m scripts.learning                          # demo model, port 8080
    uv run python -m scripts.learning --backend torch          # PyTorch track
    uv run python -m scripts.learning --model resource/models/torch_real --port 9000
    uv run python -m scripts.learning --host 0.0.0.0           # allow LAN access

The page's Model dropdown switches the SERVED CHECKPOINT (not just the
backend): "moe" (default — the learning_demo checkpoint with routed + always-on
shared experts, so the diagram shows the router/top-k/shared cells) and
"dense" (the learning_tool checkpoint, plain SwiGLU FFN). Either checkpoint
runs on any backend selected in-page.

If the default demo checkpoint is missing it is trained first (a few minutes).
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from impl._np import learning_server  # noqa: E402
from shared.utils.logger_setup import setup_logging  # noqa: E402

DEFAULT_MODEL = "resource/models/learning_tool"

# Model variants the page can switch between (the in-page Model dropdown):
#   moe   — the char-level learning_demo checkpoint (E routed experts +
#           always-on shared experts): the diagram shows the router, the
#           top-k selection, the renormalized weights, and the shared cell.
#   dense — the default learning_tool checkpoint (plain SwiGLU FFN).
# MoE is the DEFAULT so the page opens with the richer architecture.
VARIANTS = {
    "moe": "resource/models/learning_demo",
    "dense": "resource/models/learning_tool",
}
DEFAULT_VARIANT = "moe"


DEFAULT_COMPARE = [
    ("base", "resource/models/learning_base"),
    ("sft", "resource/models/learning_sft"),
]


def _port_free(host: str, port: int) -> bool:
    """Return True if (host, port) can be bound right now."""
    with socket.socket() as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _ensure_demo_model(model_dir: str) -> None:
    """Train the demo-model pipeline (train_demo_model_v2) if the checkpoint
    does not exist yet — it writes resource/models/{learning_base,
    learning_sft,learning_tool} in one run."""
    if (Path(model_dir) / "model.npz").is_file():
        return
    print(f"demo checkpoint not found at {model_dir} — training the demo pipeline first (a few minutes)...")
    subprocess.run([sys.executable, "-m", "scripts.train_demo_model_v2"], check=True)


def _ensure_moe_demo(model_dir: str) -> None:
    """Train the MoE demo checkpoint (train_demo_model) if missing — the
    page's default variant needs it (router + top-k + shared experts)."""
    if (Path(model_dir) / "model.npz").is_file():
        return
    print(f"MoE demo checkpoint not found at {model_dir} — training it first (a few minutes)...")
    subprocess.run([sys.executable, "-m", "scripts.train_demo_model"], check=True)


def _ensure_drafters(model_dir: str) -> None:
    """Bootstrap the speculative-decoding drafter sidecars on first run
    (ADR 0003): the learning page ships with MTP active by default, so the
    sidecars are part of the demo — trained by distillation from the frozen
    target (a few minutes on GPU)."""
    from shared.draft import DSHARK, MTP, sidecar_dir

    if all((sidecar_dir(model_dir, f) / "draft.npz").is_file() for f in (MTP, DSHARK)):
        return
    print("drafter sidecars not found — distilling MTP + DSpark first (a few minutes)...")
    subprocess.run([sys.executable, "-m", "scripts.train_drafters", "--model", model_dir], check=True)


def _ensure_variant_drafters(variant_dirs: dict[str, str]) -> None:
    """Every switchable checkpoint needs its OWN drafter sidecars: the
    drafters are distilled from (and verify against) their target, so the
    moe variant cannot reuse the dense model's sidecars. Ensures each
    registered variant has draft_mtp/ + draft_dspark/."""
    for vdir in variant_dirs.values():
        _ensure_drafters(vdir)


def _resolve_spec(model_dir: str, cli_spec: str) -> str:
    """Resolve the startup spec mode: an explicit --spec wins; ``auto``
    picks mtp when the sidecars exist, plain otherwise (the trained demo
    ships speculative; plain is the comparison mode)."""
    if cli_spec != "auto":
        return cli_spec
    from shared.draft import MTP, sidecar_dir

    return "mtp" if (sidecar_dir(model_dir, MTP) / "draft.npz").is_file() else "plain"


def _variant_registry(model_dir: str) -> dict[str, str]:
    """The page's switchable checkpoints: moe (router + top-k + shared
    experts, the DEFAULT so the page opens with the richer architecture) and
    dense (plain SwiGLU). Registered only when serving the default model
    family; a custom --model serves a single checkpoint (dropdown hidden)."""
    if model_dir != DEFAULT_MODEL:
        return {}
    for label, vdir in VARIANTS.items():
        if label == "moe":
            _ensure_moe_demo(vdir)
    return dict(VARIANTS)


def main() -> int:
    """Entry point — parse arguments, load the model, serve forever."""
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="learning.py",
        description="Host the learning-mode web page (interactive transformer visualization).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default 8080)")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Interface to bind (default 0.0.0.0, all interfaces — allows local-network access)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Track to materialize the model (default numpy)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Checkpoint directory (default: the SFT learning_tool checkpoint)",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="auto",
        choices=["auto", "plain", "mtp", "dspark"],
        help="Decoding mode the page starts in (auto: mtp when drafter sidecars exist, else plain)",
    )
    parser.add_argument(
        "--compare",
        action="append",
        metavar="label=path",
        default=[],
        help="Register a named secondary model for the compare tab (default: the two-stage SFT demo models)",
    )
    args = parser.parse_args()

    default_compare = list(DEFAULT_COMPARE)
    if args.compare:
        specs = []
        for arg in args.compare:
            if "=" in arg:
                label, path = arg.split("=", 1)
                specs.append((label, path))
            else:
                specs.append((f"m{len(specs)}", arg))
        default_compare.extend(specs)
    args.compare = default_compare

    if args.model == DEFAULT_MODEL:
        _ensure_demo_model(args.model)
    variant_dirs = _variant_registry(args.model)
    _ensure_drafters(args.model)
    _ensure_variant_drafters(variant_dirs)
    spec = _resolve_spec(args.model, args.spec)
    if not _port_free(args.host, args.port):
        print(f"error: port {args.port} is already in use (override with --port)", file=sys.stderr)
        return 1

    model, vocab, _cfg, _tok = learning_server.load_learning_model(args.model, backend=args.backend)
    # compare tab: register additional named (pre-trained + fine-trained) models
    compare_models = {}
    compare_vocabs = {}
    compare_tokens = {}
    for label, mpath in args.compare:
        try:
            m, v, _, tok = learning_server.load_learning_model(mpath, backend=args.backend)
        except (OSError, ValueError) as e:
            print(f"compare model {label}={mpath} skipped: {e}")
            continue
        compare_models[label] = m
        compare_vocabs[label] = v
        compare_tokens[label] = tok
    server = learning_server.start_server(
        model,
        vocab,
        host=args.host,
        port=args.port,
        backend=args.backend,
        models=compare_models,
        vocabs=compare_vocabs,
        tokenizer=_tok,
        tokenizers=compare_tokens,
        model_dir=args.model,
        default_spec=spec,
        variant_dirs=variant_dirs or None,
        default_variant=DEFAULT_VARIANT if variant_dirs else "dense",
    )
    print(
        f"Learning mode: listening on {args.host}:{args.port}  "
        f"(model: {args.model}, backend: {args.backend}, spec: {spec})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
