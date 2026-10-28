#!/usr/bin/env python3
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
backend). Four variants, two controlled comparisons:

- **moe-rl** (default) — ``learning_moe_rl``: the full v3 pipeline
  (pre-train → code SFT → tool SFT → thinking SFT → GRPO RL → drafter
  distillation, ADR 0004) on the param-matched MoE of ADR 0005: 3 layers
  with E=8 routed + 1 always-on shared expert at dim 32 (282,560 total
  params, 153,536 active/token — the router scores, top-1 selection,
  and shared branch render per block).
- **dense-rl** — ``learning_rl``: the same five stages on the dense
  TWIN: 6 layers with a thin FFN (dim 96) — total-matched to the MoE
  (275,264 vs 282,560, the Qwen3-32B-deeper-than-30B-A3B shape), so
  moe-rl vs dense-rl compares the FFN geometry at matched budget.
- **moe** — ``learning_moe_think``: the MoE family's SFT-final
  checkpoint (everything except RL), so *-rl vs * isolates RL's delta
  (RL started from exactly this model — the comparison is controlled).
- **dense** — ``learning_think``: the dense family's SFT-final.

Either checkpoint runs on any backend selected in-page.

Bootstrap: the DEFAULT family (moe) trains at startup if missing or
stale (``"pipeline": 4`` marker in config.json — older checkpoints,
including the 3-layer dense pair, retrain); the dense family trains in
a BACKGROUND thread started right after the server comes up, so the
page is usable while the second family builds — switching to a
still-training variant answers "training, retry shortly" instead of
hanging the page. One family is six stages (~10-15 min on the Jetson:
the GRPO stage dominates).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from impl._np import learning_server  # noqa: E402
from shared.utils.logger_setup import setup_logging  # noqa: E402

DEFAULT_MODEL = "resource/models/learning_moe_rl"

# The demo-pipeline marker (ADRs 0004/0005): checkpoints whose
# config.json lacks this (or carries an older one) are stale and retrain.
PIPELINE_MARKER = 4

# Model variants the page can switch between (the in-page Model dropdown):
#   moe-rl   — learning_moe_rl: the FULL pipeline (pre → SFT-code →
#              SFT-tool → SFT-think → GRPO RL → drafters) on the
#              param-matched MoE (ADR 0005). The DEFAULT: the page opens
#              on the final model with the richest architecture.
#   dense-rl — learning_rl: the full pipeline, dense FFN.
#   moe      — learning_moe_think: the MoE family at SFT-final (pre-RL),
#              the exact starting point of the moe-rl GRPO run.
#   dense    — learning_think: the dense family at SFT-final.
# The two comparisons are both controlled: *-rl vs * isolates RL (same
# start), moe vs dense isolates the ARCHITECTURE split (depth vs expert
# bank, same total budget) at BOTH levels.
VARIANTS = {
    "moe-rl": "resource/models/learning_moe_rl",
    "dense-rl": "resource/models/learning_rl",
    "moe": "resource/models/learning_moe_think",
    "dense": "resource/models/learning_think",
}
DEFAULT_VARIANT = "moe-rl"


def _port_free(host: str, port: int) -> bool:
    """Return True if (host, port) can be bound right now."""
    with socket.socket() as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _checkpoint_stale(model_dir: str) -> bool:
    """True when the checkpoint is missing or pre-v4 (no/older marker).

    Older checkpoints (the 3-layer dense pair, the v2-era geometry)
    break the param-parity story the page tells — serving one silently
    would show two SAME-SHAPE models, so staleness retrains instead."""
    cfg_path = Path(model_dir) / "config.json"
    if not (Path(model_dir) / "model.npz").is_file() or not cfg_path.is_file():
        return True
    try:
        return int(json.loads(cfg_path.read_text()).get("pipeline", 0)) < PIPELINE_MARKER
    except (ValueError, KeyError):
        return True


def _train_family(variant: str) -> None:
    """Train one full family (dense or moe) with the v3 pipeline."""
    family = "moe" if variant.startswith("moe") else "dense"
    subprocess.run([sys.executable, "-m", "scripts.train_demo_model_v3", "--variant", family], check=True)


def _ensure_drafters(model_dir: str) -> None:
    """Bootstrap the speculative-decoding drafter sidecars on first run
    (ADR 0003): the learning page ships with MTP active by default, so
    the sidecars are part of the demo — trained by distillation from the
    frozen target (a few minutes)."""
    from shared.draft import DSHARK, MTP, sidecar_dir

    if all((sidecar_dir(model_dir, f) / "draft.npz").is_file() for f in (MTP, DSHARK)):
        return
    print("drafter sidecars not found — distilling MTP + DSpark first (a few minutes)...")
    subprocess.run([sys.executable, "-m", "scripts.train_drafters", "--model", model_dir], check=True)


def _resolve_spec(model_dir: str, cli_spec: str) -> str:
    """Resolve the startup spec mode: an explicit --spec wins; ``auto``
    picks mtp when the sidecars exist, plain otherwise (the trained demo
    ships speculative; plain is the comparison mode)."""
    if cli_spec != "auto":
        return cli_spec
    from shared.draft import MTP, sidecar_dir

    return "mtp" if (sidecar_dir(model_dir, MTP) / "draft.npz").is_file() else "plain"


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
        help="Checkpoint directory (default: the RL learning_moe_rl checkpoint)",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="auto",
        choices=["auto", "plain", "mtp", "dspark"],
        help="Decoding mode the page starts in (auto: mtp when drafter sidecars exist, else plain)",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Do not train missing/stale checkpoints (fail on missing instead)",
    )
    args = parser.parse_args()

    if args.model == DEFAULT_MODEL:
        if _checkpoint_stale(args.model) and not args.no_bootstrap:
            print(
                f"checkpoint {args.model} missing or stale — training the moe family first "
                f"(6 stages: pre → code → tool → think → GRPO RL → drafters; ~10-15 min on the Jetson)..."
            )
            _train_family(DEFAULT_VARIANT)
        elif _checkpoint_stale(args.model):
            print(f"error: checkpoint {args.model} missing/stale and --no-bootstrap given", file=sys.stderr)
            return 1
    variant_dirs = dict(VARIANTS) if args.model == DEFAULT_MODEL else {}

    if not _port_free(args.host, args.port):
        print(f"error: port {args.port} is already in use (override with --port)", file=sys.stderr)
        return 1

    _ensure_drafters(args.model)

    if variant_dirs and not args.no_bootstrap:
        _start_background_bootstrap(variant_dirs)

    # Drafters for the SERVE-ready variants the pipeline actually distills
    # for (the *-rl checkpoints — ADR 0003's per-target contract). The
    # SFT-final variants (moe / dense) ship without sidecars on purpose:
    # they exist to show RL's delta, not decoding speed, and the page
    # falls back to plain decoding for them with a warning (never a hard
    # failure). This loop is the safety net for a deleted sidecar.
    for label, vdir in variant_dirs.items():
        if not _checkpoint_stale(vdir) and label.endswith("-rl"):
            _ensure_drafters(vdir)

    model, vocab, _cfg, _tok = learning_server.load_learning_model(args.model, backend=args.backend)
    spec = _resolve_spec(args.model, args.spec)
    server = learning_server.start_server(
        model,
        vocab,
        host=args.host,
        port=args.port,
        backend=args.backend,
        tokenizer=_tok,
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


def _start_background_bootstrap(variant_dirs: dict[str, str]) -> None:
    """Train the OTHER family (the one the startup model is not in) in a
    daemon thread, started right after the server is up — the first run
    stays "a few minutes" (the default family is ready), and a page
    switch to a still-training variant gets the server's "training,
    retry shortly" answer instead of a hung request."""
    default_family = DEFAULT_VARIANT.split("-")[0]  # "moe" or "dense"
    other = [v for v in variant_dirs if not v.startswith(default_family)]
    pending = [v for v in other if _checkpoint_stale(variant_dirs[v])]
    if not pending:
        return
    print(f"background: training the other family for variants {pending} (the page stays usable)...")

    def _bg() -> None:
        try:
            _train_family(pending[0])
        except Exception as exc:  # pragma: no cover — best-effort
            print(f"background family training failed: {exc}", file=sys.stderr)

    threading.Thread(target=_bg, daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
