#!/usr/bin/env python3
# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Learning mode — host the interactive transformer visualization page.

Served checkpoints run per-block mixers (ADR 0007,
``n_deltanet_sublayers=3``): every block contains 3 linear-attention mixers
(Gated DeltaNet, recurrent S state) + 1 full RoPE attention mixer. The page
renders a ``deltanet``/``attn`` module per mixer (projections, forget/output
gates, the per-position S matrices, the read-out) and labels each mixer's
cache node "S state" or "K/V cache". BF16 is the running serving dtype on the
backends (the page's records recompute the displayed math in float64);
NumPy computes the reference in float64/float32.

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

- **moe-rl** (default) — ``learning_moe_rl``: the full v7 pipeline
  (pre-train → code SFT → tool SFT → thinking SFT → multi-turn agentic
  GRPO RL (ADR 0010) → drafter distillation) on the MoE of ADR 0005 with
  SWA w=32 (ADR 0008) and group-of-2 alternating MLA/MHA global mixers
  (ADRs 0009/0011, L=16): 4 layers with E=10 routed + 1 always-on shared
  expert at dim 20 — **543K** (router scores, top-1 selection, and the
  shared branch render per block; even blocks show the latent c cache,
  odd blocks the K/V cache).
- **dense-rl** — ``learning_rl``: the same five stages on the dense
  TWIN: 8 layers with a thin FFN (dim 48) — **752K** all-active. Same
  MLA/MHA interleave as the MoE (both use `mla_group_size=2`), so
  moe-rl vs dense-rl compares the FFN geometry at matched interleave.
- **moe** — ``learning_moe_think``: the MoE family's SFT-final
  checkpoint (543K, everything except RL), so *-rl vs * isolates
  RL's delta (RL started from exactly this model — controlled).
- **dense** — ``learning_think``: the dense family's SFT-final (752K).

Either checkpoint runs on any backend selected in-page, and every
variant ships its own drafter sidecars (distilled per checkpoint in
stage 6 of the v7 pipeline), so mtp/dspark speculative decoding works
on RL-trained AND SFT-trained models of both geometries.

This is a PURE SERVING script - it loads pre-trained checkpoints and
serves them. Training happens separately via scripts.train_demo_model_v3
(run before serving). Missing checkpoints cause a clear error telling
the user to run the training pipeline first.

Dtype note: the served checkpoints carry ``dtype: bfloat16`` in config.json
because the GPU tracks (torch/triton/cuda) TRAIN in BF16 — AMP matmuls on the
GPU tensor cores (~2x FP32 throughput on Ampere+). The weights themselves are
stored FP32 in model.npz, and this page SERVES every backend in float32
(load_learning_model casts torch/triton/cuda models to float32), so the
in-page records (float64 display path) and the float32 drafter sidecars all
share one dtype. The NumPy backend computes the reference math in
float64/float32 and never uses BF16. The page header shows a dtype badge
explaining exactly this.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from impl._np import learning_server  # noqa: E402
from shared.utils.logger_setup import setup_logging  # noqa: E402

DEFAULT_MODEL = "resource/models/learning_moe_rl"

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
    args = parser.parse_args()

    variant_dirs = dict(VARIANTS) if args.model == DEFAULT_MODEL else {}

    if not _port_free(args.host, args.port):
        print(f"error: port {args.port} is already in use (override with --port)", file=sys.stderr)
        return 1

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


if __name__ == "__main__":
    raise SystemExit(main())
