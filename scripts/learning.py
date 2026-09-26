#!/usr/bin/env python
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

If the default demo checkpoint is missing it is trained first (~100 s).
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
    """Train the learning-demo checkpoint if it does not exist yet."""
    if (Path(model_dir) / "model.npz").is_file():
        return
    print(f"demo checkpoint not found at {model_dir} — training it first (~100s)...")
    subprocess.run([sys.executable, "-m", "scripts.train_demo_model"], check=True)


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
        help="Interface to bind (default 127.0.0.1; 0.0.0.0 allows local-network access)",
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
        help="Checkpoint directory (default: the learning demo model)",
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
    if not _port_free(args.host, args.port):
        print(f"error: port {args.port} is already in use (override with --port)", file=sys.stderr)
        return 1

    model, vocab, _cfg = learning_server.load_learning_model(args.model, backend=args.backend)
    # compare tab: register additional named (pre-trained + fine-trained) models
    compare_models = {}
    compare_vocabs = {}
    for label, mpath in args.compare:
        try:
            m, v, _ = learning_server.load_learning_model(mpath, backend=args.backend)
        except (OSError, ValueError) as e:
            print(f"compare model {label}={mpath} skipped: {e}")
            continue
        compare_models[label] = m
        compare_vocabs[label] = v
    server = learning_server.start_server(
        model,
        vocab,
        host=args.host,
        port=args.port,
        backend=args.backend,
        models=compare_models,
        vocabs=compare_vocabs,
    )
    print(f"Learning mode: listening on {args.host}:{args.port}  (model: {args.model}, backend: {args.backend})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
