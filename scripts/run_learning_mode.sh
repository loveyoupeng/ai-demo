#!/usr/bin/env bash
# Run the learning-mode web page (NumPy track).
#
# Serves the interactive page — prompt + generation + architecture
# visualization with per-block numbers + inference-record download — at
# http://127.0.0.1:8080 by default.
#
# Usage:
#   bash scripts/run_learning_mode.sh                  # default: port 8080, demo model
#   bash scripts/run_learning_mode.sh --port 9000      # extra args pass through
#   bash scripts/run_learning_mode.sh --model resource/models/some_ckpt
#
# If the demo checkpoint is missing it is trained first (~100s, reproducible).
set -euo pipefail

# Run from the repo root so relative paths (resource/..., impl/...) resolve.
cd "$(dirname "$0")/.."

MODEL_DIR="resource/models/learning_demo"
PORT=8080
ARGS=()

# Pull --port out of the args (rest passes through untouched).
while [[ $# -gt 0 ]]; do
    case "$1" in
    --port)
        PORT="$2"
        shift 2
        ;;
    --port=*)
        PORT="${1#--port=}"
        shift
        ;;
    *)
        ARGS+=("$1")
        shift
        ;;
    esac
done

if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/model.npz" ]]; then
    echo "demo checkpoint not found — training it first (about 100s)..."
    uv run python -m scripts.train_demo_model
fi

if ! uv run python - "$PORT" <<'EOF' 2>/dev/null
import socket
import sys

port = int(sys.argv[1])
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
EOF
then
    echo "error: port $PORT is already in use (override with --port, e.g. --port 9000)" >&2
    exit 1
fi

echo "learning mode: http://127.0.0.1:$PORT  (model: $MODEL_DIR)"
echo "stop with Ctrl-C"
exec uv run python -m impl._np.cli --learning --port "$PORT" --model "$MODEL_DIR" "${ARGS[@]}"
