#!/bin/bash
set -e

# Set PYTHONPATH to include src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

DATA_FILE="tiny_data.txt"
CHECKPOINT_NAME="shakespeare_model"

echo "--- Starting E2E Verification ---"

# 1. Train NumPy model
echo "[1/7] Training NumPy model on $DATA_FILE..."
uv run src/train.py train \
    --data "$DATA_FILE" \
    --backend numpy \
    --epochs 2 \
    --checkpoint_name "${CHECKPOINT_NAME}_np" \
    --embed_dim 32 \
    --layers 1 \
    --heads 2 \
    --batch_size 16 \
    --max_context 128 \
    --seq_len 32

# 2. Train PyTorch model
echo "[2/7] Training PyTorch model on $DATA_FILE..."
uv run src/train.py train \
    --data "$DATA_FILE" \
    --backend torch \
    --epochs 2 \
    --checkpoint_name "${CHECKPOINT_NAME}_torch" \
    --embed_dim 32 \
    --layers 1 \
    --heads 2 \
    --batch_size 16 \
    --max_context 128 \
    --seq_len 32

# 3. NumPy inference (NumPy train)
echo "[3/7] NumPy inference (NumPy train)..."
NP_OUTPUT=$(uv run src/train.py infer \
    --checkpoint_name "${CHECKPOINT_NAME}_np" \
    --backend numpy \
    --prompt "ROMEO:" \
    --num_new_tokens 50 \
    --temperature 0.7)
echo "$NP_OUTPUT"

# 4. PyTorch inference (PyTorch train)
echo "[4/7] PyTorch inference (PyTorch train)..."
PT_OUTPUT=$(uv run src/train.py infer \
    --checkpoint_name "${CHECKPOINT_NAME}_torch" \
    --backend torch \
    --prompt "ROMEO:" \
    --num_new_tokens 50 \
    --temperature 0.7)
echo "$PT_OUTPUT"

# 5. Cross-load: NumPy train -> PyTorch inference
echo "[5/7] Cross-load: NumPy train -> PyTorch inference..."
uv run src/train.py infer \
    --checkpoint_name "${CHECKPOINT_NAME}_np" \
    --backend torch \
    --prompt "ROMEO:" \
    --num_new_tokens 50 \
    --temperature 0.7

# 6. Cross-load: PyTorch train -> NumPy inference
echo "[6/7] Cross-load: PyTorch train -> NumPy inference..."
uv run src/train.py infer \
    --checkpoint_name "${CHECKPOINT_NAME}_torch" \
    --backend numpy \
    --prompt "ROMEO:" \
    --num_new_tokens 50 \
    --temperature 0.7

# 7. Run pytest for detailed assertions
echo "[7/7] Running cross-load pytest tests..."
uv run pytest tests/model/test_cross_load_checkpoint.py -v

echo "--- E2E Verification Completed Successfully ---"
