#!/bin/bash
set -e

# Set PYTHONPATH to include src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

DATA_FILE="tiny_data.txt"
CHECKPOINT_NAME="shakespeare_model"

echo "--- Starting E2E Verification ---"

# 1. Train the model
echo "[1/2] Training model on $DATA_FILE..."
uv run src/train.py train \
    --data "$DATA_FILE" \
    --epochs 2 \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --embed_dim 32 \
    --layers 1 \
    --heads 2 \
    --batch_size 16 \
    --max_context 128 \
    --seq_len 32

# 2. Run inference
echo "[2/2] Running inference..."
uv run src/train.py infer \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --prompt "ROMEO:" \
    --num_new_tokens 50 \
    --temperature 0.7

echo "--- E2E Verification Completed Successfully ---"
