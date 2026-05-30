#!/bin/bash
set -e

# Set PYTHONPATH to include src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

DATA_FILE="tiny_data.txt"
CHECKPOINT_NAME="shakespeare_model"

echo "--- Starting E2E Verification ---"

# 1. Train the model
echo "[1/2] Training model on $DATA_FILE..."
uv run src/training/app.py train \
    --data_path "$DATA_FILE" \
    --epochs 2 \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --embed_dim 32 \
    --layers 1 \
    --heads 2 \
    --batch_size 16 \
    --max_context 128 \
    --seq_len 32 \

# 2. Run inference
echo "[2/2] Running inference..."
uv run src/training/app.py infer \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --prompt "ROMEO:" \
    --gen_len 50 \
    --temp 0.7

echo "--- E2E Verification Completed Successfully ---"
