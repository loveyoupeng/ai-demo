# AI Transformer Demo (NumPy Implementation)

This repository is a pedagogical implementation of a decoder-only Transformer model built from scratch using NumPy. It is designed to demonstrate the internal mechanics of Large Language Models (LLMs), including Multi-Head Attention, Mixture of Experts (MoE), and manual backward passes for training.

## Features
- **Pure NumPy**: No heavy frameworks like PyTorch; all math and gradients are manual.
- **Mixture of Experts (MoE)**: Implements a routing mechanism to use a subset of experts per token.
- **Autoregressive Generation**: Supports text generation with temperature sampling.
- **Model Checkpointing**: Easily save and load trained models and tokenizers.

## Installation

Ensure you have `uv` installed.
```bash
uv sync
```

## Usage

The project uses a CLI entry point in `src/training/app.py`.

### 1. Training the Model

To train the model on a text dataset, use the `train` command. You can configure hyperparameters like embedding dimension, number of layers, and MoE configuration.

```bash
# Basic training with default parameters
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uv run src/training/app.py train --checkpoint_name my_model

# Advanced training with custom parameters
uv run src/training/app.py train \
    --embed_dim 64 \
    --layers 4 \
    --heads 8 \
    --experts 8 \
    --max_context 128 \
    --epochs 10 \
    --lr 0.001 \
    --checkpoint_name advanced_model \
    --data_path path/to/your/text_file.txt
```

*Note: If `--data_path` is not provided, a small toy dataset is used for demonstration.*

### 2. Running Inference

Once a model is trained, you can generate text using the `infer` command.

```bash
# Basic inference
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uv run src/training/app.py infer --checkpoint_name my_model --prompt "the"

# Advanced inference with temperature and generation length
uv run src/training/app.py infer \
    --checkpoint_name my_model \
    --prompt "Once upon a" \
    --gen_len 50 \
    --temp 0.8
```

## Project Structure
- `src/model/`: Core Transformer components (Attention, MoE, Layers).
- `src/training/`: Training orchestration and data loading.
- `src/tokenizer/`: Character-level tokenizer.
- `src/utils/`: Utility functions like checkpointing.
- `src/inference.py`: Autoregressive generation logic.
