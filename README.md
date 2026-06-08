# AI Transformer Demo (NumPy + PyTorch Dual-Backend)

This repository is a pedagogical implementation of a decoder-only Transformer model with dual backends: NumPy (pure manual math) and PyTorch (explicit autograd). It is designed to demonstrate the internal mechanics of Large Language Models (LLMs),
including Multi-Head Attention, Mixture of Experts (MoE), and manual backward passes for training.

## Features

- **Dual Backends**: Pure NumPy and PyTorch implementations with cross-backend parity testing (verified to < 1e-6 float64).
- **Mixture of Experts (MoE)**: Implements a routing mechanism to use a subset of experts per token.
- **Manual Backward Passes**: Gradients are computed explicitly in both backends for educational clarity.
- **Autoregressive Generation**: Supports text generation with temperature sampling and KV caching.
- **Model Checkpointing**: Easily save and load trained models and tokenizers.
- **Cross-Backend Validation**: E2E tests verify that both backends produce identical forward and backward results.

## Installation

Ensure you have `uv` installed.

```bash
uv sync
```

## Usage

### Training the Model

Train the model on a text dataset using the CLI entry point:

```bash
# Basic training with default parameters
uv run src/train.py train --checkpoint_name my_model

# Advanced training with custom parameters
uv run src/train.py train \
    --embed_dim 64 \
    --layers 4 \
    --heads 8 \
    --experts 8 \
    --max_context 128 \
    --epochs 10 \
    --lr 0.001 \
    --checkpoint_name advanced_model \
    --backend numpy
```

### Running Inference

Generate text from a trained checkpoint:

```bash
# Basic inference
uv run src/train.py infer --checkpoint_name my_model --prompt "the"

# Advanced inference with temperature and generation length
uv run src/train.py infer \
    --checkpoint_name my_model \
    --prompt "Once upon a" \
    --gen_len 50 \
    --temp 0.8
```

## Cross-Backend Validation

This project provides an E2E validation script that ensures the NumPy and PyTorch backends produce numerically identical results:

```bash
# Run all 4 validation scenarios (forward/backward parity + cross-load)
uv run src/validate_e2e.py
```

### Scenarios

| #   | Description                                                  | Verified                     |
| --- | ------------------------------------------------------------ | ---------------------------- |
| 1   | Standalone layer parity (NumPy vs PyTorch)                   | All 5 layer types           |
| 2   | Transformer block forward + backward parity                   | Parameter gradients         |
| 3   | Cross-load PyTorch → NumPy                                   | Forward + backward match    |
| 4   | Cross-load NumPy → PyTorch                                   | Forward + backward match    |

All scenarios pass with a maximum difference of < 0.5e-6 (float64).

## Test Suite

```bash
# Run all tests
uv run pytest tests/ -v

# Run cross-backend parity tests
uv run pytest tests/test_cross_backend.py -v

# Run E2E validation
uv run pytest tests/test_e2e_cross_backend.py -v
```

### Test Categories

- **NumPy backend**: 60+ tests covering all layers, transformer, training loop.
- **PyTorch backend**: 40+ tests mirroring NumPy tests.
- **Cross-backend**: 6 parity tests ensuring NumPy and PyTorch produce matching results (with tiered tolerances: standalone rtol=1e-4, single chain rtol=1e-3, full chain rtol=1e-2 per AGENTS.md).
- **E2E validation**: 4 cross-load scenarios verifying bidirectional parameter loading.

## Project Structure

```
src/
├── train.py                 # Main CLI entry point (train/infer/generate)
├── validate_e2e.py          # E2E cross-backend validation
├── model/
│   ├── layers.py            # NumPy: TokenEmbedding, LayerNorm, FeedForward, PositionalEmbedding
│   ├── attention.py         # NumPy: MultiHeadAttention
│   ├── moe.py               # NumPy: Router, Expert, MoELayer
│   ├── transformer.py       # NumPy: TransformerBlock, Transformer
│   ├── pytorch/
│   │   ├── layers.py        # PyTorch: PyTorchTokenEmbedding, PyTorchLayerNorm, PyTorchFeedForward, PyTorchPositionalEmbedding
│   │   ├── attention.py     # PyTorch: PyTorchMultiHeadAttention
│   │   ├── moe.py           # PyTorch: PyTorchRouter, PyTorchExpert, PyTorchMoELayer
│   │   └── transformer.py   # PyTorch: PyTorchTransformerBlock, PyTorchTransformer
├── training/
│   ├── data_loader.py       # TextDataLoader
│   ├── trainer.py           # Training loop
│   └── config.py            # CLI argument definitions
├── tokenizer/               # Character-level tokenizer
└── inference.py             # Autoregressive generation logic
tests/
├── test_cross_backend.py    # 6 parity tests (NumPy vs PyTorch)
├── test_e2e_cross_backend.py # E2E cross-load validation
└── test_numpy_*.py          # NumPy backend tests
└── test_pytorch_*.py        # PyTorch backend tests
```

