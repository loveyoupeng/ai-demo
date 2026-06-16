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

### Running Inference

Generate text from a small model:

```bash
# NumPy model inference
uv run python -m impl._np.cli --prompt "hello" --max_new_tokens 10

# PyTorch model inference
uv run python -m impl._torch.cli --prompt "hello" --max_new_tokens 10

# With custom parameters
uv run python -m impl._torch.cli \
    --prompt "hello" \
    --max_new_tokens 20 \
    --temperature 0.8 \
    --embed_dim 32 \
    --n_layers 2
```

### Testing

Run the test suite:

```bash
# Run all tests
uv run pytest tests/ -v

# Run NumPy backend tests
uv run pytest tests/unit/_np/ -v

# Run PyTorch backend tests
uv run pytest tests/unit/_torch/ -v

# Run cross-backend parity tests
uv run pytest tests/cross_backend/ -v
```

### Project Structure

```
impl/
├── _np/                    # NumPy implementation
│   ├── __init__.py
│   ├── cli.py              # CLI entry point (training, inference)
│   ├── inference.py        # Autoregressive generation logic
│   ├── layers.py           # TokenEmbedding, LayerNorm, FeedForward, etc.
│   ├── model.py            # NumPyModel: full transformer
│   ├── training.py         # Training loop
│   └── cross_entropy.py    # Cross-entropy loss
│
├── _torch/                 # PyTorch implementation
│   ├── __init__.py
│   ├── cli.py              # CLI entry point (inference)
│   ├── cross_entropy.py    # Cross-entropy loss (F.cross_entropy)
│   ├── inference.py        # Autoregressive generation with greedy/sampled/top-k
│   ├── kv_cache.py         # Naive KV cache
│   ├── layers.py           # Embedding, RMSNorm, SwiGLU, RoPE, MHA, MoE, AdamW
│   ├── model_config.py     # ModelConfig dataclass + TorchModel stub
│   ├── training.py         # Training loop (autograd)
│   └── turboquant_kv_cache.py  # 1-bit compressed KV cache
│
shared/                     # Shared utilities (config, constants)
tests/
├── cross_backend/          # Cross-backend parity tests
├── unit/
│   ├── _np/               # NumPy backend tests (~75 tests)
│   └── _torch/            # PyTorch backend tests (~70+ tests)
scripts/
└── download_tinystories.py # Dataset download utility
```

## CLI Commands

All commands run via `uv run python -m`:

```bash
# NumPy inference
uv run python -m impl._np.cli --prompt "the" --max_new_tokens 10 --temperature 0.0

# PyTorch inference
uv run python -m impl._torch.cli --prompt "the" --max_new_tokens 10 --temperature 0.0

# With all options
uv run python -m impl._torch.cli \
    --prompt "Once upon a" \
    --max_new_tokens 50 \
    --temperature 0.9 \
    --top_k 20 \
    --embed_dim 64 \
    --n_layers 4 \
    --n_heads 8
```

## Tests

- **NumPy backend**: 70+ tests covering all layers, transformer, training loop, CLI.
- **PyTorch backend**: 65+ tests mirroring NumPy tests. Tests include layers, training loop, KV cache (naive + TurboQuant), inference engine, CLI.
- **Cross-backend**: Parity tests at three tiers per AGENTS.md:
  - **Standalone layers** (isolated): rtol=1e-4, atol=1e-4
  - **Single chain** (one layer chain): rtol=1e-3, atol=1e-3
  - **Multi-layer chains**: rtol=1e-2, atol=1e-2

## Development

Use `uv` for dependency management and running scripts:

```bash
# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright .
```
