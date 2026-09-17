# AI Transformer Demo (Four-Backend)

A pedagogical implementation of a decoder-only transformer with **four
equivalent backends**: NumPy (pure manual math, the teaching reference),
PyTorch (the production idiom), Triton (GPU kernels over a PyTorch model),
and CUDA (bare-metal NVRTC kernels). All four tracks share the same
architecture, the same key scheme, and the same config — a model trained on
one track loads and runs on any other.

See [CONTEXT.md](CONTEXT.md) for the domain glossary,
[docs/specs/architecture-fixes.md](docs/specs/architecture-fixes.md) for the
architecture spec and progress, and
[docs/seam_triton_to_torch.md](docs/seam_triton_to_torch.md) for the
documented Triton→PyTorch shared seam.

## Features

- **Four backends, one model**: NumPy (analytic backward, the math
  reference), PyTorch (autograd + `F.scaled_dot_product_attention` +
  `torch.optim`), Triton (FlashAttention-style online-softmax kernel), CUDA
  (NVRTC kernels for RMSNorm, RoPE, attention, FFN).
- **Standard architecture**: pre-norm LLaMA block
  (`h = x + attn(ln1(x)); out = h + mlp(ln2(h))`), RMSNorm, GQA, RoPE,
  SwiGLU (dense by default; MoE is opt-in), causal mask, no weight tying, no
  biases.
- **KV cache**: the per-token step path is exact (`forward_prefill` +
  `forward_step` == full re-forward). The generators step one token per
  iteration (O(1) per step). TurboQuant (1-bit quantized cache) is wired as
  an alternative with a parity-budget test.
- **Analytic backward** (NumPy): every operator has a closed-form
  `backward(dout, x) -> (dinput, dparams)`; `check_model_gradients` verifies
  it against finite differences (float64, ~1e-10).
- **Cross-backend parity**: three-tier tolerance policy (see AGENTS.md rule
  2); 42 cross-backend tests cover dense/GQA/MoE parity, GPU parity, and a
  3-way equivalence demo.
- **Checkpoint interchange**: one flat-dict key scheme
  (`shared.constants.Keys`) + a parameter registry
  (`shared/registry.py`) that validates shape and key-set on load.

## Installation

```bash
uv sync
```

## Usage

### Inference

```bash
# NumPy
uv run python -m impl._np.cli --prompt "the" --max_new_tokens 10

# PyTorch
uv run python -m impl._torch.cli --prompt "the" --max_new_tokens 10

# Triton (GPU)
uv run python -m impl._triton.cli --prompt "the" --max_new_tokens 10

# CUDA (GPU)
uv run python -m impl._cuda.cli --prompt "the" --max_new_tokens 10

# With custom parameters
uv run python -m impl._torch.cli \
    --prompt "Once upon a" --max_new_tokens 50 \
    --temperature 0.9 --top_k 20 \
    --embed_dim 64 --n_layers 4 --n_heads 8
```

### Training

```bash
uv run python -m scripts.train --backend numpy|torch|triton|cuda
```

### Real-data checkpoints (TinyStories)

The 4-backend "real" checkpoints (`resource/models/{backend}_real/`) train all four
backends on the TinyStories dataset (GPT-2 BPE, vocab 50,257) from an identical
weight start, then save one checkpoint per backend. They total ~150 MB (the
50,257-wide embedding + lm_head dominate — the transformer itself is only ~1 MB),
so they are **git-ignored**, not committed. Recreate any time:

```bash
# Full reproduction (all 4 backends, ~150 MB, ~1-2 min on the GPU)
uv run python -m scripts.train_real_tinystories

# Quick small check (2 backends, 5 steps, throwaway suffix)
uv run python -m scripts.train_real_tinystories --backends numpy,torch --num_batches 5 --suffix tmp
```

### Equivalence verification

```bash
uv run python -m scripts.verify_equivalence
```

### Testing

```bash
# All CPU unit tests (NumPy + PyTorch)
uv run pytest tests/unit/ -q --timeout=120 \
    --ignore=tests/unit/_cuda --ignore=tests/unit/_triton

# Triton unit tests (GPU)
uv run pytest tests/unit/_triton/ -q --timeout=120

# CUDA unit tests (GPU; one file per invocation for NVRTC context isolation)
for f in tests/unit/_cuda/test_*.py; do
    uv run pytest "$f" -q --timeout=120
done

# Cross-backend parity tests (GPU)
uv run pytest tests/cross_backend/ -q --timeout=120
```

### Project structure

```
impl/
├── _np/         # NumPy track (math reference; analytic backward)
├── _torch/      # PyTorch track (production idiom; autograd + SDPA)
├── _triton/     # Triton track (kernels over a PyTorch model)
├── _cuda/       # CUDA track (NVRTC kernels)
shared/
├── config.py    # TransformerConfig (single source of truth)
├── constants.py # Keys scheme + Attn/LayerNorm/Mlp constants
├── registry.py  # ParameterRegistry (checkpoint format owner)
docs/
├── specs/architecture-fixes.md   # Spec + progress
├── seam_triton_to_torch.md       # Documented shared seam
└── docstring_style.md            # numpydoc + shape convention
scripts/
├── train.py             # Training loop (all backends)
├── infer.py             # Inference (all backends)
├── verify_equivalence.py # 6-scenario parity check
tests/
├── unit/           # Per-track unit tests
└── cross_backend/  # Parity tests (dense/GQA/MoE, GPU, 3-way)
```

## Development

```bash
uv run ruff format . && uv run ruff check .
uv run pyright .
uv run pytest tests/ -v
```

All code must be free of `ruff` and `pyright` issues; all unit tests must
carry a `pytest-timeout` timeout (see AGENTS.md).
