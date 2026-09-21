# AI Transformer Demo (Four-Backend)

A pedagogical implementation of a decoder-only transformer with **four
equivalent backends**: NumPy (pure manual math, the teaching reference),
PyTorch (the production idiom), Triton (GPU kernels over a PyTorch model),
and CUDA (bare-metal NVRTC kernels). All four tracks share the same
architecture, the same key scheme, and the same config — a model trained on
one track loads and runs on any other. A **learning mode** (NumPy or PyTorch
backend) adds an interactive web page that visualizes the architecture, shows
every intermediate tensor of an inference, and exports full inference records.

## What each track teaches

Each track answers a different question about the *same* model:

| Track | Question it answers | Read it for |
| --- | --- | --- |
| **NumPy** (`impl/_np/`) | *How does the math work?* | Hand-derived forward + analytic backward, formula citations, shape comments on every matrix op. The teaching reference. |
| **PyTorch** (`impl/_torch/`) | *How do you implement it properly with a framework?* | Production idiom: `nn.Module` composition, autograd, `F.scaled_dot_product_attention`, `torch.optim`. |
| **Triton** (`impl/_triton/`) | *How do you write the kernels?* | Memory-traffic-aware GPU kernels (online softmax / FlashAttention) over a PyTorch skeleton. |
| **CUDA** (`impl/_cuda/`) | *What does the metal see?* | Bare-metal NVRTC kernels, explicit launches, no framework. |

Because all four implement bit-comparable behavior on one checkpoint format,
you can trace the same tensor through four increasingly low-level lenses.

See [CONTEXT.md](CONTEXT.md) for the domain glossary,
[docs/theory/transformer-walkthrough.md](docs/theory/transformer-walkthrough.md)
for a stage-by-stage tour of the forward pass (equations + code pointers),
[docs/specs/architecture-fixes.md](docs/specs/architecture-fixes.md) for the
architecture spec and progress,
[docs/seam_triton_to_torch.md](docs/seam_triton_to_torch.md) for the
documented Triton→PyTorch shared seam, and [AGENTS.md](AGENTS.md) for
repository guidelines and development rules.

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
  it against finite differences (float64, ~1e-10, MoE top-k kink-aware).
- **Cross-backend parity**: three-tier tolerance policy (see AGENTS.md rule
  2); 42 cross-backend tests cover dense/GQA/MoE parity, GPU parity, and a
  3-way equivalence demo; `scripts.verify_equivalence` runs 6 end-to-end
  scenarios.
- **Checkpoint interchange**: one flat-dict key scheme
  (`shared.constants.Keys`, HF-Llama naming) + a parameter registry
  (`shared/registry.py`) that validates shape and key-set on load
  (stale checkpoints fail fast).
- **Learning mode** (NumPy or PyTorch backend): `scripts/learning.py
  [--backend numpy|torch]` hosts a web page (stdlib HTTP server, no
  dependencies) with prompt + generation, an architecture diagram with
  flow navigation (click a node to light up its predecessors/successors) and
  per-block numbers, and downloadable JSON inference records of every
  intermediate tensor. Both record adapters emit the same JSON shapes, so
  the page consumes either backend interchangeably.
- **Real-data training**: TinyStories (GPT-2 BPE, vocab 50,257) dataset
  pipeline in `shared/dataset.py`, plus a char-level tokenizer for the tiny
  demo model; unified train/infer scripts for all four backends.

## Installation

```bash
uv sync
```

Python 3.10; torch is pinned to `2.13.0+cu132` via a custom uv index.

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

# Unified script (any backend, any checkpoint)
uv run python -m scripts.infer --model resource/models/torch_real/ --backend torch --prompt "hello"
```

### Learning mode (web page)

```bash
# Serves http://127.0.0.1:8080; auto-trains the demo model first run (~100 s)
uv run python -m scripts.learning

# Backend selects which track materializes the model: numpy [default] or
# torch; port/model overridable
uv run python -m scripts.learning --backend torch
uv run python -m scripts.learning --port 9000 --model resource/models/torch_real

# Rebuild the demo model (tiny char-level MoE: D=8, H=4, L=3, E=3, V=20)
uv run python -m scripts.train_demo_model
uv run python -m scripts.train_demo_model --backend cuda   # GPU backends too
```

### Training

```bash
# Unified training (all backends)
uv run python -m scripts.train --backend numpy|torch|triton|cuda

# Options: --synthetic (no dataset), --n_layers, --embed_dim, --n_experts,
# --save_dir, --config resource/models/config.json, ...
uv run python -m scripts.train --backend torch --synthetic --epochs 3
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

The TinyStories dataset itself lives in `resource/` (also git-ignored):
`uv run python -m scripts.download_tinystories` fetches it.

### Equivalence verification

```bash
# 7 scenarios: dense_np_torch, gqa_np_torch, moe_np_torch, moe_shared_experts_np_torch, gqa_torch_triton, cuda_shared_weights, all_four_backends
uv run python -m scripts.verify_equivalence

# Quick mode / single scenario
uv run python -m scripts.verify_equivalence --fast
uv run python -m scripts.verify_equivalence --scenario gqa
```

### Testing

```bash
# All CPU unit tests (NumPy + PyTorch + shared/root)
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

# Non-GPU tests anywhere
uv run pytest tests/ -q -m "not gpu"
```

### Project structure

```text
impl/
├── _np/         # NumPy track (math reference; analytic backward; learning mode + web/)
├── _torch/      # PyTorch track (production idiom; autograd + SDPA)
├── _triton/     # Triton track (kernels over a PyTorch model; flash_attn.py)
├── _cuda/       # CUDA track (NVRTC kernels)
shared/
├── config.py    # TransformerConfig (single source of truth)
├── constants.py # Keys scheme + Attn/LayerNorm/Mlp constants
├── registry.py  # ParameterRegistry (checkpoint format owner)
├── checkpoint.py# save/load (config.json + model.npz)
├── init.py      # canonical cross-backend weight initialization
├── tokenizer.py # GPT-2 BPE + char-level tokenizers
├── dataset.py   # TinyStories pipeline
├── config_utils.py  # CLI > env > config file > defaults
└── utils/logger_setup.py
docs/
├── specs/architecture-fixes.md   # Spec + progress
├── seam_triton_to_torch.md       # Documented shared seam
├── docstring_style.md            # numpydoc + shape convention
├── adr/0001-gated-residual-abandonment.md
└── design.md
scripts/
├── train.py               # Training loop (all backends)
├── infer.py               # Inference (all backends)
├── verify_equivalence.py  # 7-scenario parity check
├── learning.py            # learning-mode web page server
├── train_real_tinystories.py
├── train_demo_model.py    # learning-mode demo model
└── download_tinystories.py
tests/
├── unit/           # Per-track unit tests + shared/root tests
└── cross_backend/  # Parity tests (dense/GQA/MoE, GPU, 3-way; 43 tests)
resource/           # git-ignored: TinyStories data + model checkpoints
```

## Development

```bash
uv run ruff format . && uv run ruff check .
uv run pyright .   # pyright config strictly includes shared/
uv run pytest tests/ -v
```

All code must be free of `ruff` and `pyright` issues; all unit tests must
carry a `pytest-timeout` timeout (see AGENTS.md).
