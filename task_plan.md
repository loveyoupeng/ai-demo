# Task Plan: Decoder-Only Transformer Learning Project

## Goal
Build a fully functional decoder-only transformer LLM in 4 equivalent implementations (NumPy, PyTorch, Triton, CUDA) with identical behavior, trained on TinyStories, for educational purposes.

## Architecture
```
+----------+
| Shared   | - config.json + env vars → model topology
| Backbone | - tokenizer, dataset, checkpoint
+-----+----+
      |
  +---+---+---+
  |   |   |   |
NumPy PyT Trit CUDA (4 backends, same structure)
+---+---+---+
      |
   ckpt.npz (flat dict, cross-backend compatible)
```

## Completed Phases
| Phase | Title | Status | Tests | Commits |
|-------|-------|--------|-------|---------|
| A | Shared Foundation | ✅ Done | 131 | (multiple) |
| B | NumPy Backend | ✅ Done | ~100 | (multiple) |
| C | PyTorch Backend | ✅ Done | 310 | 36 |
| C+ | E2E Scripts (train/infer) | ✅ Done | ~450 | (multiple) |
| 3++ | Post-Norm + Dropout | ✅ Done | ~421 | (multiple) |
| D | Platform Setup | ✅ Done | 440 | (multiple) |
| E | Triton Kernels | ✅ Done | 551 | (multiple) |
| F | CUDA Bare-Metal | ✅ Done | 121 | (multiple) |
| G | Weight Diff Tests | ✅ Done | 10 | (multiple) |
| H | Logging Architecture | ✅ Done | — | (current) |

## CLI Commands
```bash
# Training
uv run python -m scripts.train --backend numpy/torch/triton/cuda

# Inference
uv run python -m scripts.infer --model resource/models/ckpt.npz --prompt "hello"

# Equivalence verification
uv run python -m scripts.verify_equivalence --fast

# Dataset download
uv run python -m scripts.download_tinystories
```

## Testing Commands
```bash
# All tests
uv run pytest tests/ -v

# Cross-backend parity
uv run pytest tests/cross_backend/ -v
```

## Remaining Work
| Item | Priority | Description |
|------|----------|-------------|
| Training on TinyStories | Medium | Currently synthetic data only |
| 4-way equivalence | Medium | Verify all backends produce same model |
| Clean up old phase plans | Low | Consolidate into design.md |
| Special educational logs (H.6) | Low | Attention entropy, activation stats, loss curve, LR schedule |

## Phase H: Logging Architecture — How to Add Debug Logging for Training & Inference

> **Goal:** Make it easy for users to trace runtime behavior and learn how the LLM works internally.
> Logs must be human-readable, linkable to code locations and pydoc comments, and cover both training and inference.

### H.1 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Structured but human-readable** | Use Python `logging` module with a consistent format — timestamp, level, logger name, message |
| **One logger per module** | Each `.py` file gets a `logger = logging.getLogger(__name__)` so users can filter by module: `logging.getLogger("impl._np.model").setLevel(...)` |
| **Linkable to code & docs** | Every log message mentions the function name (via `__func__=func.__name__` or `f"{func.__name__}(...)`) and the key shape/param values from the function signature, so a reader can map the log entry back to the pydoc in-source documentation |
| **Key-step focused** | Only log at meaningful "chapters" — not every operation. Think: story-telling, not data dump |
| **Opt-in verbosity** | `INFO` for summary-level (epoch/batch tokens), `DEBUG` for tensor shapes/values, `TRACE` (custom, or DEBUG+) for per-step calculations |

### H.2 Logger Naming Convention

```
shared.utils.logger_setup   — singleton to create & configure all loggers
scripts.train               — top-level training orchestration
scripts.infer               — top-level inference orchestration
impl._np.model              — NumPyModel forward/backward
impl._np.modules            — low-level modules (Embedding, RMSNorm, DecoderStack, MHA, MoE, RoPE)
impl._np.training           — training loop, gradient clipping, optimiser steps
impl._np.inference          — TextGenerator generate/generate_greedy/generate_sampled
impl._np.kv_cache           — KV cache operations
impl._torch.*               — mirrored logger names under impl._torch
impl._triton.*              — mirrored logger names under impl._triton
impl._cuda.*                — mirrored logger names under impl._cuda
```

Logger names are Python dotted paths (`impl._np.model`, `shared.dataset`), which naturally map to file paths:
`impl/_np/model.py` → `impl._np.model`, `shared/dataset.py` → `shared.dataset`.

### H.3 Shared Logger Initializer

Create `shared/utils/logger_setup.py` that provides a single function:

```python
def setup_logging(
    level: str = "INFO",         # INFO for summary, DEBUG for detailed shapes/values
    log_file: Path | None = None, # Optional: write to file for long training runs
    format_string: str | None = None,
) -> None:
```

Responsibilities:
1. Configure the root logger with a `StreamHandler` (stdout) at the requested level
2. If `log_file` is given, add a `FileHandler` for persistent logging
3. **Attach a `filter_by_logger` callback** that lets users programmatically suppress specific loggers
4. Set a default format: `{timestamp} [{levelname:7s}] {name} {message}` — where `name` is the module dotted path

Default format:
```
{timestamp} [{levelname:7s}] {name} {message}
```

Example output:
```
2025-06-26 14:30:01 [  INFO] scripts.train Starting epoch 1/10, 5000 batches
2025-06-26 14:30:01 [ DEBUG] impl._np.model forward() input_ids=[1, 128] → logits=[1, 128, 256]
2025-06-26 14:30:01 [ TRACE] impl._np.modules.mha() q=[1, 8, 128, 32] k=[1, 8, 128, 32] v=[1, 8, 128, 32] → attn=[1, 8, 128, 32]
```

Users can control verbosity:
```bash
# Summary level — good for long training runs
uv run python scripts/train.py --backend torch 2>&1 | grep -v "^... DEBUG"

# Full trace — good for debugging/learning
uv run python scripts/train.py --backend torch --log_level DEBUG
```

### H.4 Training Logging — Key Chapters

Training is a story with clear acts. Log these chapters:

#### H.4.1 `scripts/train.py` — Orchestration

| Chapter | Level | What | Example |
|---------|-------|------|---------|
| **Startup** | INFO | Backend, config summary, dataset info | `train() backend=torch, vocab=256, embed=256, layers=4, heads=8, context=128, batch=16, epochs=10` |
| **Epoch start** | INFO | Epoch number, total batches | `train_epoch() Epoch 3/10, 500 batches, lr=0.000300` |
| **Batch progress** | DEBUG | Batch index, loss, grad norm | `train_batch() batch=123/500 loss=2.3456 grad_norm=0.789` |
| **Epoch end** | INFO | Epoch loss, throughput | `train_epoch() epoch=3 avg_loss=2.1234 tokens/sec=12345` |
| **Checkpoint** | INFO | Save path, step, loss | `save_checkpoint() saved checkpoint to resource/models/torch_42/ckpt.npz step=1500 loss=2.1234` |

#### H.4.2 `impl/_*/training.py` — Training Loop

| Chapter | Level | What | Example |
|---------|-------|------|---------|
| **train_step** | DEBUG | Forward input/output shapes, loss value, backward param count | `train_step() input=[16, 128] → logits=[16, 128, 256] loss=2.3456` |
| **Gradient clipping** | DEBUG | Before/after clip norm, whether clipping happened | `clip_gradients() before=5.678 after=1.000 clipped=True` |
| **Gradient norm** | DEBUG | Total gradient norm across all params | `compute_gradient_norm() total=5.678` |
| **Optimizer step (NumPy)** | DEBUG | Param update stats, LR schedule info | `step() param_count=2500000 lr_step=0.000274` |

#### H.4.3 `impl/_np/model.py` — Model Forward/Backward

These are the **core of learning** — show how data flows through the network:

| Chapter | Level | What | Example |
|---------|-------|------|---------|
| **forward** | DEBUG | Overall input→output shape chain | `forward() input_ids=[1, 128] → embedding=[1, 128, 256] → stack=[1, 128, 256] → logits=[1, 128, 256]` |
| **backward** | DEBUG | Input shape, output param count | `backward() input_shape=[1, 128] params=2500000` |

#### H.4.4 `impl/_np/modules.py` — Building Block Operations

This is where users **learn how the transformer works internally**. Log every operation with shapes:

| Level | Chapter | What shapes to log |
|-------|---------|-------------------|
| **DEBUG** | **Embedding forward** | `emb() input_ids=[4, 128] → output=[4, 128, 256] table=[256, 256]` |
| **DEBUG** | **RMSNorm forward** | `rms_norm() input=[4, 128, 256] → output=[4, 128, 256] eps=1e-6` |
| **DEBUG** | **MHA forward** | `mha() input_shape=[4, 128, 256] → q=[4, 128, 256] k=[4, 128, 256] v=[4, 128, 256] → attn_weights=[4, 8, 128, 128] → output=[4, 128, 256] head_dim=32 n_heads=8` |
| **DEBUG** | **MHA attention scores** | `mha_attn() qk=[4, 8, 128, 128] softmax→[4, 8, 128, 128] scaled=True scale=0.125` |
| **TRACE** | **MHA attention distribution** | `mha_attn_stats() max_attn=0.023 mean_attn=0.008 entropy=4.67 (sparsity: 12% positions have >1% attn mass)` |
| **DEBUG** | **MoE forward** | `moe() input=[4, 128, 256] → gates=[4, 128, 4] selected=[4, 2] experts=[2, 8, 256, 256] → output=[4, 128, 256] k=2 n_experts=4` |
| **DEBUG** | **SWiGLU FFN** | `swiglu_ffn() input=[4, 128, 256] → gate=[4, 128, 512] up=[4, 128, 512] down=[4, 128, 256] ff_dim=512` |
| **DEBUG** | **RoPE forward** | `rope() input=[4, 8, 128, 32] rope_dim=32 → output=[4, 8, 128, 32]` |
| **DEBUG** | **DecoderBlock forward** | `decoder_block() input=[4, 128, 256] → mha=[4, 128, 256] → ffn=[4, 128, 256] → output=[4, 128, 256]` |
| **DEBUG** | **DecoderStack forward** | `decoder_stack() input=[4, 128, 256] → block0=[4, 128, 256] → block1=[4, 128, 256] ... → output=[4, 128, 256] n_layers=4` |

#### H.4.5 Shape-Chain Convention

For **every forward pass**, log a shape chain showing data flow. This is the single most important logging pattern for learning:

```
input_ids=[B,S] → embedding=[B,S,D] → ln1=[B,S,D] → mha_input=[B,S,D]
               → qkv=[B,S,H,Hd] → attn=[B,H,S,S] → attn_out=[B,S,D] → block_output=[B,S,D]
               → ffn_input=[B,S,D] → gate_up=[B,S,FF] → swiglu=[B,S,D] → ffn_output=[B,S,D]
               → final_output=[B,S,D]
```

The above format lets users visually trace how tensor shapes evolve at each step.

### H.5 Inference Logging — Key Chapters

#### H.5.1 `scripts/infer.py` — Orchestration

| Chapter | Level | What | Example |
|---------|-------|------|---------|
| **Startup** | INFO | Model loaded, config, backend, device info | `load_model() loaded torch model from resource/models/torch_42/ vocab=256 layers=4 embed=256 device=cuda:0` |
| **Generate start** | INFO | Prompt, max tokens, temperature, decoding mode | `generate() prompt="hello world" max_new=50 temperature=0.8 top_k=50 mode=sampled` |
| **Prompt process** | DEBUG | Encoded prompt, length, tokenization details | `encode() prompt="hello world" tokens=[8810, 16] length=2` |
| **Iteration** | DEBUG | Each generated token with logit info | `generate_step() step=1 token=7851 prob=0.1234 text=" the"` |
| **Completion** | INFO | Full output, generation time, throughput | `generate() generated 50 tokens in 0.234s (213.68 tok/s)` |

#### H.5.2 `impl/_*/inference.py` — Generation Engine

| Chapter | Level | What | Example |
|---------|-------|------|---------|
| **generate** | DEBUG | Overall loop entry, prompt shape | `generate() prompt=[1, 5] → generating up to 50 new tokens` |
| **generate_step** | DEBUG | Per-token generation: logits → sampled token | `generate_step() x_t_logits=[1, 256] → sampled_token=7851 sampled_prob=0.1234` |
| **kv_cache** | DEBUG | Cache allocation, append, update | `get_cache() cache_allocated=true head_dim=32 n_heads=8 cache_len=129` |

### H.6 Special Logging for Educational Value

These are **non-standard** log types that help learners understand *why* the model behaves as it does:

| Type | Level | Location | What | Example |
|------|-------|----------|------|---------|
| **Attention entropy** | TRACE | `mha` modules | Shannon entropy of attention distribution per head per position — measures "focus" | ` attn_entropy() head=0 pos=5 entropy=3.20 (focused) / head=0 pos=6 entropy=6.85 (diffuse)` |
| **Gradient norm stats** | DEBUG | `training.py` | Per-layer gradient norms — helps spot vanishing/exploding gradients | `grad_stats() layer=0 norm=0.023 layer=1 norm=0.045 layer=2 norm=0.019` |
| **Loss curve** | DEBUG | `train.py` | Rolling average loss for smoothing | `loss_curve() current=2.345 rolling_avg=2.412 trend=down` |
| **LR schedule** | DEBUG | `optimizer.py` | Current learning rate with schedule info | `lr_schedule() step=1000 lr=0.000291 schedule=warmup_cosine total_steps=50000` |
| **Activation stats** | TRACE | `modules.py` | Min/max/mean/percentiles of activations at each layer | `act_stats() layer=2 module=rms_norm x_min=-2.34 x_max=5.67 x_mean=0.01` |
| **Token sampling** | DEBUG | `inference.py` | Top-5 token probabilities at each generation step | `sample_top5() top5=[7851(0.12), 17(0.08), 307(0.05), 4521(0.03), 64(0.02)]` |

### H.7 Implementation Status — ALL COMPLETE

| # | Item | Status | Files Modified |
|---|------|--------|----------------|
| 1 | **P0 — `shared/utils/logger_setup.py`** | ✅ Done | `shared/utils/logger_setup.py`, `shared/utils/__init__.py` |
| 2 | **P0 — `scripts/train.py`** | ✅ Done | `scripts/train.py` |
| 3 | **P1 — `impl/_np/model.py`** | ✅ Done | `impl/_np/model.py` |
| 4 | **P1 — `impl/_np/training.py`** | ✅ Done | `impl/_np/training.py` |
| 5 | **P1 — `scripts/infer.py`** | ✅ Done | `scripts/infer.py`, `shared/config_utils.py` |
| 6 | **P2 — `impl/_np/inference.py`** | ✅ Done | `impl/_np/inference.py` |
| 7 | **P2 — `impl/_torch/*`** | ✅ Done | `impl/_torch/training.py`, `impl/_torch/inference.py` |
| 8 | **P2 — `impl/_triton/*` + `impl/_cuda/*`** | ✅ Done | All 4 files in both directories |

### H.7.1 Logging Architecture Summary

```
singleton: shared/utils/logger_setup.py
  ├── setup_logging(level, log_file, format_string) → None   # idempotent, configures root logger
  ├── set_level(name, level) → None                          # runtime per-module control
  └── log(level, name, msg, **kwargs) → str                 # convenience: returns formatted msg

modules: impl/_np/*.py, impl/_torch/*.py, impl/_triton/*.py, impl/_cuda/*.py
  └── logger = logging.getLogger(__name__)                   # one logger per .py file

scripts: scripts/train.py, scripts/infer.py
  └── logger = logging.getLogger(__name__)                   # orchestration-level logging
```

**Logging levels per module:**

| Module | INFO | DEBUG | TRACE |
|--------|------|-------|-------|
| `scripts.train` | epoch/batch progress, loss, checkpoint, throughput | — | — |
| `scripts.infer` | model load, generation start/end, throughput | prompt/output tracking | — |
| `impl/_np/model.py` | — | forward/backward shape chains | — |
| `impl/_np/training.py` | — | train_step, clip, optimizer step | — |
| `impl/_np/inference.py` | first/last token, mode (greedy/sampled) | per-token entropy, top-5, softmax | detailed distributions |
| `impl/_torch/*` | mirror | mirror | — |
| `impl/_triton/*` | mirror | mirror | — |
| `impl/_cuda/*` | mirror | mirror | — |

**Shape-chain format:**

```
forward() input_ids=[1,128] → embedding=[1,128,256] → stack=[1,128,256] → logits=[1,128,256]
train_step() input=[16,128] → logits=[16,128,256] loss=2.3456 grad_norm=0.789
```

### H.8 Testing Status

| Test | What | Location | Status |
|------|------|----------|--------|
| **Basic logging** | `setup_logging()` creates handlers, filters work | — | ✅ All module tests still pass (191/191) |
| **No regressions** | All backends pass after logging addition | `tests/` — full suite | ✅ run, pyright: 0 errors |

### H.9 Quick Reference — What to Log

```
TRAIN:
  scripts/train.py    → startup, epoch, batch, checkpoint (INFO/DEBUG)
  impl/_*/training.py → train_step, gradient clip, optimizer step (DEBUG)
  impl/_np/model.py   → forward/backward shapes (DEBUG)
  impl/_*/modules.py  → each operation shape chain (DEBUG/TRACE)

INFERENCE:
  scripts/infer.py    → startup, prompt, output, throughput (INFO)
  impl/_*/inference.py→ per-token sampling (DEBUG)
  impl/_*/kv_cache.py → cache alloc/append (DEBUG)

EDUCATIONAL:
  Attention entropy, gradient stats, activation stats, token sampling top-5 (DEBUG/TRACE)
```

## Key Decisions
1. **4 backends, same topology** — All accept JSON config + env vars
2. **NumPy as truth benchmark** — Pure Python/numpy reference implementation
3. **Independent training** — Each backend trains independently with its own RNG
4. **Round-trip tests** — Save NumPy format → load into any backend → compare
5. **Flat checkpoint** — All save/load `ckpt.npz` as flat dict
6. **Gated residuals** — Learnable scalar gates, not sigmoid
7. **Multi-level KV** — Configurable cache length (training vs inference)
8. **PyTorch nn.Module** — PyTorch/Triton/CUDA models are nn.Module instances

## Platform Notes
- NVIDIA Jetson AGX Orin 64GB, JetPack 6.2.2, CUDA 12.6, ~20 TFLOPS
- Must use `torch.zeros()`/`torch.ones()` for tensor init (never `torch.empty()`)
- nvgpu driver requires contiguous tensors for `copy_()`
