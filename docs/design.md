# Design Document: Decoder-Only Transformer Learning Project

**Date:** 2026-06-14
**Goal:** Build a fully functional decoder-only transformer LLM in 4 equivalent implementations (NumPy, PyTorch, Triton, CUDA) for educational purposes.

---

## 1. Project Overview

Build a decoder-only text-to-text transformer that can be trained on the TinyStories dataset and used for text generation. The same model architecture will be implemented across 4 backends, each producing identical results given the same seed and input.

**Philosophy:**
- **NumPy** = "Read to learn" — Every math operation explained with comments, every matrix dimension annotated
- **PyTorch** = "Production ready" — Clean OOP, proper interfaces, docstrings, minimal comments
- **Triton** = "Kernel learning" — Custom CUDA kernels written in Triton DSL for attention, MoE, normalization
- **CUDA** = "Bare metal" — Lowest-level GPU programming, manual memory management, device code

## 2. Architecture Specifications

### 2.1 Configurable Parameters

```
TransformerConfig:
  vocab_size: int          = 4096       # Token vocabulary size (BPE)
  context_length: int      = 256         # Max sequence length
  embed_dim: int           = 512         # Hidden embedding dimension
  n_layers: int            = 8           # Number of transformer blocks
  n_heads: int             = 8           # Number of query heads
  n_groups: int            = 8           # KV groups (1=GQA, n_heads=self-attn)
  rope_dim: int            = 0           # 0=full, >0=partial RoPE
  n_experts: int           = 4           # Number of MoE experts
  top_k: int               = 2           # Top-k experts per token
  expert_dim: int          = 0           # 0=4×embed_dim, >0=override
  max_length: int          = 2048        # Max generation length (inference)
  quant_type: str          = "1-bit"     # "none", "1-bit", "2-bit", "4-bit"
  qkv_cache_type: str      = "naive"     # "naive", "turboquant"
  load_balance_loss: float = 0.0         # Weight for MoE load balance loss
  seed: int                = 42          # Random seed
```

### 2.2 Model Architecture

```
Input (tokens: [batch, seq_len])
    │
    ▼
┌─────────────────────┐
│  Token Embedding     │ shape: [B, S, D]
├─────────────────────┤
│  RMSNorm             │
├─────────────────────┤
│  RoPE                │ inject position info into Q, K
├─────────────────────┤
│  Stream 1: MHA       │ → [B, S, D]
│    ├── Q, K, V proj  │
│    ├── Multi-head     │
│    ├── (GQA toggle)   │
│    └── Output proj    │
├─────────────────────┤
│  Stream 2: MoE       │ → [B, S, D]
│    ├── Top-k routing  │
│    ├── Expert 1       │ FFN(4D → D)
│    ├── Expert 2       │ FFN(4D → D)
│    │  ...             │
│    └── Expert N       │
├─────────────────────┤
│  RMSNorm             │
├─────────────────────┤
│  Output LM Head      │ → [B, S, V]
└─────────────────────┘

Each "Stream" follows a residual block pattern:
  h = h + Stream(h)
```

### 2.3 Attention Mechanics

**Self-Attention (default, GQA disabled):**
```
Q = X @ W_Q   → [B, S, n_heads, head_dim]
K = X @ W_K   → [B, S, n_heads, head_dim]
V = X @ W_V   → [B, S, n_heads, head_dim]

scores = Q @ K^T / sqrt(head_dim)   → [B, n_heads, S, S]
attn = softmax(scores)             → [B, n_heads, S, S]
output = attn @ V                   → [B, n_heads, S, head_dim]
```

**GQA (enabled, n_groups < n_heads):**
```
Q = [B, S, n_heads, head_dim]
K = [B, S, n_groups, head_dim]   # Shared across groups
V = [B, S, n_groups, head_dim]   # Shared across groups
Q reshaped to [B, S, n_groups, group_size, head_dim]
```

**KV Cache (during inference):**
- **Naive:** Store K_cache, V_cache as full-precision tensors, append new positions
- **TurboQuant:** Store compressed 1-bit K, V tensors with per-channel scaling factors

## 3. Project Structure

```
project-root/
├── AGENTS.md                    # Dev guidelines (existing)
├── pyproject.toml               # Dependencies & config
├── task_plan.md                 # Roadmap
├── findings.md                  # Research & decisions
├── progress.md                  # Session log
│
├── shared/                      # Common code shared by ALL backends
│   ├── __init__.py
│   ├── config.py                # TransformerConfig dataclass
│   ├── tokenizer.py             # BPE + CharLevelTokenizer
│   ├── dataset.py               # TinyStories loader
│   ├── checkpoint.py            # Load/save in .npz format
│   └── constants.py             # Parameter name constants
│
├── impl/                        # 4 backend implementations
│   ├── numpy/                   # Learning-focused, heavily commented
│   │   ├── __init__.py
│   │   ├── model.py             # Full decoder model (all layers internal)
│   │   ├── layers.py            # Core layers: Embedding, LayerNorm, etc.
│   │   ├── attention.py         # MHA + GQA
│   │   ├── moe.py               # MoE + routing
│   │   ├── kvcache.py           # Naive + TurboQuant KV caches
│   │   ├── loss.py              # CrossEntropy, FocalLoss if added
│   │   ├── optimizer.py         # SGD, AdamW
│   │   ├── train.py             # Full training loop
│   │   ├── infer.py             # Autoregressive inference engine
│   │   └── cli.py               # CLI interface
│   │
│   ├── torch/                   # Production-ready PyTorch
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── layers.py            # nn.Module equivalents
│   │   ├── attention.py
│   │   ├── moe.py
│   │   ├── kvcache.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── cli.py
│   │
│   ├── triton/                  # GPU kernels in Triton DSL
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── kernels.py           # Custom kernel implementations
│   │   ├── attention.py
│   │   ├── moe.py
│   │   ├── kvcache.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── cli.py
│   │
│   └── cuda/                    # Bare-metal GPU code
│       ├── __init__.py
│       ├── model.py
│       ├── kernels.py           # CUDA C device code + python bindings
│       ├── attention.py
│       ├── moe.py
│       ├── kvcache.py
│       ├── loss.py
│       ├── optimizer.py
│       ├── train.py
│       ├── infer.py
│       └── cli.py
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   │
│   ├── unit/                    # Tests per module
│   │   ├── test_config.py
│   │   ├── test_tokenizer.py
│   │   ├── test_dataset.py
│   │   ├── test_checkpoint.py
│   │   ├── numpy/
│   │   │   ├── test_layers.py
│   │   │   ├── test_attention.py
│   │   │   ├── test_moe.py
│   │   │   ├── test_kvcache.py
│   │   │   ├── test_model.py
│   │   │   └── test_train.py
│   │   ├── torch/
│   │   │   ├── test_layers.py
│   │   │   └── ...
│   │   ├── triton/
│   │   │   └── test_kernels.py
│   │   └── cuda/
│   │       └── test_kernels.py
│   │
│   └── cross_backend/           # Equivalence tests
│       ├── test_layer_parity.py       # Standalone layer → rtol=1e-4
│       ├── test_single_chain.py       # 1-layer model → rtol=1e-3
│       ├── test_multilayer_chain.py   # 2+ layers → rtol=1e-2
│       ├── test_checkpoint_roundtrip.py
│       ├── test_inference_equivalence.py
│       └── test_cli_equivalence.py
│
├── scripts/
│   ├── train.sh                 # Quick train script
│   ├── infer.sh                 # Quick inference script
│   └── benchmark.sh             # Performance comparison
│
├── examples/
│   ├── train.sh                 # Example commands
│   └── cli_usage.md             # Documentation
│
├── models/                      # Saved checkpoints go here
└── outputs/                     # Generated text samples
```

## 4. Cross-Backend Equivalence Strategy

### 4.1 Checkpoint Format

**File 1: `config.json`**
```json
{
  "model_type": "decoder_transformer",
  "vocab_size": 4096,
  "context_length": 256,
  "embed_dim": 512,
  "n_layers": 8,
  "n_heads": 8,
  "n_groups": 8,
  "rope_dim": 0,
  "n_experts": 4,
  "top_k": 2,
  "expert_dim": 0,
  "max_length": 2048,
  "quant_type": "none",
  "seed": 42
}
```

**File 2: `model.npz`** — Binary numpy arrays for every parameter:
```
embeddings  : [vocab_size, embed_dim]
layers.0.attention.qkv_proj  : [embed_dim, 3*n_heads*head_dim]
layers.0.attention.o_proj    : [embed_dim, embed_dim]
layers.0.moe.router          : [embed_dim, n_experts]
layers.0.moe.experts.0.w1    : [embed_dim, expert_dim]
layers.0.moe.experts.0.w2    : [expert_dim, embed_dim]
layers.0.ln.weight           : [embed_dim]
layers.0.ln.bias             : [embed_dim]
... (repeat for all layers)
final_ln.weight              : [embed_dim]
output_proj                  : [embed_dim, vocab_size]
```

**File 3: `vocab.json`** — tokenizer vocabulary

**Compatibility:**
- NumPy can write `.npz` and Torch can read it via `torch.from_numpy(np.load())`
- Torch can write via `torch.save(model.state_dict(), ...)` and NumPy reads by converting tensors to numpy
- All backends read `config.json` then map keys to their own parameter store

### 4.2 Parameter Naming Convention

All backends use the same hierarchy for parameter names:
```
layers.{i}.{module}.{name}
├── layers.0.attention.qkv_proj.weight
├── layers.0.attention.o_proj.weight
├── layers.0.moe.experts.0.w1.weight
├── layers.0.moe.experts.0.w2.weight
├── layers.0.moe.router.weight
├── layers.0.ln.weight
├── layers.0.ln.bias
├── layers.0.attention.qkv_proj.bias     (if applicable)
└── ...
```

### 4.3 Random Seed Handling

```python
# Global seed set in config.seed
# Every random call prefixed with seed:
import numpy as np
import torch
import random

def set_global_seed(seed: int):
    random.seed(seed)            # Python RNG
    np.random.seed(seed)         # NumPy RNG
    torch.manual_seed(seed)      # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # All CUDA GPUs

# For reproducibility in training:
# 1. Shuffle indices uses same RNG state
# 2. MoE routing randomness (if any) uses same RNG
# 3. Dropout/quantization noise (if any) uses same RNG
# 4. Forward pass is deterministic (no batch norm variance issue)
```

### 4.4 Equivalence Test Matrix

| Test | NumPy vs PyTorch | NumPy vs Triton | NumPy vs CUDA | PyTorch vs Triton | PyTorch vs CUDA | Triton vs CUDA |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standalone Layer | ✅ | — | — | — | — | — |
| Model Forward | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Model Backward | ✅ | — | — | — | — | — |
| Training Step | ✅ | — | — | — | — | — |
| Inference Output | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Training Convergence | ✅ | — | — | — | — | — |

**Tolerances (following AGENTS.md tiered policy):**
- Standalone layer (tested in isolation): `rtol=1e-4, atol=1e-4`
- 1-layer model (single residual chain): `rtol=1e-3, atol=1e-3`
- 2+ layer model (multi-chain): `rtol=1e-2, atol=1e-2`
- Inference output match: exact token string match (integer comparison)

## 5. Implementation Priority Order

```
Phase A: Shared Foundation (NumPy-first approach) ✅ COMPLETE
Phase B: NumPy Implementation (complete) ✅ COMPLETE
Phase C: PyTorch Implementation ⏳ PLANNED (Not Started)
Phase D: Triton Implementation 🔲 Not Started
Phase E: CUDA Implementation 🔲 Not Started
Phase F: Integration & E2E 🔲 Not Started
```

### Phase A: Shared Foundation ✅ COMPLETE
**Goal:** Create shared code that all 4 backends will import
**Prerequisite:** NumPy implementation MUST exist first (backend for reference)

1. `config.py` — TransformerConfig dataclass
2. `constants.py` — Parameter name string constants (no raw magic strings allowed)
3. `tokenizer.py` — BPE + CharLevel tokenizers
4. `dataset.py` — TinyStories streaming loader → local JSON (no external downloads)
5. `checkpoint.py` — Load/Save in shared format
6. `conftest.py` — Pytest fixtures shared across backends

### Phase B: NumPy Implementation ✅ COMPLETE
**Goal:** Reference implementation, fully tested
**Principle:** Educational, every operation explained
**Result:** 21 commits (b0–b19), ~70 tests, all tests pass

1. Layer primitives: Embedding, RMSNorm, SiLU, SwiGLU, MLP
2. RoPE position encoding (configurable dimension)
3. MHA (configurable heads)
4. MoE (configurable experts, top-k)
5. TransformerBlock → combined residual stream
6. DecoderStack (layer stacking)
7. Full model: embedding → blocks → RMSNorm → output_proj
8. Forward + backward pass (manual gradient computation)
9. Loss: CrossEntropy
10. Optimizer: SGD, AdamW
11. Naive KV Cache (full precision)
12. TurboQuant KV Cache (1-bit compression with calibration)
13. Training loop (full: data loading → batch → forward → loss → backward → step)
14. Inference engine (autoregressive with KV cache)
15. CLI: `train.py` + `infer.py` integration
16. Unit tests for each component
17. Cross-backend reference tests (numpy is the baseline)

### Phase C: PyTorch Implementation ⏳ PLANNED (Not Started)
**Goal:** Production-ready, mirrors NumPy behavior exactly
**Status:** Plan exists at `docs/phase_c_plan.md` — execution ready

1. nn.Module equivalents of all layers
2. Same model construction with same parameter mapping
3. Automatic backward (torch.autograd) replaces manual gradients

4. Same training loop (uses torch.optim, torch.utils.data)
5. Same inference engine (torch-based KV cache)
6. Same CLI interface
7. Cross-backend parity tests (numpy vs torch)
8. Performance comparison (optional)

### Phase D: Triton Implementation 🔲 Not Started
**Goal:** GPU kernels for compute-heavy operations

1. Triton kernels for: LayerNorm, Attention, MoE routing, SiGLU
2. Model construction using Triton kernels where applicable
3. Non-Triton parts use standard CUDA/PyTorch
4. Same training + inference loops
5. Cross-backend parity tests
6. Profiling comparison

### Phase E: CUDA Implementation 🔲 Not Started
**Goal:** Lowest-level GPU programming

1. nvidia/cuda-python bindings (no PyTorch dependency)
2. CUDA kernel source files (.cu) for all compute operations
3. Python wrapper to call CUDA kernels directly
4. Full model construction from kernels
5. Same training + inference loops
6. Cross-backend parity tests

### Phase F: Integration & E2E 🔲 Not Started
**Goal:** Verify everything works end-to-end

1. Train on TinyStories with each backend (NumPy, PyTorch)
2. Save checkpoints, verify cross-load (torch→numpy, numpy→torch)
3. Run inference on same prompt, verify identical output
4. Generate demo text samples from each backend
5. Final verification script runs all tests

## 6. Training Pipeline

### 6.1 Data Flow

```
TinyStories raw text
    │
    ▼
[Tokenizer] → token IDs
    │
    ▼
[Dataset] → (tokens: [seq_len + padding]) with proper attention masking
    │
    ▼
[Batches] → token_ids: [B, S], labels: [B, S]
    │
    ▼
[Forward] → logits: [B, S, V]
    │
    ▼
[Loss = CrossEntropy(predictions, labels)]
    │
    ▼
[Backward] → gradients for all parameters
    │
    ▼
[Optimizer Step] → update parameters
```

### 6.2 Training Configuration (Default)

```python

```

## 7. Inference Pipeline

```
Input: text prompt "Once upon a time"
    │
    ▼
[Tokenizer.encode] → token_ids = [12, 45, 23, 67, 89]
    │
    ▼
[Forward] → logits[:, -1, :] → softmax → sample → new_token
    │
    ▼
[Store K, V in KV Cache]
    │
    ▼
[Next token] new_input = [new_token]
    │
    ▼
[Forward with cached K, V] → new_token_2
    │
    ▼
[Repeat until] EOS detected or max_length reached
    │
    ▼
[Tokens: [12, 45, 23, 67, 89, tok_A, tok_B, ...]]
    │
    ▼
[Tokenize.decode] → "Once upon a time A, B, ..."
```

### 7.1 Decoding Strategies
- **Greedy:** `argmax(logits)` — deterministic, same output every time
- **Sampling with temperature:** probabilistic, controlled creativity (default for demos)

### 7.2 KV Cache Usage During Inference

**Naive Cache:**
```python
class NaiveKVCache:
    def __init__(self, max_length: int, n_layers: int, n_heads: int, head_dim: int):
        self.k_cache = torch.zeros(n_layers, n_heads, max_length, head_dim)
        self.v_cache = torch.zeros(n_layers, n_heads, max_length, head_dim)
        self.current_pos = 0

    def update(self, k: Tensor, v: Tensor, pos: int):
        """Append single step KV to cache at given position."""
        self.k_cache[:, :, pos:pos+1, :] = k
        self.v_cache[:, :, pos:pos+1, :] = v
        self.current_pos = pos + 1

    def get(self) -> tuple[Tensor, Tensor]:
        """Return entire cached K, V tensors."""
        return self.k_cache[:, :, :self.current_pos, :], self.v_cache[:, :, :self.current_pos, :]
```

**TurboQuant Cache:**
```python
class TurboQuantKVCache:
    def __init__(self, max_length: int, n_layers: int, n_heads: int, head_dim: int, quant_type: str = "1-bit"):
        self.max_length = max_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.quant_type = quant_type  # "1-bit", "2-bit", "4-bit"

        # Quantized storage
        bit_depth = {"1-bit": 1, "2-bit": 2, "4-bit": 4}[quant_type]
        self.k_bits = torch.zeros(n_layers, n_heads, max_length, head_dim, dtype=torch.uint8)
        self.v_bits = torch.zeros(n_layers, n_heads, max_length, head_dim, dtype=torch.uint8)
        self.k_scale = torch.ones(n_layers, n_heads, 1, head_dim)  # Per-channel scaling
        self.v_scale = torch.ones(n_layers, n_heads, 1, head_dim)

        self.current_pos = 0

    def calibrate(self, k_stats: Tensor, v_stats: Tensor):
        """Pre-compute scaling factors from validation data."""
        self.k_scale = compute_scale(k_stats, quant_type)
        self.v_scale = compute_scale(v_stats, quant_type)

    def update(self, k: Tensor, v: Tensor, pos: int):
        """Quantize and store."""
        self.k_bits[:, :, pos:pos+1, :] = quantize(k, self.k_scale, self.quant_type)
        self.v_bits[:, :, pos:pos+1, :] = quantize(v, self.v_scale, self.quant_type)
        self.current_pos = pos + 1

    def get(self) -> tuple[Tensor, Tensor]:
        """Dequantize cached values."""
        k = dequantize(self.k_bits[:, :, :self.current_pos, :], self.k_scale, self.quant_type)
        v = dequantize(self.v_bits[:, :, :self.current_pos, :], self.v_scale, self.quant_type)
        return k, v
```

Compression benefit: 1-bit stores 32 values in 4 bytes instead of 128 bytes (fp32) → 32× savings.

## 8. Test Strategy (TDD Approach)

### 8.1 Test Categories

| Category | Scope | Tolerance | Run When |
|----------|-------|-----------|----------|
| Layer unit tests | Single layer in isolation | 1e-4 | After every layer change |
| Component tests | Layer + attention or MoE | 1e-3 | After component is complete |
| Model forward tests | Full forward no grad | 1e-3 | After model is complete |
| Model backward tests | Full backward | 1e-2 (multi-layer) | After backward is implemented |
| Cross-backend tests | Backend A vs Backend B | 1e-4 to 1e-2 | After each backend is complete |
| E2E tests | Train → Save → Load → Infer | Exact match | After integration |

### 8.2 Test Naming Convention

```python
# Unit tests: what + behavior + expected
def test_layer_norm_stabilizes_variance():
    """LayerNorm should produce zero mean, unit variance."""

def test_rope_applies_positional_rotation():
    """RoPE should rotate Q and K by position-dependent angles."""

def test_moe_selects_top_k():
    """MoE router should select exactly top_k experts."""

# Cross-backend tests: what + between + equivalence
def test_attention_numpy_torch():
    """Same input → same output for MHA layer (numpy vs torch)."""

def test_decoder_stack_equivalent():
    """Same model, same seed → identical forward pass across backends."""
```

### 8.3 Test-Driven Workflow

```
1. Write failing test         (red)
2. Implement minimal code     (green)
3. Refactor + cross-backend   (blue)
4. Repeat for next component
```

## 9. Build Sequence (Phase-by-Phase)

### Phase A: Shared Foundation ✅ COMPLETE
```
1. config.py + constants.py
2. tokenizer.py
3. dataset.py (TinyStories loader)
4. checkpoint.py
5. conftest.py
```
*Acceptance:* All shared code imports without errors.

### Phase B: NumPy (15 sub-phases) ✅ COMPLETE
**Status:** 21 commits (b0-b19), ~70 tests pass, all quality gates clean
```
B1: Basic layers (Embedding, RMSNorm, SiLU, SwiGLU)
B2: RoPE
B3: MHA (without GQA)
B4: MHA (with GQA toggle)
B5: MoE routing + expert FFN
B6: TransformerBlock (attention + MoE + residual)
B7: DecoderStack (multiple blocks)
B8: Full model (embedding + blocks + output)
B9: Forward pass (full model, verified)
B10: Backward pass (full model, verified)
B11: Loss + Optimizer
B12: Training loop
B13: Naive KV Cache
B14: TurboQuant KV Cache
B15: CLI + Inference
```
*Acceptance:* 100% unit tests pass. Model trains on TinyStories. Generates coherent text.

### Phase C: PyTorch ⏳ PLANNED (Not Started)
**Status:** Execution ready in `docs/phase_c_plan.md` — 14 sub-phases (C0-C14), 20+ commits, ~65-70 tests
```
C1: Layer wrappers (nn.Module)
C2: Full model construction
C3: Cross-backend parity tests
C4: Training loop (automatic diff)
C5: Inference engine
C6: CLI integration
```
*Acceptance:* PyTorch matches NumPy on all parity tests.

### Phase D: Triton 🔲 Not Started
```
D1: Custom kernels (LayerNorm, Attention, MoE, Activation)
D2: Model using Triton kernels
D3: Cross-backend parity tests
D4: Training + Inference
```
*Acceptance:* Triton matches NumPy/PyTorch on parity tests.

### Phase E: CUDA 🔲 Not Started
```
E1: CUDA kernels (all compute ops)
E2: Python wrapper for kernels
E3: Full model
E4: Cross-backend parity tests
E5: Training + Inference
```
*Acceptance:* CUDA matches NumPy/PyTorch/Triton on parity tests.

### Phase F: Integration 🔲 Not Started
```
F1: Train model on TinyStories (NumPy + PyTorch)
F2: Checkpoint cross-load (torch→numpy, numpy→torch)
F3: Inference equivalence (same prompt → same output)
F4: Demo generation samples
F5: Final verification script
```

## 10. Dataset Strategy

### 10.1 TinyStories
- **Source:** `allenai/tinystories` (HuggingFace)
- **Size:** ~8MB (all of it, not a subset)
- **Content:** Simple English stories (AI-generated, clean language)
- **Training subset:** 95% train / 5% val

### 10.2 Tokenizer

```python
# Default: BPE with ~4096 tokens
# Fallback: CharLevel for tiny demos (< 128 tokens)
# Vocab size configurable: minimal(128) → demo(512) → small(1024) → normal(4096)

# Training tokenization:
tokenizer.train(text_corpus, vocab_size=config.vocab_size, special_tokens=["<PAD>", "<EOS>"])

# Usage:
tokens = tokenizer.encode(text)           # str → list[int]
text = tokenizer.decode(tokens)           # list[int] → str
```

### 10.3 Training Data Pipeline

```python
# Dataset creates sliding windows of context_length tokens
# Each forward pass: [B, S] → labels shifted by 1
# Prediction: predict token at position t based on tokens 0..t-1

# Example (context_length=4):
# Input:  [12, 45, 67]       Target: [45, 67, 89]
# (predict 45 from 12, predict 67 from [12,45], etc.)

# Batching:
# Stack multiple sequences, pad to same length, apply attention mask
# Loss only computed on non-padded positions
```

## 11. CLI Interface

### 11.1 Training

```bash
# Quick training (defaults)
uv run src/train.py train

# Full custom training
uv run src/train.py train \
    --backend numpy \
    --vocab_size 4096 --context_length 256 --embed_dim 512 \
    --n_layers 8 --n_heads 8 --n_groups 8 \
    --n_experts 4 --top_k 2 \
    --max_length 512 --quant_type none \
    --epochs 10 --batch_size 32 --lr 0.001 \
    --seed 42 --save_path ./models/tiny_llm
```

### 11.2 Inference

```bash
# Quick inference
uv run src/train.py inference --prompt "Once upon a time"

# Full options
uv run src/train.py inference \
    --model_path ./models/tiny_llm/ckpt.npz \
    --backend torch \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 50 \
    --kv_cache_type naive
```

### 11.3 Verification

```bash
# Cross-backend equivalence check
uv run src/verify.py

# Performance benchmark
uv run src/bench.py
```

## 12. Dependencies

### 12.1 Required

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.24 | Math operations, array operations |
| torch | ≥ 2.0 | PyTorch implementation |
| datasets | ≥ 2.14 | TinyStories dataset loader (HuggingFace) |
| tokenizers | ≥ 0.15 | BPE tokenizer |

### 12.2 Optional (backend-specific)

| Package | Version | Purpose |
|---------|---------|---------|
| triton | ≥ 2.2 | GPU kernel implementation |
| cuda-python | ≥ 12.0 | Bare-metal GPU kernels |

### 12.3 Dev Tooling

| Package | Purpose |
|---------|---------|
| pytest | Testing framework |
| pytest-timeout | Ensure tests don't hang |
| ruff | Linting, formatting, import sorting |
| pyright | Type checking (strict mode) |

## 13. Quality Gates

Each phase must pass these gates before proceeding:

| Gate | Requirement |
|------|-------------|
| **Code Quality** | pyright passes, ruff passes (zero warnings) |
| **Test Coverage** | All unit tests for the phase pass |
| **Cross-Backend** | Existing backends still match after new backend |
| **Documentation** | Key classes/functions have docstrings |
| **No Regressions** | Existing tests still pass after changes |

## 14. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| TinyStories too small | Model can't learn meaningful patterns | Use full dataset, longer training, smaller config |
| Cross-backend drift | NumPy vs Torch produce different results | TDD: always write cross-backend test first |
| CUDA/Triton complexity | Hard to debug GPU code | Start with reference NumPy, verify before adding GPU |
| TurboQuant complexity | Hard to get right quantization | Implement naive KV cache first, validate correctness then add quantization |
| Memory issues | Large models on small GPUs | Ensure NumPy/PyTorch work on CPU first; CUDA/Triton scale independently |
| Seed non-determinism | Results vary across runs | Single seed source, explicit RNG state management |

## 15. Implementation Order Justification

```
Why NumPy first?
├─ Everyone can read and understand it without CUDA expertise
├─ Each line maps directly to a mathematical operation
├─ Provides reference implementation for all cross-backend tests
├─ Debugging is easy (no GPU, no framework black boxes)
│
Why PyTorch second?
├─ Direct equivalent of NumPy implementation
├─ Same API surface, just different backend
├─ Easiest cross-backend test (torch ↔ numpy)
│
Why Triton third?
├─ Requires PyTorch knowledge (uses torch for data)
├─ Only replaces compute kernels, not model architecture
├─ GPU-only, harder to debug without CPU fallback
│
Why CUDA last?
├─ Hardest to implement (no framework helpers)
├─ Requires CUDA toolkit installation
├─ Reference implementations already exist (numpy + torch + triton)
└─ Best as a learning exercise after understanding the abstractions
```
