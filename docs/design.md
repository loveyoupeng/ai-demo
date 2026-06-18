# Design Document: Decoder-Only Transformer Learning Project

**Date:** 2026-06-14 (last synced: 2026-06-18)
**Goal:** Build a fully functional decoder-only transformer LLM in 4 equivalent implementations (NumPy, PyTorch, Triton, CUDA) for educational purposes.

---

## 1. Project Overview

Build a decoder-only text-to-text transformer that can be trained on the TinyStories dataset and used for text generation. The same model architecture will be implemented across 4 backends, each producing identical results given the same seed and input.

**Philosophy:**
- **NumPy** = "Read to learn" — Every math operation explained with comments, every matrix dimension annotated
- **PyTorch** = "Production ready" — Clean OOP, proper interfaces, docstrings, minimal comments
- **Triton** = "Kernel learning" — Custom CUDA kernels written in Triton DSL for attention, MoE, normalization
- **CUDA** = "Bare metal" — Lowest-level GPU programming, manual memory management, device code

---

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

Config loading priority: CLI args → env vars → config file → defaults (via `shared/config_utils.py`).

### 2.2 Model Architecture

**Current implementation: Post-Norm with gated residuals + dropout**

```
Input (tokens: [batch, seq_len])
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DecoderStack (n_layers)                                             │
│  ┌──────────── TransformerBlock ──────────┐                          │
│  │  Input: x [B, S, D]                     │                          │
│  │                                          │                          │
│  │  Stream 1: Attention                    │                          │
│  │    attn_out = MHA(x)                    │ [B, S, D]               │
│  │    h = x + attn_out                     # residual add FIRST       │
│  │    h = RMSNorm(h)                       # post-norm → [B, S, D]   │
│  │    h = h + sigmoid(gate1) * h           # gated residual          │
│  │    h = dropout(h)                       # dropout (training only) │
│  │                                          │                          │
│  │  Stream 2: MoE                          │                          │
│  │    moe_out = MoE(h)                     │ [B, S, D]               │
│  │    out = h + moe_out                    # residual add            │
│  │    out = RMSNorm(out)                   # post-norm → [B, S, D]   │
│  │    out = out + sigmoid(gate2) * out     # gated residual          │
│  │    out = dropout(out)                   # dropout                 │
│  │                                          │                          │
│  │  Output: out [B, S, D]                   │                          │
│  └──────────────────────────────────────────┘                          │
│      block_0 → block_1 → ... → block_{n_layers-1}                     │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────┐
│  RMSNorm (final)    │ [B, S, D]
├─────────────────────┤
│  Output LM Head     │ → [B, S, V]
└─────────────────────┘
```

**Key differences from original design:**
- **Post-Norm** instead of Pre-Norm: residual add first, then RMSNorm, then gated residual
- **2 gates**: gate1 for attention stream, gate2 for MoE stream. `sigmoid(0) = 0.5` at init → partial gating, gate learns to open during training
- **Dropout**: 0.05 rate by default, active only in training mode. Disabled in eval mode for deterministic inference
- **Gradient clipping**: Applied after `loss.backward()` before optimizer step in both backends

**Per-block architecture (code formula):**
```python
h = x + MHA(x)                  # residual add
h = RMSNorm(h)                  # post-norm
h = h + sigmoid(gate1) * h      # gated residual
h = dropout(h)                  # dropout (training only)
moe_out = MoE(h)
out = h + moe_out               # residual add
out = RMSNorm(out)              # post-norm
out = out + sigmoid(gate2) * out  # gated residual
out = dropout(out)              # dropout
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

**PyTorch MHA biases:** 4 biases total (Wq/bq, Wk/bk, Wv/bv, Wo/bo). Note: Wk.bias has a mathematically zero gradient due to softmax attention weight property.

### 2.4 MoE Architecture

```
MoE(x [B, S, D]) → [B, S, D]

Step 1: Compute routing scores
  logits = Gate(x [B, S, D]) @ W_router [D, n_experts] → [B, S, n_experts]
  top_k_indices = argmax(logits, k=top_k) → [B, S, top_k]
  top_k_weights = softmax(logits for top_k) → [B, S, top_k]

Step 2: Route tokens to experts
  For each token position (b, s):
    For each expert k in top_k:
      expert_output[k] = Expert_k(x[b, s])  # Each expert = SwiGLU FFN

Step 3: Weighted sum
  out[b, s] = Σ_k (top_k_weights[b, s, k] * expert_output[k])
```

Each expert implementation:
- SwiGLU: splits input into 3 parts (W1, W2, gate), applied element-wise
- `expert_dim`: feedforward hidden dimension (default = 4 × embed_dim)

---

## 3. Project Structure

**Date:** 2026-06-14 (last synced: 2026-06-18)

**Note:** Only 2 of 4 planned backends have been implemented. Triton and CUDA are not yet started.

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
│   ├── config_utils.py          # Unified config reader with source tracking
│   ├── tokenizer.py             # BPE + CharLevelTokenizer
│   ├── dataset.py               # TinyStories loader
│   ├── checkpoint.py            # Load/save in .npz format
│   └── constants.py             # Parameter name constants (no raw strings)
│
├── impl/                        # Backend implementations
│   ├── _np/                     # NumPy — learning-focused, heavily commented
│   │   ├── __init__.py
│   │   ├── modules.py           # All layers: Embedding, RMSNorm, MHA, MoE, TransformerBlock, DecoderStack
│   │   ├── model.py             # Full decoder model (all layers internal)
│   │   ├── cross_entropy.py     # Cross-entropy loss
│   │   ├── optimizer.py         # AdamW optimizer
│   │   ├── training.py          # Training loop
│   │   ├── inference.py         # Autoregressive inference engine
│   │   ├── kv_cache.py          # Naive KV cache
│   │   ├── turboquant_kv_cache.py  # TurboQuant 1-bit KV cache
│   │   ├── cli.py               # CLI interface
│   │   └── utils/               # Utility package
│   │
│   ├── _torch/                  # PyTorch — production-ready
│   │   ├── __init__.py
│   │   ├── layers.py            # nn.Module equivalents (attention, MoE, TransformerBlock, etc.)
│   │   ├── model_config.py      # Model config + TorchModel with save/load/load_from_numpy
│   │   ├── cross_entropy.py     # Cross-entropy loss
│   │   ├── optimizer.py         # AdamW optimizer (via torch.optim)
│   │   ├── training.py          # Training loop (autograd)
│   │   ├── inference.py         # Inference engine
│   │   ├── kv_cache.py          # Naive KV cache
│   │   ├── turboquant_kv_cache.py  # TurboQuant 1-bit KV cache
│   │   └── cli.py               # CLI entry point (argparse)
│   │
│   ├── triton/                  # GPU kernels in Triton DSL (NOT STARTED)
│   │   └── ...
│   │
│   └── cuda/                    # Bare-metal GPU code (NOT STARTED)
│       └── ...
│
├── tests/                       # Test suite (63 files, 421+ tests)
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── cross_backend/
│   │   └── test_parity.py       # NumPy vs PyTorch parity (5 tests, rtol=1e-3)
│   ├── unit/
│   │   ├── shared/
│   │   │   ├── test_config.py
│   │   │   ├── test_tokenizer.py
│   │   │   ├── test_constants.py
│   │   │   ├── test_dataset.py
│   │   │   └── test_checkpoint.py
│   │   ├── _np/                 # NumPy backend tests (22 files)
│   │   │   ├── test_architecture_improvements.py  # Post-norm + gated residuals + dropout
│   │   │   ├── test_modules.py
│   │   │   ├── test_attn.py
│   │   │   ├── test_moe.py
│   │   │   ├── test_transformer_block.py
│   │   │   ├── test_decoder_stack.py
│   │   │   ├── test_model.py
│   │   │   ├── test_rmsnorm.py
│   │   │   ├── test_rope.py
│   │   │   ├── test_silu.py
│   │   │   ├── test_swiglu.py
│   │   │   ├── test_naive_kvcache.py
│   │   │   ├── test_turboquant_kvcache.py
│   │   │   ├── test_cross_entropy.py
│   │   │   ├── test_optimizer.py
│   │   │   ├── test_training.py
│   │   │   ├── test_inference.py
│   │   │   ├── test_full_pipeline.py
│   │   │   └── test_gradient_clipping.py
│   │   ├── _torch/              # PyTorch backend tests (20 files)
│   │   │   └── (mirrors _np structure)
│   │   └── scripts/             # Script tests (10 files)
│   │       ├── test_train_script.py
│   │       ├── test_infer_script.py
│   │       ├── test_verify_equivalence.py
│   │       └── test_auto_test_equivalence.py
│   └── cross_backend/           # Cross-backend parity (merged into main)
│
├── scripts/
│   ├── train.py                 # Unified training entry point (--backend numpy/torch)
│   ├── infer.py                 # Unified inference CLI with context status
│   ├── download_tinystories.py  # Dataset download script
│   ├── verify_equivalence.py    # 6-scenario cross-backend verification
│   └── auto_test_equivalence.py # 8-test automation matrix
│
├── models/                      # Saved checkpoints go here
└── outputs/                     # Generated text samples
```

---

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
embeddings                    : [vocab_size, embed_dim]
blocks.0.mha.q_proj.weight    : [embed_dim, embed_dim]
blocks.0.mha.k_proj.weight    : [embed_dim, embed_dim]
blocks.0.mha.v_proj.weight    : [embed_dim, embed_dim]
blocks.0.mha.o_proj.weight    : [embed_dim, embed_dim]
blocks.0.mha.q_proj.bias      : [embed_dim]
blocks.0.mha.k_proj.bias      : [embed_dim]
blocks.0.mha.v_proj.bias      : [embed_dim]
blocks.0.mha.o_proj.bias      : [embed_dim]
blocks.0.moe.router.weight    : [embed_dim, n_experts]
blocks.0.moe.router.bias      : [n_experts]
blocks.0.moe.experts.0.w1     : [embed_dim, expert_dim]
blocks.0.moe.experts.0.w2     : [expert_dim, embed_dim]
blocks.0.moe.gate1            : [1]      ← NEW: gate for attention stream
blocks.0.moe.gate2            : [1]      ← NEW: gate for MoE stream
blocks.0.ln1_gamma            : [embed_dim]
blocks.0.ln2_gamma            : [embed_dim]
```

**File 3: `vocab.json`** — tokenizer vocabulary

### 4.2 Parameter Naming Convention

Both backends use `shared/constants.py` for all parameter name strings — no raw literals.

**NumPy model keys:** Flat dictionary with keys like:
- `blocks.{i}.mha.q_proj.weight`, `blocks.{i}.mha.k_proj.bias`, etc.
- `blocks.{i}.moe.experts.{j}.w1`, `blocks.{i}.moe.router.weight`, etc.
- `blocks.{i}.ln1_gamma`, `blocks.{i}.gate1`, `blocks.{i}.gate2`

**PyTorch model keys:** Flat dictionary (from `save_as_numpy()`) matches NumPy keys exactly.

**Shared constants:** Defined in `shared/constants.py`:
- Block: `BLOCK_MHA_PREFIX`, `BLOCK_MOE_PREFIX`, `BLOCK_LN`
- LayerNorm: `LAYERNORM_GAMMA`
- Expert: `EXPERT_W1`, `EXPERT_W2`
- Final: `LN_GAMMA`, `EMBEDDING`, `LM_HEAD`

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
```

### 4.4 Equivalence Test Matrix

| Test | NumPy vs PyTorch | NumPy vs Triton | NumPy vs CUDA | PyTorch vs Triton | PyTorch vs CUDA | Triton vs CUDA |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standalone Layer | ✅ | — | — | — | — | — |
| Model Forward | ✅ | — | — | — | — | — |
| Model Backward | ✅ | — | — | — | — | — |
| Training Step | ✅ | — | — | — | — | — |
| Inference Output | ✅ | — | — | — | — | — |
| Training Convergence | ✅ | — | — | — | — | — |
| Checkpoint Round-trip | ✅ | — | — | — | — | — |

**Status:** All NumPy vs PyTorch rows verified — 5 parity tests pass, `weight_diff=0.0`.
Triton: 🔶 Phase E ready to start (GPU confirmed, plan complete).
CUDA: 🔲 Not started.

**Tolerances (following AGENTS.md tiered policy):**
- Standalone layer (tested in isolation): `rtol=1e-4, atol=1e-4`
- 1-layer model (single residual chain): `rtol=1e-3, atol=1e-3`
- 2+ layer model (multi-chain): `rtol=1e-2, atol=1e-2`
- Inference output match: exact token string match (integer comparison)

---

## 5. Implementation Priority Order

```
Phase A: Shared Foundation ✅ COMPLETE
Phase B: NumPy Implementation ✅ COMPLETE
Phase C: PyTorch Implementation ✅ COMPLETE
Phase C+: E2E Training/Inference ✅ COMPLETE
Phase C++: Normalization Improvements ✅ COMPLETE
Phase D: Equivalence Verification ✅ COMPLETE
Phase E: Triton Implementation 🔲 Not Started
Phase F: CUDA Implementation 🔲 Not Started
Phase G: Integration & E2E 🔲 Not Started
```

### Phase A: Shared Foundation ✅ COMPLETE
1. `config.py` — TransformerConfig dataclass
2. `constants.py` — Parameter name constants (no raw strings)
3. `tokenizer.py` — BPE + CharLevel tokenizers
4. `dataset.py` — TinyStories streaming loader → local cache
5. `checkpoint.py` — Load/Save in shared format
6. `config_utils.py` — Unified config reader with source tracking

### Phase B: NumPy Implementation ✅ COMPLETE
1-15. All core layers, RoPE, MHA, MoE, TransformerBlock, DecoderStack
16. Full model: embedding → blocks → RMSNorm → output_proj
17. Forward + backward pass (manual gradient computation)
18. Loss: CrossEntropy, Optimizer: AdamW
19. Training loop (full: data loading → batch → forward → loss → backward → step)
20. Naive KV Cache + TurboQuant KV Cache
21. Inference engine (autoregressive with KV cache)
22. CLI, unit tests, gradient clipping

### Phase C: PyTorch Implementation ✅ COMPLETE
1-15. All layers as nn.Module, same model construction
16. Automatic backward via torch.autograd
17. Same training loop with torch.optim
18. Same inference engine (torch-based KV cache)
19. Cross-backend parity tests (numpy vs torch) — rtol=1e-3, atol=1e-3
20. Save/load (save_as_numpy, load_from_numpy_dict)

### Phase C+: E2E Training/Inference & Equivalence ✅ COMPLETE
1. Unified config system (`shared/config_utils.py`)
2. `scripts/train.py` — single entry, `--backend numpy/torch`
3. `scripts/infer.py` — interactive CLI with context status
4. `scripts/verify_equivalence.py` — 6-prompt equivalence check
5. `scripts/auto_test_equivalence.py` — 8-combination matrix
6. Cross-backend weight diff, token match, distribution check all pass

### Phase E: Triton 🔶 READY TO START
- **GPU confirmed:** CUDA 12.6, cuDNN 9.3, cuBLAS 12.6, 8x Orin GPU
- Custom kernels (LayerNorm, Attention, MoE, Activation) — production-quality with detailed learning comments
- **Goal:** Every kernel documented for learning Triton DSL, memory patterns, numerical stability
- Full model using Triton kernels + cross-backend parity tests + Training + Inference
- **Reference:** `docs/phase_e_plan.md` — 12-stage plan (E0–E11), ~60-80 tests, ~15 commits

### Phase F: CUDA 🔲 Not Started
1. CUDA kernels (all compute ops)
2. Python wrapper for kernels
3. Full model
4. Cross-backend parity tests
5. Training + Inference

---

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
[Gradient Clipping] → clip gradients to stable range
    │
    ▼
[Optimizer Step] → update parameters
```

### 6.2 Training Configuration (Default)

Config defaults from `shared/config.py` + defaults in `shared/config_utils.py`:
- Learning rate, batch size, epochs, etc. configurable via CLI/env/file
- Synthetic data generation for fast iteration (no TinyStories dependency needed)
- Variable-length batch handling with padding and attention masking

---

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
[Tokenizer.decode] → "Once upon a time A, B, ..."
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

---

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

---

## 9. Build Sequence (Phase-by-Phase)

### Phase A: Shared Foundation ✅ COMPLETE
```
1. config.py + constants.py
2. tokenizer.py
3. dataset.py (TinyStories loader)
4. checkpoint.py
5. config_utils.py (unified config reader)
```

### Phase B: NumPy ✅ COMPLETE
**21 commits (b0-b19), ~70 tests pass**
```
Core layers → RoPE → MHA → MHA(GQA) → MoE → TransformerBlock → DecoderStack
→ Full model → Forward/Backward → CrossEntropy → AdamW → Training loop
→ Naive KV → TurboQuant KV → Inference → CLI → Gradient clipping → Post-norm gates
```

### Phase C: PyTorch ✅ COMPLETE
**36 commits (c0-c36), 129 tests pass**
```
C0: Project scaffolding
C1-C6: All layers as nn.Module (Embedding → RMSNorm → RoPE → MHA → MoE → TransformerBlock → DecoderStack)
C7: TorchModel full (save/load/load_from_numpy)
C8: CrossEntropyLoss + AdamW
C9: Training Loop (autograd)
C10: Naive KV Cache + TurboQuant KV Cache
C11: Inference Engine (greedy + sampled + top-k)
C12: CLI (argparse entry point)
C13: Full training pipeline
C14: Cross-backend parity (rtol=1e-3, atol=1e-3)
```

### Phase C+: E2E Training/Inference & Equivalence ✅ COMPLETE
**8 commits (c37-c44), 90+ tests pass**
```
C+: Unified config system (CLI > env > config file)
C+: scripts/train.py — single entry point, --backend numpy/torch
C+: scripts/infer.py — interactive mode with context status
C+: scripts/verify_equivalence.py — 6 scenario equivalence check
C+: scripts/auto_test_equivalence.py — 8-combination matrix test
```

### Phase C++: Normalization Improvements ✅ COMPLETE
**3 commits (d0-d2), 21 tests pass**
```
C++: Post-Norm (residual add → norm → gate)
C++: Gated residuals (gate1 for attention, gate2 for MoE)
C++: Dropout (train/eval mode)
C++: Gradient clipping in both backends
C++: Cross-backend parity maintained (save/load/load_from_numpy handle gates)
```

### Phase D: Equivalence Verification ✅ COMPLETE
```
Fixed: MoE router bias, weight sync, verify script, zero-size arrays, dropout mode
Result: All 6/6 scenarios pass with weight_diff=0.0, identical tokens, KL=0.0
```

### Phase E: Triton 🔶 READY TO START
**GPU confirmed** — CUDA 12.6, cuDNN 9.3, Orin GPU, 8x GPUs available. Reference: `docs/phase_e_plan.md`

```
E0: Scaffolding (directories + import test)          — 1 commit
E1: SiLU kernel (element-wise)                       — 1 commit
E2: RMSNorm kernel (reduction)                       — 1 commit
E3: RoPE kernel (trig, indexing)                     — 1 commit
E4: SwiGLU kernel (SiLU + matmul)                    — 1 commit
E5: MHA kernel (attention + GQA)                     — 1 commit
E6: MoE kernel (top-k routing)                       — 1 commit
E7: TransformerBlock (Python wiring)                 — 1 commit
E8: DecoderStack (Python wiring)                     — 1 commit
E9: Full TritonModel (save/load/parity)              — 1 commit
E10: Inference + Training scripts                     — 1 commit
E11: Cross-backend parity (NumPy/PyTorch vs Triton)  — 1 commit
```

**Execution order:** Sequential by wave (~12 sub-phases, ~15 commits, ~60-80 tests)
**Goal:** Production-quality Triton code — every kernel documented with math explanations, memory patterns, numerical stability techniques

### Phase F: CUDA 🔲 Not Started
```
E1: CUDA kernels (all compute ops)
E2: Python wrapper for kernels
E3: Full model
E4: Cross-backend parity tests
E5: Training + Inference
```

---

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

---

## 11. CLI Interface

### 11.1 Training

```bash
# Quick training (defaults)
uv run python -m scripts.train

# Full custom training
uv run python -m scripts.train --backend numpy \
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
uv run python -m scripts.infer --prompt "Once upon a time"

# Full options
uv run python -m scripts.infer \
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
uv run python -m scripts.verify_equivalence

# Automated equivalence matrix
uv run python -m scripts.auto_test_equivalence
```

---

## 12. Dependencies

### 12.1 Required

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.24 | Math operations, array operations |
| torch | ≥ 2.0 | PyTorch implementation |

### 12.2 Optional (backend-specific)

| Package | Version | Purpose |
|---------|---------|---------|
| triton | ≥ 2.2 | GPU kernel implementation |
| cuda-python | ≥ 12.0 | Bare-metal GPU kernels |
| datasets | ≥ 2.14 | TinyStories dataset loader (HuggingFace) |
| tokenizers | ≥ 0.15 | BPE tokenizer |

### 12.3 Dev Tooling

| Package | Purpose |
|---------|---------|
| pytest | Testing framework |
| pytest-timeout | Ensure tests don't hang |
| ruff | Linting, formatting, import sorting |
| pyright | Type checking (strict mode) |

---

## 13. Quality Gates

Each phase must pass these gates before proceeding:

| Gate | Requirement |
|------|-------------|
| **Code Quality** | pyright passes, ruff passes (zero warnings) |
| **Test Coverage** | All unit tests for the phase pass |
| **Cross-Backend** | Existing backends still match after new backend |
| **Documentation** | Key classes/functions have docstrings |
| **No Regressions** | Existing tests still pass after changes |

---

## 14. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| TinyStories too small | Model can't learn meaningful patterns | Use full dataset, longer training, smaller config |
| Cross-backend drift | NumPy vs Torch produce different results | TDD: always write cross-backend test first |
| CUDA/Triton complexity | Hard to debug GPU code | Start with reference NumPy, verify before adding GPU |
| TurboQuant complexity | Hard to get right quantization | Implement naive KV cache first, validate correctness then add quantization |
| Memory issues | Large models on small GPUs | Ensure NumPy/PyTorch work on CPU first; CUDA/Triton scale independently |
| Seed non-determinism | Results vary across runs | Single seed source, explicit RNG state management |

---

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
Why Triton third? (🔶 READY — GPU confirmed, plan complete)
├─ Requires PyTorch knowledge (uses torch for data)
├─ Only replaces compute kernels, not model architecture
├─ GPU-only, harder to debug without CPU fallback
├─ GPU confirmed: CUDA 12.6, Orin 8x, PyTorch 2.11.0 (CUDA)
└─ Phase E plan ready: 12 stages, ~60-80 tests, production-quality learning focus
│
Why CUDA last?
├─ Hardest to implement (no framework helpers)
├─ Requires CUDA toolkit installation
├─ Reference implementations already exist (numpy + torch + triton)
└─ Best as a learning exercise after understanding the abstractions
```
