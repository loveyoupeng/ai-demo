# Phase F: CUDA Bare-Metal Implementation — Execution Plan

**Status:** 🔲 Not Started
**Start Date:** —
**End Date:** —
**Total Stages:** 12 sub-phases (sequential, one at a time)

## Environment

| Component | Version |
|-----------|---------|
| CUDA | 12.6 |
| cuDNN | 9.3 |
| cuBLAS | 12.6 |
| PyTorch | 2.11.0 (with CUDA 12.6) |
| cuda-python | ≥ 12.0 (to be installed) |
| GPU Hardware | Orin, compute capability 8.x |
| GPUs | 8 (64GB shared memory) |
| CUDA API Check | `torch.cuda.is_available()` → True |

**Dependency:** `nvidia/cuda-python` package — Python bindings for CUDA C Runtime API.
**Learning focus:** Manual memory management, PTX/Sass, kernel launches, stream ordering, shared memory.

## Progress Summary

| Stage | Status | Tests | Description |
|-------|--------|-------|-------------|
| F0: Project scaffolding | 🔲 Ready | 0 | Directories + import test |
| F1: SiLU kernel | 🔲 Ready | ~4 | Element-wise activation |
| F2: RMSNorm kernel | 🔲 Ready | ~8 | Reduction normalization |
| F3: RoPE kernel | 🔲 Ready | ~6 | Positional encoding |
| F4: SwiGLU kernel | 🔲 Ready | ~6 | Gated feedforward |
| F5: MHA kernel | 🔲 Ready | ~10 | Multi-head attention |
| F6: MoE kernel | 🔲 Ready | ~6 | Expert routing |
| F7: TransformerBlock | 🔲 Ready | ~5 | Assembly layer |
| F8: DecoderStack | 🔲 Ready | ~3 | Stacked blocks |
| F9: Full CUDAModel | 🔲 Ready | ~5 | Full model integration |
| F10: Inference + Training | 🔲 Ready | ~5 | Pipeline scripts |
| F11: Cross-backend parity | 🔲 Ready | ~8 | 4-way equivalence |

---

## Goal

Build **bare-metal CUDA kernels** using `nvidia/cuda-python` that produce **numerically identical** results to NumPy/PyTorch at `float64` precision. This is NOT a wrapper around cuBLAS/cuDNN — this is hand-written CUDA C compiled at runtime via `cuda-python`.

**Philosophy:** Same as Phase E, but lower level. CUDA requires manual memory management, kernel launch configuration, and stream ordering. Every line of CUDA C is visible and可控.

**Key difference from Triton:**
- **Triton:** High-level DSL, automatic memory management, Python-only
- **CUDA:** Low-level C API, manual malloc/free, PTX compilation, explicit streams
- **CUDA** teaches you how GPUs actually work — shared memory, warps, coalesced access, occupancy

## Architecture Overview

```
Same model architecture, bare-metal compute backend:

CUDAModel (wraps cuda-python kernel calls)
├── embedding          → cudaMalloc + cudaMemcpy (no compute)
├── stack.layers[i]    → CUDATransformerBlock
│   ├── mha            → CUDA MHA kernel (attention)
│   ├── rope           → CUDA RoPE kernel
│   └── moe.experts    → CUDA MoE kernel + CUDA expert kernel
├── final_ln           → CUDA RMSNorm kernel
└── output             → CUDA SiLU + CUDA GatedLinear

All kernels are hand-written .cu files compiled at runtime via nvrtc (or pre-compiled .so).
```

## Directory Structure

```
impl/
├── _cuda/                    # NEW — Bare-metal CUDA implementation
│   ├── __init__.py           # Package init + public API
│   ├── kernels/              # CUDA C source files
│   │   ├── activation.cu     # SiLU, gating kernels
│   │   ├── layernorm.cu      # RMSNorm kernel
│   │   ├── rope.cu           # RoPE kernel
│   │   ├── ffn.cu            # SwiGLU kernel
│   │   ├── attention.cu     # MHA kernel
│   │   └── moe.cu           # MoE routing kernel
│   ├── compiler.py           # CUDA source → PTX → runtime compilation
│   │   │   # Wrapper functions (Python) → kernel launch config → kernels
│   │   ├── activation.py     # SiLU/CUDA wrapper
│   │   ├── layernorm.py      # RMSNorm/CUDA wrapper
│   │   ├── rope.py           # RoPE/CUDA wrapper
│   │   ├── ffn.py            # SwiGLU/CUDA wrapper
│   │   ├── attention.py     # MHA/CUDA wrapper
│   │   ├── moe.py            # MoE/CUDA wrapper
│   │   ├── transformer.py    # TransformerBlock wrapper
│   │   └── model.py          # CUDAModel → model integration
│   ├── inference.py          # Inference engine (same API as other backends)
│   └── training.py           # Training loop (same API as other backends)
│
tests/
├── unit/
│   └── _cuda/                # NEW — CUDA-specific tests
│       ├── __init__.py
│       ├── test_activation.py # SiLU kernel tests
│       ├── test_layernorm.py  # RMSNorm kernel tests
│       ├── test_rope.py       # RoPE kernel tests
│       ├── test_ffn.py        # SwiGLU kernel tests
│       ├── test_attention.py  # MHA kernel tests
│       ├── test_moe.py        # MoE kernel tests
│       ├── test_transformer.py # TransformerBlock tests
│       ├── test_model.py      # Full model tests
│       └── test_inference.py  # Inference tests
│
├── cross_backend/
│   └── test_parity_cuda.py   # NEW — 4-way NumPy/PyTorch/Triton vs CUDA parity
│
scripts/
└── train.py / infer.py       # No changes (uses --backend flag)
```

---

## TDD Process

Each stage follows the same exact pattern. **One test file, one commit, one thing at a time.**

```
Step 1: Write test → ALL FAIL
   Write test file. Run:
   PYTHONPATH=shared PYTHONPATH=impl uv run pytest tests/unit/_cuda/test_*.py -v --timeout=60
   Check: ALL tests fail (MissingFile, MissingFunction, or actual assertion failures).
   If any test passes unexpectedly — the task isn't done.

Step 2: Implement → all PASS
   Write minimal CUDA kernel + Python wrapper.
   Run: ALL pass.
   If any test fails — fix it, don't skip it.

Step 3: Quality check
   ruff check impl/_cuda/<file>.py tests/unit/_cuda/test_<file>.py
   pyright impl/_cuda/<file>.py tests/unit/_cuda/test_<file>.py

Step 4: Commit
   git add -A && git commit -m "f<stage>: <component> — <N> tests pass"
```

**Rules:**
- NEVER implement without failing tests first
- ONE component per commit (atomic)
- ONE test file per component
- Tests must run on GPU for CUDA — use `pytest.mark.gpu`
- Skip tests when no GPU is available (`pytest.skip(reason="No GPU")`) — but always check
- CUDA kernel compilation can be slow — cache compiled PTX

- **Tolerance policy (AGENTS.md tiered):**
  - Standalone kernels (tested in isolation): `rtol=1e-4, atol=1e-4`
  - Kernels in single chain: `rtol=1e-3, atol=1e-3`
  - Full model (multi-layer chain): `rtol=1e-2, atol=1e-2`

---

## Stages (12 sub-phases)

### Phase F0: Project scaffolding

**What it does:** Create the directory and package structure. Zero tests expected.

- [ ] **F0.1** | `impl/_cuda/`, `tests/unit/_cuda/`, `tests/unit/_cuda/__init__.py` | pytest discovers 0 tests | 0

No implementation code. Just directories.

- [ ] **F0.2** | `impl/_cuda/__init__.py` + `tests/unit/_cuda/test_import.py` | import test fails with ModuleNotFoundError | 0

```python
# tests/unit/_cuda/test_import.py
import pytest

class TestImport:
    @pytest.mark.timeout(10)
    def test_cuda_package(self):
        import impl._cuda
        assert hasattr(impl._cuda, "__file__")
```

**Gate:** 1 test → commit.

---

### Phase F1: SiLU Activation Kernel

**What it does:** Element-wise `x * sigmoid(x)`. Simplest possible CUDA kernel — pure element-wise mapping.

```python
# CUDA kernel (kernels/activation.cu)
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = silu(input[idx]);
    }
}
```

```python
# Python wrapper (activation.py)
import torch, ctypes

def silu(x: torch.Tensor) -> torch.Tensor:
    """x * sigmoid(x) — element-wise, same as torch.nn.functional.silu(x)"""
    assert x.dtype == torch.float32 or x.dtype == torch.float64
    out = torch.empty_like(x)
    size = x.numel()
    # ... launch CUDA kernel via ctypes/cuda-python
    return out
```

**Tests:** `tests/unit/_cuda/test_activation.py`
- `test_silu_matches_torch_float32` — rtol=1e-4
- `test_silu_matches_torch_float64` — rtol=1e-4
- `test_silu_input_gradient` — same as `torch.nn.functional.silu` gradient
- `test_silu_shapes` — 1D, 2D, 3D

**Gate:** 4 tests pass → commit.

---

### Phase F2: RMSNorm Kernel

**What it does:** Root Mean Square Layer Normalization. Reduction kernel across the last dimension.

```c
// kernels/layernorm.cu
__device__ float rmsnorm(const float* x, const float* weight, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i];
    }
    float rms = rsqrtf(sum / size + 1e-6f);
    float out = 0.0f;
    for (int i = 0; i < size; i++) {
        out += weight[i] * x[i] * rms;
    }
    return out;
}
```

**Tests:** `tests/unit/_cuda/test_layernorm.py`
- `test_rmsnorm_matches_torch` — rtol=1e-4
- `test_rmsnorm_gradient` — gradient matches torch
- `test_rmsnorm_shapes` — various batch/seq/dim sizes
- `test_rmsnorm_numerical_stability` — large/small input values

**Gate:** ~8 tests pass → commit.

---

### Phase F3: RoPE Kernel

**What it does:** Rotary Position Embeddings — 2D rotation on Q and K pairs.

```c
// kernels/rope.cu
__device__ void apply_rope(float* q, float* k, const float* freqs, int head_dim, int pos) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = freqs[i / 2];
        float cos = cosf(pos * freq);
        float sin = sinf(pos * freq);
        float q0 = q[i] * cos - q[i+1] * sin;
        float q1 = q[i] * sin + q[i+1] * cos;
        float k0 = k[i] * cos - k[i+1] * sin;
        float k1 = k[i] * sin + k[i+1] * cos;
        q[i] = q0; q[i+1] = q1;
        k[i] = k0; k[i+1] = k1;
    }
}
```

**Tests:** `tests/unit/_cuda/test_rope.py`
- `test_rope_matches_torch` — rtol=1e-4
- `test_rope_gradient` — gradient through RoPE
- `test_rope_positional_sensitivity` — different positions → different outputs
- `test_rope_partial` — partial dim RoPE

**Gate:** ~6 tests pass → commit.

---

### Phase F4: SwiGLU Kernel

**What it does:** SwiGLU FFN: `silu(W1 @ x) @ W2`. Three weight matrices, element-wise SiLU gating.

```c
// kernels/ffn.cu
// GEMM + SiLU + GEMM — uses cuBLAS sgemm or manual tile-based matmul
```

For SWiGLU, we have two options:
- **Option A:** Manual tile-based matmul in pure CUDA (most educational, most code)
- **Option B:** Use `cublasSgemm` for GEMM, element-wise for SiLU (more practical)

**Decision:** Use option A (manual) for first expert, validate. If performance is unacceptable, fall back to option B.

**Tests:** `tests/unit/_cuda/test_ffn.py`
- `test_swiglu_matches_torch` — rtol=1e-4
- `test_swiglu_gradient` — gradient through W1, W2
- `test_swiglu_shapes` — various hidden expansion factors

**Gate:** ~6 tests pass → commit.

---

### Phase F5: MHA Kernel

**What it does:** Multi-head self-attention: Q @ K^T / sqrt(d) → softmax → @ V. The most complex kernel in the model.

```c
// kernels/attention.cu
// Contains:
// 1. GEMM for Q, K, V projections
// 2. Tiled scaled dot-product attention kernel
// 3. Stable softmax (max-subtract-then-exp)
// 4. Weighted sum (attn @ V)
// 5. GQA support (grouped query)
```

**Key CUDA-specific challenges:**
- Shared memory tiling for the attention matrix
- Warp-level reductions for softmax
- Grid-stride loops for variable sequence lengths
- Flash attention optimization (optional, later)

**Tests:** `tests/unit/_cuda/test_attention.py`
- `test_mha_matches_torch` — rtol=1e-3 (attn is harder to match exactly)
- `test_mha_backward` — gradient norm within tolerance
- `test_mha_gqa` — GQA pattern matches expected
- `test_mha_batched` — batch dimension

**Gate:** ~10 tests pass → commit.

---

### Phase F6: MoE Kernel

**What it does:** Top-k expert routing with weighted combination.

```c
// kernels/moe.cu
// Contains:
// 1. Router: Linear -> softmax -> top-k selection
// 2. Expert dispatch: scatter/gather tokens to experts
// 3. Expert computation: SwiGLU per expert
// 4. Weighted combine: weighted sum of expert outputs
```

**Tests:** `tests/unit/_cuda/test_moe.py`
- `test_moe_matches_torch` — rtol=1e-3
- `test_moe_top_k` — correctly selects top-k
- `test_moe_gradient` — router + expert gradients
- `test_moe_load_balance` — routing is fair across experts

**Gate:** ~6 tests pass → commit.

---

### Phase F7: TransformerBlock

**What it does:** Assembles MHA + RMSNorm + MoE + SiLU into one TransformerBlock.

```python
# transformer.py

class CUDATransformerBlock(nn.Module):
    def __init__(self, config):
        self.mha = CUDA_MHA(config)
        self.ln1 = CUDA_RMSNorm(config)
        self.moe = CUDA_MoE(config)
        self.ln2 = CUDA_RMSNorm(config)
        self.gate1 = nn.Parameter(torch.zeros(1))
        self.gate2 = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        h = x + self.mha(x)          # residual add
        h = self.ln1(h)               # post-norm
        h = h + sigmoid(self.gate1) * h  # gated residual
        moe_out = self.moe(h)
        out = h + moe_out            # residual add
        out = self.ln2(out)          # post-norm
        out = out + sigmoid(self.gate2) * out
        return out
```

**Tests:** `tests/unit/_cuda/test_transformer.py`
- `test_block_matches_torch` — rtol=1e-3
- `test_block_backward` — gradient within tolerance
- `test_block_forward_shape` — input/output shapes match

**Gate:** ~5 tests pass → commit.

---

### Phase F8: DecoderStack

**What it does:** Sequential chain of TransformerBlocks.

```python
# decoder.py

class CUDADecoderStack(nn.Module):
    def __init__(self, config, n_layers):
        self.layers = nn.ModuleList(
            [CUDATransformerBlock(config) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

**Tests:** `tests/unit/_cuda/test_decoder.py`
- `test_stack_matches_torch` — rtol=1e-2 (multi-layer chain)
- `test_stack_backward` — gradient within tolerance
- `test_stack_shape` — shapes match

**Gate:** ~3 tests pass → commit.

---

### Phase F9: Full CUDAModel

**What it does:** Complete model — embedding → DecoderStack → RMSNorm → output_proj → logits.

```python
# model.py

class CUDAModel(nn.Module):
    def __init__(self, config):
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.stack = CUDADecoderStack(config, config.n_layers)
        self.final_ln = CUDA_RMSNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.stack(x)
        x = self.final_ln(x)
        logits = self.output_proj(x)
        return logits
```

**Tests:** `tests/unit/_cuda/test_model.py`
- `test_model_matches_torch` — rtol=1e-2
- `test_model_forward_backward` — gradient norm matches torch
- `test_model_save_load` — save_as_numpy / load_from_numpy_dict
- `test_model_named_parameters` — key names match other backends

**Gate:** ~5 tests pass → commit.

---

### Phase F10: Inference + Training Scripts

**What it does:** Training loop and inference engine matching the API of other backends.

```python
# training.py
def train_step(model, batch, lr, clip_grad):
    """Full training step: forward → loss → backward → clip → step"""
    ...

# inference.py
class CUDATextGenerator:
    def generate(self, text, max_length, temperature, top_k):
        """Autoregressive generation with CUDA KV cache"""
        ...
```

**Tests:** `tests/unit/_cuda/test_training.py` + `test_inference.py`
- `test_training_step` — loss reduction
- `test_inference_generation` — token match
- `test_kv_cache_cuda` — CUDA KV cache works

**Gate:** ~5 tests pass → commit.

---

### Phase F11: Cross-Backend Parity (4-Way)

**What it does:** NumPy ↔ PyTorch ↔ Triton ↔ CUDA — all produce identical results.

```python
# tests/cross_backend/test_parity_cuda.py

@pytest.mark.gpu
class TestCUDAParity:
    def test_numpy_torch_cuda_forward(self):
        """3-way forward match"""
        ...
    
    def test_torch_triton_cuda_forward(self):
        """3-way forward match (no NumPy)"""
        ...
    
    def test_all_four_backward(self):
        """All 4 backends produce same gradient norm"""
        ...
```

**Gate:** 4-way parity tests pass → commit.

---

## Execution Notes

### Order of Operations

```
1. F0 (Scaffolding) — directories and import test
2. F1-F6 (Standalone kernels) — SiLU → RMSNorm → RoPE → SwiGLU → MHA → MoE
3. F7-F8 (Assembly) — TransformerBlock → DecoderStack
4. F9 (Full model) — CUDAModel integration
5. F10 (Scripts) — training + inference
6. F11 (Parity) — 4-way equivalence tests
```

### CUDA-Specific Concerns

1. **Compilation speed:** Each CUDA file is compiled at runtime. Add caching (`impl/_cuda/.cache/`) to avoid recompilation.
2. **Memory management:** Manual `cudaMalloc`, `cudaMemcpy`, `cudaFree`. Always `cudaDeviceSynchronize()` before comparing.
3. **Streams:** Separate streams for compute and memory transfers for better throughput.
4. **Error handling:** Every CUDA API call can fail. Check for `cudaError_t`.

### TDD Discipline

- Write test → run → see failure → implement → see pass
- Never skip tests. Never "fix the test." The implementation is wrong if the test fails.
- GPU tests use `pytest.mark.gpu` and `pytest.skip()` when no GPU available.
- After each stage: run ALL previous tests to verify no regression.

---

## Estimation

| Stage | Commits | Hours | Risk |
|-------|---------|-------|------|
| F0 | 1 | 0.5 | Low |
| F1 (SiLU) | 1 | 1 | Low |
| F2 (RMSNorm) | 1 | 1.5 | Low |
| F3 (RoPE) | 1 | 1 | Low |
| F4 (SwiGLU) | 2 | 2 | Medium (GEMM) |
| F5 (MHA) | 3 | 4 | High (tiled attention, shared memory) |
| F6 (MoE) | 2 | 3 | Medium (dispatch complexity) |
| F7 (Block) | 1 | 1.5 | Low |
| F8 (Stack) | 1 | 1 | Low |
| F9 (Model) | 1 | 2 | Medium |
| F10 (Scripts) | 1 | 1.5 | Low |
| F11 (Parity) | 1 | 2 | Medium |
| **Total** | **~15** | **~21** | **Medium-High** |

---

## Completion Criteria

Phase F is done when ALL of the following are true:

- [ ] **551 + N tests pass** — new CUDA test count, all passing (N ≈ 13 new tests)
- [ ] **All CUDA kernels CUDA C** — hand-written `.cu` files, not wrappers around cuBLAS/cuDNN
- [ ] **Memory management correct** — no leaks, proper sync before comparisons
- [ ] **4-way parity** — NumPy, PyTorch, Triton, CUDA produce identical outputs
- [ ] **ruff check passes** — zero errors in `impl/_cuda/`
- [ ] **pyright check passes** — zero errors (or expected documented)
- [ ] **commits are clean** — one per stage, meaningful messages

**Tolerance for CUDA parity:**
- Standalone kernels: `rtol=1e-4, atol=1e-4`
- 1-layer model: `rtol=1e-3, atol=1e-3`
- 2+ layer model: `rtol=1e-2, atol=1e-2` (CUDA float32 drift can be slightly higher)

---

## Phase F Estimation Table

| Stage | Files | Tests | Hours | Risk | Status |
|-------|-------|-------|-------|------|--------|
| F0: Scaffolding | 2 dirs | 1 | 0.5 | Low | Not started |
| F1: SiLU activation kernel | 4 | 4 | 1 | Low | Not started |
| F2: RMSNorm kernel | 4 | 8 | 1.5 | Low | Not started |
| F3: RoPE kernel | 4 | 6 | 1 | Low | Not started |
| F4: SwiGLU FFN kernel | 4 | 6 | 2 | Medium | Not started |
| F5: MHA kernel | 5 | 10 | 4 | High | Not started |
| F6: MoE kernel | 4 | 6 | 3 | Medium | Not started |
| F7: TransformerBlock | 2 | 5 | 1.5 | Low | Not started |
| F8: DecoderStack | 2 | 3 | 1 | Low | Not started |
| F9: Full CUDAModel | 3 | 5 | 2 | Medium | Not started |
| F10: Inference + Training | 2 | 5 | 1.5 | Low | Not started |
| F11: Cross-backend parity (4-way) | 1 | 8 | 2 | Medium | Not started |

---

*This plan follows the same TDD discipline as Phase E: one test, one commit, one thing at a time. Tests are the source of truth — never reason about correctness without test results.*