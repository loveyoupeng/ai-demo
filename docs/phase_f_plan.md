# Phase F: CUDA Bare-Metal Implementation — REDESIGNED for JetPack 6.2.2

**Status:** 🔶 REDESIGNED — original plan broken, new approach defined
**Start Date:** —
**End Date:** —
**Total Stages:** 10 sub-phases (reduced, hybrid approach)
**Platform:** Jetson AGX Orin 64GB, JetPack 6.2.2, CUDA 12.6
**Critical Issue:** `cuda-python` 12.6.2.post1 — `cuLaunchKernel`/`cuLaunchKernelEx` broken

## Environment — Reality Check

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 12.6 | OK |
| cuDNN | 9.3 | OK |
| cuBLAS | 12.6 | OK |
| PyTorch | 2.11.0 | ✅ `torch.cuda.is_available()` works |
| cuda-python | 12.6.2.post1 | ❌ **New CUDA driver API broken** |
| GPU Hardware | Orin, compute capability 8.x | OK |
| GPUs | 8 (64GB shared memory) | OK |

### Working API (Legacy cuLaunch ONLY)
```python
# Only these work reliably — no new API at all:
cuParamSetSize(kernel, size)        # → returns (status,) — sets parameter buffer size
cuParamSetv(kernel, offset, bytes, count)  # → (status,) — passes bytes at offset
cuFuncSetBlockShape(kernel, x, y, z)    # → (status,) — sets fixed block dimensions
cuLaunch(kernel)                         # → launches with the set block shape
```

### Working API (nvrtc ONLY)
```python
nvrtcCreateProgram(src, name, ...)    # → (status, program)
nvrtcCompileProgram(prog, ...)        # → (status,) — compiles PTX
nvrtcGetPTX(prog, buffer)             # → (status,) — writes in-place to bytearray
nvrtcGetProgramLogSize(prog)          # → (status, size)
nvrtcGetProgramLog(prog, buffer)      # → (status,) — in-place write
```

### Working API (cudaMalloc/Free/Memcpy)
```python
# These work:
cudaMalloc(byref(ptr), size)  # → device pointer via ctypes c_void_p
cudaMemcpy(dst, src, count)   # → host↔device memory transfer
cudaFree(ptr)                 # → free device memory
```

### What is BROKEN (do NOT attempt)
- `cuModuleGetFunction` with new-style handles
- `cuLaunchKernel` — always `TypeError: an integer is required`
- `cuLaunchKernelEx` — always `TypeError: an integer is required`
- `cuContext` API — all fail
- `cuStream` API — all fail
- `cuEvent` API — all fail
- Any function that requires `CUmodule`, `CUfunction`, `CUstream` objects from new API

## Architecture: nvrtc + PyTorch Custom Kernels (Option A)

```
CUDAModel (PyTorch nn.Module with CUDA kernels)
├── embedding          → nn.Embedding (no compute, just lookup)
├── stack.layers[i]    → CUDATransformerBlock (uses CUDA kernels)
│   ├── mha            → softmax_kernel (CUDA) + mm (PyTorch)
│   ├── rope           → apply_rope_kernel (CUDA)
│   └── moe            → routing_kernel (CUDA) + weighted_sum (CUDA)
├── final_ln           → rmsnorm_kernel (CUDA)
└── output             → silu_kernel (CUDA) + linear (PyTorch)

All CUDA C kernels:
- Hand-written .cu files compiled at runtime via nvrtc
- PTX cached in impl/_cuda/.cache/
- Dispatched via PyTorch (torch.tensor → kernel → torch.tensor)
- Backward pass: PyTorch autograd (automatic, no CUDA backward kernels)
```

**Why this approach:**
- nvrtc compilation + CUDA C source = preserves "learning how GPUs work"
- PyTorch dispatcher = practical, all CUDA features available through PyTorch
- Manual memory → PyTorch tensor memory = acceptable tradeoff
- Backward pass → PyTorch autograd = correct gradients, no manual derivation
- The learning focus (shared memory, warp reduction, coalesced access) is ALL in the .cu files

## Directory Structure (REDESIGNED)

```
impl/
├── _cuda/                    # CUDA kernels with nvrtc + PyTorch dispatch
│   ├── __init__.py
│   ├── compiler.py           # nvrtc compile → PTX → cache (already written ✅)
│   ├── kernels/              # CUDA C source files
│   │   ├── activation.cu     # SiLU kernel (+ device function for backward)
│   │   ├── layernorm.cu      # RMSNorm kernel (+ backward)
│   │   ├── rope.cu           # RoPE kernel
│   │   ├── ffn.cu            # SiLU element-wise (FFN uses PyTorch mm)
│   │   ├── attention.cu      # Stable softmax + weighted sum (NOT full MHA)
│   │   └── moe.cu            # Top-k routing + weighted sum
│   ├── activation.py         # silu(tensor) — nvrtc + PyTorch dispatcher
│   ├── layernorm.py          # rmsnorm(tensor, weight) — nvrtc + PyTorch
│   ├── rope.py               # apply_rope(tensor, freqs) — nvrtc + PyTorch
│   ├── ffn.py                # swiglu(tensor) — SiLU kernel + torch matmul
│   ├── attention.py          # mha(tensor) — softmax kernel + torch mm
│   ├── moe.py                # moe_router(tensor, weights) — nvrtc + PyTorch
│   ├── model.py              # CUDAModel — full model with CUDA kernels
│   ├── training.py           # train_step — forward via CUDA, backward via torch
│   └── inference.py          # CUDATextGenerator — inference with CUDA kernels
│
tests/
├── unit/
│   └── _cuda/                # CUDA-specific tests
│       ├── __init__.py       # Already exists ✅
│       ├── test_activation.py # SiLU kernel tests — written, NOT passing yet
│       ├── test_layernorm.py  # RMSNorm kernel tests — NOT written
│       ├── test_rope.py       # RoPE kernel tests — NOT written
│       ├── test_ffn.py        # SwiGLU kernel tests — NOT written
│       ├── test_attention.py  # Attention kernel tests — NOT written
│       ├── test_moe.py        # MoE kernel tests — NOT written
│       ├── test_model.py      # Full model tests — NOT written
│       ├── test_training.py   # Training tests — NOT written
│       └── test_inference.py  # inference tests — NOT written
│
├── cross_backend/
│   └── test_parity_cuda.py   # 4-way NumPy/Torch/Triton/CUDA parity — NOT written
│
scripts/
└── train.py / infer.py       # No changes (uses --backend flag)
```

### Key Changes from Original Plan

| Original | Redesign | Reason |
|----------|----------|--------|
| `cuModuleGetFunction` + `cuLaunchKernel` | nvrtc + PyTorch custom op | New API broken |
| `cudaMalloc`/`cudaMemcpy`/`cudaFree` | PyTorch tensor memory | Practical necessity |
| Manual memory management → manual everything | Manual CUDA C kernel + PyTorch dispatcher | Platform limitation |
| 32-bit `int` in kernel params via `cuParamSeti` | 32-bit `int` via `cuParamSetv` or PyTorch dispatcher | `cuParamSeti` broken |
| Pure bare-metal (no framework) | PyTorch as dispatcher | `cuda-python` too broken |
| MHA: full CUDA attention | MHA: softmax in CUDA, QKV/proj in PyTorch | Simplifies kernel, still educative |
| Backward: manual CUDA gradients | Backward: PyTorch autograd | Same as Triton phase |
| Stream ordering (learning goal) | Warp reduction, coalesced access, shared memory (still learnable) | Streams not feasible |

## TDD Process (Same as Original, Adapted)

```
Step 1: Write test → ALL FAIL
   Write test file. Run:
   PYTHONPATH=. uv run pytest tests/unit/_cuda/test_*.py -v --timeout=60
   Check: ALL tests fail (MissingFunction or actual assertion failures).

Step 2: Implement → all PASS
   Write CUDA C kernel + Python wrapper using nvrtc compile + PyTorch dispatch.
   Run: ALL pass.
   If any test fails — fix it, don't skip it.

Step 3: Quality check
   ruff check impl/_cuda/<file>.py tests/unit/_cuda/test_<file>.py
   pyright impl/_cuda/<file>.py tests/unit/_cuda/test_<file>.py

Step 4: Commit
   git add -A && git commit -m "f<stage>: <component> — <N> tests pass"
```

**Tolerance policy (same as original):**
- Standalone kernels (tested in isolation): `rtol=1e-4, atol=1e-4`
- Kernels in single chain: `rtol=1e-3, atol=1e-3`
- Full model (multi-layer chain): `rtol=1e-2, atol=1e-2`

## Stages (10 sub-phases — Reducd from 12)

### Phase F0: Project scaffolding ✅ COMPLETE

- [x] **F0.1** | `impl/_cuda/`, `tests/unit/_cuda/` | directories created |
- [x] **F0.2** | `impl/_cuda/__init__.py` + `tests/unit/_cuda/test_import.py` | import test passes |

**Gate:** 1 test passes → committed: `f0: project scaffolding — 1 tests pass`

---

### Phase F1: SiLU Activation Kernel

**What it does:** Element-wise `x * sigmoid(x)`. Simplest CUDA kernel — pure element-wise mapping. No reduction, no shared memory. A "hello world" kernel for learning nvrtc compilation.

```cuda
// kernels/activation.cu
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));  // numerically stable sigmoid
}

__global__ void silu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = silu(input[idx]);
    }
}

// Backward: d_input = grad_output * silu(x) = grad_output * output / x * (1 - silu(x))
// Simplified: d_input = grad_output * output * (1.0f - output / input)
// But we just need element-wise multiply for the backward — PyTorch handles it
```

**Python wrapper:**
```python
# impl/_cuda/activation.py
def silu(x: torch.Tensor) -> torch.Tensor:
    """Element-wise SiLU via nvrtc-compiled CUDA kernel.
    Backward pass uses PyTorch's built-in autograd."""
    # 1. Allocate output tensor (PyTorch memory)
    # 2. Compile CUDA C if not cached
    # 3. Launch kernel via PyTorch custom op or direct nvrtc load
    # 4. Return output tensor
```

**Tests:** `tests/unit/_cuda/test_activation.py` (WRITTEN, NOT PASSING — needs implementation)
- `test_silu_matches_torch_float32` — rtol=1e-4
- `test_silu_matches_torch_float64` — rtol=1e-4
- `test_silu_input_gradient` — gradient matches torch
- `test_silu_shapes` — 1D, 2D, 3D

**Gate:** 4 tests pass → commit.

---

### Phase F2: RMSNorm Kernel

**What it does:** Root Mean Square Layer Normalization. A reduction kernel across the last dimension. Teaches warp reduction, shared memory.

```cuda
// kernels/layernorm.cu
__device__ float rms_norm(const float* x, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i];
    }
    return rsqrtf(sum / size + 1e-6f);
}
```

**Tests:** `tests/unit/_cuda/test_layernorm.py`
- `test_rmsnorm_matches_torch_float32` — rtol=1e-4
- `test_rmsnorm_matches_torch_float64` — rtol=1e-4
- `test_rmsnorm_shapes` — various batch/seq/dim sizes
- `test_rmsnorm_numerical_stability` — large/small input values

**Gate:** ~4 tests pass → commit.

---

### Phase F3: RoPE Kernel

**What it does:** Rotary Position Embeddings — 2D rotation on Q and K pairs. Teaches trig functions, indexing patterns.

```cuda
// kernels/rope.cu
__global__ void apply_rope_kernel(const float* q, const float* k,
                                   const float* freqs, float* q_out, float* k_out,
                                   int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * num_heads * head_dim / 2) {
        // apply 2D rotation at position idx / (num_heads * head_dim / 2)
        // at even position idx * 2 (cos) and odd position idx * 2 + 1 (sin)
    }
}
```

**Tests:** `tests/unit/_cuda/test_rope.py`
- `test_rope_matches_torch_float32` — rtol=1e-4
- `test_rope_matches_torch_float64` — rtol=1e-4
- `test_rope_positional_sensitivity` — different positions → different outputs

**Gate:** ~3 tests pass → commit.

---

### Phase F4: SwiGLU Kernel

**What it does:** SwiGLU FFN: `silu(W1 @ x) @ W2`. Three weight matrices. The SiLU part is a CUDA kernel; the matmul is done by PyTorch's cuBLAS. This teaches the hybrid approach.

```cuda
// kernels/ffn.cu
// SiLU is the same as activation.cu — reuse via include or recompile
// GEMM is handled by PyTorch — this kernel focuses on the element-wise part
```

**Python wrapper:**
```python
def swiglu_ffn(x, w1, w2, w3):
    # 1. PyTorch GEMM: gate = x @ w1.T, query = x @ w3.T
    # 2. CUDA SiLU: output = silu(gate) * query
    # 3. PyTorch GEMM: return output @ w2.T
```

**Tests:** `tests/unit/_cuda/test_ffn.py`
- `test_swiglu_matches_torch` — rtol=1e-4
- `test_swiglu_shapes` — various hidden expansion factors

**Gate:** ~2 tests pass → commit.

---

### Phase F5: Attention Kernel

**What it does:** Stable softmax (max-subtract-then-exp) + weighted sum. NOT full MHA — QKV projections and output projection use PyTorch's `nn.Linear` and `torch.mm`. The kernel focuses on the numerically tricky part: stable softmax over the attention matrix.

```cuda
// kernels/attention.cu
__global__ void stable_softmax_kernel(float* logits, int rows, int cols) {
    // Each thread block handles one row
    // 1. Find max of row (warp reduction)
    // 2. Subtract max (numerical stability)
    // 3. expf each element
    // 4. Sum (warp reduction)
    // 5. Divide by sum
}

__global__ void weighted_sum_kernel(const float* attn, const float* v, float* out,
                                     int rows, int attn_cols, int v_cols) {
    // out[i][j] = sum_k(attn[i][k] * v[k][j])
    // Each row is a separate reduction
}
```

**Tests:** `tests/unit/_cuda/test_attention.py`
- `test_softmax_stable` — max-subtract-then-exp produces valid probabilities
- `test_weighted_sum` — matches einsum
- `test_attention_matches_torch` — rtol=1e-3 (stable softmax ≠ PyTorch exactly)

**Gate:** ~3 tests pass → commit.

---

### Phase F6: MoE Kernel

**What it does:** Top-k expert routing with weighted combination. Teaches scattering, gathering, and indexed access — patterns common in GPU programming.

```cuda
// kernels/moe.cu
__global__ void topk_routing_kernel(const float* scores, int* indices,
                                     float* weights, int batch_seq, int n_experts, int top_k) {
    // 1. For each token, find top-k indices (naive comparison — not optimized)
    // 2. Apply softmax to top-k scores
    // 3. Store indices and weights (interleaved: [i0, w0, i1, w1, ...])
}

__global__ void weighted_sum_kernel(const float* expert_outputs, const float* weights,
                                     float* out, int batch_seq, int top_k, int dim) {
    // out[b] = sum_k(weights[b][k] * expert_outputs[indices[b][k]])
}
```

**Tests:** `tests/unit/_cuda/test_moe.py`
- `test_topk_matches_torch` — rtol=1e-3
- `test_weighted_sum_matches` — rtol=1e-3

**Gate:** ~2 tests pass → commit.

---

### Phase F7: Full CUDAModel

**What it does:** Complete model — embedding → TransformerBlocks → RMSNorm → SwiGLU → Linear → logits. Integrates all kernels above.

```python
class CUDAModel(nn.Module):
    def __init__(self, config):
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.stack = nn.ModuleList([CUDATransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = RMSNormWithCUDA(config.embed_dim)  # CUDA kernel
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.stack:
            x = block(x)  # uses CUDA kernels internally
        x = self.final_ln(x)  # CUDA RMSNorm kernel
        return self.output_proj(x)
```

**Tests:** `tests/unit/_cuda/test_model.py`
- `test_model_matches_torch` — rtol=1e-2 (2+ layers)
- `test_model_named_parameters` — key names match other backends

**Gate:** ~2 tests pass → commit.

---

### Phase F8: Training + Inference Scripts

What it does: Training loop and inference engine matching the API of other backends.

**Tests:** `tests/unit/_cuda/test_training.py` + `test_inference.py`
- `test_training_step` — loss reduction
- `test_inference_generation` — token match

**Gate:** ~2 tests pass → commit.

---

### Phase F9: 4-Way Cross-Backend Parity

What it does: NumPy ↔ PyTorch ↔ Triton ↔ CUDA produce identical results.

```python
@pytest.mark.gpu
class TestCUDAParity:
    def test_all_four_forward(self):
        """All 4 backends produce same output for given input"""
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
1. F0 (Scaffolding) — done ✅
2. F1-F6 (Standalone kernels) — SiLU → RMSNorm → RoPE → SwiGLU → Attention → MoE
3. F7-F8 (Full model + scripts) — CUDAModel → training + inference
4. F9 (Parity) — 4-way equivalence tests
```

### CUDA-Specific Concerns (REDUCED)
1. **Compilation speed:** nvrtc compiles at runtime. Caching in `.cache/` essential.
2. **Memory management:** PyTorch tensors manage memory. No manual free needed.
3. **Streams:** PyTorch streams handle concurrent execution. Not manual.
4. **Error handling:** PyTorch exceptions instead of `cudaError_t`.

### What CUDA Learning is LOST
- Manual `cudaMalloc`/`cudaMemcpy`/`cudaFree`
- Manual `cuLaunch`/`cuStream`/`cuEvent`
- Stream ordering and overlap
- `cudaError_t` error handling

### What CUDA Learning is PRESERVED
- Hand-written CUDA C in `.cu` files
- Shared memory usage (in RMSNorm, MHA kernels)
- Warp-level reduction (in RMSNorm, softmax)
- Coalesced memory access patterns
- Grid/block/thread configuration
- PTX compilation via nvrtc
- Numerical stability techniques (stable softmax)

## Estimation (REDUCED)

| Stage | Commits | Hours | Risk | Status |
|-------|---------|-------|------|--------|
| F0 | 0 | 0 | — | ✅ Done |
| F1 (SiLU) | 1 | 2 | Low | 🔶 Test written |
| F2 (RMSNorm) | 1 | 2 | Low | 🔶 Not started |
| F3 (RoPE) | 1 | 2 | Low | 🔶 Not started |
| F4 (SwiGLU) | 1 | 2 | Low (hybrid) | 🔶 Not started |
| F5 (Attention) | 2 | 3 | Medium | 🔶 Not started |
| F6 (MoE) | 1 | 2 | Medium | 🔶 Not started |
| F7 (Model) | 1 | 2 | Low (wires kernels) | 🔶 Not started |
| F8 (Scripts) | 1 | 2 | Low | 🔶 Not started |
| F9 (Parity) | 1 | 2 | Medium | 🔶 Not started |
| **Total** | **~9** | **~17** | **Low-Medium** | |

### Why Reduced from Original (21 hours, 15 commits)
- **Fewer kernels:** SiLU, RMSNorm, RoPE, SwiGLU (hybrid), Attention (hybrid), MoE = 6 vs original 6+MHA+TransformerBlock+DecoderStack+Model
- **Hybrid approach:** SwiGLU and Attention use PyTorch for GEMM, only CUDA for element-wise/stable algorithms
- **No 3-way:** Original had MHA, TransformerBlock, DecoderStack, Model — now simplified

## Completion Criteria

Phase F is done when ALL of the following are true:

- [ ] **CUDA test count** — ~19 new CUDA tests, all passing
- [ ] **All CUDA kernels are hand-written CUDA C** — `.cu` files, not wrappers around cuBLAS/cuDNN
- [ ] **nvrtc compilation works** — PTX compiled at runtime, cached
- [ ] **ruff check passes** — zero errors in `impl/_cuda/`
- [ ] **pyright check passes** — zero errors or expected documented
- [ ] **commits are clean** — one per stage, meaningful messages
- [ ] **4-way parity** — NumPy, PyTorch, Triton, CUDA produce matching outputs

**Tolerance for CUDA parity:**
- Standalone kernels: `rtol=1e-4, atol=1e-4`
- 1-layer model: `rtol=1e-3, atol=1e-3`
- 2+ layer model: `rtol=1e-2, atol=1e-2`

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| `cuLaunchKernel` TypeError | 10+ | Documented: BROKEN on JetPack 6.2.2, do NOT use |
| `cuLaunchKernelEx` TypeError | 10+ | Same as above |
| 3-param kernel with int via `cuParamSeti` | 4 | `int` params don't work via legacy API |
| Phase F as planned | N/A | REDESIGNED: nvrtc + PyTorch custom ops instead of bare-metal |

---

## Decision Required from User

**The original Phase F plan is broken.** Three options:

1. **Option A: nvrtc + PyTorch custom kernels** (planned above)
   - Preserves CUDA C learning (shared memory, warp reduction, coalesced access)
   - Uses PyTorch as practical dispatcher
   - ~17 hours, ~9 commits, 19 tests
   - This is what I recommend

2. **Option B: Legacy API only (`cuLaunch`)**
   - Only 2-param kernels, 1D fixed grids
   - No streams, no events, no grid config
   - MHA kernel impossible
   - ~10 hours, ~5 commits, 8 tests
   - Only SiLU, RMSNorm, RoPE

3. **Option C: Skip Phase F entirely**
   - Add more to existing backends instead
   - Better training/inference scripts
   - Data pipeline improvements

**Please let me know which option to proceed with.**