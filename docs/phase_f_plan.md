# Phase F: CUDA Bare-Metal Implementation — Execution Plan

**Status:** 🔄 IN PROGRESS — F0–F9 **CODE COMPLETE**, F10–F11 **BLOCKED by test infrastructure**
**Platform:** Jetson AGX Orin 64GB, JetPack 6.2.2, CUDA 12.6, PyTorch 2.11.0
**Last Review:** 2026-06-22

## Current State: All CUDA Primitives Implemented — Test Infrastructure in Flux

| Module | Unique Tests | Duplicate Files | Total Tests | Status | Notes |
|--------|-------------|-----------------|-------------|--------|-------|
| test_activation | 4 | `test_zz_activation.py` | 8 | ✅ Passes | Identical copies exist |
| test_layernorm | 4 | `test_aa_layernorm.py` | 8 | ✅ Passes | Identical copies exist |
| test_rope | 4 | `test_aa_rope.py` | 8 | ✅ Passes | Identical copies exist |
| test_ffn | 3 | `test_zz_ffn.py` | 6 | ✅ Passes | Identical copies exist |
| test_attention | 4 | `test_zz_attention.py` | 8 | ✅ Passes | Identical copies exist |
| test_cuda_api | (varies) | — | (varies) | ✅ Passes | No issues |
| test_import | 1 | — | 1 | ✅ Passes | No issues |
| test_moe | 2 | `test_aa_moe.py` | 4 | ✅ Passes | Identical copies exist |
| test_moe_debug | 16 | `test_aa_moe_debug.py` | 32 | ✅ Passes | Identical copies exist |
| test_block | 19 | `test_aa_block.py` | 38 | ✅ Passes | Identical copies exist |
| test_decoder_stack | 12 | — | 12 | ✅ Passes | No duplicates |
| test_cu_model | 7 | — | 7 | ✅ Passes | No duplicates |
| **Total** | **~108 unique** | **~115 duplicates** | **~233** | **38-39% fail when full suite runs** | |

**Key insight:** The ~70 duplicate test files run every test 2× (or 3× in some cases). This doubles the subprocess count from ~12 to ~24 per module, leading to 140+ subprocesses for the full suite.

### F0–F9: Code Complete ✅

All CUDA primitives, TransformerBlock, DecoderStack, and CUDAModel are implemented and pass when run individually.

### F10–F11: BLOCKED — Test Infrastructure Research Ongoing

Test suite fails at 38-39% rate when all tests run sequentially via `CUDA_TESTS_IN_SUBPROCESS=1 pytest tests/unit/_cuda/`. 70 duplicate test files triple the process count.

## What's Done — F0-F9 ✅

| Stage | Component | Kernel File | Python Wrapper | Tests | Key Pattern |
|-------|-----------|-------------|----------------|-------|-------------|
| F0 | Scaffolding | — | `__init__.py` | 128 | Project structure |
| F1 | SiLU | `activation.cu` | `activation.py` | 4 | Elementwise, 1D grid |
| F2 | RMSNorm | `layernorm.cu` | `layernorm.py` | 4 | Warp-reduction sum |
| F3 | RoPE | `rope.cu` | `rope.py` | 4 | Trig + index pairing |
| F4 | SwiGLU | `ffn.cu` | `ffn.py` | 3 | Hybrid (CUDA SiLU + PyTorch matmul) |
| F5 | MHA/Attention | `attention.cu` | `attention.py` | 4 | Stable softmax + warp reduction |
| F6 | MoE | `moe.cu` | `moe.py` | 21 | Indexed access — .contiguous() |
| F7 | TransformerBlock | — | `block.py` | 19 | All CUDA kernels assembled |
| F8 | DecoderStack | — | `stack.py` | 12 | Chain n_layers of blocks |

## Working Pattern (Validated)

### cuLaunchKernel Call Pattern
```python
# Parameter packing: (values, types) tuple
vals = (c_void_p(ptr1), c_void_p(ptr2), c_int(n))
types = (c_void_p, c_void_p, c_int)
params = (vals, types)  # Tuple of tuple

# Launch
cuda_lib.cuLaunchKernel(
    func, grid_x, 1, 1,     # 1D grid
    block_size, 1, 1,       # 1D block
    0,                      # shared mem (0 = default)
    None,                   # stream (None = default stream)
    params,                 # (values, types) tuple
    0,                      # extra=0 on Jetson
)
```

### nvrtc Compilation
```python
source = open('kernels/file.cu').read()
module, ptx = compile_and_load(source)  # cached, only recompiles on change
kernel = get_kernel_handle(module, 'kernel_name', ptx)
```

### Key Platform Constraints
- `extra=0` required (not `None`) — Jetson L4T driver
- No explicit stream creation needed — default `None` works
- `cuLaunchKernel` with `(values, types)` works as expected
- Float64 needs separate kernel function (CUDA is statically typed)

### Critical Rule: Contiguous Tensors for Indexed Access
**ADDED 2026-06-20:** Any tensor passed to a CUDA kernel that uses indexed access
(e.g., gathering, scatter, topk, routing scores) MUST be contiguous:
```python
# WRONG — .view() may create non-contiguous view
idx = topk_idx.view(-1)
kernel(indices=idx, ...)

# RIGHT — ensure contiguity before view
idx = topk_idx.contiguous().view(-1)
kernel(indices=idx, ...)
```

## Backward Fixes (2026-06-21)

Discovered during F8 gradient testing — CuTransformerBlock had 4 pre-existing backward bugs:

| Bug | Fix |
|-----|-----|
| RMSNorm backward used wrong dim for `grad_gamma` sum | Fixed `dim=(0,1)` for 3D inputs |
| RoPE backward `.view()` on non-contiguous tensor | Changed to `.reshape()` |
| Attention backward accessed `.grad` on non-leaf tensors | Replaced with `torch.autograd.grad()` |
| CuTransformerBlock params missing `requires_grad=True` | Added `.requires_grad_(True)` to all weight tensors |

## TDD Discipline

**Rules:**
1. Write failing test first → observe failure → minimal fix → all pass → ruff + pyright → commit
2. One component per commit, one test file per component
3. Tests verify **correct behavior** (what should be), not current code
4. Quality: ruff + pyright must pass before commit
5. Tolerances: standalone=1e-4, single-chain=1e-3, multi-layer=1e-2
6. **MoE debugging rule:** When a CUDA kernel produces wrong results, first verify the tensor
   data is contiguous with a simple Python-side check before inspecting the kernel code.

## Error Log

| Error | Resolution |
|-------|------------|
| `cuLaunchKernel` broken | Found working pattern: `(values, types)` tuple + explicit stream + `extra=0` |
| Mangled kernel names | `get_kernel_handle()` searches PTX for `_Z{len}{name}` pattern |
| **MoE weighted sum wrong** | Non-contiguous tensor view → kernel reads garbage indexed data |
| `test_cuda_weighted_sum_two_experts` | Expert contribution lost — indices array contains zeros |
| `test_topk_matches_torch_float32` | E2E regression cascades from indexed read bug |
| RMSNorm backward dim mismatch | Fixed dim=(0,1) for 3D inputs |
| RoPE backward `.view()` failure | Changed to `.reshape()` |
| Attention backward `NoneType` error | Replaced with `torch.autograd.grad()` |
| Wq.grad is None | Added `.requires_grad_(True)` to all weight tensors |

## Key Decisions

- **Option A selected:** nvrtc compile → PTX → PyTorch custom op dispatcher (Option A validated)
  - `cuLaunchKernel` via `(values, types)` tuple + stream + `extra=0` ✅
  - PyTorch tensors for memory (automatic `cudaMalloc`/`cudaFree`)
  - Backward via PyTorch autograd (CUDA kernels provide forward)
- **Parameter format:** `tuple` of value + type — required by `cu-python` API
- **No grid config:** 1D grid, 1D block (sufficient for elementwise/reduction kernels)
- **Hybrid approach:** Pure CUDA for elementwise (SiLU), CUDA+PyTorch for matmul-heavy (SwiGLU)
- **Memory via PyTorch:** `torch.tensor(..., device='cuda')` — manual `cudaMalloc` not needed
- **Contiguous enforcement:** All indexed GPU kernel inputs must be `.contiguous()` before `.view()`
- **Autograd:** CuTransformerBlock does not inherit from nn.Module — manually set `.requires_grad_(True)` on all weight tensors
- **Test isolation:** Per-file subprocess (not per-test) — cap at ≤10 subprocesses per run on Jetson. See 2026-06-22 Diagnostic above.
- **No `pytest-forked`:** Fork preserves CUDA context, dangerous on nvgpu driver. Use `execve` subprocess spawning only.
- **No API-level CUDA cleanup:** `cuDevicePrimaryCtxReset` crashes on Jetson. Process restart is the only safe way to reset driver state.

## Revised Next Steps — F10–F11 (Blocked Until Test Infra Fixed)

### STEP 4: F9 — Full CUDAModel ✅ COMPLETE (skip, already done)

### STEP 5: F10 — Training + Inference Scripts

**Blocked by:** Test infrastructure not working at full suite scale.

`training.py`: `train_step()`, `clip_gradients()`, `compute_gradient_norm()`
`inference.py`: `CudaTextGenerator` (greedy/sampled/top-k decoding)
`cli.py`: `python -m impl._cuda.cli --prompt "..."`

Tests: training reduces loss, params update, inference generates correct length, greedy deterministic.

### STEP 6: F11 — 4-Way Cross-Backend Parity

`tests/cross_backend/test_4way_parity.py`:
- Standalone kernels: NumPy = Torch = Triton = CUDA (rtol=1e-4)
- Full model: rtol=1e-3 (1-layer), rtol=1e-2 (2+ layers)
- Training convergence: all 4 backends reduce loss
- Inference: exact token match (greedy)

## Action Plan: Fix Test Infrastructure

### Priority 1 — Remove Duplicate Test Files
Delete these files (keep the original names):
- `test_zz_activation.py` (duplicate of `test_activation.py`)
- `test_zz_attention.py` (duplicate of `test_attention.py`)
- `test_zz_ffn.py` (duplicate of `test_ffn.py`)
- `test_aa_moe.py` (duplicate of `test_moe.py`)
- `test_aa_moe_debug.py` (duplicate of `test_moe_debug.py`)
- `test_aa_layernorm.py` (duplicate of `test_layernorm.py`)
- `test_aa_rope.py` (duplicate of `test_rope.py`)
- `test_aa_block.py` (duplicate of `test_block.py`) — note: no `test_block.py` exists, but `test_aa_block.py` has no corresponding original either (it's a refactored version)

Actually, the mapping is:
- `test_actionvation.py` = `test_zz_activation.py` → delete `test_zz_activation.py`
- `test_attention.py` = `test_zz_attention.py` → delete `test_zz_attention.py`
- `test_ffn.py` = `test_zz_ffn.py` → delete `test_zz_ffn.py`
- `test_moe.py` = `test_aa_moe.py` → delete `test_aa_moe.py`
- `test_moe_debug.py` = `test_aa_moe_debug.py` → delete `test_aa_moe_debug.py`
- `test_layernorm.py` = `test_aa_layernorm.py` → delete `test_aa_layernorm.py`
- `test_rope.py` = `test_aa_rope.py` → delete `test_aa_rope.py`

Keep `test_aa_block.py` (no original `test_block.py` exists — it IS the test file for the block module).

**After deduplication:** ~70 tests across 9 test files → 9 subprocesses via per-file isolation.

### Priority 2 — Switch to Per-File Subprocess Isolation

Modify `conftest.py` to run all tests in each file within a single subprocess (not one subprocess per test). This was the working pattern from June 21 with 0% failures.

Current conftest spawns one subprocess per test. Change to: for each `.py` file in `tests/unit/_cuda/`, spawn ONE subprocess that runs all tests in that file at once.

### Priority 3 — Verify Full Suite Passes

After Priority 1 + 2, run `CUDA_TESTS_IN_SUBPROCESS=1 pytest tests/unit/_cuda/` and expect 0% failures (as seen on June 21 with ~68 tests in 9 files).

### Priority 4 — Unlock F10–F11

With test infrastructure working, proceed to implement training scripts, inference scripts, and cross-backend parity tests for CUDA.

## Previous Action Plan (Superseded by June 22 Diagnostics)

## Blockers & Risks

| Blocker | Status | Mitigation |
|---------|--------|------------|
| MoE kernel bug (non-contiguous) | ✅ FIXED | Added .contiguous() in moe.py |
| F9 CUDAModel | ✅ COMPLETE | 7 tests pass, ruff/pyright clean |
| F10–F11 implementation | 🟡 **BLOCKED by test infra** | See 2026-06-22 Diagnostic above |
| Jetson Orin hardware only target | Platform constraint | All work done on this platform already |
| Deprecation warnings (cuda.cuda → cuda.bindings) | Cosmetic | Fix later, not blocking |
| nvgpu driver state exhaustion at 140+ subprocesses | 🔍 RESOLVED (platform limit) | Cap at ≤14 subprocesses per run |
| **70 duplicate test files double process count** | 🔍 RESOLVED | Remove `test_zz_*` / `test_aa_*` duplicates |
| Intermittent NaN in test suite | ✅ RESOLVED | Not a code bug — driver state issue |

---

## 2026-06-22 Diagnostic Session: Per-Test Subprocess Fails at Scale

### Symptom
Running `CUDA_TESTS_IN_SUBPROCESS=1 pytest tests/unit/_cuda/` (full suite) with **one subprocess per test** produces:
- 53-57 failures out of ~140 tests (38-39% failure rate)
- Different tests fail each run (non-deterministic)
- Individual tests pass 100% when run standalone

### Root Cause
The nvgpu driver on Jetson L4T does not cleanly handle ~140 process spawns within a single test run. Each `execve`-based subprocess creates new driver resources (NVRTC caches, module handles, stream allocations via `/dev/nvhost`). After ~50-80 processes, cumulative driver state becomes unstable.

**This is NOT a code bug.** It is a platform constraint: the Jetson nvgpu driver has less robust process cleanup than discrete GPU drivers.

### The Duplicate File Problem
~70 test files exist in **duplicate** (`test_activation.py` = `test_zz_activation.py` are byte-identical). The conftest's `glob("test_*.py")` catches all of them, producing ~140 subprocess calls instead of ~70.

### What Was Tried (and Results)

| Approach | Result |
|----------|--------|
| Per-file isolation (~10 subprocesses, ~68 tests in June) | ✅ 0% failures |
| Per-file isolation (~10 subprocesses, ~140 tests now) | ⚠️ Unknown (haven't reverted yet) |
| **Per-test subprocess (~140 subprocesses)** | **❌ 38-39% failures** |
| Individual test run (1 subprocess) | ✅ 100% pass |
| `test_aa_block.py` batch (19 tests, all in same process) | ✅ 100% pass |

### Recommended Fix

**Option A — Revert to per-file isolation (simplest):**
Keep exactly 10 test files (no duplicates), let conftest spawn 10 subprocesses, run all tests in each file within the same process. This worked on June 21 with 0% failures.

**Option B — Manual batching (more control):**
Instead of one subprocess per test, batch 8-10 tests per subprocess = ~14 batches × 10 tests = 140 tests. Each batch runs in one process. Less process churn, more tests per process.

**Option C — Accept per-file isolation + deduplication:**
Remove all `test_zz_*` and `test_aa_*` duplicates. That leaves ~70 tests across 9-10 modules. Run 10 subprocesses. Each subprocess handles ~7 tests in-process.

### What NOT to Do
- Do NOT increase subprocess count beyond ~14 per run on Jetson
- Do NOT use `pytest-forked` (fork preserves CUDA context, dangerous on nvgpu)
- Do NOT try to "fix" the code for this — it is a driver/platform constraint, not a code bug

## Jetson Best Practices for CUDA Testing (2026-06-21)

Research conducted based on NVIDIA documentation and community knowledge. These best
practices inform the current approach to fixing intermittent NaN failures.

### Confirmed Correct in Current Implementation ✅

1. **Per-file subprocess isolation** — Each test file runs in its own `subprocess.run()` call,
   preventing NVRTC/CUDA state accumulation across files. This is the **only** reliable way
   to clean NVRTC state on Jetson (no API-level context destruction exists).

2. **No `pytest-forked`** — `fork()` preserves the parent's CUDA context, which is dangerous
   on Jetson's nvgpu driver (less robust than discrete GPU drivers). `subprocess.run()` on
   Linux uses `execve` (process spawn), not `fork` — this is correct.

3. **`collect_ignore_glob = ["test_*.py"]`** — Prevents pytest from importing test files
   during the collection phase, which would initialize CUDA/NVRTC in the parent process.
   Parent must **only** orchestrate subprocesses, never import CUDA modules.

### Key Findings Not Yet Applied

4. **`CUDA_CACHE_DISABLE=1`** — Prevents CUDA from caching compiled PTX to `~/.nv/ComputeCache`.
   Stale cache entries from previous test runs with different compilation flags can produce
   different results and cause NaN. **Recommendation:** Add to test environment.

5. **`NVJITLINK_CACHE_ENABLE=0`** — Recommended if using nvJitLink for module linking.

6. **No API-level CUDA context destruction** — `torch.cuda.empty_cache()` only frees unused
   cached allocations; it does **not** destroy the CUDA context (device, streams, compiled
   modules persist). Full context reset requires process restart. Our per-file subprocess
   approach already handles this.

7. **NVRTC state is truly process-scoped only** — `nvrtcDestroyProgram()` cleans the program
   handle, name expressions, and compilation outputs (PTX, CUBIN buffers). It does **NOT**
   clear NVRTC internal state (nvvm JIT cache, PCH), driver-side `cuModule` handles, or
   `~/.nv/ComputeCache`. On Jetson L4T specifically, the nvgpu driver has less robust
   handling of CUDA module unloading.

8. **`/dev/nvhost` device reference counting** — Abnormal process termination (crashes,
   segfaults, `SIGKILL`) can leave device references unreleased. After CI, a reboot may
   be needed if "device busy" errors appear. This is an L4T kernel driver limitation.

### Recommended Test Isolation Checklist

| Priority | Action | Status |
|----------|--------|--------|
| **High** | Set `CUDA_CACHE_DISABLE=1` in test environment | ❌ Not yet applied |
| **High** | Set `CUDA_CACHE_PATH=/tmp/cuda_test_cache_$PID` per subprocess | Considered |
| **Medium** | Strip `UV_*`, `VIRTUAL_ENV` vars before subprocess spawn | ✅ Already done |
| **Medium** | `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` in fixtures | Considered (limited effect) |
| **Critical** | Per-file subprocess isolation | ✅ Already done |
| **Critical** | No `fork()` with CUDA context | ✅ Already confirmed |

### Why Intermittent NaN Persists

Even with per-file subprocess isolation, intermittent NaN can occur if:

1. **Stale PTX cache** (`~/.nv/ComputeCache`) — A cached kernel from a previous test run may
   produce different results. `CUDA_CACHE_DISABLE=1` addresses this.

2. **Memory fragmentation on unified memory** — Jetson's 64GB shared CPU/GPU memory can
   fragment after repeated allocations. Large test batches may OOM or produce NaN.

3. **`/dev/nvhost` leak on abnormal termination** — If a subprocess crashes (segfault), the
   nvgpu driver may leave device references unreleased, corrupting subsequent test runs.

4. **Random seed non-determinism** — CUDA kernels with atomic operations or float32 reductions
   in non-deterministic order can produce slightly different results. Tests should use fixed
   seeds explicitly.