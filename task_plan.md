# Task Plan: Decoder-Only Transformer Learning Project

## Goal
Build a fully functional decoder-only transformer LLM from scratch in 4 implementations (NumPy, PyTorch, Triton, CUDA) with identical behavior, trained on TinyStories, featuring RoPE, MHA, MoE, GQA, and multi-level KV caching for educational purposes.

## Current Phase
**Phase 3 (PyTorch) is ✅ COMPLETE.**
**Phase 3+ (E2E Training/Inference) is ✅ COMPLETE.**
**Phase 3++ (Normalization Improvements) is ✅ COMPLETE.**
**Phase D (Cross-Backend Equivalence) is ✅ COMPLETE.**
**Phase E (Triton GPU Kernels) is ✅ COMPLETE.** — 538 tests pass
**Phase E+ (Cleanup & Refinement) is ✅ COMPLETE.** — 551 tests pass

**Next step: Phase F (CUDA Bare-Metal) — F0+F1 Complete, F2–F11 Ready**

---

## High-Level Roadmap

### Phase 2: NumPy Implementation (Learning-Focused) ✅ COMPLETE
- All core layers, RoPE, MHA, MoE, TransformerBlock, DecoderStack
- Loss functions, optimizers, KV Cache (Naive + TurboQuant)
- Training loop, inference engine, checkpoint save/load
- CLI, unit tests, cross-backend reference tests
- **Result:** 21 commits (b0–b19), TDD-style re-implementation, all tests pass

### Phase 3: PyTorch Implementation (Production-Ready) ✅ COMPLETE
- Same architecture as NumPy, `nn.Module` based
- Cross-backend parity tests, benchmarks
- **Plan:** `docs/phase_c_plan.md`
- **Status:** 36 commits, 310 total tests — **all pass**, ruff/pyright clean
- **Key artifacts:** `impl/_torch/` (22 files), `tests/unit/_torch/` (15 files), `tests/cross_backend/` (2 files)
- **Key fixes:** Wk.bias zero-gradient (mathematical property of softmax attention); weight transpose on Linear loading; `nn.Linear` for Wk/Wv to preserve bias gradients

### Phase 3+: E2E Training, Inference & Equivalence ✅ COMPLETE
- Unified training/inference scripts (NumPy + PyTorch)
- Interactive inference CLI with context status
- 8-combination automated equivalence matrix
- **Plan:** `docs/phase_c_plus_plan.md`
- **Status:** Complete — all 6 steps done, 400 tests pass, ruff/pyright clean

### Phase 3++: Normalization Improvements ✅ COMPLETE
- ✅ Post-Norm architecture (residual add → norm → gate)
- ✅ Gated residuals (sigmoid-scaled skip connections, gate init=0 → identity at start)
- ✅ Dropout (train/eval mode, deterministic inference)
- ✅ Gradient clipping in both backends
- ✅ Cross-backend parity maintained (save/load/load_from_numpy handle gate1/gate2)
- ✅ All inference tests updated with eval() for dropout deterministic behavior
- **Status:** all 421 tests pass, ruff/pyright clean

### Phase E: Triton Implementation (GPU Kernel Optimization) ✅ COMPLETE
- Custom GPU kernels: SiLU, RMSNorm, RoPE, SwiGLU, MHA, MoE+Expert, TransformerBlock, DecoderStack
- Full model: `TritonModel` — embedding → DecoderStack → RMSNorm → SwiGLU → Linear → logits
- Training loop: `train_step`, `clip_gradients`, `compute_gradient_norm`
- CLI inference: `TritonTextGenerator` with greedy/sampled/top-k decoding
- Cross-backend parity: Triton ↔ PyTorch (forward, backward, training) — rtol=1e-3
- Parity via `save_as_numpy()` → `load_from_numpy_dict()` (transposed output_proj weights for compat)
- **Total tests:** 538 (all pass), ruff + pyright clean

### Phase E+: Cleanup & Refinement ✅ COMPLETE
- Zero magic strings in codebase (all constants in `shared/constants.py`)
- All 551 tests pass, ruff + pyright clean
- Consistent naming across all 3 backends
- Comprehensive documentation for all Triton kernels
- 3-way equivalence: NumPy/Torch/Triton produce identical outputs

### Phase F: CUDA Implementation (Bare-Metal) 🔶 IN PROGRESS
- Platform: Jetson AGX Orin 64GB, JetPack 6.2.2, CUDA 12.6, PyTorch 2.11.0
- **Working API Pattern:** nvrtc compile → PTX → PyTorch custom op dispatcher (Option A)
  - `cuLaunchKernel` via `(values, types)` tuple + explicit stream + `extra=0` ✅
  - PyTorch tensors for memory (automatic `cudaMalloc`/`cudaFree`)
  - Backward via PyTorch autograd (CUDA kernels provide forward)
- **F0: Scaffolding** ✅ — `impl/_cuda/` + `tests/unit/_cuda/` created, import test passes
- **F1: SiLU** ✅ — nvrtc compile + PyTorch custom op, 4 tests pass, both fp32/fp64
  - Hand-written `.cu` source with f32/f64 kernels, nvrtc compilation + caching
  - `impl/_cuda/compiler.py` — nvrtc compile → PTX → cache in `impl/_cuda/.cache/`
  - `impl/_cuda/activation.py` — PyTorch custom_op dispatcher
  - `tests/unit/_cuda/test_activation.py` — 4 tests, all pass
- **F2–F11:** Ready to start RMSNorm → RoPE → SwiGLU → MHA → MoE → model wiring → training/inference → parity
- **Learning focus:** warp reduction, shared memory, coalesced access, grid/block/threads, PTX

### Phase G: Integration & Verification 🔲 NOT STARTED
- Train on TinyStories per backend -> save/load cross-validation -> identical outputs -> final e2e script

---

## Phase E: Triton Implementation (GPU Kernel Optimization) ✅ COMPLETE

**Status:** ✅ Complete — all 12 stages done (E0–E11) + E+ cleanup, 551 tests pass total

**Completed Stages:**
- `E0` — Scaffolding: `impl/_triton/` + `tests/unit/_triton/`
- `E1–E3` — Standalone kernels: SiLU (`activation.py`), RMSNorm (`layernorm.py`), RoPE (`rope.py`) 
- `E4` — SwiGLU FFN (`ffn.py`)
- `E5` — MHA kernel (`attn.py`) with pad/tile/softmax/weighted-sum + autograd wrapper
- `E6` — MoE + Expert (`moe.py`) with top-k routing + stable softmax
- `E7` — TransformerBlock (`transformer.py`) — MHA + MoE + SiLU + RMSNorm + residual + dropout
- `E8` — DecoderStack with sequential layer chaining via list
- `E9` — TritonModel: embedding → DecoderStack → RMSNorm → SwiGLU output → Linear logits
- `E10` — Training loop: `train_step` (reshape + forward + backward + clip + step)
- `E11` — CLI (`cli.py`) + inference engine (`inference.py`): greedy/sampled/top-k/text conversion
- `E12` — Cross-backend parity: Triton ↔ PyTorch forward (rtol=1e-3), backward (norm ratio < 1.1), training loss reduction

**Cross-backend parity fixes:**
- `save_as_numpy()` transposes `output_proj_w` (D, V) for NumPy convention
- `load_from_numpy_dict()` accepts TorchModel keys (`final_gamma`, transposed Matrices)
- `train_step` reshapes logits/targets for CrossEntropyLoss compatibility
- Parity via `save_as_numpy()` → `load_from_numpy_dict()` for weight sync

**Environment:** CUDA 12.6, cuDNN 9.3, cuBLAS 12.6, 8x Orin GPU, PyTorch 2.11.0, Triton 3.6.0

**Reference:** `docs/phase_e_plan.md` contains the full 12-stage spec.

---

## Phase 1A: Shared Foundation ✅ COMPLETE
**This phase is done.** See `docs/phase_a_plan.md` for full details.

**Summary:** Create `shared/` module with config, constants, tokenizer, dataset loaders for all 4 backends.
- ✅ `shared/config.py` — 41 tests pass
- ✅ `shared/tokenizer.py` — 21 tests pass
- ✅ `shared/constants.py` — 79 tests pass (strict TDD: class-by-class)
- ✅ `shared/dataset.py` — 12 tests pass (cache in resource/)
- ✅ `shared/checkpoint.py` — 11 tests pass (save/load config + npz)
- ✅ Integration tests — 11 tests (full pipeline: config→save→load→verify)

## Key Questions
1. ~~Tokenizer choice for TinyStories~~ (confirmed: BytePair + Char fallback)
2. ~~Dataset source~~ (confirmed: TinyStories, ~8MB from HuggingFace)
3. ~~KV cache approach~~ (confirmed: naive full-precision + TurboQuant 1-bit)
4. ~~GQA~~ (confirmed: opt-in, toggle via config)
5. ~~MoE: top-k (default 2) or all experts?~~ (confirmed: top-2)
6. ~~Training: which loss + optimizer? default?~~ (confirmed: CrossEntropy + AdamW)
7. ~~Project structure: shared code vs per-backend standalone?~~ (confirmed: shared `shared/` + per-backend `impl/_np/`, `impl/_torch/`)
8. **Residual connections needed?** — User flagged "lack of residual connection" but current code has post-norm residuals with gating.
9. **Post-Norm vs Pre-Norm?** — Implemented in Phase 3++: post-norm (residual add → norm → gate)

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| NumPy first, then torch/triton/cuda | Learning path; NumPy is the "source of truth" — everyone learns from it first |
| TinyStories dataset | Small, clean, ideal for demo |
| Shared config + tokenizer | Single source of truth across backends |
| TurboQuant for KV | Google's approach, dramatic memory savings for long sequences |
| All backends produce identical results | Deterministic with same seed |
| **Post-Norm architecture** | Residual add → RMSNorm → gated residual + dropout — standard for stable training |
| Gated residuals | Learnable `sigmoid(gate)` scaling on each residual, initialized to 0.5 |
| Dropout | 0.05 rate by default, disabled in eval mode for deterministic inference |
| Single train script with --backend flag | Less duplication, easier maintenance |
| Greedy decoding = 100% deterministic | Exact token match across backends; sampling uses KL divergence |
| Gradient clipping | Added for training stability in both backends |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Previous TDD agents ignored "test-first" requirement | 1 | User enforced: write ALL tests first, run to confirm fail, then implement |
| Tests timeout downloading TinyStories dataset | 1 | Tests already written; dataset loading expected to take ~1 min first time |
| Pyright error: `savez_compressed` argument type | 1 | Added `# pyright: ignore[reportArgumentType]` to dict unpacking |

# Phase 3++: Normalization Improvements

## Goal
Implement architecture improvements for faster training and better gradient flow.

## Current Architecture (Post-Norm with Gated Residuals + Dropout)
Both backends use post-normalization (RMSNorm AFTER the residual add):
```
# Stream 1: Attention
attn_out = MHA(x)
h = x + attn_out              # residual add
h = RMSNorm(h)                 # post-norm
h = h + sigmoid(gate1) * h     # gated residual
h = dropout(h)                 # dropout (training only)

# Stream 2: MoE
moe_out = MoE(h)
out = h + moe_out              # residual add
out = RMSNorm(out)             # post-norm
out = out + sigmoid(gate2) * out  # gated residual
out = dropout(out)             # dropout (training only)
```

Gated residuals:
- `gate1`, `gate2`: parameters initialized to 0 in both backends
- Sigmoid activation: `sigmoid(0) = 0.5` at init → partial gating, learns to open during training

## Implementation Status
All improvements implemented in both backends:
1. ✅ Post-Norm (residual add → norm → gate)
2. ✅ Gated residuals (sigmoid-scaled skip connections)
3. ✅ Dropout (train/eval mode, deterministic inference)
4. ✅ Gradient clipping (both backends)
5. ✅ Cross-backend parity maintained (save/load/load_from_numpy handle gate1/gate2)

## TDD Approach
- Write failing test first (gradient norm check, training speed comparison)
- Implement changes in both backends in parallel
- Verify cross-backend parity maintained
- Small, focused commits per sub-change

## Current Project State
| Phase | Tests | Commits | Status |
|-------|-------|---------|--------|
| A (Shared) | 111 | N/A | ✅ Complete |
| B (NumPy) | ~70 | 21 (b0-b19) | ✅ Complete |
| C (PyTorch) | 129 | 36 (c0-c36) | ✅ Complete |
| C+ (E2E) | 90 | 8 (c37-c44) | ✅ Complete |
| C++ (Norm) | 21 | 3 (d0-d2) | ✅ Complete |
| D (Equivalence) | 0 | 1 (e0) | ✅ Complete |
| E (Triton) | 538 | ~53 | ✅ Complete |
| E+ (Cleanup) | +13 | ~15 | ✅ Complete |
| **F0** | **128** | **1 (f0)** | **✅ Complete** |
| **F1** | **4** | **1 (f1)** | **✅ Complete** |
| F2–F11 | 0 | 0 | 🔶 In Progress |
| **Total** | **558** | **~130** | **557 pass, ruff/pyright clean** |

---

## Phase D: Cross-Backend Equivalence Verification

**Issue:** `scripts/verify_equivalence.py` reported 0/6 scenarios passed with `weight_diff ≈ 0.42`

**Root causes identified and fixed:**
1. PyTorch MoE router had `bias=False` but NumPy had bias → `load_from_numpy()` skipped bias
2. `verify_equivalence.py` never called `torch_model.load_from_numpy()` → models differed
3. `verify_equivalence.py` used `state_dict()` (nested keys) instead of `save_as_numpy()` (flat keys)
4. `weight_diff()` crashed on zero-size expert arrays → added skip for `size == 0`
5. Greedy inference ran in training mode → dropout made PyTorch output non-deterministic
6. Distribution check called `.numpy()` on gradient tensor → crashes

**Changes made:**
- `impl/_torch/layers.py:526` — `Linear(embed_dim, n_experts, bias=True)`
- `impl/_torch/layers.py:924-925` — Load MoE bias in `load_from_numpy()`
- `impl/_torch/layers.py:1017-1019` — Save MoE bias in `save_as_numpy()`
- `impl/_torch/layers.py:1103-1104` — Load MoE bias in `load_from_numpy_dict()`
- Fixed duplicate `return` + dead code block in `load_from_numpy_dict()` (removed lines 1125-1179)
- `scripts/verify_equivalence.py:398` — Use `save_as_numpy()` instead of `state_dict()`
- `scripts/verify_equivalence.py:440` — Add `torch_model.load_from_numpy(np_model)`
- `scripts/verify_equivalence.py:244-245` — Skip zero-size arrays in `weight_diff()`
- `scripts/verify_equivalence.py:483-493` — Add `torch.no_grad()` + `eval()` for greedy
- `scripts/verify_equivalence.py:515-521` — Use `.detach().numpy()` for distribution check

**Result:** All 6/6 scenarios pass with `weight_diff=0.0`, identical tokens, KL=0.0