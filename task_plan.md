# Task Plan: Decoder-Only Transformer Learning Project

## Goal
Build a fully functional decoder-only transformer LLM from scratch in 4 implementations (NumPy, PyTorch, Triton, CUDA) with identical behavior, trained on TinyStories, featuring RoPE, MHA, MoE, GQA, and multi-level KV caching for educational purposes.

## Current Phase
**Phase 3 — PyTorch Implementation is PLANNED (not started). Execution ready at `docs/phase_c_plan.md`.**

**Phase 2 (NumPy) is complete.** Next step: execute Phase 3 plan.

---

## High-Level Roadmap

### Phase 2: NumPy Implementation (Learning-Focused) ✅ COMPLETE
- All core layers, RoPE, MHA, MoE, TransformerBlock, DecoderStack
- Loss functions, optimizers, KV Cache (Naive + TurboQuant)
- Training loop, inference engine, checkpoint save/load
- CLI, unit tests, cross-backend reference tests
- **Result:** 21 commits (b0–b19), TDD-style re-implementation, all tests pass 📍 `docs/phase_b_plan.md`

### Phase 3: PyTorch Implementation (Production-Ready) ✅ COMPLETE
- Same architecture as NumPy, `nn.Module` based
- Cross-backend parity tests, benchmarks
- **Plan:** `docs/phase_c_plan.md`
- **Status:** 36/20 commits, 310 tests — **all pass**, ruff/pyright clean
- **Key artifacts:** `impl/_torch/` (22 files), `tests/unit/_torch/` (15 files), `tests/cross_backend/` (2 files)
- **Key fixes:** Wk.bias zero-gradient (mathematical property of softmax attention); weight transpose on Linear loading; `nn.Linear` for Wk/Wv to preserve bias gradients

### Phase 3+: E2E Training, Inference & Equivalence ✅ COMPLETE
- Unified training/inference scripts (NumPy + PyTorch)
- Interactive inference CLI with context status
- 8-combination automated equivalence matrix
- **Plan:** `docs/phase_c_plus_plan.md`
- **Status:** Complete — all 6 steps done, 400 tests pass, ruff/pyright clean

### Phase 4: Triton Implementation (GPU Kernel Optimization)
- Custom kernels: LayerNorm, attention, MoE routing, activations
- Parity tests, profiling vs NumPy/PyTorch

### Phase 5: CUDA Implementation (Lowest Level)
- `nvidia/cuda-python` bindings, same architecture
- Parity tests, benchmarks

### Phase 6: Integration & Verification
- Train on TinyStories per backend -> save/load cross-validation -> identical outputs -> final e2e script

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
5. MoE: top-k (default 2) or all experts?
6. Training: which loss + optimizer? default?
7. Project structure: shared code vs per-backend standalone?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| NumPy first, then torch/triton/cuda | Learning path; NumPy is reference implementation |
| TinyStories dataset | Small, clean, ideal for demo |
| Shared config + tokenizer | Single source of truth across backends |
| TurboQuant for KV | Google research, 1-bit compression |
| All backends produce identical results | Deterministic with same seed |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Previous TDD agents ignored "test-first" requirement | 1 | User enforced: write ALL tests first, run to confirm fail, then implement |
| Tests timeout downloading TinyStories dataset | 1 | Tests already written; dataset loading expected to take ~1 min first time |
