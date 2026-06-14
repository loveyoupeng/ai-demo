# Task Plan: Decoder-Only Transformer Learning Project

## Goal
Build a fully functional decoder-only transformer LLM from scratch in 4 implementations (NumPy, PyTorch, Triton, CUDA) with identical behavior, trained on TinyStories, featuring RoPE, MHA, MoE, GQA, and multi-level KV caching for educational purposes.

## Current Phase
Phase 1A — Shared Foundation (Execution in `docs/phase_a_plan.md`)

---

## High-Level Roadmap

### Phase 2: NumPy Implementation (Learning-Focused)
- All core layers, RoPE, MHA, MoE, TransformerBlock, DecoderStack
- Loss functions, optimizers, KV Cache (Naive + TurboQuant)
- Training loop, inference engine, checkpoint save/load
- CLI, unit tests, cross-backend reference tests

### Phase 3: PyTorch Implementation (Production-Ready)
- Same architecture as NumPy, `nn.Module` based
- Cross-backend parity tests, benchmarks

### Phase 4: Triton Implementation (GPU Kernel Optimization)
- Custom kernels: LayerNorm, attention, MoE routing, activations
- Parity tests, profiling vs NumPy/PyTorch

### Phase 5: CUDA Implementation (Lowest Level)
- `nvidia/cuda-python` bindings, same architecture
- Parity tests, benchmarks

### Phase 6: Integration & Verification
- Train on TinyStories per backend -> save/load cross-validation -> identical outputs -> final e2e script

---

## Phase 1A: Shared Foundation (Canonical Plan)
📍 **This is the active execution plan.** See `docs/phase_a_plan.md` for full details, agent statuses, and gates.

**Summary:** Create `shared/` module with config, constants, tokenizer, dataset loaders for all 4 backends.
- ✅ `shared/config.py` — 41 tests pass
- ⏳ `shared/tokenizer.py` — 21 tests pass
- ❌ `shared/constants.py` — 52/79 tests fail (needs full impl)
- ❌ `shared/dataset.py` — tests timeout (download issue)
- ⏳ `shared/checkpoint.py` — BLOCKED by constants + dataset

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
