# Progress Log

## Session: 2026-06-16

### Phase 0: Project Initialization
- **Status:** complete
- Cleaned existing codebase, created planning files

### Phase 1A: Shared Foundation
- **Status:** complete
- `shared/config.py` — 41 tests pass
- `shared/tokenizer.py` — 21 tests pass
- `shared/constants.py` — 79 tests pass (strict TDD)
- `shared/dataset.py` — 12 tests pass (resource cache, no external downloads)
- `shared/checkpoint.py` — 11 tests pass
- Integration tests (conftest.py + shared_pipeline.py) — 11 tests pass

### Phase 2: NumPy Implementation
- **Status:** complete
- 21 commits (b0–b19), entirely re-implemented in TDD style
- All core layers built from scratch, each with separate failing-then-passing commits:
  - b0: Project scaffolding
  - b1: Token Embedding (3 tests)
  - b2: RMSNorm (6 tests — 4 forward + 2 backward)
  - b3: SiLU activation (4 tests)
  - b4: SwiGLU FFN (5 tests)
  - b5: RoPE position encoding (4 tests)
  - b7: MHA with GQA (5 tests)
  - b8: MoE with top-k routing (4 tests)
  - b9: TransformerBlock (4 tests)
  - b10: DecoderStack (3 tests)
  - b11: NumPyModel full (5 tests)
  - b12: CrossEntropy Loss (4 tests)
  - b13: AdamW Optimizer (4 tests)
  - b14: Training Loop (11 tests)
  - b15: Naive KV Cache (4 tests)
  - b16: TurboQuant KV Cache (4 tests)
  - b17: Autoregressive Inference (4 tests)
  - b18: CLI interface (6 tests)
  - b19: Full Training Pipeline (4 tests)
- Quality: ruff check clean, pyright clean on all NumPy modules

### Phase 3: PyTorch Implementation
- **Status:** PLANNED but NOT EXECUTED
- `docs/phase_c_plan.md` created — 14 sub-phases (C0–C14), 20+ commits, ~65-70 tests
- No `_torch/` directory exists yet
- Ready to begin execution

---

## Test Results
| Module | Tests | Status |
|--------|-------|--------|
| shared/ | 111 | ✅ all pass |
| impl/_np/ (21 modules) | ~70 | ✅ all pass (TDD re-impl) |
| impl/_torch/ | — | ⏳ not started |
| tests/cross_backend/ | — | ⏳ not started |
| **Total** | **223** | **✅ all pass** |

## Plan File Hierarchy
| File | Purpose |
|------|---------|
| `task_plan.md` | High-level 6-phase roadmap |
| `docs/phase_a_plan.md` | Phase 1A (Shared Foundation) — **complete** |
| `docs/phase_b_plan.md` | Phase 2 (NumPy re-implementation) — **complete** |
| `docs/phase_c_plan.md` | Phase 3 (PyTorch implementation) — **planned, not started** |
| `findings.md` | Research findings, design decisions, validation strategy |
| `progress.md` | This file — session logs, test results, reboot check |
| `docs/design.md` | Full architecture design document |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 3 plan created but not started. Phase 1A and Phase 2 are complete. |
| Where am I going? | Execute Phase 3: PyTorch implementation with TDD, parity tests, benchmarks |
| What's the goal? | Build decoder-only transformer in 4 backends (NumPy, PyTorch, Triton, CUDA) |
| What have I learned? | TDD test-first is mandatory; resource caching avoids external downloads; planning files must be synced after each phase |
| What have I done? | Shared foundation complete, NumPy re-impl complete (21 commits, 223 tests pass), PyTorch plan created |

## Phase Completion Summary
| Phase | Name | Status | Tests | Commits | Plan File |
|-------|------|--------|-------|---------|-----------|
| A | Shared Foundation | ✅ Complete | 111 | N/A | `docs/phase_a_plan.md` |
| B | NumPy Implementation | ✅ Complete | ~70 | 21 (b0-b19) | `docs/phase_b_plan.md` |
| C | PyTorch Implementation | ⏳ Planned | — | 0 | `docs/phase_c_plan.md` |
| D | Triton Implementation | 🔲 Not Started | — | — | — |
| E | CUDA Implementation | 🔲 Not Started | — | — | — |
| F | Integration & E2E | 🔲 Not Started | — | — | — |
