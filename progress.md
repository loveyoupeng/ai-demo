# Progress Log

## Session: 2026-06-17

### Phase 3+: E2E Training/Inference Scripts
- **Status:** ✅ Complete
- Finalized commit for `auto_test_equivalence.py` (pyright fix + 18 tests)
- Updated `task_plan.md` and `progress.md` with completion status
- 400/400 tests pass, ruff + pyright clean

### Phase 3++: Normalization Improvements — PLANNED
- **Status:** in progress
- Created `docs/phase_c_plus2_plan.md` with detailed plan
- Investigated current residual connections — **already present** (pre-norm)
- Identified 3 candidate improvements:
  1. **Post-Norm** (residual add first, then norm)
  2. **Gated Residuals** (learnable `gate * residual`)
  3. **Dropout** (regularization, currently absent)
- **Blocked on user clarification** — need to confirm which improvement to implement

---

## Test Results
| Module | Tests | Status |
|--------|-------|--------|
| shared/ | 111 | ✅ all pass |
| impl/_np/ (21 modules) | ~70 | ✅ all pass |
| impl/_torch/ (22 files) | ~129 | ✅ all pass |
| tests/cross_backend/ (2 files) | 7 | ✅ all pass |
| **Total** | **400** | **✅ all pass** |

## Plan File Hierarchy
| File | Purpose |
|------|---------|
| `task_plan.md` | High-level 6-phase roadmap + Phase 3++ |
| `docs/phase_a_plan.md` | Phase 1A (Shared Foundation) — **complete** |
| `docs/phase_b_plan.md` | Phase 2 (NumPy re-implementation) — **complete** |
| `docs/phase_c_plan.md` | Phase 3 (PyTorch implementation) — **complete** |
| `docs/phase_c_plus_plan.md` | Phase 3+ (E2E training/inference/equivalence) — **complete** |
| `docs/phase_c_plus2_plan.md` | Phase 3++ (Normalization improvements) — **planned, waiting** |
| `findings.md` | Research findings, design decisions, validation strategy |
| `progress.md` | This file — session logs, test results, reboot check |
| `docs/design.md` | Full architecture design document |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 3++ planning — Phase 3+ complete with 400 tests. Residual connections already exist (pre-norm architecture). Clarifying whether user wants post-norm, gated residuals, or dropout before implementation. |
| Where am I going? | Implement normalization improvement once user confirms which one. |
| What's the goal? | Build decoder-only transformer in 4 backends (NumPy, PyTorch, Triton, CUDA) with identical behavior; currently adding architecture improvements for faster training and better gradient flow. |
| What have I learned? | Residual connections are already in place (h = x + attn_out, out = h + moe_out). User's concern about "lack of residual" may be about post-norm vs pre-norm, gated residuals, or dropout. Need clarification before implementing. |
| What have I done? | Updated task_plan.md, findings.md, progress.md, and docs/phase_c_plus2_plan.md with complete status and detailed improvement plan. |

## Session: 2026-06-16

### Phase 3+: E2E Training/Inference Scripts
- **Status:** complete
- c37: `shared/config_utils.py` — unified config reader with source tracking (20 tests)
- c38: `scripts/train.py` fixes — variable-length batch padding, CLI aliases, synthetic data generation, test fixes (9 tests)
  - Fixed `run_training_numpy`/`run_training_torch` to pad variable-length sequences before stacking
  - Added `--context_length`/`--embed_dim`/`--n_layers`/`--n_heads`/`--n_groups` as CLI aliases
  - Fixed synthetic data generation for small `context_length` values
  - Fixed `test_run_training_numpy` to use smaller model (NumPy finite-diff is O(params))
  - Switched `test_main_with_synthetic` to PyTorch backend (autograd, fast enough)
  - All 338+ tests pass, ruff/pyright clean

---

## Phase Completion Summary
| Phase | Name | Status | Tests | Commits | Plan File |
|-------|------|--------|-------|---------|-----------|
| A | Shared Foundation | ✅ Complete | 111 | N/A | `docs/phase_a_plan.md` |
| B | NumPy Implementation | ✅ Complete | ~70 | 21 (b0-b19) | `docs/phase_b_plan.md` |
| C | PyTorch Implementation | ✅ Complete | 129 | 36 (c0-c36) | `docs/phase_c_plan.md` |
| C+ | E2E Training/Inference | ✅ Complete | 90 | 8 (c37–c44) | `docs/phase_c_plus_plan.md` |
| C++ | Normalization Improvements | 🟡 Planned | — | — | `docs/phase_c_plus2_plan.md` |
| D | Triton Implementation | 🔲 Not Started | — | — | — |
| E | CUDA Implementation | 🔲 Not Started | — | — | — |
| F | Integration & E2E | 🔲 Not Started | — | — | — |
