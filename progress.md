# Progress Log

## Session: 2026-06-14

### Phase 0: Project Initialization
- **Status:** complete
- Actions taken:
  - Cleaned existing codebase as requested by user
  - Created task_plan.md, findings.md, progress.md
  - Present design plan for user review
  - Implemented shared/config.py (Agent 1) — 41 tests pass, pyright/ruff clean
  - Implemented shared/tokenizer.py (Agent 3) — 21 tests pass, pyright/ruff clean
- Files created/modified:
  - task_plan.md, findings.md, progress.md (created)
  - shared/config.py, shared/tokenizer.py, shared/__init__.py (created)
  - tests/unit/test_config.py, tests/unit/test_tokenizer.py (created)

### Phase 1A: Constants Module
- **Status:** incomplete — 52/79 tests fail
- What's done: `shared/constants.py` has old `Attention` (4 attrs), `MoE` (2 attrs), `param()` (1 function)
- What's missing: 
  - New `Attention` attributes (Q, K, V, O, weights, biases — 13 new)
  - New `MoE` attributes (W1/W2/W3, GATE_WEIGHT, EXPERT_W1/W2/W3 — 7 new)
  - `LayerNorm` class (GN_BIAS, GN_GAMMA, LN_BIAS, LN_GAMMA)
  - `Transformer` class (EMBEDDING, LM_HEAD_WEIGHT, LM_HEAD_BIAS, TRANSFORMER_LN_*)
  - 6 helper functions: block_param, attention_param, layer_norm_param, moe_param, transformer_param, get_all_params
- Tests: `tests/unit/test_constants.py` — 27 pass, 52 fail

### Phase 1B: Dataset Module
- **Status:** incomplete — tests timeout on dataset download
- What's done: `shared/dataset.py` — 208 lines, complete implementation of `load_tinystories()`, `TextDataset`, `get_dataloader_sequences()`
- What's needed: Tests need to handle the ~1 min download time; may need caching/fixtures to avoid re-downloading every test run
- Tests: `tests/unit/test_dataset.py` — 14 tests, all timeout waiting for TinyStories download

### Phase 1C: Integration
- **Status:** not started
- Needs: Phase 1A + 1B → ruff + pyright → commit

## Test Results
| Module | Tests | Pass | Fail | Status |
|--------|-------|------|------|--------|
| config | 41 | 41 | 0 | ✅ |
| tokenizer | 21 | 21 | 0 | ✅ |
| constants | 79 | 27 | 52 | ❌ |
| dataset | 14 | 0 | 0 (timeout) | ❌ |

## Plan File Hierarchy
| File | Purpose |
|------|---------|
| `docs/phase_a_plan.md` | **CANONICAL** — Active execution plan for Phase 1A |
| `task_plan.md` | High-level 6-phase roadmap, points to phase_a_plan.md |
| `findings.md` | Research findings, design decisions, validation strategy |
| `progress.md` | This file — session logs, test results, reboot check |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1A — Shared Foundation, `docs/phase_a_plan.md` is the canonical current plan |
| Where am I going? | Complete constants.py → fix dataset tests → checkpoint → Phase 2 (NumPy backend) |
| What's the goal? | Build decoder-only transformer in 4 backends, starting with shared infrastructure |
| What have I learned? | TDD test-first is mandatory; dataset download is slow first time; planning files separated by role |
| What have I done? | config.py + tokenizer.py complete; constants.py + dataset.py incomplete |
