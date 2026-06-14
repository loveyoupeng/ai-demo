# Phase A: Shared Foundation — Execution Plan

## Goal
Create `shared/` module with config, constants, tokenizer, dataset loaders for all 4 backend implementations to import from.

## Execution Stages

### Stage 0: Directory Structure ✅
`shared/` and `tests/unit/` created (checkpoint commit)

### Stage 1: Parallel Tasks (4 agents, ZERO inter-dependencies)

| Agent | Files | Gate | Status |
|-------|-------|------|--------|
| **Agent 1** | `shared/config.py` + `tests/unit/test_config.py` | pyright + ruff + pytest (41 tests) | ✅ COMPLETE |
| **Agent 2** | `shared/constants.py` + `tests/unit/test_constants.py` | pyright + ruff + pytest (79 tests) | ❌ 52 FAIL — full impl needed |
| **Agent 3** | `shared/tokenizer.py` + `tests/unit/test_tokenizer.py` | pyright + ruff + pytest (21 tests) | ✅ COMPLETE |
| **Agent 4** | `shared/dataset.py` + `tests/unit/test_dataset.py` | pyright + ruff + pytest (14 tests) | ❌ TIMEOUT — dataset download |

### Stage 2: Integration (depends on ALL Stage 1 passing)

| Agent | Files | Gate | Status |
|-------|-------|------|--------|
| **Agent 5** | `shared/checkpoint.py` + `tests/unit/test_checkpoint.py` + `tests/unit/test_shared_pipeline.py` + `tests/conftest.py` | pytest all + ruff + pyright + import test | ⏳ BLOCKED by Agent 2 + 4 |

---

## Current Issues

### Agent 2: constants.py (52/79 tests fail)
**What exists (19 lines):** `Attention` (4 attrs), `MoE` (2 attrs), `param()` helper

**What's missing:**
- `LayerNorm` class: `GN_BIAS`, `GN_GAMMA`, `LN_BIAS`, `LN_GAMMA`
- `Transformer` class: `EMBEDDING`, `LM_HEAD_WEIGHT`, `LM_HEAD_BIAS`, `TRANSFORMER_LN_GAMMA`, `TRANSFORMER_LN_BIAS`
- New `Attention` attrs: `Q`, `K`, `V`, `O`, `Q_WEIGHT`, `K_WEIGHT`, `V_WEIGHT`, `O_WEIGHT`, `Q_BIAS`, `K_BIAS`, `V_BIAS`, `O_BIAS`
- New `MoE` attrs: `W1`, `W2`, `W3`, `GATE_WEIGHT`, `EXPERT_W1`, `EXPERT_W2`, `EXPERT_W3`
- 6 helper functions: `block_param()`, `attention_param()`, `layer_norm_param()`, `moe_param()`, `transformer_param()`, `get_all_params()`

**Tests:** `tests/unit/test_constants.py` — 27 pass, 52 fail (all NewClassesExist, ValuesAllStrings, BlockParam, AttentionParam, LayerNormParam, MoeParam, TransformerParam, GetAllParams, Completeness classes fail)

### Agent 4: dataset.py (tests timeout)
**What exists (208 lines):** Complete implementation — `load_tinystories()`, `TextDataset` class, `get_dataloader_sequences()`

**What's needed:** Tests handle the ~1 min download time on first run. Tests use `load_tinystories()` with small subsets (50-100 stories) but still trigger the same download each time. May need:
- `pytest` fixtures with `@pytest.fixture(scope="session")` to download once per session
- `--timeout=120` on all dataset tests (already set)
- Or skip tests if dataset already cached

---

## Execution Order (TDD discipline enforced)
1. Constants first — pure code, no network deps, fast feedback
2. Dataset second — network dep, may need fixture refactoring  
3. If any test fails: add smaller isolated test case → fix → re-run (DO NOT over-reason, USE TESTS)
4. After both pass: ruff + pyright → commit

## TDD Rules (MANDATORY — user enforced)
1. **Test file FIRST** — write complete test file before touching implementation
2. **Run all tests → confirm all fail** before implementing
3. **Run all tests → confirm all pass** after implementing
4. **If any fail**: add smaller focused test to isolate → fix → re-run (no over-reasoning)
5. **ruff + pyright clean** required after each module

## Final Gate
```bash
PYTHONPATH=shared uv run pytest tests/unit/ -v --timeout=120 && ruff check shared/ && pyright shared/ && python -c "import shared; print('OK')"
```
