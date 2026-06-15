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
| **Agent 2** | `shared/constants.py` + `tests/unit/test_constants.py` | pyright + ruff + pytest (35 tests) | ✅ COMPLETE — strict TDD: class-by-class, no magic strings |
| **Agent 3** | `shared/tokenizer.py` + `tests/unit/test_tokenizer.py` | pyright + ruff + pytest (21 tests) | ✅ COMPLETE |
| **Agent 4** | `shared/dataset.py` + `tests/unit/test_dataset.py` | pyright + ruff + pytest (12 tests) | ✅ COMPLETE — cache in resource/, fixed get_sequences batching + target length |

### Stage 1: ✅ COMPLETE — All 109 tests pass.

**Stage 2: Integration (Agent 5) — next step:**

| Agent | Files | Gate | Status |
|-------|-------|------|--------|
| **Agent 5** | `shared/checkpoint.py` + `tests/unit/test_checkpoint.py` + `tests/unit/test_shared_pipeline.py` + `tests/conftest.py` | pytest all + ruff + pyright + import test | 🔄 NEXT — no more blockers |

---

## Completed Work

### constants.py (Agent 2) — 35 tests, strict TDD
- Wrote test file FIRST (all 16 class-attr tests failed)
- Implemented `Attention`, `LayerNorm`, `Transformer`, `MoE` classes — no magic strings
- Built 5 helper functions: `block_param()`, `attention_param()`, `layer_norm_param()`, `moe_param()`, `transformer_param()` — each test→implement→verify cycle
- Added `get_all_params()` using ONLY constants, never raw strings
- 35 tests, all pass. Zero ruff/pyright errors in `shared/constants.py`

### dataset.py (Agent 4) — 12 tests, caching + batching fix
- Added `resource/` directory to `.gitignore` for dataset cache
- `load_tinystories()` downloads once to `resource/`, caches with pickle
- `get_sequences()` returns batches (not individual sequences) — fixed API mismatch
- Fixed input/target length mismatch (target now includes next token, same length as input)
- Fixed indentation bug in test that masked proper batch validation
- 12 tests, all pass. Zero ruff/pyright errors in `shared/dataset.py`

---


### Agent 4: dataset.py (tests timeout)
**What exists (208 lines):** Complete implementation — `load_tinystories()`, `TextDataset` class, `get_dataloader_sequences()`

**What's needed:** Tests handle the ~1 min download time on first run. Tests use `load_tinystories()` with small subsets (50-100 stories) but still trigger the same download each time. May need:
- `pytest` fixtures with `@pytest.fixture(scope="session")` to download once per session
- `--timeout=120` on all dataset tests (already set)
- Or skip tests if dataset already cached

---

## Final Gate (after Stage 2)
```bash
PYTHONPATH=shared uv run pytest tests/unit/ -v --timeout=120 && ruff check shared/ tests/ && pyright shared/ && python -c "import shared; print('OK')"
```
