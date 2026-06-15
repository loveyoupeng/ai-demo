# Phase A: Shared Foundation — Execution Plan

## Goal
Create `shared/` module with config, constants, tokenizer, dataset, checkpoint for all 4 backend implementations to import from.

## Execution Stages

### Stage 0: Directory Structure ✅
`shared/` and `tests/unit/` created (checkpoint commit)

### Stage 1: ✅ COMPLETE — All 120 tests pass, ruff+pyright clean on shared/

| Agent | Files | Test Count | Status |
|-------|-------|------------|--------|
| **Agent 1** | `shared/config.py` | 41 | ✅ complete |
| **Agent 2** | `shared/constants.py` | 35 | ✅ complete — strict TDD: class-by-class, no magic strings |
| **Agent 3** | `shared/tokenizer.py` | 21 | ✅ complete |
| **Agent 4** | `shared/dataset.py` | 12 | ✅ complete — cache in resource/, fixed batching |
| **Agent 5** | `shared/checkpoint.py` | 11 | ✅ complete — save/load config + npz |

### Stage 2: Next — Integration (Agent 6)

| Agent | Files | Gate | Status |
|-------|-------|------|--------|
| **Agent 6** | `tests/conftest.py` + `tests/unit/test_shared_pipeline.py` | pytest all + ruff + pyright + import test | ⏳ NEXT |

## Final Gate (after Stage 2)
```bash
PYTHONPATH=shared uv run pytest tests/unit/ -v --timeout=120 && ruff check shared/ tests/ && pyright shared/ && python -c "import shared; print('OK')"
```
