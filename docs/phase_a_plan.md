# Phase A: Shared Foundation — Execution Plan

## Goal
Create `shared/` module with config, constants, tokenizer, dataset, checkpoint for all 4 backend implementations to import from.

## Execution Stages

### Stage 1: ✅ COMPLETE — 131 tests pass, ruff+pyright clean on shared/

| Agent | Files | Test Count | Status |
|-------|-------|------------|--------|
| **Agent 1** | `shared/config.py` | 41 | ✅ complete |
| **Agent 2** | `shared/constants.py` | 35 | ✅ complete — strict TDD: class-by-class, no magic strings |
| **Agent 3** | `shared/tokenizer.py` | 21 | ✅ complete |
| **Agent 4** | `shared/dataset.py` | 12 | ✅ complete — cache in resource/, fixed batching |
| **Agent 5** | `shared/checkpoint.py` | 11 | ✅ complete — save/load config + npz |

### Stage 2: ✅ COMPLETE — Integration tests for full pipeline

| Agent | Files | Test Count | Status |
|-------|-------|------------|--------|
| **Agent 6** | `tests/conftest.py` + `tests/unit/test_shared_pipeline.py` | 11 | ✅ complete — full pipeline: config→save→load→verify |

## Final Gate Results
```bash
PYTHONPATH=shared uv run pytest tests/unit/ -v --timeout=120   # 131 passed
PYTHONPATH=shared ruff check shared/                             # All checks passed
PYTHONPATH=shared pyright shared/                                # 0 errors
PYTHONPATH=shared python -c "import shared; print('OK')"         # OK
```

## Phase A: DONE
All shared Foundation modules are complete and tested. Ready for Phase B: NumPy Implementation.
