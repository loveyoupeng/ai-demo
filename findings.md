# findings.md

## Architecture

- Core transformer components in NumPy (`src/model/`)
- PyTorch parity implementations in `src/model/pytorch/`
- MoE implementation in `src/model/moe.py`
- Transformer in `src/model/transformer.py` with full attention+FFN+LayerNorm
- Backends: NumPy (baseline) → PyTorch → Triton → CUDA (planned)

## Current Discoveries

### Test Status (2026-06-03)
- **85/87 tests passing** (97.7%)
- **2 failing**: MoE backward numerical issues (`test_expert_backward_numerical`, `test_moe_layer_params_numerical`)
- All parity tests for individual layers pass (29/29 originally, now 31/29 with additions)

### Pyright Configuration
- Source files (`src/`): 0 pyright errors ✅
- Test files (`tests/`): 44 errors (pyright doesn't read pytest `pythonpath` from pyproject.toml)
- Tests import from flat relative paths (e.g., `from model.layers import ...`), which pyright resolves as if `tests/` is the root
- **Solution**: Configure pyright to only check `src/` explicitly, or add a pyrightconfig.json

### Test File Structure
- **Parity tests** (`tests/parity/`): 7 files, test NumPy vs PyTorch parity
- **Model tests** (`tests/model/`): 9 files, component-specific tests
- **Integration tests** (root `tests/*.py`): 5 files
- **Unit tests** (`tests/tokenizer/`, `tests/evaluation/`, `tests/inference/`): 4 files
- Total: 25 test Python files (excluding `__init__.py`)

### Import Patterns
- All `src/` code uses flat imports: `from model.layers import ...`, `from optimizer import ...`
- No `src.` prefix used anywhere (pythonpath = ["src"] in pyproject.toml)
- Tests import from flat packages too: `from src.model.layers import ...` (after fix)

## Errors Encountered

| Error | Count | Category |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tests'` | 1 | Import path |
| Pyright: "Import could not be resolved" | 44 | Tool config |
| MoE backward numerical mismatch | 2 | Test failure |