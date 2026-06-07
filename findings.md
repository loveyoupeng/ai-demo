# findings.md

## Architecture

- Core transformer components in NumPy (`src/model/`) — pedagogical with detailed comments
- Parallel NumPy implementations in `src/model/numpy/` — production API (`get_params`/`set_params`, registry)
- PyTorch implementations in `src/model/pytorch/` — manual backward (no autograd) for parity
- Backend abstraction: `src/backends/numpy/numpy_backend.py` wraps NumPy Transformer
- Training: `src/trainer.py`, `src/training/app.py` (not yet tested E2E)
- Two parallel NumPy implementations is intentional — one for learning, one for production patterns

## Codebase Structure

```
src/model/         # Pedagogical implementations (comments, manual grad)
  layers.py        # TokenEmbedding, PositionalEmbedding, FeedForward, LayerNorm
  attention.py     # MultiHeadAttention
  moe.py           # Router, Expert, MoELayer
  transformer.py   # TransformerBlock, Transformer
src/model/numpy/   # Production API implementations
  layers.py        # NumPy* prefixed versions with get_params/set_params
  moe.py           # Registry-integrated MoE (459 lines)
  transformer.py   # NumPyTransformerBlock (175 lines)
src/model/pytorch/ # PyTorch manual backward versions
  layer.py, attention.py, moe.py, transformer.py
src/backends/      # Backend wrapper layer
src/training/      # Training orchestration
src/utils/         # Checkpoint, profiler
tests/
  parity/          # NumPy ↔ PyTorch parity tests
  model/           # Component tests
```

## Current Discoveries

### Test Status (2026-06-07)
- **98/109 tests passing (90%)**
- **11 failing** — all backward gradient parity for LayerNorm parameters
- LayerNorm `gamma`/`beta` backward gradients diverge between NumPy and PyTorch at ~0.001 max diff
- Full Transformer backward has additional MoE `W1` gradient mismatch

### Key Structural Discovery

The codebase has **two complete NumPy implementation sets**:

| Feature | `src/model/` | `src/model/numpy/` |
|---------|-------------|-------------------|
| Purpose | Pedagogical learning | Production API |
| LayerNorm | 313 lines | 146 lines |
| Layers files | classes without `NumPy` prefix | classes with `NumPy*` prefix |
| API | Direct attribute access | `get_params()`/`set_params()`/registry |
| Test source | `tests/model/*.py`, training pipeline | `tests/parity/*.py` |

**Why both exist**: One is for showing the raw math, the other for demonstrating how a real production framework manages parameters. This is intentional for the educational goal.

### LayerNorm Backward Analysis

All 11 failures are in backward gradient computation for LayerNorm:
- `test_backward_gamma_parity` / `test_backward_beta_parity` in `test_layernorm.py` (2 tests)
- 4x backward parity in `test_transformer_block.py` (ln1 ln2 gamma/beta)
- 4x backward parity in `test_transformer.py` (ln1 ln2 gamma/beta + MoE W1)

The error magnitude (~0.001) suggests:
- Not a formula bug (formula bugs produce larger errors)
- Likely accumulation of float32/float64 boundary issues
- Possible difference in how epsilon interacts with variance computation

### Pyright Configuration
- Source files (`src/`): 0 pyright errors ✅
- Test files (`tests/`): ~20 errors (cross-imports pyright can't resolve)
- Solution: pyright only checks `src/`, configured in pyproject.toml

## Errors Encountered

| Error | Count | Category |
|-------|-------|----------|
| LayerNorm backward gradient mismatch | 11 | Test failure (all backward parity for ln params) |
| MoE W1 backward gradient mismatch | 1 | Test failure (chain gradient in transformer) |
| ModuleNotFoundError: No module named 'model' | ~50 | During test refactoring |
