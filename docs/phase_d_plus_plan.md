# Phase D+: GPU PyTorch Setup for Jetson AGX Orin

## Objective

Replace CPU-only PyTorch with NVIDIA-optimized GPU PyTorch for JetPack 6.2.2 on Jetson AGX Orin 64GB, enabling GPU-accelerated training, inference, and Triton development.

## Current System State

| Component | Current | Target |
|-----------|---------|--------|
| Platform | Jetson AGX Orin 64GB | Same |
| JetPack | R36.5.0 (6.2.2) | Same |
| CUDA | 12.6 (nvcc installed) | 12.5 (JetPack 6.2) |
| OS | Ubuntu 22.04 | Same |
| System Python | 3.10.12 | 3.10 |
| Project Python | 3.14.5 (via uv) | 3.10 |
| PyTorch | 2.12.0+cpu | 2.8.0a0+nv25.06 (JetPack 6.2 wheel) |
| PyTorch CUDA | N/A | CUDA 12.5 |
| GPU Detection | `torch.cuda.is_available()` → False | True |

## Why Downgrade to Python 3.10

1. **NVIDIA PyTorch wheels for JetPack 6.2 are built for Python 3.10**: The official wheel naming is `cp310-cp310-linux_aarch64.whl` — it only supports Python 3.10.
2. **Python 3.14 is too new**: Many Jetson-optimized packages (cusparselt, TensorRT, cuDNN bindings) are not available for Python 3.14.
3. **Project compatibility**: `pyproject.toml` has `requires-python >=3.10` and ruff `target-version = "py310"`. Python 3.10 is the intended minimum.

## Execution Plan

### Step 1: Remove CPU-Only PyTorch Index

**Files to modify:** `pyproject.toml`, `uv.lock`

**Rationale:** The current `pyproject.toml` forces CPU-only PyTorch via a custom uv index. This must be removed.

```diff
- [[tool.uv.index]]
- name = "pytorch-cpu"
- url = "https://download.pytorch.org/whl/cpu"
-
- [tool.uv.sources]
- torch = { index = "pytorch-cpu" }
```

### Step 2: Configure pyproject.toml for JetPack 6.2

**Rationale:** Direct uv to use the NVIDIA-hosted PyTorch wheel for JetPack 6.2 on aarch64.

```toml
[tool.uv.sources]
torch = { url = "https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/torch-2.8.0a0+5228986c39.nv25.06.13854593-cp310-cp310-linux_aarch64.whl" }

[project.optional-dependencies]
gpu = ["torch>=2.8"]
# Remove "cuda" extra — not needed when using official wheel
```

### Step 3: Create Python 3.10 Virtual Environment

```bash
# Remove old venv
rm -rf .venv

# Create new venv with Python 3.10
uv python install 3.10
uv venv --python 3.10

# Verify
.venv/bin/python --version  # Should print Python 3.10.x
```

### Step 4: Install Dependencies

```bash
# Install from pyproject.toml (should use NVIDIA PyTorch wheel)
uv sync

# Verify PyTorch installation
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Expected output:
# 2.8.0a0+5228986c39
# True
```

### Step 5: Verify CUDA Environment

```bash
# Check CUDA device listing
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cudnn version: {torch.backends.cudnn.version()}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'Device {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB')
    print(f'  Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}')
"
```

**Expected:** 8 GPU cores, 12GB+ memory each on AGX Orin (shared 64GB system memory).

### Step 6: Verify Existing Code Works with GPU PyTorch

Run the test suite to check for compatibility issues:

```bash
uv run pytest tests/ -v --timeout=300
```

**Potential issues to watch for:**
- `torch.backends.cudnn.enabled` may return `False` — handle gracefully
- Any hardcoded CPU device selection — should be parameterized
- NumPy float64 tests may behave differently with GPU tensors (test with `device='cpu'`)

### Step 7: Update Test Fixtures for Cross-Backend Testing

**Files to modify:** `tests/cross_backend/test_parity.py`, `tests/conftest.py`

Ensure all backends (NumPy, PyTorch, Triton) can compare results:
- GPU PyTorch results run on CPU (`.cpu()`) before comparison
- NumPy backends unaffected (CPU-only)
- PyTorch `device='cuda'` tests should check `torch.cuda.is_available()`

### Step 8: Verify Triton Compatibility

Trition requires CUDA. Verify it's available after PyTorch install:

```bash
uv pip install triton  # or use uv optional dependency
uv run python -c "import triton; print('Triton version:', triton.__version__)"
```

**Note:** Triton on Jetson may require specific CUDA toolkit paths. If issues arise:
```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda
```

### Step 9: Update AGENTS.md (If Needed)

If the CLI command for PyTorch backend needs updating to use GPU:

```bash
# Current CLI command already uses PyTorch — verify it detects GPU
uv run python -m impl._torch.cli --prompt "the" --max_new_tokens 5
```

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | **Modify** | Remove CPU index, add NVIDIA wheel URL, update deps |
| `.venv` | **Recreate** | Python 3.10 venv |
| `uv.lock` | **Regenerate** | Auto-generated by `uv lock` |
| `tests/conftest.py` | **Review** | Ensure GPU-aware fixtures |
| `tests/cross_backend/test_parity.py` | **Review** | Ensure GPU parity tests |
| `impl/_torch/` | **Review** | Any hardcoded CPU devices |

## Rollback Plan

If GPU PyTorch installation fails:

```bash
# Restore CPU-only setup
1. Update pyproject.toml to restore `[[tool.uv.index]]` for pytorch-cpu
2. `rm -rf .venv && uv sync`
3. `uv run python -c "import torch; assert not torch.cuda.is_available()"`
```

## Progress Log

### Commit 1: GPU PyTorch Setup (`35e0c02`)
- PyTorch 2.11.0 from Jetson-ai-lab PyPI index (`pypi.jetson-ai-lab.io/jp6/cu126`)
- Python 3.10.12 on JetPack R36.5.0 (6.2.2)
- CUDA 12.6, cuDNN 9.3, cuBLAS 12.6
- GPU: Orin, 64GB shared, compute capability 8.7

### Commit 2: GPU Cross-Backend Parity (`3e673d2`)
- 11 tests in `tests/cross_backend/test_gpu_parity.py`
- GPU forward parity against NumPy: `rtol=1e-2` (tier 3, float32 CUDA)
- GPU backward parity: gradient chaining, magnitude, training loop
- GPU/CPU weight exchange via `state_dict`: bit-for-bit forward match

## Success Criteria

- [x] `torch.cuda.is_available()` → `True`
- [ ] `torch.cuda.device_count()` → 8 (AGX Orin) — detected as 1 GPU with 64GB shared (system config)
- [ ] Run `uv run python -m impl._torch.cli --prompt "hello"` produces output
- [x] `uv run pytest tests/ -v` all pass (440 tests, 0 failures)
- [x] `uv run ruff check .` clean
- [x] `uv run pyright` clean
- [ ] Triton importable after GPU PyTorch install
- [x] Cross-backend parity tests pass for PyTorch-GPU ↔ NumPy (11 tests)

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Python 3.10 breaks existing code | Medium | Run `ruff` and `pyright` before/after; keep CPU wheel as backup |
| Triton not compatible with JetPack 6.2 | High | Use system CUDA (`/usr/local/cuda`) for Triton build; fallback to PyTorch-only |
| cuDNN not configured | Medium | `torch.backends.cudnn.enabled` may be False; add CPU fallback in code |
| 64GB shared memory limitation | Low | Small model sizes; monitor OOM in tests |

## Timeline Estimate

| Step | Estimated Time |
|------|---------------|
| 1-2: Modify config files | 5 min |
| 3: Create Python 3.10 venv | 5 min |
| 4: Install dependencies (torch ~2GB) | 10-15 min |
| 5-6: Verify CUDA and run tests | 10 min |
| 7-8: Update tests, verify Triton | 15 min |
| 9: Final validation | 5 min |
| **Total** | **~50 min** |