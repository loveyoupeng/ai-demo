# task_plan.md

## Goal

Build a decoder-only transformer demo from scratch in Python using NumPy, with TDD. Implement all components (Attention, MoE, etc.) with manual backward passes, then compare against PyTorch for parity.

---

## Current Status

**Tests: 87 collected | 85 passing (97.7%) | 2 failing**

**Pyright: 0 errors on src/**

### Current Blockers

1. **MoE Numerical Failures** (2 tests):
   - `tests/model/numpy/test_moe_layers.py::test_expert_backward_numerical`
   - `tests/model/numpy/test_moe_layers.py::test_moe_layer_params_numerical`
   - These test backward pass gradient accuracy for MoE component
   - **Priority**: Fix before moving to Phase 2 (TransformerBlock)

---

## Phases

### Phase 0: Infrastructure ✅ COMPLETE

- [x] Project structure (`src/`, `tests/`)
- [x] Core utilities (Tokenizer, Loss, Optimizer, Trainer)
- [x] Backend infrastructure (BaseBackend, ParameterRegistry)
- [x] Import/config: `pythonpath = ["src", "tests"]` in pyproject.toml
- [x] Pyright config: `include = ["src"]` (exclude tests due to import resolution)
- [x] All `__init__.py` files for Python packages

### Phase 1: NumPy Core + PyTorch Parity 🔄 IN PROGRESS

**Passed Parity Tests (all passing):**
- TokenEmbedding: 1/1 ✅ `test_token_embedding_parity`
- LayerNorm: 4/4 ✅ (forward + backward)
- FeedForward: 6/6 ✅ (forward + backward)
- PositionalEmbedding: 4/4 ✅ (matrix parity + forward/backward)
- MultiHeadAttention: 7/7 ✅ (forward + all backward params)
- **MoELayer** (parity): 7/7 ✅ through `tests/parity/test_moe_layer.py`

**Failing MoE Tests:**
- `test_expert_backward_numerical` — Numerical mismatch in expert gradients
- `test_moe_layer_params_numerical` — Parameter gradient mismatch

**Current Test Breakdown (87 total):**
- **Parity tests**: 31 passing (FeedForward 6 + LayerNorm 4 + MoE 7 + MHA 7 + PE 4 + Token 1 + LMHead 2)
- **Model tests**: 31 tests (9 in model/numpy, 8 in model/test, 14 in test_parity/mock/utils, 3 in test_pytorch_components, 4 in test_optimizer)
- **Tokenizer**: 4 tests
- **Evaluation**: 2 tests
- **Inference**: 2 tests

### Phase 2: TransformerBlock & Full Transformer 🔲 TODO

- [ ] TransformerBlock (attention + FFN + LayerNorm composition)
- [ ] Full Transformer (composing TransformerBlocks)
- [ ] Transformer parity NumPy vs PyTorch

### Phase 3: Training Orchestration (NumPy Backend) 🔲 TODO

- [ ] Trainer integration (already exists — `src/trainer.py`)
- [ ] Training app (`src/training/app.py`)
- [ ] E2E training test with Shakespeare/tiny_data

### Phase 4: Evaluation 🔲 TODO

- [ ] Perplexity metric
- [ ] Accuracy metrics

### Phase 5: Backend Wrapper (Level 2: PyTorch) 🔲 TODO

- [ ] `src/backends/pytorch/pytorch_backend.py`
- [ ] Wrapper around PyTorch Transformer for parity comparison

### Phase 6: Backend Wrappers (Level 3: Triton/CUDA) 🔲 TODO

- [ ] `src/backends/triton/triton_backend.py`
- [ ] `src/backends/cuda/cuda_backend.py`

### Phase 7: Benchmarking & Profiling 🔲 TODO

- [ ] Latency/throughput/memory comparison across backends

### Phase 8: Educational Synthesis 🔲 TODO

- [ ] Concept-to-Tool mapping guide
- [ ] Architecture diagram

---

## Test Inventory

| Category | File | Tests | Parity |
|----------|------|-------|--------|
| **Parity** | `tests/parity/test_feedforward.py` | 6 | ✅ |
| **Parity** | `tests/parity/test_layernorm.py` | 4 | ✅ |
| **Parity** | `tests/parity/test_moe_layer.py` | 7 | ✅ |
| **Parity** | `tests/parity/test_multihead_attention.py` | 7 | ✅ |
| **Parity** | `tests/parity/test_positional_embedding.py` | 4 | ✅ |
| **Parity** | `tests/parity/test_token_embedding.py` | 1 | ✅ |
| **Model** | `tests/model/numpy/test_moe_layers.py` | 9 | ❌ (2 failing) |
| **Model** | `tests/model/test_attention.py` | 2 | |
| **Model** | `tests/model/test_layers.py` | 4 | |
| **Model** | `tests/model/test_transformer.py` | 3 | |
| **Model** | `tests/model/test_trainer.py` | 4 | |
| **Model** | `tests/model/test_moe_numpy.py` | 4 | |
| **Model** | `tests/model/test_trainer.py` | 4 | |
| **Integration** | `tests/test_optimizer.py` | 4 | |
| **Integration** | `tests/test_backend_interface.py` | 2 | |
| **Integration** | `tests/test_parity.py` | 4 | |
| **Integration** | `tests/test_parity_mock.py` | 1 | |
| **Integration** | `tests/test_parity_utils.py` | 1 | |
| **Integration** | `tests/test_pytorch_components.py` | 6 | |
| **Tokenizer** | `tests/tokenizer/test_char_tokenizer.py` | 4 | |
| **Evaluation** | `tests/evaluation/test_evaluation.py` | 2 | |
| **Inference** | `tests/inference/test_generator.py` | 2 | |

---

## Decisions

1. **Flat import from src/** — `pythonpath = ["src", "tests"]` in pyproject.toml
2. **Tests at root** — All tests in `tests/`, parity tests in `tests/parity/`
3. **Numbers first** — NumPy for core math, PyTorch for parity/verification
4. **TDD** — Write tests before implementation
5. **Pyright** — Only check `src/` (tests have cross-imports pyright can't resolve)
