# task_plan.md

## Goal

Build a decoder-only transformer demo from scratch in Python using NumPy, with TDD. Implement all components (Attention, MoE, etc.) with manual backward passes, then compare against PyTorch for parity.

---

## Current Status

**Tests: 92 collected | 92 passing (100%) | 0 failing**

**Pyright: 0 errors on src/**

### Fixed ✅

- **MoE tests rewritten** — Replaced broken numerical differentiation with parity-based tests vs PyTorch. All 18 tests pass. The old tests had a fundamental bug — they computed a single scalar `(loss_p - loss_m) / (2*eps)` for an entire array parameter, then compared that scalar to the full gradient array. The actual expert/MoE backward pass is correct as verified by PyTorch parity.
- **All parity tests passing** — 31 parity tests covering TokenEmbedding, LayerNorm, FeedForward, PositionalEmbedding, MultiHeadAttention, MoELayer

---

## Phases

### Phase 0: Infrastructure ✅ COMPLETE

- [x] Project structure (`src/`, `tests/`)
- [x] Core utilities (Tokenizer, Loss, Optimizer, Trainer)
- [x] Backend infrastructure (BaseBackend, ParameterRegistry)
- [x] Import/config: `pythonpath = ["src", "tests"]` in pyproject.toml
- [x] Pyright config: `include = ["src"]` (exclude tests due to import resolution)
- [x] All `__init__.py` files for Python packages

### Phase 1: NumPy Core + PyTorch Parity ✅ COMPLETE

**All 31 Parity Tests Passing:**
- TokenEmbedding: 1/1 ✅
- LayerNorm: 4/4 ✅ (forward + backward)
- FeedForward: 6/6 ✅ (forward + backward)
- PositionalEmbedding: 4/4 ✅ (matrix parity + forward/backward)
- MultiHeadAttention: 7/7 ✅ (forward + all backward params)
- **MoELayer**: 7/7 ✅
- **MoELayer (unit)**: 11/11 ✅ (rewritten, all parity-based)

### Phase 2: TransformerBlock & Full Transformer 🔲 IN PROGRESS

- [ ] TransformerBlock (attention + FFN + LayerNorm composition)
- [ ] Full Transformer (composing TransformerBlocks)
- [ ] Transformer parity NumPy vs PyTorch
- [ ] `src/model/pytorch/` — Create `PyTorchTransformerBlock` and `PyTorchTransformer`
- [ ] `tests/parity/` — Create parity tests for both

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
| **Model** | `tests/model/numpy/test_moe_layers.py` | 18 | ✅ (rewritten, parity-based) |
| **Integration** | `tests/test_optimizer.py` | 4 | ✅ |
| **Integration** | `tests/test_backend_interface.py` | 2 | ✅ |
| **Integration** | `tests/test_parity.py` | 4 | ✅ |
| **Integration** | `tests/test_parity_mock.py` | 1 | ✅ |
| **Integration** | `tests/test_parity_utils.py` | 1 | ✅ |
| **Integration** | `tests/test_pytorch_components.py` | 6 | ✅ |
| **Tokenizer** | `tests/tokenizer/test_char_tokenizer.py` | 4 | ✅ |
| **Evaluation** | `tests/evaluation/test_evaluation.py` | 2 | ✅ |
| **Inference** | `tests/inference/test_generator.py` | 2 | ✅ |

---

## Decisions

1. **Flat import from src/** — `pythonpath = ["src", "tests"]` in pyproject.toml
2. **Tests at root** — All tests in `tests/`, parity tests in `tests/parity/`
3. **Numbers first** — NumPy for core math, PyTorch for parity/verification
4. **TDD** — Write tests before implementation
5. **Pyright** — Only check `src/` (tests have cross-imports pyright can't resolve)
