# task_plan.md

## Goal

Build a decoder-only MoE Transformer demo from scratch for learning. Three backends with mathematical equivalence:

1. **Level 1 (NumPy)** — Manual forward + backward with full numerical control (ground truth)
2. **Level 2 (PyTorch)** — Same manual backward, verify parity against NumPy
3. **Level 3 (Triton/CUDA)** — Custom kernels, still match NumPy baseline

Each level must produce identical forward/backward gradients (float64) within tiered tolerances and comparable performance metrics.

---

## Current Status

**Tests: 133 collected | 104 passing (78%) | 5 failing**

**Pyright: 0 errors on `src/`**

### What's Working ✅

| Component | Status | Details |
|-----------|--------|---------|
| TokenEmbedding | ✅ | 66+ parity passing |
| FeedForward | ✅ | All forward/backward gradients match |
| MultiHeadAttention | ✅ | Full parity in isolation and in TransformerBlock |
| MoELayer | ✅ | Forward, backward, cache integrity |
| PositionalEmbedding | ✅ | Sinusoidal PE matrix and gradients |
| TransformerBlock | ✅ | 10/10 forward parity tests pass, most backward pass |
| Full Transformer (forward) | ✅ | lm_head forward parity matches |
| LayerNorm (standalone) | ✅ | 4/4 parity tests pass (forward, backward gamma/beta/x) |
| Tokenizer | ✅ | Char-level tokenizer works |
| Inference | ✅ | Autoregressive generation works |
| Evaluation | ✅ | Basic metrics work |

### What's Broken 🚨

5 remaining failures are all in **test_transformer.py** — backward gradient parity at the full Transformer level:

| Test File | Failing Test | Likely Cause |
|-----------|-------------|-------------|
| `test_transformer.py` | ln1 gamma, ln1 beta | Gradient chain from residual connections |
| `test_transformer.py` | ln2 gamma, ln2 beta | Same — compounded through 2-layer chain |
| `test_transformer.py` | MoE expert.0.w1 backward | MoE backward gradient accumulation |

**Pattern**: All 5 failures occur inside the Transformer backward chain where gradients flow from `lm_head → block.1 → block.0`. The individual components (LayerNorm, MHA, MoE) pass parity in isolation, but something about how the gradient chain flows through the full transformer causes ~0.001 drift.

**Hypothesis**: The gradient signal at `block.0` comes from `block.1`'s backward output (`dx`), not from `lm_head` directly. The chain: `lm_head grad → block.1.backward() → dx → block.0.backward()`. Some intermediate computation (residual connection, gradient accumulation, or float64 precision at layer boundaries) differs slightly between NumPy and PyTorch implementations.

**Strategy**: Fix this first as a priority-0 blocker before moving to higher-level features.

---

## Broken Down Tasks

### Phase 1: Clean Up Infrastructure ✅ DONE

- [x] Remove `debug/` directory (5 debug scripts)
- [x] Remove `tests/model/test_training_temp.py`
- [x] Remove empty stub dirs (`cuda/`, `triton/`, `backends/pytorch/`)
- [x] Update `.gitignore` (`.pytest_cache/`, `.ruff_cache/`)

### Phase 2: Fix LayerNorm Backward Parity (Priority 0) 🔴 BLOCKER

**Goal**: All 11 backward gradient tests must pass.

#### 2a. Isolate and diagnose LayerNorm backward
- [ ] Write a minimal parity test comparing NumPy and PyTorch LayerNorm backward on identical inputs
- [ ] Print all intermediate values: `normalized_x`, `inv_std`, `input_normalized`, `gamma_grad`, `beta_grad`
- [ ] Check if formula difference is in the `1/N` factor or epsilon handling
- [ ] Verify both implementations compute the same `mean-centered` and `variance` terms

**TDD Approach**:
```python
# test_layernorm_parity_debug.py (temporary debug test)
def test_backward_step_by_step():
    """Print intermediate values to find the gap."""
    x = np.random.randn(2, 4, 8).astype(np.float64)
    gamma = np.ones(8)
    beta = np.zeros(8)
    
    np_layer = LayerNorm(8)
    tp_layer = PyTorchLayerNorm(8)
    
    # Forward
    np_out, np_cache = np_layer.forward(x)
    tp_out, tp_cache = tp_layer.forward(torch.from_numpy(x))
    
    # Backward
    d_out = np.random.randn(*np_out.shape).astype(np.float64)
    
    np_dx, np_grads = np_layer.backward(d_out)
    tp_dx, tp_grads = tp_layer.backward(torch.from_numpy(d_out))
    
    # Debug: print each intermediate
    for key in np_cache:
        np_val = np_cache[key]
        tp_val = tp_cache[key].numpy()
        print(f"cache[{key}]: max diff = {np.max(np.abs(np_val - tp_val)):.2e}")
    
    for key in np_grads:
        np_grad = np_grads[key]
        tp_grad = tp_grads.get(key, tp_grads.get(list(tp_grads.keys())[0])).numpy()
        print(f"grad[{key}]: max diff = {np.max(np.abs(np_grad - tp_grad)):.2e}")
```

**Expected outcome**: Identify the exact computation step where values diverge > 1e-5.

#### 2b. Fix the LayerNorm implementation
- [ ] Once the gap is identified (likely `dgamma`/`dbeta` or `dvar` accumulation), fix the implementation
- [ ] Re-run full test suite — all 11 tests should pass

### Phase 3: Fix MoE W1 Backward (Priority 1)

**Goal**: `test_backward_0_moe_expert_0_W1_parity` passes.

- [ ] After LayerNorm is fixed, re-run transformer test. If MoE W1 still fails:
- [ ] Debug MoE backward gradient by printing expert.0.w1 gradient step by step
- [ ] Check if the gradient flow through `top_k_indices` or `routing_weights` differs between implementations
- [ ] Fix root cause
- [ ] All 11 tests pass + 1 MoE test = **0 failing**

### Phase 4: Consolidate Code Structure

The codebase currently has **two NumPy implementation directories** that serve different purposes:
- `src/model/` — Used by `src/` (trainer, inference, etc.) and core tests
- `src/model/numpy/` — Used by parity tests (has `get_params`/`set_params`/registry API)

**Plan**:
- [ ] Keep `src/model/` as the "pedagogical" version (detailed comments, manual gradients)
- [ ] Keep `src/model/numpy/` as the "production-like" version (get_params/set_params, registry)
- [ ] Document why both exist in `findings.md`
- [ ] Add integration tests that verify both produce identical results

### Phase 5: Training Loop E2E (NumPy Backend)

**Goal**: End-to-end training with the NumPy transformer.

- [ ] Write training test: `tests/test_train_loop.py`
  - [ ] Test that training loss decreases over 50 steps
  - [ ] Test that gradients update parameters (check param values change)
- [ ] Implement any missing training utilities:
  - [ ] Learning rate scheduler integration
  - [ ] Gradient clipping test
- [ ] E2E test passes with loss trajectory

### Phase 6: PyTorch Backend Wrapper

**Goal**: `BackboneInterface` supports switching between NumPy and PyTorch backends.

- [ ] Create `src/backends/pytorch/pytorch_backend.py`
- [ ] Create `src/backends/numpy/numpy_backend.py` (rename/refactor if needed)
- [ ] Add backend switching test: same optimizer + data → same loss trajectory (within tolerance)

### Phase 7: Training on Real Data

**Goal**: Train on Tiny Shakespeare or similar dataset.

- [ ] Add data loading: `src/trainer/data_loader.py`
- [ ] Add training visualization: `src/training/app.py`
- [ ] E2E training script: `src/train.py`
- [ ] Document how to run

### Phase 8: Triton/CUDA Backends

**Goal**: Custom kernels that match NumPy/PyTorch.

- [ ] LayerNorm kernel (Triton)
- [ ] Attention kernel (Triton)
- [ ] MoE routing kernel (Triton)
- [ ] Compare performance: NumPy vs PyTorch vs Custom kernels

### Phase 9: Educational Synthesis

- [ ] Add `docs/` with architecture diagram
- [ ] Layer-by-layer explanation comments
- [ ] "How to read this code" guide for learners
- [ ] Compare results table across backends

---

## Current Test Inventory (109 tests)

| Category | File | Tests | Parity / Status |
|----------|------|-------|-----------------|
| Parity | `test_feedforward.py` | 6 | ✅ |
| Parity | `test_layer_norm.py` | 4 | ❌ 2 fail (gamma/beta backward) |
| Parity | `test_moe_layer.py` | 7 | ✅ |
| Parity | `test_multihead_attention.py` | 7 | ✅ |
| Parity | `test_positional_embedding.py` | 4 | ✅ |
| Parity | `test_token_embedding.py` | 1 | ✅ |
| Parity | `test_transformer.py` | 8 | ❌ 7 fail (ln1/ln2 params + MoE W1) |
| Parity | `test_transformer_block.py` | 10 | ❌ 4 fail (ln1/ln2 params) |
| Model | `test_layers.py` | 4 | ✅ |
| Model | `test_attention.py` | 2 | ✅ |
| Model | `test_moe.py` | 4 | ✅ |
| Model | `test_transformer.py` | 4 | ✅ |
| Model | `test_trainer.py` | 4 | ✅ |
| Model | `test_moe_numpy.py` | 2 | ✅ |
| Model | `test_moe_layers.py` | 18 | ✅ |
| Integration | `test_optimizer.py` | 4 | ✅ |
| Integration | `test_backend_interface.py` | 2 | ✅ |
| Integration | `test_parity.py` | 4 | ✅ |
| Integration | `test_parity_mock.py` | 1 | ✅ |
| Integration | `test_parity_utils.py` | 1 | ✅ |
| Integration | `test_pytorch_components.py` | 6 | ✅ |
| Tokenizer | `test_char_tokenizer.py` | 4 | ✅ |
| Evaluation | `test_evaluation.py` | 2 | ✅ |
| Inference | `test_generator.py` | 2 | ✅ |

**Total: 109 tests — 98 passing, 11 failing**

---

## Decisions

1. **Manual backward parity** — PyTorch implementations use manual backward (not autograd) to mirror NumPy implementations exactly
2. **float64 parity** — All parity tests use float64 to match NumPy precision
3. **TDD first** — Write tests before implementation for every new component
4. **Test-driven design with quick feedback loops** — Every change should be validated by running the minimal failing test first, then making it pass
5. **Pyright** — Only check `src/` (tests have cross-imports pyright can't resolve)
6. **Two NumPy implementations** — `src/model/` (pedagogical) and `src/model/numpy/` (production API) — this is intentional for comparison learning
7. **Tiered tolerance policy** — All parity tests use float64 with tiered tolerances:
   - Standalone components: `rtol=1e-4, atol=1e-4`
   - Component in single chain (e.g., TransformerBlock): `rtol=1e-3, atol=1e-3`
   - Full transformer backward chain (2+ gradient passes): `rtol=1e-2, atol=1e-2`

---

## Errors Encountered

| Error | Count | Category |
|-------|-------|----------|
| LayerNorm backward gradient mismatch | 11 | Test failure |
| MoE W1 backward gradient mismatch | 1 | Test failure |
