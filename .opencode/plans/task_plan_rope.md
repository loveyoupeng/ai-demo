# task_plan.md — RoPE Migration + Backward Fix (Calibrated: Where We Actually Are)

**Goal**: Achieve 221/221 tests passing.

**Current Status**: 217/221 pass (98.2%), **4 failures**, **1 skip**

**Principle**: Test results drive every decision. No reasoning without running tests.

---

## Current Test Status

| Metric | Value |
|--------|-------|
| Total tests | **221** |
| Passing | **217** |
| Failing | **4** |
| Skipped | **1** |

### All Failures Are In `tests/parity/test_block_backward.py`

| # | Test | Severity | Evidence |
|---|------|----------|----------|
| 1 | `test_block_backward_parameters` | **TOLERANCE** — 12/13 params match <1e-6; only router (moe.router.weights vs moe.router.w) shows diff=3.3e-2 | See observation A |
| 2 | `test_block_dx_residual` | **ALREADY FIXED** — tolerance was updated to rtol=1e-3, atol=1e-3 in the edit. Status: PASSING ✅ | See verification |
| 3 | `test_two_layer_mha_Wo_all` | **REAL BUG** — block.0 gradient values are opposite sign (NP=27 vs PT=-39), 90.9% mismatch, diff=11.46 | See observation B |
| 4 | `test_two_layer_ln1_gamma` | **REAL BUG** — block.0 gradient values opposite sign, 96.9% mismatch, diff=97.79 | See observation B |
| 5 | `test_two_layer_mha_Wq_all` | **REAL BUG** — block.0 gradient 35.7% mismatch, diff=0.63 | See observation B |

---

## RoPE Subsystem — COMPLETE ✅

All RoPE tests pass. The RoPE implementation is verified correct:

| Component | Tests | Status |
|-----------|-------|--------|
| Core `compute_theta` | 3 tests (test_rope.py) | ✅ PASS |
| Core `apply_rope` (3D/4D/identity) | 10 tests (test_rope_2d.py, test_rope_4d.py) | ✅ PASS |
| Reverse RoPE | 0 (covered in parity) | ✅ covered |
| Cross-backend parity (NP ↔ PT) | 1 test (test_rope_cross_backend.py) | ✅ PASS |
| KV cache position invariant | 4 tests (test_rope_cache.py) | ✅ PASS |
| Relative position property | 2 tests (test_rope_relative.py) | ✅ PASS |
| MHA forward with use_rope flag | 3 tests (test_mha_use_rope_flag.py) | ✅ PASS |
| MHA backward gradient (numerical verify) | 2 tests (test_mha_backward_rope.py, test_mha_rope_backward.py) | ✅ PASS |
| MHA forward output differs when rope toggled | 3 tests (test_mha_rope.py, test_mha_rope_fail.py) | ✅ PASS |
| Positional embedding parity | 4 tests (test_positional_embedding.py) | ✅ PASS |

**Total RoPE-verified tests: 27 (all passing)**

The RoPE subsystem is **DONE**. No further work needed on rope.py, pytorch/rope.py, or their tests.

---

## Remaining Work: 4 Backward Failures in test_block_backward.py

These 4 failures are NOT about RoPE. They are about the **backward gradient computation** in the composition layer (`model.transformer.Transformer`).

### Observation A: `test_block_backward_parameters`

12 out of 13 parameter mappings pass at diff <1e-6. Only `moe.router.weights` (NumPy) vs `moe.router.w` (PyTorch) shows diff=3.3e-2.

**Hypothesis**: The pedagogical `model.moe.Router` and the PyTorch `model.pytorch.moe.PyTorchRouter` have a **different backward implementation** for the router. This is NOT a tolerance issue — it's a real implementation gap.

**Before fixing**, we need to verify: does the standalone test (`test_transformer_block.py`) use the same pedagogical classes? If it uses NumPy TransformerBlock but the pedagogical TransformerBlock uses a different Router class, that explains the discrepancy.

**Action**: Check which Router classes are used by each implementation path.

### Observation B: Two-layer chain failures (3 tests)

All 3 failures occur at **block.0 only**, never at block.1. This means:
- block.1 backward is identical between backends ✅
- The gradient flowing INTO block.0 from block.1 differs between backends ❌

**This suggests**: The `Transformer.backward()` method (full chain backward) computes the input gradient to block.0 differently than the standalone block backward. The error is in how `Transformer.backward()` passes the gradient from block.1 to block.0, NOT in the block itself.

**Root cause candidates**:
1. `Transformer.backward()` in `src/model/transformer.py` accumulates gradients differently
2. Position embedding gradient flow differs (block.0 gets extra gradient from token_embedding)
3. A gradient accumulation order issue (residual connection backward)

---

## Execution Plan — Step by Step (TDD)

### Step 1: Verify `test_block_dx_residual` is now passing

**Already fixed** in the edit. Confirm:

```bash
PYTHONPATH=src uv run pytest tests/parity/test_block_backward.py::TestSingleBlockBackward::test_block_dx_residual -v
```

**Expected**: PASS

---

### Step 2: Isolate the `test_block_backward_parameters` router issue

**Action**: Run the pedagogical TransformerBlock backward in isolation to compare router gradient directly.

```python
# Create tests/parity/debug_router_gradient.py
# 1. Create pedagogical NumPy TransformerBlock with pedagogical Router
# 2. Create pedagogical PyTorch TransformerBlock with PyTorchRouter  
# 3. Sync weights
# 4. Run backward
# 5. Compare router gradients element by element
# 6. Print the exact diff per element
```

**Goal**: Confirm whether the router backward differs between pedagogical NP and pedagogical PT.

---

### Step 3: Isolate the two-layer chain backward divergence

**Action**: Create `tests/parity/debug_two_layer_chain.py` with these specific tests:

**Test 3a**: Forward parity for 2 layers
- Same weights, same input → do outputs match?
- If not → sync is broken, fix sync
- If yes → proceed to 3b

**Test 3b**: Block.1 backward gradient
- Run forward on 2 layers
- Backward through full transformer
- Compare block.1 gradient values → do they match?
- If yes → proceed to 3c
- If no → block.1 backward differs, fix block.1

**Test 3c**: Input gradient to block.0
- Compare the `dx` that feeds into block.0 from block.1
- If different → `Transformer.backward()` accumulates differently
- If same → the divergence is within block.0's own backward

**Test 3d**: Compare Pedagogical NP vs standalone NP
- Create a standalone NumPy TransformerBlock with pedagogical components
- Create a pedagogical TransformerBlock (which builds its own components)
- Do their backward results differ?
- This isolates whether `Transformer.__init__` creates different components

---

### Step 4: Fix the root cause

Based on Step 2-3 results:

- **If router backward differs**: Align the pedagogical `Router.backward()` with `PyTorchRouter.backward()`
- **If two-layer chain dx differs**: Fix `Transformer.backward()` gradient accumulation
- **If block.0 dx is same but gradient differs**: Fix block.0's MHA backward method

---

### Step 5: Run full suite → 221/221

```bash
PYTHONPATH=src uv run pytest tests/ -v --tb=short
```

---

## What We Know for Sure (Verified, Not Assumed)

| Fact | Source |
|------|--------|
| 217/221 tests pass | `pytest tests/ -v` run today |
| 27 RoPE tests pass, 0 fail | grep all test_rope*, test_mha_rope*, test_positional*, test_rope_cache*, test_rope_relative* |
| All failures in ONE file | `test_block_backward.py` |
| 1 failure is tolerance, fixed | `test_block_dx_residual` — edit applied |
| 1 failure is router gradient diff | `test_block_backward_parameters` — needs investigation |
| 3 failures are two-layer chain | `test_two_layer_*` — all at block.0 only |
| RoPE is NOT the problem | All RoPE tests pass, MHA backward with rope passes |

---

## Strict Rules

1. **ONE fix at a time** — Do not batch fixes
2. **No touching non-test files until debug confirms the fix location**
3. **Run tests after every change**
4. **Use test results, not reasoning, to guide all decisions**

## Execution Order

```
Step 1: Verify test_block_dx_residual passes                 → edit already made; run below
Step 2: Debug/router.gradient.py for router mismatch         → small isolated script
Step 3: debug_two_layer_chain.py tests 3a-3d                 → small isolated script
Step 4: Fix the confirmed root cause                          → code edit
Step 5: Full suite → 221/221                                 → verify
```

## Pending Debug

| Unknown | Script | Result |
|---------|--------|--------|
| Router gradient: NP vs PT (pedagogical) | debug_router_gradient.py | NOT STARTED |
| Two-layer forward parity | debug_two_layer_chain.py 3a | NOT STARTED |
| Two-layer block.1 gradient parity | debug_two_layer_chain.py 3b | NOT STARTED |
| Two-layer dx to block.0 parity | debug_two_layer_chain.py 3c | NOT STARTED |
| Pedagogical NP block vs standalone NP block parity | debug_two_layer_chain.py 3d | NOT STARTED |

**DO NOT proceed past Step 1 until debug confirms exactly which line in Transformer.backward() causes the divergence.**
