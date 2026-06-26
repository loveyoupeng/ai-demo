# Phase G+++: Auto Test Weight Diff — TDD Plan

## Goal

Fix the `auto_test_equivalence.py` weight diff tests so that cross-backend weight comparison is accurate and informative. Work uses small, focused test cases to drive each fix.

## Current State

| Test | Result | Expected |
|------|--------|----------|
| numpy vs torch | max_diff=0.338 FAIL | Divergent weights (expected) |
| numpy vs triton | max_diff=? | Divergent weights (expected) |
| numpy vs cuda | max_diff=? | Divergent weights (expected) |
| torch vs triton | max_diff=? | Divergent weights (expected) |
| torch vs cuda | max_diff=0.000 PASS? | TBD |
| triton vs cuda | max_diff=? | Divergent weights (expected) |
| Two-way inference | PASS | Same weights → same outputs |
| Training dynamics | PASS | Both converge |
| Round-trip torch→numpy | diff=0.000 PASS | Exact match |
| Round-trip numpy→torch | diff=0.000 PASS | Exact match |

**Key finding:** Round-trip tests produce 0.000 diff (exact match). This validates that cross-backend load/save is working correctly. The "weight drift" after independent training is expected — different backends use different RNG implementations.

## Approach: Test-First, Small & Focused

Each step below is a self-contained test case. Run, verify, commit. Never batch fixes.

---

### Test Case 1: Verify Key Normalization Maps Correctly

**What to test:** `normalize_params_to_torch()` correctly maps all NumPy/Triton/CUDA parameter keys → Torch keys. No keys lost, no keys added.

**How to verify:**
```python
# Train both backends with seed=42, n_layers=1
# Call normalize_params_to_torch on both
# Compare dicts

np_params = train_numpy(seed=42, n_layers=1, embed_dim=32, vocab_size=64)
torch_params = train_torch(seed=42, n_layers=1, embed_dim=32, vocab_size=64)

# Normalize both
torch_norm = normalize_params_to_torch(torch_params)  # identity
np_norm = normalize_params_to_torch(np_params)        # remap

# Assertions
assert set(torch_norm.keys()) == set(np_norm.keys()), f"Key mismatch: extra={set(np_norm.keys())-set(torch_norm.keys())}, missing={set(torch_norm.keys())-set(np_norm.keys())}"
assert len(set(torch_norm.keys())) == 24, f"Expected 24 keys, got {len(set(torch_norm.keys()))}"
```

**Expected result:** All 24 keys map correctly. Torch keys are the canonical set.

**Commit condition:** Key sets are identical. No orphaned keys in either direction.

---

### Test Case 2: Shared Weights → Identical Outputs (Two-Way Inference)

**What to test:** Same weights loaded into different backends → same output for same input.

**How to verify:**
```python
# 1. Train one model (e.g., NumPy)
np_model = create_and_train_numpy(seed=42)
np_weights = np_model.save_as_numpy()

# 2. Load same weights into PyTorch, Triton, CUDA
torch_model = create_torch_model(config)
torch_model.load_from_numpy_dict(np_weights)

triton_model = create_triton_model(config)
triton_model.load_from_numpy_dict(np_weights)

cuda_model = create_cuda_model(config)
cuda_model.load_from_numpy_dict(np_weights)

# 3. Same input
input_tokens = torch.randint(0, 64, (1, 16))

# 4. Forward pass (all in eval mode)
np_out = np_model.eval_forward(input_tokens)
torch_out = torch_model.eval_forward(input_tokens)
triton_out = triton_model.eval_forward(input_tokens)
cuda_out = cuda_model.eval_forward(input_tokens)

# Assertions
assert np.max(np.abs(np_out - torch_out.numpy())) < 1e-4, "NumPy vs Torch output mismatch"
assert np.max(np.abs(np_out - triton_out.numpy())) < 1e-4, "NumPy vs Triton output mismatch"
# CUDA: structural check only (different numerical precision)
assert cuda_out.sum() > -1e6 and cuda_out.sum() < 1e6, "CUDA output exploded"
```

**Expected result:** NumPy ≈ PyTorch ≈ Triton output (max_diff < 1e-4). CUDA may differ but no NaN.

**Commit condition:** All three backends produce bitwise-identical (NumPy/Torch) or near-identical (Triton) outputs.

---

### Test Case 3: Weight Diff After Independent Training Is Non-Zero

**What to test:** Independently trained models produce DIFFERENT weights (this is expected behavior).

**How to verify:**
```python
# Train two models independently with same seed
model_a = create_and_train_numpy(seed=42)
model_b = create_and_train_numpy(seed=42)

# They should be identical (same code, same RNG, same seed)
params_a = normalize_params_to_torch(model_a.save_as_numpy())
params_b = normalize_params_to_torch(model_b.save_as_numpy())

for key in params_a:
    diff = np.abs(params_a[key].astype(np.float64) - params_b[key].astype(np.float64)).max()
    assert diff < 1e-10, f"Self-reproducibility failed for {key}: diff={diff}"
```

**Expected result:** Self-reproducibility holds — identical seed → identical weights.

**Commit condition:** Repeated runs with same seed produce bitwise-identical weights.

---

### Test Case 4: Torch vs Triton Weight Drift After Independent Training

**What to test:** When both backends use PyTorch-based RNG (`torch.manual_seed`), weight drift should be SMALLER than NumPy vs Torch.

**How to verify:**
```python
# Both use torch.manual_seed(seed) + torch.nn.init.kaiming_uniform_
torch_params = train_torch(seed=42)
triton_params = train_triton(seed=42)

torch_norm = normalize_params_to_torch(torch_params)
triton_norm = normalize_params_to_torch(triton_params)

common_keys = set(torch_norm.keys()) & set(triton_norm.keys())
max_diff = 0
for key in common_keys:
    diff = np.abs(torch_norm[key].cpu().numpy().astype(np.float64) - triton_norm[key].astype(np.float64)).max()
    max_diff = max(max_diff, diff)

print(f"torch vs triton weight diff: {max_diff:.4f}")

# Expected: MUCH less than numpy vs Torch (0.33)
# Likely range: 0.05 - 0.15 (same RNG, slightly different numerics)
assert max_diff < 0.5, f"Expected < 0.5 diff for torch vs triton, got {max_diff}"
```

**Expected result:** torch vs triton diff < 0.5 (ideally < 0.15). Both use same RNG → similar init + divergence only from numerical differences in training loop.

**Commit condition:** torch vs triton weight diff < 0.5.

---

### Test Case 5: Round-Trip Equivalence (Trusted Baseline)

**What to test:** This is the CANONICAL equivalency test. If round-trip fails, everything else is suspect.

**Already working — verify it still passes:**
```python
# Train torch → save_npz → load numpy → compare
torch_model = create_and_train_torch(seed=42)
npz_path = torch_model.save_as_npz('/tmp/rt_test/')

numpy_model = create_numpy_model('/tmp/rt_test/')
params_np = normalize_params_to_torch(numpy_model.get_all_parameters())
params_torch = normalize_params_to_torch(torch_model.save_as_numpy())

common = set(params_torch.keys()) & set(params_np.keys())
for key in common:
    diff = np.abs(params_torch[key].cpu().numpy().astype(np.float64) - params_np[key].astype(np.float64)).max()
    assert diff < 1e-10, f"Round-trip failed {key}: diff={diff}"
```

**Expected result:** Exact match (diff ≈ 0.0000).

**Commit condition:** This test MUST pass. If it doesn't, the save_as_numpy / load_from_numpy_dict path is broken.

---

### Test Case 6: CUDA Key Mapping (No Map Needed for CUDA)

**What to test:** CUDA model uses different attribute names (`Wq` instead of `mha.Wq`) but weight diff comparison should still work.

**How to verify:**
```python
cuda_params = train_cuda(seed=42)

# CUDA uses expanded flat attributes
# MHA: Wq, Wk, Wv, Wo (flat on block) + bk, bo, bq, bv
# MoE: expert_weights[NxDxD], expert_bias[NxD], routing_weights, router (scalar)
# NOT the same structure as NumPy/Torch

# Check what keys exist
print("CUDA keys:", sorted(cuda_params.keys())[:20])
# "blocks.0.Wq", "blocks.0.Wk", "blocks.0.Wv", "blocks.0.Wo",
# "blocks.0.Wk.bias", "blocks.0.Wq.bias", etc.

# Compare with torch
torch_params = train_torch(seed=42)
torch_norm = normalize_params_to_torch(torch_params)

common = set(torch_norm.keys()) & set(cuda_params.keys())
print(f"Common torch ↔ cuda keys: {len(common)}")
print(f"Common keys: {sorted(common)}")
```

**Expected result:** Few or zero common keys between CUDA and PyTorch (different structure). Weight diff may be undefined or skipped for CUDA.

**Commit condition:** CUDA weight diff is handled gracefully — either skipped or computed over a small common subset.

---

### Test Case 7: Training Dynamics Are Comparable

**What to test:** Both backends show the same qualitative behavior (loss decreases, reasonable magnitude).

**Already working — verify:**
```python
# Train both, record loss at each step
torch_losses = train_torch_record_losses(seed=42, steps=10)
triton_losses = train_triton_record_losses(seed=42, steps=10)

# Both should decrease
for i in range(len(torch_losses) - 1):
    assert torch_losses[i+1] < torch_losses[i], f"Torch loss not decreasing at step {i+1}"
    assert triton_losses[i+1] < triton_losses[i], f"Triton loss not decreasing at step {i+1}"

# Both should be in reasonable range (not exploding or stuck at 0)
for loss in torch_losses + triton_losses:
    assert 0 < loss < 100, f"Loss out of reasonable range: {loss}"
```

**Expected result:** Both backends converge, loss in [0, 100].

**Commit condition:** Both converge (already passing).

---

## Execution Order

Run test cases 1 through 7 in order. Each case is a self-contained verification:

```
Step 0: Run existing auto_test_equivalence.py to capture baseline
    $ uv run python -m scripts.auto_test_equivalence --output baseline.json

Step 1: Write/run test case 1 (key normalization)
Step 2: Write/run test case 2 (shared weights → same output)
Step 3: Write/run test case 3 (self-reproducibility)
Step 4: Write/run test case 4 (torch vs triton drift)
Step 5: Write/run test case 5 (round-trip)
Step 6: Write/run test case 6 (CUDA mapping)
Step 7: Write/run test case 7 (training dynamics)
```

After all 7 pass:
```
Final: Run full test suite
    $ uv run python -m scripts.auto_test_equivalence --compare all

Rerun for consistency (run 3 times):
    $ uv run python -m scripts.auto_test_equivalence --compare "numpy,torch"; uv run python -m scripts.auto_test_equivalence --compare "numpy,torch"; uv run python -m scripts.auto_test_equivalence --compare "numpy,torch"
```

## Success Criteria

When ALL of these are true:

| Criterion | Status |
|-----------|--------|
| 1: Key normalization — all keys map correctly | Not done |
| 2: Shared weights → same outputs across backends | Partial (round-trip only) |
| 3: Self-reproducibility — same seed → same weights | Not verified |
| 4: torch vs triton weight drift < 0.5 (same RNG) | Not verified |
| 5: Round-trip equivalence (trusted baseline) | PASS (0.0000) |
| 6: CUDA weight diff handled gracefully | Not verified |
| 7: Training dynamics comparable | PASS |
| 8: All 10 scenarios pass (weight diff, inference, training, round-trip) | Not done |
| 9: Repeated runs produce consistent results | Not done |

## Notes & Constraints

- All float64 for parity testing
- Use `--embed_dim 32` for fast iteration (small model)
- Seed must be passed through ALL creation, training, and comparison paths
- Key normalization uses `INVERSE_TRITON_MAP` — the canonical key set is Torch's
- CUDA is structurally different (flat `Wq` vs nested `blocks.0.mha.Wq`) — may need special handling
- Round-trip tests (5) are the TRULY trusted baseline — if they ever fail, STOP and fix save_as_numpy