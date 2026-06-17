# Fix Plan: NumPy/PyTorch Equivalence Verification (COMPLETED)

## Problem Statement (Resolved)

`scripts/verify_equivalence.py` was failing with `weight_diff ≈ 0.42` on all 6 scenarios.
The root cause: the script compared raw `state_dict()` keys from PyTorch against
NumPy parameter keys without syncing weights first. The models also disagreed on the
MoE router (no bias in PyTorch but yes in NumPy) and had different key naming conventions.

## Changes Applied

| # | What | File | Verified By |
|---|------|------|-------------|
| 1 | PyTorch `MixtureOfExperts.router` bias=True | `impl/_torch/layers.py:526` | `test_cross_backend_parity` passes |
| 2 | `load_from_numpy()` loads MoE bias | `impl/_torch/layers.py:924-925` | `test_cross_backend_parity` passes |
| 3 | `save_as_numpy()` saves MoE bias | `impl/_torch/layers.py:1017-1019` | `test_save_load_round_trip` passes |
| 4 | `load_from_numpy_dict()` loads MoE bias | `impl/_torch/layers.py:1103-1104` | `test_save_load_round_trip` passes |
| 5 | `verify_equivalence.py` wires sync | `scripts/verify_equivalence.py:440` | All 6/6 scenarios pass |
| 6 | `verify_equivalence.py` uses `save_as_numpy()` | `scripts/verify_equivalence.py:398` | All 6/6 scenarios pass |
| 7 | `verify_equivalence.py` skips zero-size arrays | `scripts/verify_equivalence.py:244-245` | `weight_diff` no longer crashes |
| 8 | `verify_equivalence.py` uses `no_grad()` for greedy | `scripts/verify_equivalence.py:483-493` | `greedy_match` = true |
| 9 | `verify_equivalence.py` fixes `.numpy()` on grad tensors | `scripts/verify_equivalence.py:515-521` | `distribution_match` = true |

## File Corruption (Resolved)

The `load_from_numpy_dict()` function body was duplicated after a `return` at line 1133,
creating dead code. Removed lines 1125–1179 (full duplicate section).

## Verification Results

```
pytest tests/ --tb=short → 433 passed
ruff check               → All checks passed
ruff format              → 1 file reformatted
pyright                  → 0 errors, 0 warnings, 0 informations

verify_equivalence.py    → 6/6 scenarios passed, wdiff=0.0
```
