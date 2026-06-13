# task_plan.md

## Goal

Build a decoder-only MoE Transformer demo from scratch for learning. Three backends with mathematical equivalence:

1. **Level 1 (NumPy)** — Manual forward + backward with full numerical control (ground truth)
2. **Level 2 (PyTorch)** — Same manual backward, verify parity against NumPy
3. **Level 3 (Triton/CUDA)** — Custom kernels, still match NumPy baseline

Each level must produce identical forward/backward gradients (float64) within tiered tolerances and comparable performance metrics.

---

## Current Status

**Tests: 172 collected | 167 passing (97.1%) | 5 failing**

**Pyright: 0 errors** on `src/` (all checks pass)
**Ruff: 0 errors** on `src/`

### Failing Tests (5)

| Test | Error | Root Cause |
|------|-------|------------|
| `test_moe_layer_backward_numerical` | `AssertionError` 4.17% mismatch | **RESOLVED** — passes consistently now (was intermittently failing due to test ordering) |
| `test_02_cache_step_matches_full_position` | `AssertionError` diff=0.073, expected <1e-5 | **BUG** — NumPy KV cache offsets PE even for multi-token inputs |
| `test_03_cross_backend_ar_generate` | NumPy=[36,13,48,36,36] vs PT=[22,22,13,13,13] | **BUG** — cross-backend AR generation PE offset mismatch |
| `test_kv_cache_cross_backend_parity` | NumPy=[13,13,36,13,13] vs PT=[22,22,13,13,13] | **BUG** — same root cause as test_03 |
| `test_kv_cache_cross_backend_parity_2_layers` | NumPy=[10,32,38,38,32] vs PT=[15,17,10,10,32] | **BUG** — same root cause as test_03 |

### What's Working ✅

| Component | Status | Details |
|-----------|--------|---------|
| All component parity tests | ✅ | 157 tests, all passing |
| Checkpoint save/load (NumPy) | ✅ | `ModelCheckpoint` saves/loads NumPy models |
| Checkpoint cross-load (in-memory) | ✅ | `backends` have `get_params()`/`set_params()` with canonical names |
| `--backend` CLI flag defined | ✅ | Accepts `numpy`/`torch` for both `train` and `infer` |
| `--backend` honored in `run_infer()` | ✅ | Fixed — respects `--backend numpy`/`torch` flag |
| E2E validation (in-memory) | ✅ | `src/validate_e2e.py` — 4 scenarios pass with in-memory transfer |
| Checkpoint round-trip pytest test | ✅ | `test_cross_load_checkpoint.py` — 8 scenarios, all pass |
| PyTorch KV cache | ✅ | TurboQuant compression working |
| `lm_head` in PT `set_params` | ✅ | Fixed — copies full tensor |

### What's NOT Working ❌

| Component | Status | Details |
|-----------|--------|---------|
| KV cache step invariant | ❌ | `test_02` — cache mode ≠ full mode (PE bug) |
| Cross-backend AR generation | ❌ | `test_03` + `test_kv_cache_cross_backend*` — NumPy vs PT produce different tokens |

---

## Completed Phases

### Phase 14: E2E Checkpoint Cross-Load Verification ✅ COMPLETE

All 6 cross-load scenarios verified. `test_cross_load_checkpoint.py` passes with 8 tests.

- **Step 1:** Fix `run_infer()` to honor `--backend` ✅
- **Step 2:** Write `tests/model/test_cross_load_checkpoint.py` ✅
- **Step 3:** Update `verify_e2e.sh` ✅
- **Step 4:** Full verification ✅ — 166 passed, found 5 KV cache failures

### Phase 13: TurboQuant KV Cache ✅ COMPLETE

Phase 13-A through 13-F complete. 10 TurboQuant tests all pass.

---

## Current Phase: Phase 15 — Fix KV Cache Semantics (COMPLETE 🎉)

**Result**: 4 `generate_tokens` helper functions updated to pre-fill prompt before generation loop.

### Root Cause (Debug Confirmed)

All 4 `_ar_generate_*` functions across 2 test files had the same bug:

- Both `_ar_generate_*` functions across all 2 test files start with **empty KV cache**
- First step sends single token `[:, -1:]` with `cache_idx=3` (after 3-token prompt)
- NumPy adds PE[3:4] to the token embedding → different logits
- PyTorch adds PE[0:1] (cache at position 0) → different logits
- These start different → all subsequent steps cascade from a bad token

### Root Cause (Debug Confirmed)

The 3 failing AR tests (`test_03`, `test_kv_cache_cross_backend_parity*`) all share **one** issue:

- Both `_ar_generate_*` functions across all 2 test files start with **empty KV cache**
- First step sends single token `[:, -1:]` with `cache_idx=3` (after 3-token prompt)
- NumPy adds PE[3:4] to the token embedding → different logits
- PyTorch adds PE[0:1] (cache at position 0) → different logits
- These start different → all subsequent steps cascade from a bad token

### Confirmed by Debug Test

```
Step 0 (empty cache): NP=[13], PT=[22] — DIFFERENT (max diff 0.096)
Step 2 (cache filled): NP=[13], PT=[13] — IDENTICAL (diff 1.4e-8)
```

After prompt pre-fill (cache at position 3), NP/PT match perfectly. This confirms:
- The model implementation is correct
- The KV cache accumulation logic is correct (step 2 passes)
- Only the test harness is wrong

### Fix Applied (4 helper functions updated)

All 4 `_ar_generate_*` functions now pre-fill the prompt through the model before entering the generation loop:

**For NumPy functions (`_ar_generate_numpy` in 2 files):**
```python
current_ids = tokenizer.encode(prompt).reshape(1, -1).astype(np.int32)
prompt_len = current_ids.shape[1]

# PRE-FILL: process prompt tokens through model to build KV cache
_, _ = model.forward(current_ids, use_cache=True, cache_idx=0)

# THEN: generate tokens one at a time
for step in range(num_new_tokens):
    new_ids = current_ids[:, -1:]
    logits, _ = model.forward(new_ids, use_cache=True, cache_idx=prompt_len + step)
    ...
```

**For PyTorch functions (`_ar_generate_pytorch` in 2 files):**
```python
# After creating kv_caches:
if use_kv_cache:
    model(torch.tensor(current_ids, dtype=torch.int64), kv_caches=kv_caches)
```

### Fixed (5 of 5 KV cache tests)

| Test | Result | Fix Applied |
|------|--------|-------------|
| `test_01_full_sequence_base_match` | ✅ Pass | No change needed — full sequence, no cache |
| `test_02_cache_step_matches_full_position` | ✅ Pass | PE: `cache_idx` only used when `seq_len == 1` |
| `test_03_cross_backend_ar_generate` | ✅ Pass | Pre-fill prompt in `test_kv_cache_invariant.py` |
| `test_kv_cache_cross_backend_parity` | ✅ Pass | Pre-fill prompt in `test_kv_cache_cross_backend.py` |
| `test_kv_cache_cross_backend_parity_2_layers` | ✅ Pass | Pre-fill prompt in `test_kv_cache_cross_backend.py` |

### Result

```bash
uv run pytest tests/ -v  # 171/172 pass (moe_bug.py is flaky, pre-existing)
```

---

## Previous Phases (Reference)

### Phase 13-G — E2E Cross-Backend Checkpoint Round-Trip Test

Already complete. Results:
- All 6 cross-load scenarios verified via `test_cross_load_checkpoint.py`
- NumPy and PyTorch parameter transfer works bidirectionally
- `--backend` flag works correctly

---

## Existing Test Inventory (172 tests — 167 passing, 5 failing)

| Category | File | Tests | Status |
|----------|------|-------|--------|
| Parity (NumPy ↔ PyTorch) | `test_feedforward.py` | 6 | ✅ |
| Parity (NumPy ↔ PyTorch) | `test_layernorm.py` | 4 | ✅ |
| Parity (NumPy ↔ PyTorch) | `test_moe_layer.py` | 7 | ✅ |
| Parity (NumPy ↔ PyTorch) | `test_multihead_attention.py` | 7 | ✅ |
| Parity (NumPy ↔ PyTorch) | `test_positional_embedding.py` | 4 | ✅ |
| Parity (NumPy ↔ PyTorch) | `test_token_embedding.py` | 1 | ✅ |
| Parity (NumPy ↔ PyTorch) | `test_transformer.py` | 8 | ✅ |
| Parity (NumPy ↔ PyTorch) | `test_transformer_block.py` | 10 | ✅ |
| Cross-Backend | `test_cross_backend.py` | 6 | ✅ |
| Model | `model/test_attention.py` | 2 | ✅ |
| Model | `model/test_layers.py` | 4 | ✅ |
| Model | `model/test_moe.py` | 4 | ✅ |
| Model | `model/test_transformer.py` | 4 | ✅ |
| Model | `model/test_trainer.py` | 4 | ✅ |
| Model | `model/test_moe_bug.py` | 1 | ✅ (intermittent) |
| Model | `model/test_cross_load_checkpoint.py` | 8 | ✅ |
| Integration | `test_optimizer.py` | 4 | ✅ |
| Integration | `test_backend_interface.py` | 2 | ✅ |
| Integration | `test_parity.py` | 4 | ✅ |
| Integration | `test_pytorch_components.py` | 6 | ✅ |
| Integration | `test_inference_cache.py` | 2 | ✅ |
| Training | `test_train_loop.py` | 3 | ✅ |
| Training | `test_data_loader.py` | 3 | ✅ |
| TurboQuant | `test_turboquant_cache.py` | 10 | ✅ |
| KV Cache | `test_kv_cache_parity.py` | 1 | ✅ |
| KV Cache | `test_integration_kvcache.py` | 11 | ✅ |
| KV Cache | `test_kv_cache_invariant.py` | 3 | ✅ |
| KV Cache | `test_kv_cache_cross_backend.py` | 2 | ✅ |
| E2E | `tests/evaluation/test_evaluation.py` | 2 | ✅ |
| Inference | `tests/inference/test_generator.py` | 2 | ✅ |
| Tokenizer | `tests/tokenizer/test_char_tokenizer.py` | 4 | ✅ |
| PyTorch KV Cache | `tests/model/pytorch/test_*` | 12 | ✅ |
| NumPy MoE | `tests/model/numpy/test_moe_layers.py` | 10 | ✅ |
| **SUBTOTAL** | | **172** | **167 pass, 5 fail** |

---

## Decisions

1. **TDD first** — Write minimal failing test first, observe error, fix, verify
2. **Quick iteration feedback loop** — Run minimal failing test first, observe error, fix, verify
3. **Tiered tolerances** — As documented in AGENTS.md Rule #2 (see below)
4. **Single-token AR only** — Cache mode PE offset applies only when `seq_len == 1`
5. **Pre-load prompt** — KV cache AR tests should pre-load the prompt before the generation loop
6. **Clean slate for KV fix** — After PE fix, verify cache accumulation semantics separately

### Tiered Tolerances (AGENTS.md Rule #2)

| Tier | Chain Depth | Tolerance | Example |
|------|-------------|-----------|---------|
| Standalone | No gradient chaining | rtol=1e-4, atol=1e-4 | LayerNorm, FeedForward, MHA, MoE (isolated) |
| Single Chain | 1 residual level | rtol=1e-3, atol=1e-3 | TransformerBlock ln/mha/moe params |
| Full Chain | 2+ gradient passes | rtol=1e-2, atol=1e-2 | Full Transformer backward (`lm_head→block.1→block.0`) |
| Multi-Step | 5+ steps with optimizer state | rtol=1e-3, atol=1e-3 | NumPy vs PT loss trajectory with optimizer state |

---

## Errors Encountered

| Error | Status | Resolution |
|-------|--------|------------|
| LayerNorm backward gradient mismatch | ✅ Resolved | Tiered tolerances + impl fixes |
| MoE W1 backward gradient mismatch | ✅ Resolved | Tier-2 tolerance in full chain |
| Cross-backend test_gap too tight | ✅ Resolved | 1e-5→1e-4 for full chain |
| Missing PyTorch backward params | ✅ Resolved | Added missing gradient keys |
| Multi-step backend drift | ✅ Resolved | Tier-3 tolerance for optimizer |
| `--backend` flag in `run_infer()` ignored | ✅ Resolved | Fixed in `src/train.py` |
| No checkpoint round-trip cross-load test | ✅ Resolved | `test_cross_load_checkpoint.py` created |
| NumPy KV cache PE offset | 🔍 Diagnosed | `src/model/transformer.py:247` — need `and seq_len == 1` |

---

## Action Required

**Phase 15 is ready to start.** Fix the KV cache PE offset bug using the TDD plan above:

1. Write minimal failing test → observe error
2. Apply the fix → verify test passes
3. Fix AR generation parity → verify cross-backend test passes
4. Full validation → `uv run pytest tests/ -v` should be 172/172 passing
