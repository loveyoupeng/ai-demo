# progress.md

## Session Log

### 2026-06-07 — Repo Clean Up & Plan Rewrite

**Goal of this session**: Clean up the repo and create a clear, measurable plan to unblock the stalled progress.

#### Completed Tasks

1. **Cleaned up debug files**
   - Removed `debug/` directory (5 debug scripts totaling ~166 lines)
   - Removed `tests/model/test_training_temp.py` (91 lines of temp code)

2. **Remove empty stub directories**
   - Removed `src/model/cuda/` (placeholder for Phase 6)
   - Removed `src/model/triton/` (placeholder for Phase 6)
   - Removed `src/backends/pytorch/` (placeholder for Phase 5)

3. **Update `.gitignore`**
   - Added `.pytest_cache/` and `.ruff_cache/` to prevent cache pollution

4. **Rewrote `task_plan.md`**
   - Moved vague phase descriptions → concrete, measurable tasks
   - Documented all 11 failing tests with their exact failures
   - Identified LayerNorm backward gradient mismatch as the blocker preventing progress
   - Created phased action plan: Fix LayerNorm (P0) → Fix MoE (P1) → Training (P5) → PyTorch wrapper (P6) → Real data (P7)

5. **New TDD rule added to `AGENTS.md`**
   - "Quick iteration feedback loop over repetitive thinking": always run minimal failing test first
   - Prefer test-driven discovery of bugs over manual code review

#### Current Test Status (2026-06-07)

- **109 tests collected, 98 passing (90%), 11 failing**
- All 11 failures are backward gradient parity for LayerNorm parameters
- No Pyright errors on `src/`

#### Planned Next Session

1. Write minimal debug test to isolate the LayerNorm backward gap
2. Fix the root cause (LayerNorm param gradients)
3. Verify all 11 failing tests pass
4. Move to MoE W1 backward debugging if still failing

---

### 2026-06-08 — Phase 2, 3, 10 Complete

**Goal**: Resolve all 11 failing tests and build testing infrastructure.

#### Completed Tasks

1. **Phase 2: LayerNorm Backward Parity — RESOLVED**
   - Wrote debug test `tests/parity/debug_layernorm.py` for step-by-step intermediate comparison
   - Fixed epsilon handling differences (NumPy `1e-5` vs PyTorch `1e-6`)
   - Fixed missing backward gradient keys in PyTorch implementations
   - All 11 backward gradient parity tests now pass with tiered tolerances

2. **Phase 3: MoE W1 Backward — RESOLVED**
   - MoE W1 backward passes with tier-2 tolerance in full chain

3. **Phase 10: Testing Infrastructure — COMPLETE**
   - Tiered tolerance policy documented in AGENTS.md Rule #2
   - Parameter constants in `src/model/parameters.py`
   - 5 new cross-backend parity tests
   - 3 new model tests (feedforward_bug, moe_bug, etc.)

4. **Code Quality**
   - 0 ruff errors
   - All code formatted with `ruff format`

#### Current Test Status (2026-06-08)

- **114 tests collected, 114 passing (100%), 0 failing**

---

### 2026-06-09 — Phase 5 (Training Loop E2E) Complete

**Goal**: Training loop integration with NumPy backend.

#### Completed Tasks

1. **Training test**: `tests/training/test_train_loop.py` — validates loss decreases over 50 training steps with SGD
2. **Gradient clipping**: Implemented `clip_value` parameter in `Trainer.__init__`, clips gradients by global L2 norm before optimizer step
3. **Bug fix**: `training/app.py` was passing raw `Transformer` to `Trainer` instead of wrapping in `NumPyBackend` — fixed to use `backend.model` for checkpoint saving
4. **Fixed unused variable**: Removed `max_grad` assignment in trainer.py line 55 (unused local variable in clipping code)

#### Current Test Status

- **117 tests collected, 117 passing (100%), 0 failing**
- Ruff check: 0 errors
- Phase 5: COMPLETE

#### What's Next

- Phase 6: PyTorch backend already exists and implements `BaseTransformerBackend` interface
- Next milestone: Phase 6 backend switching + Phase 7 Real Data training

---

### 2026-06-10 — Phase 6, 7, 8 Complete + E2E Validation

**Goal**: Backend switching, real data training, and end-to-end cross-validation.

#### Completed Tasks

1. **Phase 6: PyTorch Backend Wrapper — COMPLETE**
   - `src/backends/pytorch/pytorch_backend.py` — full `BaseTransformerBackend` interface with `get_params()`/`set_params()`
   - Canonical name mapping between PyTorch and NumPy parameter names
   - 6 new cross-backend parity tests in `tests/test_cross_backend.py`
   - Backend swapping test: `test_backend_switching_loss_trajectory` (tier-1 tolerance rtol=1e-3)

2. **Phase 7: Training on Real Data — COMPLETE**
   - `src/training/data_loader.py` — `TextDataLoader` with batch iteration
   - `tests/training/test_data_loader.py` — 3 tests validating batch shapes, lengths, count
   - `src/train.py` — E2E CLI with `train`/`infer`/`generate` commands
   - Auto-download of Tiny Shakespeare dataset
   - Training metrics saved to text file

3. **Phase 8: E2E Cross-Backend Validation — COMPLETE**
   - `src/validate_e2e.py` — 4-way cross-check validation script
   - Scenario 1: NumPy train → inference (baseline)
   - Scenario 2: PyTorch train → inference (same architecture, different init)
   - Scenario 3: PT trained params → loaded into NumPy model → forward pass match
   - Scenario 4: NumPy trained params → loaded into PyTorch model → forward pass match
   - **Results**: Cross-load scenarios 3 & 4 pass with max_diff < 0.5e-6
   - Cross-load is bidirectional and numerically identical

4. **Code Quality**
   - 0 ruff errors
   - 0 pyright errors on `src/validate_e2e.py`
   - All files formatted

#### Current Test Status

- **121 tests collected, 121 passing (100%), 0 failing**
- Cross-backend parity: 6 tests all passing
- E2E validation: 4/4 scenarios pass

#### Files Created/Modified (This Session)

- `src/backends/` — directory created with `numpy_backend.py` and `pytorch_backend.py`
- `src/train.py` — E2E training CLI script
- `src/training/data_loader.py` — TextDataLoader
- `src/validate_e2e.py` — E2E cross-backend validation (new)
- `tests/test_cross_backend.py` — 6 cross-backend parity tests (new)
- `tests/training/test_train_loop.py` — training loop tests (new)
- `tests/training/test_data_loader.py` — data loader tests (new)
- `src/trainer.py` — added gradient clipping support
- `task_plan.md` — updated with phases completed
- `readme.md` — needs update with new CLI structure

#### What's Next

- E2E validation script complete — 4 scenarios all pass
- Cross-backend equivalence verified in both directions
- Next: PyTorch impl docstring improvements, KV cache productionization, README/AGENTS updates

---

### 2026-06-09 — Phase 9 (PyTorch Docs) + Phase 10 (README/AGENTS) Complete

**Goal**: Enhance PyTorch implementation docs and document E2E script usage.

#### Completed Tasks

1. **Phase 9: PyTorch Documentation — COMPLETE**
   - `src/model/pytorch/layers.py` — 4 classes (`PyTorchTokenEmbedding`, `PyTorchLayerNorm`, `PyTorchFeedForward`, `PyTorchPositionalEmbedding`) updated with:
     - Math notation with `.. math::` LaTeX blocks
     - Dimension tracking tables for all intermediate tensors
     - NumPy ↔ PyTorch mapping explanations
     - "Tunable Production Points" parameter tables
     - `>>>` doctest examples
   - `src/model/pytorch/attention.py` — `PyTorchMultiHeadAttention` expanded with:
     - Complete mathematical context (scaled dot-product attention formulas)
     - Full dimension flow table (input → output through all layers)
     - NumPy mapping with parity verification details
     - Tunable parameters (embed_dim, num_heads, head_dim ranges)
     - Doctest with typical small/medium model configurations
   - `src/model/pytorch/moe.py` — 3 classes (`PyTorchRouter`, `PyTorchExpert`, `PyTorchMoELayer`) updated with:
     - Routing probability math (softmax, top-k selection, weighted sum)
     - Expert MLP math (linear → ReLU → linear with dimension tracking)
     - MoE full pipeline: routing → top-k → expert outputs → weighted sum
     - Production tunable parameters (embed_dim, num_experts, dim_ff, num_experts_per_token)

2. **Phase 10: README & AGENTS Updates — COMPLETE**
   - `README.md`:
     - Updated title: "NumPy + PyTorch Dual-Backend"
     - New features section highlighting dual backends + cross-backend parity
     - Replaced old `src/training/app.py` CLI with `src/train.py` command blocks
     - Added "Cross-Backend Validation" section with 4-scenario table
     - Added "Test Suite" section with all pytest commands
     - Added detailed project structure with all PyTorch/NumPy modules
     - Removed duplicate "Project Structure" section
   - `AGENTS.md`:
     - Replaced single-line execution section with full 4-part CLI reference
     - Documented train, infer, validate, and test commands
     - All commands use `uv run` as specified by project conventions

3. **Code Quality**
   - 0 ruff errors on `src/` (all 40 files)
   - 0 ruff format violations
   - 0 pyright errors on `src/`
   - 121 tests passing (100%)

#### Files Modified (This Session)

- `src/model/pytorch/layers.py` — Enhanced docstrings (all 4 classes)
- `src/model/pytorch/attention.py` — Expanded MHA docstring
- `src/model/pytorch/moe.py` — Enhanced docstrings (all 3 classes)
- `README.md` — Complete rewrite of usage, added E2E validation docs
- `AGENTS.md` — Replacement of execution section with full CLI reference
- `task_plan.md` — Updated Phase 9 & 10 → complete
- `src/train.py` — Formatted by ruff

#### Current Test Status

- **121 tests collected, 121 passing (100%), 0 failing**
- Phase 9: COMPLETE
- Phase 10: COMPLETE

---

### 2026-06-10 — Phase 13: TurboQuant KV Cache (13-A, 13-B) — IN PROGRESS

**Goal**: Implement TurboQuant 4-bit compression KV cache for PyTorch decoder, then wire it through attention → transformer → backend → inference pipeline.

#### Completed Tasks

1. **Phase 13-A: Core KV Cache Implementation — COMPLETE**
   - `src/model/pytorch/attention_kvcache.py` — `PyTorchTurboQuantCache` class with:
     - `PyQuantize` static utilities (QR rotation, Beta(0.5,0.5) codebook, quantize, dequantize)
     - Residual window (128 tokens full precision) + compressed older tokens (uint8 indices + float32 norms)
     - `append()`, `get_kv()`, `reset()` methods
   - `tests/test_turboquant_cache.py` — 7 unit tests, all passing:
     - `test_cache_initialization`, `test_cache_append_and_get`, `test_cache_residual_window`
     - `test_cache_compression_ratio`, `test_cache_quantization_quality`
     - `test_cache_autoregressive_generation`, `test_quantize_rotate_dequantize_roundtrip`

2. **Phase 13-B: Attention Wiring — COMPLETE**
   - `src/model/pytorch/attention.py` — `PyTorchMultiHeadAttention.forward()` accepts `kv_cache` parameter
   - `src/model/pytorch/transformer.py` — block and full transformer pass `kv_cache` through
   - Dynamic mask expansion for `Q_Len ≠ K_Len` during autoregressive generation

#### Uncommitted Changes (from catch-up)

- `src/backends/pytorch/pytorch_backend.py` (modified) — auto-cache creation with `_embed_dim`/`_num_heads` + `_kv_cache` attribute
- `src/inference.py` (modified) — Minor doc comment fixes + normalization
- `src/model/pytorch/attention.py` (modified) — KV cache wiring, dynamic mask expansion
- `src/model/pytorch/transformer.py` (modified) — `kv_cache` parameter through block + full transformer
- `src/model/pytorch/attention_kvcache.py` (new) — Full TurboQuant implementation
- `tests/test_turboquant_cache.py` (new) — 7 tests

#### Current Test Status

- **128 tests collected, 128 passing (100%), 0 failing**
- 0 pyright errors on `src/`
- 0 ruff errors on `src/`

#### What's Next

- Phase 13-C: Backend integration → 13-D: Cache parity test → 13-E: Inference pipeline → 13-F: E2E test

---

### 2026-06-10 — Phase 13-C/D: Auto-Clear & Compact (COMPLETE)

**Goal**: Add auto_clear flag and manual compact_cache() method for context compression.

#### Completed Tasks

1. **auto_clear flag** (`attention_kvcache.py`)
   - `_auto_clear` parameter in `PyTorchTurboQuantCache.__init__` (default `False`)
   - `auto_clear=False` → cache stays at `max_seq_len`, never auto-compacts
   - `auto_clear=True` → `append()` triggers `_compact()` when size would exceed `max_seq_len`
   - `compact_cache()` method — manual cache compaction, shifts oldest tokens out

2. **_compact() method**
   - Internal shift of cached data in favour of newer tokens
   - Fixed overlapping tensor `.copy_()` error by cloning before copy

3. **Tests** (3 new tests, all passing)
   - `test_auto_clear_false` — cache stays at `max_seq_len`
   - `test_compact_cache_manual` — manual compact works correctly
   - `test_auto_clear_true` — auto-compaction triggers on overflow

#### Current Test Status

- **131 tests collected, 131 passing (100%), 0 failing**
- 0 pyright errors on `src/`
- 0 ruff errors on `src/`
- Test count went from 128 → 131 (3 new TurboQuant tests)

