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

#### Current Test Status

- **109 tests collected, 98 passing (90%), 11 failing**
- All 11 failures are backward gradient parity for LayerNorm parameters
- No Pyright errors on `src/`
- No new code written — this session focused entirely on cleanup, diagnosis, and planning

#### Key Discovery

The codebase has a **structural problem**: two separate NumPy implementations (`src/model/` and `src/model/numpy/`) that serve different purposes but have evolved independently. The parity tests use `src/model/numpy/` while the training pipeline uses `src/model/`. This split has made debugging hard — when tests break, it's unclear which implementation is wrong.

#### What's Blocked

- Phase 2 (Transformer backward) is blocked on fixing LayerNorm backward parity
- Without fixing the 11 failing tests, moving to training/data phases would accumulate more unknown-to-known bugs
- The fix must start with minimal TDD reproduction, not code review

#### Planned Next Session

1. Write minimal debug test to isolate the LayerNorm backward gap
2. Fix the root cause (LayerNorm param gradients)
3. Verify all 11 failing tests pass
4. Move to MoE W1 backward debugging if still failing
