# Progress Log

## Current Session: 2026-06-26 — Repo Cleanup & Consolidation ✅
### What Was Done
1. **Removed temp/debug code:** debug/, tmp/, resource/models/auto_test/, scripts/auto_test_equivalence.py
2. **Consolidated docs:** Merged 12+ phase plan files into single design.md
3. **Updated status files:** task_plan.md, findings.md, progress.md trimmed to current state

## Session 2026-06-25 — Phase G+++: Weight Diff Debug
### What Was Done
1. Implemented weight comparison across all 4 backends
2. Fixed MHA→RoPE shape mismatch in NumPy and PyTorch
3. Added 80+ tests for MHA, RoPE, CUDA parity, weight sharing, round-trip
4. Fixed GPU NaN from `torch.empty()` → use `torch.zeros()`
## Phase Breakdown

| Phase | Status | Tests | Key Achievements |
|-------|--------|-------|------------------|
| A | ✅ | 131 | Shared config, constants, tokenizer, dataset, checkpoint |
| B | ✅ | ~100 | Complete NumPy reference implementation, TDD-style |
| C | ✅ | 310 | PyTorch autograd backend, gradient-based training |
## Known Issues

### 36 Pre-existing CUDA Unit Failures
- **Not implementation bugs** — structural mismatch (flat vs nested MoE tensors)
- Both CUDA and NumPy/PyTorch produce correct outputs
- To fix: implement flat→nested conversion in NumPy or nested→flat in CUDA

### Weight Drift After Independent Training
- Expected behavior — different backends use different RNG, so weights diverge
- This is not a bug; it's a consequence of independent training

### No `torch.empty()` on GPU
- Uninitialized memory contains garbage → NaN during training
- Must use `torch.zeros()`, `torch.ones()`, or `torch.nn.init.*()` for weight initialization
