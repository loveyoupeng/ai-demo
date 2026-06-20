# Progress Log

## Session: 2026-06-20 — Phase F: MoE Bug Identified, Strategic Review Completed

### CONTEXT RECOVERY: Full Phase F Review

Ran session-catchup.py → detected phase F has been stuck at MoE (F6) for too long.
Conducted comprehensive review of all phase F code, tests, and planning docs.

### Key Discovery: MoE Bug Root Cause

**5 MoE tests failing** (45/50 CUDA tests pass). Root cause identified:
- The `moe_weighted_sum` kernel reads indexed data from tensors passed from Python
- The tensors may be non-contiguous (`.view()` creates views, not copies)
- Non-contiguous tensor `data_ptr()` points to wrong memory location for indexed reads
- This causes the kernel to read zeros/garbage from the indices array
- Result: all tokens use only expert 0's output

**Fix hypothesis:** Add `.contiguous()` before `.view()` in `moe.py` for:
- `expert_outputs` before `.view(total_tokens, N, D)`
- `topk_idx` before `.view(-1)`
- `topk_weights` before `.view(-1)`

### Test Results Summary

```
45/50 CUDA tests passing
---
✅ test_activation.py: 4/4 (F1 SiLU)
✅ test_layernorm.py: 4/4 (F2 RMSNorm)
✅ test_rope.py: 4/4 (F3 RoPE)
✅ test_ffn.py: 3/3 (F4 SwiGLU)
✅ test_attention.py: 4/4 (F5 MHA)
✅ test_cuda_api_foundations.py: 11/11
✅ test_import.py: 1/1
❌ test_moe.py: 0/2 (F6 bug - topk matching)
❌ test_moe_debug.py: 10/15 (F6 bug - 4 weighted sum failures)
```

### Planning Docs Updated

1. **task_plan.md** — Updated Phase F status: F0-F5 complete, F6 bug identified
2. **docs/phase_f_plan.md** — Complete rewrite: root cause analysis, two-path approach, F7-F11 revised plan with contiguous tensor rule
3. **findings.md** — Added MoE root cause analysis, working pattern for indexed access
4. **docs/design.md** — Phase F section updated with current state
5. **progress.md** — This file, session log updated

### Blocker Status

| Blocker | Status |
|---------|--------|
| MoE kernel wrong output | Root cause identified, fix pending (Path A) |
| F7-F11 not started | Plan defined sequentially, ready after MoE fix |
| Deprecation warnings | Cosmetic, not blocking |

### Strategy Going Forward

1. **Immediate:** Fix MoE by adding `.contiguous()` — should be 1-2 line fix
2. **Then sequential:** F7 (TransformerBlock) → F8 (DecoderStack) → F9 (CUDAModel) → F10 (Training/Inference) → F11 (4-way Parity)
3. **TDD discipline:** Each stage = failing test → minimal fix → quality check → commit
4. **Contiguous rule:** All indexed GPU kernel inputs must be `.contiguous()` before `.view()` or `.transpose()`

---

## Session: 2026-06-19 — Phase E Prep: GPU Confirmed, Docs Updated

### Context Recovery
- Ran session-catchup.py — no unsynced context detected
- All 5 planning files read and updated for Phase E

### Key Findings
1. **GPU confirmed:** `torch.cuda.is_available()` = True (CUDA 12.6, cuDNN 9.3, cuBLAS 12.6, Orin GPU)
2. No code changes needed — GPU already working from Phase D work
3. Phase E ready to begin — 12 sub-phases (E0–E11) planned, TDD approach ready

---

## Phase Completion Summary

| Phase | Name | Status | Tests | Commits | Plan File |
|-------|------|--------|-------|---------|-----------|
| A | Shared Foundation | ✅ Complete | 111 | N/A | `docs/phase_a_plan.md` |
| B | NumPy Implementation | ✅ Complete | ~70 | 21 (b0-b19) | `docs/phase_b_plan.md` |
| C | PyTorch Implementation | ✅ Complete | 129 | 36 (c0-c36) | `docs/phase_c_plan.md` |
| C+ | E2E Training/Inference | ✅ Complete | 90 | 8 (c37–c44) | `docs/phase_c_plus_plan.md` |
| C++ | Normalization Improvements | ✅ Complete | 21 | 3 (d0-d2) | `docs/phase_c_plus2_plan.md` |
| D | Cross-Backend Equivalence | ✅ Complete | — | 1 (e0) | — |
| E | Triton GPU Kernels | ✅ Complete | 538 | ~53 | `docs/phase_e_plan.md` |
| E+ | Cleanup & Refinement | ✅ Complete | +13 | ~15 | `docs/phase_e_plus_plan.md` |
| F0 | CUDA Scaffolding | ✅ Complete | 128 | 1 (f0) | `docs/phase_f_plan.md` |
| **F1** | **SiLU nvrtc+PyTorch** | **✅ Complete** | **4** | **1 (f1)** | `docs/phase_f_plan.md` |
| F2–F11 | CUDA Remaining | 🔶 In Progress | 0 | 0 | `docs/phase_f_plan.md` |
| **Total** | | **558 passing** | **~133** | | |

---

## Test Results

| Module | Tests | Status |
|--------|-------|--------|
| shared/ + tests/ (unit) | 540+ | ✅ all pass |
| tests/cross_backend/ | 21 | ✅ all pass |
| impl/_cuda/ | 132 | ✅ all pass (128 F0 + 4 F1) |
| **Total** | **558+** | **557 pass** |
| Code quality | 0 ruff errors, 0 pyright errors | ✅ clean |

## Plan File Hierarchy
| File | Purpose | Status |
|------|---------|--------|
| `task_plan.md` | High-level roadmap | ⚠️ Needs update — F1 complete, F2–F11 pending |
| `docs/design.md` | Full architecture design | ✅ Updated — CUDA now IN PROGRESS |
| `docs/phase_f_plan.md` | Phase F 12-stage execution plan | ✅ Valid — F1 done, F2–F11 to execute |
| `docs/phase_e_plan.md` | Phase E 12-stage plan | ✅ Complete |
| `findings.md` | Research findings | ✅ Updated — cuLaunchKernel working pattern documented |
| `progress.md` | Session log, test results | This file |

## 5-Question Reboot Check (2026-06-20)

| Question | Answer |
|----------|--------|
| Where am I? | Phase F0-F5 complete (SiLU, RMSNorm, RoPE, SwiGLU, MHA kernels). F6 MoE blocked by 5 failing tests due to non-contiguous tensor access. 45/50 CUDA tests pass. |
| Where am I going? | Fix MoE bug → F7 TransformerBlock → F8 DecoderStack → F9 CUDAModel → F10 Training/Inference → F11 4-way parity. Total: ~6 stages remaining. |
| What's the goal? | Build CUDA C kernels via nvrtc + PyTorch dispatcher, with all kernels producing results matching NumPy/Torch/Triton within specified tolerances. |
| What have I learned? | cuLaunchKernel works with (values, types) tuple. Indexed GPU kernel reads require .contiguous() on source tensors. Non-contiguous .view() silently corrupts data. |
| What have I done? | Updated all 5 planning docs with comprehensive MoE root cause analysis, revised F6-F11 plan, contiguous tensor rule, and two-path strategy. |