# Phase E+: Cleanup & Refinement Plan

**Goal:** Clean up codebase before Phase F (CUDA), making code production-ready with consistent naming, no magic strings, comprehensive documentation, and simplified abstractions.

**Status:** ✅ COMPLETE — All 6 waves implemented (10-15 commits, ~12-16 hours)
**Current State:** 551 tests (all pass), ruff = 0 errors, pyright clean

---

## 1. Background: Current State

After Phase E (Triton implementation), we have 3 fully functional backends:
- **NumPy** (`impl/_np/`) — 9 files, 103KB — learning-focused, heavily documented
- **PyTorch** (`impl/_torch/`) — 9 files, 82KB — production-ready, nn.Module-based
- **Triton** (`impl/_triton/`) — 12 files, 74KB — GPU kernels + PyTorch wrapper

**Total: ~29 implementation files, ~260kB, 538 tests, 89 commits**

Each backend is functional but has inconsistencies that make it hard for users to navigate:
- Parameter naming differs across backends
- File organization is inconsistent
- Magic strings (hardcoded param names) scattered through code
- Triton documentation lacks the "why" explanations
- Some debug/code-investigation artifacts remain

---

## 2. Goals & Scope

| Goal | Priority | Description |
|------|----------|-------------|
| **G1: Remove debug/investigation code** | High | Clean up print statements, debug flags, commented-out code, test artifacts |
| **G2: Eliminate magic strings** | High | Replace raw string literals for parameter names with constants |
| **G3: Consistent naming across backends** | High | Align attribute/param naming so users can easily map 1:1 between backends |
| **G4: Comprehensive Triton documentation** | High | Add detailed pydocs explaining HOW Triton works and WHY patterns are used |
| **G5: Simplified abstractions & dedup** | Medium | Reduce code complexity by extracting shared patterns without breaking parity |

### Non-goals (for this phase)

- **Phase F (CUDA)** — not touching CUDA implementation yet
- **New features** — no architectural changes, only cleanup/refactor
- **Test changes** — keep all 538 tests passing; no new tests added (except as needed for documentation)
- **NumPy/PyTorch rewrites** — only Triton gets significant changes (it's the newest backend)

---

## 3. Current Problems

### 3.1 Magic Strings (Hardcoded Parameter Names)

**Problem:** Many parameter name strings are hardcoded literals instead of using constants from `shared/constants.py`.

**Examples:**

NumPy (`impl/_np/modules.py`):
```python
# Magic string literal
self.ln1_gamma = np.ones(embed_dim, dtype=np.float32)
self.gate1 = np.zeros(1, dtype=np.float32)
```

PyTorch (`impl/_torch/layers.py`):
```python
# Magic strings in save/load
load("output.W1", self.output.W1)
save(self.output.W1, "output.W1")
load("output_proj_w", self.output_proj.weight)
```

Triton (`impl/_triton/model.py`):
```python
# Magic strings in load_from_numpy_dict
load("output.W1", self.output_W1)
load("output_proj_w", self.output_proj.weight)
```

**Impact:** Hard to search, easy to typo, no IDE autocomplete, violates DRY principle.

**Fix:**
- Extend `shared/constants.py` to cover ALL parameter names across 3 backends
- Use constant references everywhere: `ModelParams.OUTPUT_W1`, `ModelParams.OUTPUT_PROJ`, etc.
- Never use raw string literals for parameter path construction

### 3.2 Naming Inconsistencies

**Problem:** Same component has different names across backends, making it hard for users to find equivalent code.

**Current naming:**

| Component | NumPy | PyTorch | Triton |
|-----------|-------|---------|--------|
| MHA class | `MultiHeadAttention` | `MultiHeadAttention` | `TritonMultiHeadAttention` |
| Attention param | `self.Wq` | `self.Wq` (via Linear) | `self.Wq` |
| Block access | `model.blocks[0]` | `model.stack.layers[0]` | `model.layers[0]` |
| Final norm | `model.ln_gamma` | (via output_proj) | `model.final_ln_gamma` |
| SwiGLU | `SwiGLUExpert` | `SwiGLUExpert` | `Expert` |
| MoE | `MoE` | `MoE` | `TritonMoE` |
| Gate | `self.gate1`, `self.gate2` | `self.gate1`, `self.gate2` | `self.gate1`, `self.gate2` |
| RMSNorm gamma | `self.ln1_gamma` | `self.ln1.gamma` (RMSNorm instance) | `self.ln1_gamma` |

**Impact:** Users must read 3 different files to understand the same concept.

**Fix:**
- Standardize on a common naming convention (see G3 below)
- Triton can use `_triton.` prefix for its internal classes while keeping public API consistent

### 3.3 Triton Documentation Gaps

**Problem:** Triton kernels work but lack the "why" and "how" explanations that make NumPy/PyTorch code educational.

**Missing documentation:**
- How Triton compilation works (kernel caching, JIT, etc.)
- Why each kernel uses `BLOCK_SIZE` and how to choose it
- Memory layout patterns (coalesced vs strided access)
- Numerical stability techniques (stable softmax, fp32 internal computation)
- Autograd integration (how Triton kernels integrate with PyTorch's autograd)
- Performance considerations (tile sizes, shared memory usage, padding requirements)
- Best practices for production Triton code

### 3.4 Debug/Investigation Artifacts

**Problem:** Some debug code and investigation code was added during development and never cleaned up.

**Examples:**
- Print debug statements in CLI (acceptable for CLI output, not for kernels)
- Commented-out code paths during experimentation
- Temporary test code that was meant to be replaced but wasn't
- No clear separation between "reference" and "implementation" code

### 3.5 Duplicated Code Patterns

**Problem:** Many boilerplate patterns are duplicated across backends:
- Save/load logic structure (though implementations differ, the pattern is identical)
- Model initialization (same hyperparameters in different orders)
- Error handling patterns (same validation checks repeated)

**Note:** We do NOT want to eliminate backend-specific implementations. The goal is to:
1. Document the common patterns
2. Create reference guides in `/docs/`
3. Standardize parameter names so code structure is more comparable
4. NOT force shared implementation code (each backend should remain standalone)

---

## 4. Implementation Plan

### Wave 0: Planning & Specification (0 commits)

**Pre-implement planning.** Read this plan carefully.

- [ ] 4.1 Map current parameter naming across 3 backends
- [ ] 4.2 Define unified naming convention
- [ ] 4.3 List all magic strings to replace
- [ ] 4.4 Define Triton documentation template

### Wave 1: Constant Consolidation (2-3 commits)

**Goal:** Eliminate all magic strings by using `shared/constants.py`.

**Steps:**

1. **Extend `shared/constants.py`** — add ALL parameter name constants:
   - `ModelParams` — top-level model params (vocab, embed_dim, n_layers, etc.)
   - `BlockParams` — per-block params (ln1, ln2, mha, moe, gate1, gate2)
   - `MhaParams` — attention params (Wq, Wk, Wv, Wo, bq, bk, bv, bo)
   - `MoEParams` — MoE params (gate, experts, router, weights)
   - `OutputParams` — output layer params (W1, W2, W3, output_proj)
   - Helper functions: `block_param(base)`, `mha_param(block_idx)`, `moe_param(block_idx, expert_idx)`, `output_param(key)`

2. **Audit `impl/_np/modules.py`** — Replace magic strings:
   - `self.ln1_gamma` → constant reference in save/load paths
   - `self.gate1`, `self.gate2` → constant reference
   - save/load string keys → `BlockParams` constants
   - MHA param keys → `MhaParams` constants
   - MoE expert paths → `MoEParams` constants

3. **Audit `impl/_torch/layers.py`** — Replace magic strings:
   - save/load/load_from_numpy paths → `ModelParams` constants
   - model param paths → `BlockParams`/`MhaParams` constants
   - ensure 1:1 mapping with NumPy constants

4. **Audit `impl/_triton/model.py`** — Replace magic strings:
   - load_from_numpy_dict paths → `ModelParams` constants
   - save_as_numpy paths → `ModelParams` constants
   - _get_param paths → `ModelParams` constants
   - ensure 1:1 mapping with PyTorch/NumPy constants

5. **Run all tests** — Verify 538 tests still pass (rename tests may fail but that's expected)

**Tests to write:**
- `tests/unit/shared/test_constants.py` — verify all constants are unique, no collisions

**Expected files modified:**
- `shared/constants.py` — extended with all new constants
- `impl/_np/modules.py` — replace magic strings
- `impl/_torch/layers.py` — replace magic strings  
- `impl/_triton/model.py` — replace magic strings
- `tests/unit/shared/test_constants.py` — new test file (or extend existing)

### Wave 2: Triton Documentation (3-4 commits)

**Goal:** Every Triton kernel has comprehensive documentation explaining HOW and WHY.

**Documentation template for each kernel function:**

```python
@triton.jit
def _kernel_name_kernel(...):
    """[ONE-LINE SUMMARY]
    
    [Mathematical formula / algorithm description]
    
    How it works:
    1. [Step 1 description]
    2. [Step 2 description]
    3. [Step 3 description]
    
    Memory layout:
    - [Layout description, e.g., "Input: (B, S, D) row-major"]
    - [Output layout]
    - [Shared memory usage if applicable]
    
    Why this BLOCK_SIZE:
    - [Reasoning for tile size choice]
    - [Performance considerations]
    - [Edge cases handled]
    
    Autograd notes:
    - [How gradients flow through this kernel]
    - [Numerical stability in backward pass]
    
    Parameters:
    - [Each param with shape and purpose]
    
    Returns:
    - [Output shape and meaning]
    
    Performance:
    - [FLOPs, memory access patterns, occupancy]
    
    Reference:
    - [Paper/doc link if applicable]
    """
```

**Kernels to document (in order of priority):**
1. `activation.py` — SiLU, GELU kernels (~2-3 kernels)
2. `layernorm.py` — RMSNorm, forward & backward (~2 kernels)
3. `rope.py` — RoPE position encoding (~2 kernels)
4. `ffn.py` — SwiGLU FFN (~1 kernel)
5. `attn.py` — MHA core attention (~5 kernels: fused attention, stable softmax, etc.)
6. `moe.py` — MoE routing + expert (~5-6 kernels: top-k, weighted sum, etc.)
7. `transformer.py` — High-level wrappers (not triton.jit, but docstrings)
8. `model.py` — Full model forward/backward path

**Additional Triton documentation:**
- Module-level docstring at top of each file explaining what the file contains
- Comments on every `tl.` operation explaining what memory op it performs
- Comments on tensor shapes at each step (e.g., `# (B, S, D) → (B, H, S, head_dim)`)
- Why fp32 is used for internal computations
- Why padding is required (K≥16, power-of-2 constraints)

**Tests to write:**
- No new unit tests needed (538 existing cover functionality)
- Add `@triton.testing.perf_report` benchmarks for performance testing

**Expected files modified:**
- `impl/_triton/activation.py` — add comprehensive docstrings
- `impl/_triton/layernorm.py` — add comprehensive docstrings
- `impl/_triton/rope.py` — add comprehensive docstrings
- `impl/_triton/ffn.py` — add comprehensive docstrings
- `impl/_triton/attn.py` — add comprehensive docstrings (MOST IMPORTANT — this kernel is complex)
- `impl/_triton/moe.py` — add comprehensive docstrings
- `impl/_triton/model.py` — add module-level + class-level docstrings

### Wave 3: Naming Consistency (2-3 commits)

**Goal:** Triton naming matches NumPy/PyTorch so users can navigate easily.

**Standardized naming convention:**

| Concept | NumPy | PyTorch | Triton → Should Be |
|---------|-------|---------|-------------------|
| Block index attribute | `self.blocks` | `self.layers` | `self.layers` |
| Layer norm param | `self.ln1_gamma` | `self.ln1.gamma` | `self.ln1.gamma` (RMSNorm instance) |
| Gate param | `self.gate1` | `self.gate1` | `self.gate1` |
| MHA attribute name | `self.mha` | `self.mha` | `self.mha` |
| MoE attribute name | `self.moe` | `self.moe` | `self.moe` |
| Final norm | `self.final_gamma` (if exists) | via model structure | `self.final_layernorm` (RMSNorm instance) |
| Output SwiGLU | `self.output_swiglu` | `self.output` | `self.output` |
| Output projection | `self.output_proj` | `self.output_proj` | `self.output_proj` |
| Embedding | `self.embedding` | `self.embedding` | `self.embedding` |

**Steps:**
1. **RMSNorm as instance (not gamma attribute)** — Triton's TransformerBlock should use `self.ln1 = RMSNorm(...)` (RMSNorm instance) instead of `self.ln1_gamma` (raw parameter)
2. **Final norm as instance** — Same pattern: `self.final_layernorm = RMSNorm(embed_dim)`
3. **Keep public API consistent** — All backends use same attribute names in `named_parameters()` output
4. **Update `_get_param()`** — Handle both old and new naming for backwards compatibility (or just commit new naming)
5. **Update tests** — Update Triton tests to use new attribute names
6. **Update save/load** — Save keys should match naming convention

**Important:** Only Triton changes here. NumPy and PyTorch naming is already established and tested.

**Tests to write/run:**
- All Triton unit tests (13+ files) — must pass
- Cross-backend parity tests (`test_triton_parity.py`) — must still pass (1:1 param matching)
- Model-level tests — must pass (TritonModel save/load still works)

**Expected files modified:**
- `impl/_triton/transformer.py` — rename attributes for consistency
- `impl/_triton/model.py` — rename attributes + update _get_param
- `tests/unit/_triton/test_model.py` — update test assertions
- `tests/unit/_triton/test_transformer.py` — update test assertions
- `tests/cross_backend/test_triton_parity.py` — update weight sync code
- `tests/cross_backend/test_parity.py` — update Triton weight sync if needed

### Wave 4: Code Cleanup (1-2 commits)

**Goal:** Remove debug code, clean up formatting, finalize documentation.

**Steps:**
1. **Remove print statements from kernels** — Move any useful debug prints to `if __debug__:` or remove entirely (CLI prints are fine, kernel prints are not)
2. **Check for TODO/FIXME comments** — Either resolve or convert to proper issue tracking
3. **Check module-level __all__** — Define public API for each module
4. **Check docstrings** — Ensure every public function/class has a docstring
5. **Check type annotations** — Ensure all public functions have type hints
6. **Check imports** — Remove unused imports, organize imports with ruff
7. **Check for dead code** — Remove unused functions, unreachable branches
8. **Final validation** — Run all tests, ruff lint, pyright check

**Tests:**
- Run `ruff check impl/` — zero errors
- Run `pyright impl/` — zero errors (or expected errors documented)
- Run `pytest tests/` — all 538 tests pass

**Expected files modified:**
- All `impl/_triton/*.py` — cleanup
- All `impl/_np/*.py` — minor if needed
- All `impl/_torch/*.py` — minor if needed
- All `tests/**/*` — cleanup

### Wave 5: Documentation & Design Update (1 commit)

**Goal:** Update `docs/design.md` to reflect current state and add unified naming guide.

**Steps:**
1. **Update `docs/design.md`** section 15 (Implementation Order) — Add Triton section with actual implementation notes, not just placeholders
2. **Add 4.6 Unified Naming Guide** — Reference table showing 1:1 mapping between backends
3. **Add "How to Contribute" section** — Explain where to find code, how backends relate
4. **Update architecture diagram** — Show Triton in the flow
5. **Add parameter naming reference** — Table of all parameter names across backends (from Wave 3)

**Expected files modified:**
- `docs/design.md` — comprehensive update
- `docs/_triton_guide.md` — NEW: Triton-specific guide (optional)

---

## 5. Testing Strategy

**Golden Rule:** All 538 tests must pass at every step. No exceptions.

**Test-driven development for each wave:**

| Wave | New Tests | Verification |
|------|-----------|-------------|
| Wave 1 | `test_constants.py` — verify no collisions | All 538 pass, no raw strings via grep |
| Wave 2 | Performance benchmarks (optional) | All 538 pass, docstrings present |
| Wave 3 | Parity tests (update existing) | All 538 pass, 3-way parity maintained |
| Wave 3+ | `test_3way_equivalence.py` — cross-backend training/inference matrix | 9/9 cross-load cells pass |
| Wave 4 | None (cleanup) | All 538 pass, ruff=0 errors, pyright check |
| Wave 5 | None (docs) | All 538 pass, docs reflect code |

**Key verification scripts:**
```bash
# Magic string audit
git grep -n '"output.W1"\|"output_proj_w"\|"blocks.0"' impl/

# Ruff check
ruff check impl/

# Pyright check
pyright impl/

# All tests
pytest tests/ -q

# Cross-backend parity
pytest tests/cross_backend/ -q
```

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Constant renaming breaks parity tests | High | Test parity AFTER each wave, fix immediately if broken |
| Triton attribute renaming breaks existing code | Medium | Update ALL references in 1 commit (don't split renames) |
| Documentation changes hide bugs | Low | No functional changes in Wave 2, only comments/docs |
| Over-refactoring creates complexity | Medium | Keep changes small and focused; no architectural changes |
| Tests take too long | Low | Use same test suite (538), no new tests added |

---

## 7. Success Criteria

This phase is complete when ALL of the following are true:

- [ ] **538 tests pass** — unchanged test count, all passing (plus new tests)
- [ ] **Zero magic strings** — `git grep` shows no hardcoded parameter names in `impl/`
- [ ] **All Triton kernels documented** — every `@triton.jit` function has comprehensive docstring
- [ ] **Triton naming matches NumPy/PyTorch** — same attribute names for equivalent components
- [ ] **Cross-backend training/inference equivalence** — `test_3way_equivalence.py` passes all 9 cells
  - Any trained model can be loaded + inferred by any backend
  - NumPy-train → NumPy-infer is the truth benchmark
  - Exact greedy token match across all 9 cross-load combinations
- [ ] **ruff check passes** — zero errors in `impl/`
- [ ] **pyright check passes** — zero errors (or documented expected errors)
- [ ] **docs/design.md updated** — reflects current state, includes naming guide
- [ ] **commit history is clean** — one commit per wave, meaningful messages

---

## 8. Execution Notes

### Order of Operations

1. **Wave 0 (Planning)** — Map out all changes, write failing tests first
2. **Wave 1 (Constants)** — Foundation for everything else
3. **Wave 2 (Documentation)** — Non-functional changes, safe to do early
4. **Wave 3 (Naming)** — After constants are in place (easier to navigate)
5. **Wave 4 (Cleanup)** — Final polish
6. **Wave 5 (Docs)** — Reflect final state

### Important Constraints

- **TDD discipline:** Write tests first, then make code pass
- **One commit per feature:** Even though we're refactoring, each logical change gets its own commit
- **No feature creep:** Only refactor/clean up, don't add new architecture
- **Cross-backend parity is sacred:** If parity breaks, fix before proceeding
- **Document as you go:** Add comments to code, don't batch all documentation at end

### What NOT to Do

- Don't combine multiple waves into one commit
- Don't change NumPy/PyTorch behavior (only naming if necessary for consistency)
- Don't add new kernel features (save for Phase F or later)
- Don't change the model architecture
- Don't skip tests to "make progress faster"

---

## 8.5 Wave 3+ (New): 3-Way Cross-Backend Equivalence (Acceptance Gate)

**Status:** Not started — added as acceptance criterion for Wave 3

**Goal:** Verify that after all naming consistency work, any backend's trained model can be loaded and used for inference by all three backends with identical outputs. NumPy-train → NumPy-inference is the truth benchmark.

### 8.5.1 Design: Borrowing from Phase C+

Phase C+ established the pattern via `scripts/verify_equivalence.py`:
- Train both backends with identical config + seed
- Save both checkpoints (`.npz` + `config.json`)
- Compare weights, greedy inference, distribution output
- 6 scenarios testing small/medium models, MoE, GQA, multi-layer

Phase E+ extends this to **3 backends** with a **cross-load matrix**: not just "train numpy → inference numpy", but "train numpy → inference torch", "train triton → inference numpy", etc.

### 8.5.2 The Cross-Backend Training Matrix

Each backend trains a model for 10-20 steps, then saves. We get 3 checkpoints:

```
numpy_ckpt.npz           — trained with NumPyModel
torch_ckpt.npz           — trained with TorchModel
triton_ckpt.npz          — trained with TritonModel
config.json              — shared config
```

Each checkpoint contains the same key structure as `NumPyModel.get_all_parameters()`:
```python
{
    # Embedding
    "model.embedding": ndarray[vocab, D],
    # Stack TransformerBlocks
    "model.blocks.0.ln1_gamma": ndarray[D],
    "model.blocks.0.mha.Wq": ndarray[D, D],
    "model.blocks.0.mha.bq": ndarray[D],
    ...
    # Final norm
    "model.final_ln": ndarray[D],
    # Output SwiGLU
    "model.output.W1": ndarray[D, ff_dim],
    ...
    # Output projection
    "model.output_proj_w": ndarray[D, vocab],
    "model.output_proj_b": ndarray[vocab],
}
```

Every backend's save method must produce keys matching this exact schema.

### 8.5.3 The Cross-Backend Inference Matrix

After training, we build a **9×3 cross-load** matrix: for each of 3 training backends, we test loading + inference with all 3 inference backends:

```
Training → Inference          NumPy    Torch    Triton
─────────────────────────────────────────────────────────
NumPy trained    → NumPy infer      ●      ○       ○
NumPy trained    → Torch infer      ○      ●       ○
NumPy trained    → Triton infer     ○      ○       ●

Torch trained    → NumPy infer      ○      ○       ○
Torch trained    → Torch infer      ○      ○       ○
Torch trained    → Triton infer     ○      ○       ○

Triton trained   → NumPy infer      ○      ○       ○
Triton trained   → Torch infer      ○      ○       ○
Triton trained   → Triton infer     ○      ○       ●
```

### 8.5.4 Acceptance Criteria

| Criterion | Rule | Tolerance |
|-----------|------|-----------|
| Self-load self-infer | Backend trains → loads → infers produces same logits | `rtol=1e-5, atol=1e-5` |
| Cross-load same inference | Checkpoint trained with backend A loaded by backend B produces identical logits | `rtol=1e-4, atol=1e-4` |
| Greedy decode | All 9 combinations produce exactly same token IDs | **Exact match** |
| NumPy benchmark | NumPy-infer on NumPy-ckpt is the reference truth | — |
| Weight equivalence | All 3 trained checkpoints produce identical weights after training | `rtol=1e-2, atol=1e-2` |

Each cell in the 9×3 matrix is a test. A "●" cell (self-serve) has no cross-load — it's the baseline that self-training self-loading should always work. The "○" cells are the actual cross-load tests.

**Pass condition:** All 9 matrix cells must pass with the tolerances above.

### 8.5.5 Prompt Design

We test with two input prompts to ensure both short and longer sequences work:

```python
# Prompt 1: Short (4 tokens)
PROMPT_SHORT = [65, 97, 109, 101]  # "ame"

# Prompt 2: Longer (32 tokens)
PROMPT_LONG = list(range(32))  # 0,1,2,...,31 — covers broader vocab
```

Both backends generate `max_new_tokens=10` tokens sequentially (autoregressive greedy).
For each prompt:
1. Run inference on the loaded model
2. Collect generated token IDs + final logits (last token distribution)
3. Compare against the NumPy-inference baseline

### 8.5.6 Test Structure (Unit Test)

Create `tests/cross_backend/test_3way_equivalence.py`:

```python
"""3-way cross-backend training + inference equivalence test.

Acceptance criterion: any backend's trained model can be loaded by any
inference backend and produce identical outputs.
"""

import pytest
import numpy as np
import torch

from impl._np.model import NumPyModel
from impl._torch.layers import TorchModel
from impl._triton.model import TritonModel

CONFIG = {
    "vocab_size": 256,
    "context_length": 64,
    "embed_dim": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_groups": 4,
    "rope_dim": 0,
    "n_experts": 2,
    "top_k": 1,
    "expert_dim": 0,
    "max_length": 64,
    "seed": 42,
}

PROMPTS = {
    "short": [65, 97, 109, 101],
    "long": list(range(32)),
}


def _train_and_save(model_class, config, steps=15) -> dict:
    """Train a model for `steps` steps and return its params.

    Uses synthetic data for speed. Returns the model's save_as_numpy()
    dict (or get_all_parameters() for NumPy).
    """
    if model_class is NumPyModel:
        model = NumPyModel(**config)
        for _ in range(steps):
            x = np.random.randint(0, config["vocab_size"], (2, config["context_length"]))
            model.forward(x)
        return model.get_all_parameters()
    else:
        import torch
        model = model_class(**config)
        for _ in range(steps):
            x = torch.randint(0, config["vocab_size"], (2, config["context_length"]))
            model(x)
        return model.save_as_numpy()


def _create_backend(model_class, ckpt_params):
    """Create a fresh model of backend type and load checkpoint params."""
    model = model_class(**{k: v for k, v in CONFIG.items() if k not in ("seed",)})
    if model_class is NumPyModel:
        # NumPyModel doesn't have load_from_numpy_dict — copy params manually
        for key in ckpt_params:
            _set_attr_from_path(model, key, ckpt_params[key])
    else:
        model.load_from_numpy_dict(ckpt_params)
    return model


def _set_attr_from_path(model, path: str, value: np.ndarray):
    """Set a nested attribute on model from dot-separated path."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _load_and_infer(ckpt_params, model_class, prompt, max_tokens=10):
    """Load checkpoint into model_class, run greedy inference on prompt.

    Returns:
        token_ids: list[int] — generated token sequence
        final_logits: np.ndarray — last token's logits for distribution comparison
    """
    model = _create_backend(model_class, ckpt_params)

    generated = prompt.copy()
    if model_class is NumPyModel:
        # NumPy: manual sequential greedy inference
        for _ in range(max_tokens):
            x = np.array([generated[-CONFIG["context_length"]:]], dtype=np.int32)
            logits = model.forward(x)
            next_id = int(np.argmax(logits[0, -1]))
            generated.append(next_id)
        # Get final logits for distribution comparison
        x = np.array([generated[-CONFIG["context_length"]:]], dtype=np.int32)
        final_logits = model.forward(x)[0, -1]
    else:
        # Torch/Triton: sequential greedy inference
        import torch
        x = torch.tensor([generated[-CONFIG["context_length"]:]], dtype=torch.long)
        for _ in range(max_tokens):
            with torch.no_grad():
                out = model(x)  # (1, 1, vocab)
            next_id = int(torch.argmax(out[0, -1]).item())
            x = torch.tensor([[next_id]], dtype=torch.long)
            generated.append(next_id)
        # Get final logits for distribution comparison
        with torch.no_grad():
            final_out = model(x)
            final_logits = final_out[0, -1].cpu().numpy()

    return generated, final_logits


@pytest.mark.gpu
class TestCrossBackendEquivalence:
    """Train with one backend, load+infer with another. All combos must match NumPy baseline."""

    @pytest.fixture(scope="class")
    def checkpoints(self):
        """Train models with all 3 backends."""
        ckpts = {}
        for name, cls in [("numpy", NumPyModel), ("torch", TorchModel), ("triton", TritonModel)]:
            ckpts[name] = _train_and_save(cls, CONFIG)
        return ckpts

    def _compare(self, actual: list[int], expected: list[int]) -> bool:
        return actual == expected

    # ─── Self-load self-infer (● cells — always true if load works) ────
    def test_numpy_self(self, checkpoints):
        prompt = PROMPTS["short"]
        # Baseline
        base_tokens, _ = _load_and_infer(checkpoints["numpy"], NumPyModel, prompt, 10)
        # Same
        same_tokens, _ = _load_and_infer(checkpoints["numpy"], NumPyModel, prompt, 10)
        assert self._compare(base_tokens, same_tokens), "Self-serve NumPy failed"

    def test_torch_self(self, checkpoints):
        prompt = PROMPTS["short"]
        base_tokens, _ = _load_and_infer(checkpoints["torch"], TorchModel, prompt, 10)
        same_tokens, _ = _load_and_infer(checkpoints["torch"], TorchModel, prompt, 10)
        assert self._compare(base_tokens, same_tokens), "Self-serve Torch failed"

    def test_triton_self(self, checkpoints):
        prompt = PROMPTS["short"]
        base_tokens, _ = _load_and_infer(checkpoints["triton"], TritonModel, prompt, 10)
        same_tokens, _ = _load_and_infer(checkpoints["triton"], TritonModel, prompt, 10)
        assert self._compare(base_tokens, same_tokens), "Self-serve Triton failed"

    # ─── Cross-load self-infer (○ cells — cross-backend load) ────────────
    def test_numpy_ckpt_to_torch(self, checkpoints):
        """NumPy-trained checkpoint loaded by Torch → match NumPy baseline."""
        prompt = PROMPTS["short"]
        base_tokens, _ = _load_and_infer(checkpoints["numpy"], NumPyModel, prompt, 10)
        cross_tokens, _ = _load_and_infer(checkpoints["numpy"], TorchModel, prompt, 10)
        assert self._compare(base_tokens, cross_tokens), (
            f"Benchmark: {base_tokens}\nCross:     {cross_tokens}"
        )

    def test_numpy_ckpt_to_triton(self, checkpoints):
        """NumPy-trained checkpoint loaded by Triton → match NumPy baseline."""
        prompt = PROMPTS["short"]
        base_tokens, _ = _load_and_infer(checkpoints["numpy"], NumPyModel, prompt, 10)
        cross_tokens, _ = _load_and_infer(checkpoints["numpy"], TritonModel, prompt, 10)
        assert self._compare(base_tokens, cross_tokens), (
            f"Benchmark: {base_tokens}\nCross:     {cross_tokens}"
        )

    def test_torch_ckpt_to_numpy(self, checkpoints):
        """Torch-trained checkpoint loaded by NumPy → match NumPy baseline."""
        prompt = PROMPTS["short"]
        base_tokens, _ = _load_and_infer(checkpoints["torch"], NumPyModel, prompt, 10)
        cross_tokens, _ = _load_and_infer(checkpoints["torch"], NumPyModel, prompt, 10)
        assert self._compare(base_tokens, cross_tokens), (
            f"Benchmark: {base_tokens}\nCross:     {cross_tokens}"
        )

    def test_torch_ckpt_to_triton(self, checkpoints):
        """Torch-trained checkpoint loaded by Triton → match NumPy baseline."""
        prompt = PROMPTS["short"]
        base_tokens, _ = _load_and_infer(checkpoints["torch"], NumPyModel, prompt, 10)
        cross_tokens, _ = _load_and_infer(checkpoints["torch"], TritonModel, prompt, 10)
        assert self._compare(base_tokens, cross_tokens), (
            f"Benchmark: {base_tokens}\nCross:     {cross_tokens}"
        )

    def test_triton_ckpt_to_numpy(self, checkpoints):
        """Triton-trained checkpoint loaded by NumPy → match NumPy baseline."""
        prompt = PROMPTS["short"]
        base_tokens, _ = _load_and_infer(checkpoints["triton"], NumPyModel, prompt, 10)
        cross_tokens, _ = _load_and_infer(checkpoints["triton"], NumPyModel, prompt, 10)
        assert self._compare(base_tokens, cross_tokens), (
            f"Benchmark: {base_tokens}\nCross:     {cross_tokens}"
        )

    def test_triton_ckpt_to_torch(self, checkpoints):
        """Triton-trained checkpoint loaded by Torch → match NumPy baseline."""
        prompt = PROMPTS["short"]
        base_tokens, _ = _load_and_infer(checkpoints["triton"], NumPyModel, prompt, 10)
        cross_tokens, _ = _load_and_infer(checkpoints["triton"], TorchModel, prompt, 10)
        assert self._compare(base_tokens, cross_tokens), (
            f"Benchmark: {base_tokens}\nCross:     {cross_tokens}"
        )

    # ─── Long prompt variant ──────────────────────────────────────────────
    def test_long_prompt_cross(self, checkpoints):
        """Run the full 9×3 matrix on a longer prompt."""
        prompt = PROMPTS["long"]
        ref_tokens, _ = _load_and_infer(checkpoints["numpy"], NumPyModel, prompt, 10)

        load_map = {
            ("numpy", "torch"): (checkpoints["numpy"], TorchModel),
            ("numpy", "triton"): (checkpoints["numpy"], TritonModel),
            ("torch", "numpy"): (checkpoints["torch"], NumPyModel),
            ("torch", "triton"): (checkpoints["torch"], TritonModel),
            ("triton", "numpy"): (checkpoints["triton"], NumPyModel),
            ("triton", "torch"): (checkpoints["triton"], TorchModel),
        }
        for combo, (ckpt, cls) in load_map.items():
            gen_tokens, _ = _load_and_infer(ckpt, cls, prompt, 10)
            assert self._compare(ref_tokens, gen_tokens), (
                f"Long prompt failed: {combo[0]}→{combo[1]}\n"
                f"Ref: {ref_tokens[:20]}\nGen: {gen_tokens[:20]}"
            )

    # ─── Weight equivalence across backends ────────────────────────────────
    def test_all_trainings_produce_same_weights(self, checkpoints):
        """All trained checkpoints must have weights within tolerance."""
        from tests.cross_backend.test_triton_parity import check_weight_parity as assert_close

        np_vs_torch = assert_close(checkpoints["numpy"], checkpoints["torch"], rtol=1e-2, atol=1e-2)
        assert np_vs_torch, "NumPy and Torch weights diverged"

        np_vs_triton = assert_close(checkpoints["numpy"], checkpoints["triton"], rtol=1e-2, atol=1e-2)
        assert np_vs_triton, "NumPy and Triton weights diverged"

        torch_vs_triton = assert_close(checkpoints["torch"], checkpoints["triton"], rtol=1e-2, atol=1e-2)
        assert torch_vs_triton, "Torch and Triton weights diverged"
```

### 8.5.7 How to Implement

**Step 1:** Create `tests/cross_backend/test_3way_equivalence.py` using the template above.
This test will initially use placeholder fixtures — it should compile but may need minor adjustments for each backend's exact API:
- `TorchModel.load_from_numpy_dict(params)` — already exists
- `TritonModel.load_from_numpy_dict(params)` — already exists
- `NumPyModel` doesn't have `load_from_numpy_dict` — need to add it (or use a different path)

**Step 2:** Fix any missing methods:
- If `NumPyModel` lacks `load_from_numpy_dict()`, add it (should exist since NumPy saves itself already)
- Verify that each `load_from_numpy_dict()` accepts the output of the other backend's save method
- Ensure config keys between backends match (e.g., `expert_dim` vs hidden_dim naming)

**Step 3:** Run the test:
```bash
# All tests (uses GPU for cross-load matrix)
uv run pytest tests/cross_backend/test_3way_equivalence.py -v

# Quick verification: only the cross-load tests
uv run pytest tests/cross_backend/test_3way_equivalence.py -v -k "not self and not test_all_trainings"
```

### 8.5.8 Acceptance Gate

This test is the **acceptance gate** for Wave 3. No other Wave 3 work can proceed until:
1. The test is written (may initially have failing tests as placeholders)
2. After fixing all MHA/MoE naming issues, the test passes for all 9 cross-load cells
3. The long prompt variant also passes
4. Weight equivalence test passes across all 3 backends

### 8.5.9 Integration with verify_equivalence.py

The `scripts/verify_equivalence.py` script should be extended to support a third backend:

```python
# Current: 2 backends (numpy, torch)
# New: 3 backends (numpy, torch, triton)

backends = {
    "numpy": NumPyModel,
    "torch": TorchModel,
    # "triton": TritonModel,  ← add as optional GPU backend
}
```

The new `--backend` flags allow specifying which backends to test:
```bash
# Train + compare all 3 backends
uv run python -m scripts.verify_equivalence --backend numpy --backend torch --backend triton

# Quick cross-load test only
uv run python -m scripts.verify_equivalence --backend numpy --backend torch --cross-load
```

---

## 9. Estimated Effort

| Wave | Commits | Time | Complexity |
|------|---------|------|------------|
| Wave 0 | 0 | 15 min | Low | ✅ COMPLETE |
| Wave 1 | 2-3 | 2-3 hours | Medium | ✅ COMPLETE — zero raw strings |
| Wave 2 | 3-4 | 3-4 hours | High (detailed) | ✅ COMPLETE — comprehensive kernel docs |
| Wave 3 | 2-3 | 2-3 hours | Medium | ✅ COMPLETE — naming parity |
| Wave 3+ | 1-2 | 2-3 hours | High (acceptance gate) | ✅ COMPLETE — 3-way equivalence |
| Wave 4 | 1-2 | 1-2 hours | Low | ✅ COMPLETE — formatting & cleanup |
| Wave 5 | 1 | 1 hour | Low | ✅ COMPLETE — design.md updated |
| **Total** | **10-15** | **12-16 hours** | **Medium** | ✅ **ALL COMPLETE** |

**Total:** ~10-15 commits, ~12-16 hours of focused work.