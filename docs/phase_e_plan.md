# Phase E: Triton GPU Kernel Implementation — Execution Plan

**Status:** 🔲 NOT STARTED
**Start Date:** —
**End Date:** —
**Total Stages:** 12 sub-phases (sequential, one at a time)

## Progress Summary

| Stage | Status | Tests | Description |
|-------|--------|-------|-------------|
| E0: Project scaffolding | 🔲 Not started | 0 | Directories + import test |
| E1: SiLU activation kernel | 🔲 Not started | ~4 | Stateless element-wise |
| E2: RMSNorm kernel | 🔲 Not started | ~8 | Layer normalization |
| E3: RoPE kernel | 🔲 Not started | ~6 | Rotary position embeddings |
| E4: SwiGLU FFN kernel | 🔲 Not started | ~6 | Gated feedforward |
| E5: MHA kernel | 🔲 Not started | ~10 | Multi-head attention |
| E6: MoE kernel | 🔲 Not started | ~6 | Expert routing |
| E7: TransformerBlock | 🔲 Not started | ~5 | Attention + MoE assembly |
| E8: DecoderStack | 🔲 Not started | ~3 | Stacked blocks |
| E9: Full TritonModel | 🔲 Not started | ~5 | Full model integration |
| E10: Inference + Training | 🔲 Not started | ~5 | Pipeline scripts |
| E11: Cross-backend parity | 🔲 Not started | ~8 | NumPy/PyTorch vs Triton |

---

## Goal

Build **Triton GPU kernels** that replace PyTorch's built-in operations while producing **numerically identical** results at `float64` precision. The model architecture, save/load format, and API remain identical to `impl/_torch/` — only the compute kernels differ.

**Principle:** Write the failing test first, make it pass, move to the next. Never reason about correctness — test results are the source of truth.

---

## Architecture Overview

```
Same model architecture, different compute backend:

TritonModel (wraps PyTorch nn.Module)
├── embedding          → nn.Embedding (no Triton need)
├── stack.layers[i]    → TritonTransformerBlock
│   ├── mha            → TritonMHA kernel (attention)
│   ├── rope           → TritonRoPE kernel
│   └── moe.experts    → TritonSwiGLU kernel + Triton routing
├── final_ln           → TritonRMSNorm kernel
└── output             → TritonSiLU + TritonGatedLinear
```

**Key design:** `impl/_triton/` provides standalone `@triton.jit` kernels. A thin PyTorch wrapper (`impl/_triton/model.py`) assembles them into `TritonModel`, a subclass of `torch.nn.Module`, making it compatible with the existing training/inference scripts.

---

## Directory Structure

```
impl/
├── _triton/                    # NEW — Triton GPU kernels
│   ├── __init__.py             # Package init + public API
│   ├── activation.py           # SiLU, gating kernels
│   ├── layernorm.py            # RMSNorm kernel
│   ├── rope.py                 # RoPE kernel
│   ├── swiglu.py               # SwiGLU FFN kernel
│   ├── attention.py            # MHA kernel (full attention + GQA)
│   ├── moe.py                  # MoE routing + expert computation
│   ├── transformer.py          # TransformerBlock wrapper (Python)
│   ├── model.py                # TritonModel → PyTorch nn.Module wrapper
│   ├── inference.py            # Inference engine (same API as _torch)
│   └── training.py             # Training loop (same API as _torch)
│
tests/
├── unit/
│   └── _triton/                # NEW — Triton-specific tests
│       ├── __init__.py
│       ├── test_activation.py   # SiLU, gating tests
│       ├── test_layernorm.py    # RMSNorm tests
│       ├── test_rope.py         # RoPE tests
│       ├── test_swiglu.py       # SwiGLU FFN tests
│       ├── test_attention.py    # MHA kernel tests
│       ├── test_moe.py          # MoE kernel tests
│       ├── test_transformer.py  # TransformerBlock tests
│       ├── test_model.py        # Full model tests
│       └── test_inference.py    # Inference tests
│
├── cross_backend/
│   └── test_parity_triton.py   # NEW — NumPy/PyTorch vs Triton parity
│
scripts/
└── (no changes needed — train.py uses --backend flag)
```

---

## TDD Process

Each stage follows the same exact pattern. **One test file, one commit, one thing at a time.**

```
Step 1: Write test → ALL FAIL
   Write test file. Run:
   PYTHONPATH=shared PYTHONPATH=impl uv run pytest tests/unit/_triton/test_*.py -v --timeout=30
   Check: ALL tests fail (MissingFile, MissingFunction, or actual assertion failures).
   If any test passes unexpectedly — the task isn't done.

Step 2: Implement → all PASS
   Write the minimal Triton kernel implementation.
   Run: ALL pass.
   If any test fails — fix it, don't skip it.

Step 3: Quality check
   PYTHONPATH=shared PYTHONPATH=impl ruff check impl/_triton/<file>.py tests/unit/_triton/test_<file>.py
   PYTHONPATH=shared PYTHONPATH=impl pyright impl/_triton/<file>.py tests/unit/_triton/test_<file>.py

Step 4: Commit
   git add -A && git commit -m "e<stage>: <component> — <N> tests pass"
```

**Rules:**
- NEVER implement without failing tests first
- ONE component per commit (atomic)
- ONE test file per component
- Tests must run on CPU for parity comparisons (`float64`), GPU for correctness
- Skip tests when no GPU is available (`pytest.skip(reason="No GPU")`) — but always check
- **Tolerance policy:**
  - Standalone kernels (tested in isolation): `rtol=1e-4, atol=1e-4`
  - Kernels in single chain: `rtol=1e-3, atol=1e-3`
  - Full model (multi-layer chain): `rtol=1e-2, atol=1e-2`

---

## Stages (12 sub-phases)

### Phase E0: Project scaffolding

**What it does:** Create the directory and package structure. Zero tests expected.

- [ ] **E0.1** | `impl/_triton/`, `tests/unit/_triton/`, `tests/unit/_triton/__init__.py` | pytest discovers 0 tests | 0

No implementation code. Just directories.

- [ ] **E0.2** | `impl/_triton/__init__.py` + `tests/unit/_triton/test_import.py` | import test fails with ModuleNotFoundError | 0

This is the standard "test-first" scaffolding step. The test imports `impl._triton` and fails because the package structure doesn't exist yet, then you create it and the test passes.

```python
# tests/unit/_triton/test_import.py
import pytest

class TestImport:
    @pytest.mark.timeout(10)
    def test_triton_package(self):
        import impl._triton
        assert hasattr(impl._triton, "__file__")
```

**Gate:** 1 test → commit.

---

### Phase E1: SiLU Activation Kernel

**What it does:** Element-wise `x * sigmoid(x)`. Simplest possible Triton kernel — pure element-wise mapping with no cross-element communication. This is the warm-up to learn the Triton DSL patterns.

**Why first?** Stateless, no dependencies, tests are trivial. Confirms the Triton environment is working.

```python
class TestSiLUCKernel:
    def test_output_shape(self, device):
        # input [B, S, D] → output [B, S, D]
        ...

    def test_output_at_zero(self, device):
        # SiLU(0) = 0 * 0.5 = 0.
        ...

    def test_output_range_large_positive(self, device):
        # SiLU(10) ≈ 10 (near-identity)
        ...

    def test_output_range_negative(self, device):
        # SiLU(-10) ≈ -10 * e^(-10) ≈ 0 (suppressed)
        ...

    def test_gradient_correct(self, device):
        # autograd produces correct gradients through the kernel
        ...

    def test_parity_with_numpy(self, device):
        # Same float64 input → same output as NumPy (rtol=1e-4, atol=1e-4)
        ...

    def test_parity_with_torch(self, device):
        # Same float64 input → same output as torch.nn.SiLU()
        ...
```

**What to implement:**
```python
# impl/_triton/activation.py

@triton.jit
def _silu_kernel(
    x_ptr, y_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel: y = x * sigmoid(x)."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = x * tl.sigmoid(x)  # or: x * (1.0 / (1.0 + tl.exp(-x)))
    tl.store(y_ptr + offsets, y)
```

Plus a Python wrapper to dispatch the kernel on a 1D view of the tensor.

**Gate:** 6-7 tests → commit.

---

### Phase E2: RMSNorm Kernel

**What it does:** Normalize to unit variance, scale by learned gamma. Requires cross-element communication (reduction over the last dimension). This is the first kernel with a reduction operation.

```python
class TestRMSNormKernel:
    def test_output_shape(self, device):
        # input [B, S, D], gamma [D] → output [B, S, D]
        ...

    def test_unit_variance(self, device):
        # After normalization, per-feature mean squared ≈ 1
        ...

    def test_identity_without_gamma(self, device):
        # With gamma=1, output ≈ normalized input
        ...

    def test_learned_scale(self, device):
        # gamma controls output magnitude
        ...

    def test_gradient_shape(self, device):
        # Gradient w.r.t. input has same shape as input
        ...

    def test_gradient_correct(self, device):
        # autograd gradient check
        ...

    def test_parity_with_numpy(self, device):
        # Same float64 input → same output as NumPy RMSNorm (rtol=1e-4)
        ...

    def test_parity_with_torch(self, device):
        # Same float64 input → same output as torch.layer_norm(..., normalized_shape=[D], gamma, beta=0)
        ...
```

**What to implement:**
```python
# impl/_triton/layernorm.py

@triton.jit
def _rmsnorm_kernel(
    x_ptr, gamma_ptr, y_ptr,
    N, BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel: y = x / sqrt(mean(x^2)) * gamma."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=offsets < N)
    
    # Compute RMS (reduction over last dim happens in caller via block scheduling)
    mean_square = tl.sum(x * x) / N  (This is per-reduction block)
    rms = tl.sqrt(mean_square + 1e-6)
    
    # Load gamma and scale
    gamma = tl.load(gamma_ptr + offsets, mask=offsets < N)
    y = (x / rms) * gamma
    
    tl.store(y_ptr + offsets, y, mask=offsets < N)
```

Python wrapper handles block scheduling for large tensors.

**Gate:** 8 tests → commit.

---

### Phase E3: RoPE Kernel

**What it does:** Rotary position embeddings — rotate (odd, even) pairs of head dimensions by position-dependent angles. Requires trigonometric computations and index manipulation.

```python
class TestRoPEKernel:
    def test_output_shape(self, device):
        # Q/K [B, H, S, D] → output [B, H, S, D]
        ...

    def test_rotates_by_position(self, device):
        # Position 0 and position 1 produce different rotations
        ...

    def test_full_vs_partial(self, device):
        # rope_dim=0 (full) vs rope_dim<D (partial) behave correctly
        ...

    def test_gradient_flow(self, device):
        # All Q and K elements get non-zero gradients
        ...

    def test_parity_with_numpy(self, device):
        # Same input → same output as NumPy RoPE (rtol=1e-4)
        ...

    def test_parity_with_torch(self, device):
        # Same input → same output as PyTorch RoPE implementation
        ...
```

**What to implement:**
```python
# impl/_triton/rope.py

@triton.jit
def _rope_kernel(
    x_ptr, cos_ptr, sin_ptr,
    seq_len, head_dim, pair_dim,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Triton kernel: apply 2D rotation to (odd, even) pairs."""
    # ... block-embedded kernel for (B, H, S, D) tensor
```

Python wrapper computes `cos`/`sin` tables on CPU and passes as buffers.

**Gate:** 6 tests → commit.

---

### Phase E4: SwiGLU FFN Kernel

**What it does:** `SiLU(w1 @ x) * (w3 @ x) @ w2`. Combines matrix multiplication (for which PyTorch's native `@` is fine) with Triton's SiLU kernel and element-wise multiplication.

```python
class TestSwiGLUKernel:
    def test_output_shape(self, device):
        # [B, S, D] → [B, S, D]
        ...

    def test_gating_behavior(self, device):
        # w1 and w3 projections gated together
        ...

    def test_ff_dim_independence(self, device):
        # output does not depend on ff_dim
        ...

    def test_gradient_flow(self, device):
        # All weights get non-zero gradients
        ...

    def test_parity_with_numpy(self, device):
        # Same weights + input → same output as NumPy SwiGLU (rtol=1e-4)
        ...

    def test_parity_with_torch(self, device):
        # Same weights + input → same output as PyTorch SwiGLU FFN
        ...
```

**What to implement:**
```python
# impl/_triton/swiglu.py

# SwiGLU uses a mix of:
# 1. PyTorch matmul (w1 @ x, w3 @ x) — no Triton needed for matmul
# 2. Triton SiLU kernel — replace torch.SiLU()
# 3. Element-wise multiply — native PyTorch or Triton add-on

# The Triton part is just the SiLU kernel wrapped as a function.
# The kernel is reused from activation.py.
```

**Gate:** 6 tests → commit.

---

### Phase E5: MHA Kernel

**What it does:** Scaled dot-product attention with grouped-query attention (GQA). This is the most complex Triton kernel — combining reshaping, attention computation, softmax, and GQA broadcasting.

```python
class TestMHAKernel:
    def test_output_shape(self, device):
        # X [B, S, D] → output [B, S, D]
        ...

    def test_attention_mechanism(self, device):
        # Softmax normalized across sequence dimension
        ...

    def test_gqa_support(self, device):
        # n_groups < n_heads: K/V shared across groups
        ...

    def test_gradient_flow(self, device):
        # All weight matrices get non-zero gradients
        ...

    def test_deterministic(self, device):
        # Same input → same output
        ...

    def test_full_attn_vs_gqa(self, device):
        # GQA with n_groups=n_heads produces same output as standard MHA
        ...

    def test_gradient_correct(self, device):
        # autograd gradient check against torch MHA
        ...

    def test_parity_with_numpy(self, device):
        # Same weights + input → same output as NumPy MHA (rtol=1e-3)
        ...

    def test_parity_with_torch(self, device):
        # Same weights + input → same output as PyTorch MHA
        ...

    def test_rope_integration(self, device):
        # RoPE applied to Q/K before attention produces correct scores
        ...
```

**What to implement:**
```python
# impl/_triton/attention.py

# The attention kernel replaces the core softmax(QK^T/sqrt(d)) @ V computation.
# Most of the reshaping/projection still uses PyTorch native ops.
# The Triton kernel focuses on:
# 1. The attention score computation: Q @ K^T
# 2. Stable softmax
# 3. Score @ V multiplication

@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, score_ptr,
    n_heads, seq_len, head_dim,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Triton kernel: scaled dot-product attention."""
    # Each program handles one or more heads
    # Computation: (B, S, D) → attention → (B, S, D)
```

**Note:** If the MHA kernel proves too complex for a single pass, split it into:
- E5.1: Core attention (QK^T + softmax + @V) — no GQA
- E5.2: GQA broadcasting + integration

Only split if the kernel is clearly growing too large.

**Gate:** 8-10 tests → commit.

---

### Phase E6: MoE Kernel

**What it does:** Top-k expert routing with weighted expert outputs. Requires sorting/thresholding and weighted aggregation.

```python
class TestMoEKernel:
    def test_output_shape(self, device):
        # [B, S, D] → [B, S, D]
        ...

    def test_top_k_selection(self, device):
        # Only top-k experts get non-zero weights
        ...

    def test_expert_routing(self, device):
        # Different inputs → different expert combinations
        ...

    def test_gradient_flow(self, device):
        # Gradients flow through selected experts
        ...

    def test_parity_with_numpy(self, device):
        # Same input → same output as NumPy MoE (rtol=1e-3)
        ...

    def test_parity_with_torch(self, device):
        # Same input → same output as PyTorch MoE
        ...
```

**What to implement:**
```python
# impl/_triton/moe.py

# MoE uses:
# 1. Router: PyTorch linear → softmax → top-k (native ops)
# 2. Expert computation: SwiGLU kernels from E4
# 3. Weighted aggregation: einsum (native)

# The Triton part is minimal — mostly reuses the SwiGLU kernel.
# The routing logic (top-k, softmax) works fine with native PyTorch.
```

**Gate:** 6 tests → commit.

---

### Phase E7: TransformerBlock (Python wrapper)

**What it does:** Assembles the Triton kernels into a `TransformerBlock` PyTorch module. No new Triton kernels — this is Python wiring.

```python
class TestTransformerBlock:
    def test_output_shape(self, device):
        # X [B, S, D] → output [B, S, D]
        ...

    def test_residual_connection(self, device):
        # Output contains original input (residual pass-through)
        ...

    def test_attention_and_moe(self, device):
        # Both streams contribute to output
        ...

    def test_gradient_chaining(self, device):
        # Gradients flow through all components
        ...

    def test_parity_with_torch(self, device):
        # Same weights → same output as TorchTransformerBlock (rtol=1e-3)
        ...
```

**What to implement:**
```python
# impl/_triton/transformer.py

class TritonTransformerBlock(nn.Module):
    """TransformerBlock using Triton kernels internally."""
    def forward(self, x):
        attn_out = TritonMHA(...)(x)           # Triton attention kernel
        h = x + attn_out
        h = TritonRMSNorm(...)(h)              # Triton RMSNorm kernel
        h = h + sigmoid(gate1) * h
        h = dropout(h)
        moe_out = TritonMoE(...)(h)            # Triton SwiGLU routing
        out = h + moe_out
        out = TritonRMSNorm(...)(out)
        out = out + sigmoid(gate2) * out
        out = dropout(out)
        return out
```

This is the first stage where the **entire inference path runs through Triton kernels** in each block. Tests should verify that the assembled block matches the PyTorch reference.

**Gate:** 5 tests → commit.

---

### Phase E8: DecoderStack (Python wrapper)

**What it does:** Chains `n_layers` of `TritonTransformerBlock`. Python only — no new kernels.

```python
class TestDecoderStack:
    def test_output_shape(self, device):
        # X [B, S, D] → output [B, S, D]
        ...

    def test_gradient_chaining(self, device):
        # Gradients flow through all stacked layers
        ...

    def test_parity_with_torch(self, device):
        # Same weights → same output as TorchDecoderStack (rtol=1e-2)
        ...
```

**Gate:** 3 tests → commit.

---

### Phase E9: Full TritonModel

**What it does:** Complete model with embedding → DecoderStack (Triton) → final RMSNorm (Triton) → output projection (Triton SiLU + linear). This is the first complete, runnable model using Triton kernels throughout.

```python
class TestTritonModel:
    def test_output_shape(self, device):
        # Tokens [B, S] → logits [B, S, V]
        ...

    def test_forward_pass(self, device):
        # model(tokens) produces valid logits (finite, no NaN/Inf)
        ...

    def test_backward_pass(self, device):
        # loss.backward() → all params have valid gradients
        ...

    def test_parity_with_torch(self, device):
        # Same weights → same output as TorchModel (rtol=1e-2 for 2+ layers)
        ...

    def test_save_load(self, device):
        # save_as_numpy() → load_from_numpy_dict() → same parameters
        ...
```

**What to implement:**
```python
# impl/_triton/model.py

class TritonModel(nn.Module):
    """Complete transformer — embedding + TritonDecoderStack + Triton output."""
    
    def forward(self, tokens):
        x = self.embedding(tokens)        # nn.Embedding (PyTorch native)
        x = self.stack(x)                 # TritonTransformerBlock[n_layers]
        x = TritonRMSNorm(self.final_ln_gamma)(x)   # Triton kernel
        x = TritonSiLU()(x @ self.W1) * (x @ self.W3) @ self.W2  # Triton SiLU
        logits = self.output_proj(x)      # PyTorch linear
        return logits

    def save_as_numpy(self) -> dict:
        """Save params in standard .npz format — IDENTICAL to _torch API."""
        ...

    def load_from_numpy_dict(self, params):
        """Load params from standard format — IDENTICAL to _torch API."""
        ...
```

**Critical:** The `save_as_numpy()` and `load_from_numpy_dict()` methods must produce/accept the **exact same dictionary structure** as `TorchModel`. This is what makes cross-backend equivalence possible.

**Gate:** 5 tests → commit.

---

### Phase E10: Inference + Training Scripts

**What it does:** Drop-in `impl/_triton/inference.py` and `impl/_triton/training.py` matching the `_torch` API. Also add `impl/_triton/cli.py` for `uv run python -m impl._triton.cli --help`.

```python
# Inference tests
class TestTritonInference:
    def test_output_length(self, device):
        # Generated tokens have correct length
        ...

    def test_greedy_deterministic(self, device):
        # Same prompt → same output (no randomness)
        ...

    def test_temperature_sampling(self, device):
        # Higher temperature → more diverse outputs
        ...

    def test_parity_with_torch_inference(self, device):
        # Same prompt → same tokens as TorchTextGenerator
        ...

# Training tests
class TestTritonTraining:
    def test_training_reduces_loss(self, device):
        # 20 steps of training → loss decreases
        ...

    def test_params_update(self, device):
        # Model parameters change after training
        ...

    def test_parity_with_torch_training(self, device):
        # Same initial weights → similar loss trajectory
        ...
```

**Gate:** 6-8 tests → commit.

---

### Phase E11: Cross-backend Parity Tests

**What it does:** `tests/cross_backend/test_parity_triton.py` — NumPy/PyTorch vs Triton equivalence across all test categories.

```python
# tests/cross_backend/test_parity_triton.py

class TestTritonForwardParity:
    """Standalone kernels tested in isolation: rtol=1e-4."""
    
    @pytest.mark.timeout(30)
    def test_rmsnorm_parity(self):
        # NumPy RMSNorm = Triton RMSNorm output (rtol=1e-4)
        ...

    @pytest.mark.timeout(30)
    def test_silu_parity(self):
        # NumPy SiLU = Triton SiLU output (rtol=1e-4)
        ...

    @pytest.mark.timeout(30)
    def test_rope_parity(self):
        # NumPy RoPE = Triton RoPE output (rtol=1e-4)
        ...

    @pytest.mark.timeout(30)
    def test_swiglu_parity(self):
        # NumPy SwiGLU = Triton SwiGLU output (rtol=1e-4)
        ...


class TestTritonModelParity:
    """Model-level tests: 1-layer = rtol=1e-3, 2+ layers = rtol=1e-2."""
    
    @pytest.mark.timeout(30)
    def test_forward_match_1layer(self):
        # 1-layer model: NumPy = Triton (rtol=1e-3)
        ...

    @pytest.mark.timeout(30)
    def test_forward_match_2layer(self):
        # 2-layer model: NumPy = Triton (rtol=1e-2)
        ...

    @pytest.mark.timeout(60)
    def test_forward_multi_batch(self):
        # Batched: NumPy = Triton (rtol=1e-3)
        ...

    @pytest.mark.timeout(60)
    def test_gradient_chaining(self):
        # Triton forward+backward → valid, non-zero gradients
        ...

    @pytest.mark.timeout(60)
    def test_training_reduces_loss(self):
        # Triton training for 20 steps → loss decreases
        ...

    @pytest.mark.timeout(60)
    def test_inference_equivalence(self):
        # Same prompt → same tokens as PyTorch (exact match, greedy)
        ...


class TestTritonCheckpoint:
    """Checkpoint round-trip tests."""

    @pytest.mark.timeout(30)
    def test_save_load_roundtrip(self):
        # save_as_numpy → load_from_numpy_dict → same params
        ...

    @pytest.mark.timeout(30)
    def test_cross_backend_load(self):
        # TritonModel.load_from_numpy_dict(TorchModel.save_as_numpy()) → valid
        ...
```

**Gate:** 8-12 tests → commit. **Phase E complete.**

---

## Execution Order (Sequential, Strictly One at a Time)

```
E0: Scaffolding (directories + import)
 │
 ├──→ E1: SiLU kernel (simplest, element-wise)
 ├──→ E2: RMSNorm kernel (reduction, first parallel candidate)
 └──→ E3: RoPE kernel (no dependency on E2, can parallel with E2)
      │
      └──→ E4: SwiGLU kernel (depends on E1 SiLU kernel)
           │
           └──→ E5: MHA kernel (depends on E3 RoPE)
                │
                ├──→ E6: MoE kernel (depends on E4 SwiGLU)
                └──→ E7: TransformerBlock (depends on E5 MHA + E6 MoE)
                     │
                     └──→ E8: DecoderStack (depends on E7)
                          │
                          └──→ E9: Full TritonModel (depends on E8 — can parallel with E10)
                               │
                               ├──→ E10: Inference + Training (depends on E9)
                               │
                               └──→ E11: Cross-backend parity (depends on E9 + E10)
```

**Wave 1:** E0 (scaffolding) — 1 commit
**Wave 2:** E1+E2+E3 — 3 kernels in parallel (E3 also depends on E1, but all are simple enough to batch)
**Wave 3:** E4 (SwiGLU, uses E1 SiLU) — 1 commit
**Wave 4:** E5 (MHA, uses E3 RoPE) — 1 commit
**Wave 5:** E6+E7 in parallel (MoE+TransformerBlock) — 2 commits
**Wave 6:** E8 (DecoderStack) — 1 commit
**Wave 7:** E9 in parallel with E10 (Full model + Inference) — 2 commits
**Wave 8:** E11 (Cross-backend parity) — 1 commit

**Total:** ~12 sub-phases, ~15 commits, ~60-80 tests

---

## GPU Availability Detection

All tests that require a GPU must check for it:

```python
import pytest
import torch

def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
```

**Important:** GPU-less development is still possible for parity testing if NumPy and PyTorch are already verified and we only need to ensure Triton produces matching output. GPU availability is needed for correctness testing (kernel runs on GPU, results compared on CPU).

When running tests:
```bash
# With GPU
uv run pytest tests/unit/_triton/ -v --timeout=30

# Check GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Summary Table

| Stage | Dependency | Tests | Description |
|-------|-----------|-------|-------------|
| E0 | — | 1 | Directories + import test |
| E1 | E0 | 6-7 | SiLU kernel (element-wise) |
| E2 | E0 | 8 | RMSNorm kernel (reduction) |
| E3 | E0 | 6 | RoPE kernel (trig, indexing) |
| E4 | E1 | 6 | SwiGLU kernel (SiLU + matmul) |
| E5 | E3 | 8-10 | MHA kernel (attention + GQA) |
| E6 | E4 | 6 | MoE kernel (top-k routing) |
| E7 | E5,E6 | 5 | TransformerBlock (Python wiring) |
| E8 | E7 | 3 | DecoderStack (Python wiring) |
| E9 | E8 | 5 | Full TritonModel (save/load/parity) |
| E10 | E9 | 6-8 | Inference + training scripts |
| E11 | E9,E10 | 8-12 | Cross-backend parity (NumPy/PyTorch vs Triton) |

**Total: ~60-80 tests, ~15 commits**

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| No GPU available in dev environment | High | Test on GPU machine/CI; all parity tests need a GPU |
| Triton installation issues (CUDA version mismatch) | High | Pin `triton` version in `pyproject.toml` matching system CUDA |
| Kernel precision mismatch (float32 native, float64 for tests) | Medium | Use `tl.fp32_to_fp64` where available; test with float64 tensors |
| Attention kernel too complex for one pass | Medium | Split E5 into core attention + GQA if needed |
| Block scheduling edge cases | Low | Test small (1D tiles) and large (multi-D) shapes separately |
| Triton not in PyTorch's autograd graph | Low | Triton kernels use `triton.language` which supports autograd by default |

---

## Key Differences From NumPy/PyTorch

| Aspect | NumPy | PyTorch | Triton |
|--------|-------|---------|--------|
| Forward output dtype | float32 / float64 | Matches input dtype | `fp32` by default (use `tl.float64` for tests) |
| Execution | CPU | CPU/GPU (dispatched) | GPU (explicit `triton.jit`) |
| Parameter storage | instance attributes | `nn.Parameter` | `nn.Parameter` (same as PyTorch) |
| Cross-backend parity | reference | second reference | third equivalence point |
| Test dtype | float64 | float64 for parity | float64 tensors → GPU kernel |

---

*Plan ready for review. 12 sub-phases, ~15 atomic commits, ~60-80 tests. Strictly sequential by wave.*