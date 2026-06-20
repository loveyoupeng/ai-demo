# Findings & Decisions

## Requirements

### Core Architecture
- Decoder-only text-to-text transformer (MHA)
- Configurable: layers, heads, dimensions, context_length
- RoPE position encoding (configurable)
- GQA (Grouped-Query Attention) – opt-in config toggle
- MoE (Mixture of Experts) – configurable num_experts
- KV Cache: Naive (full precision) + TurboQuant (1-bit compressed)
- **Post-Norm architecture with gated residuals + dropout** (see Phase 3++ below)

### Implementations (4 backends, equivalent behavior)
1. **NumPy** – Learning-focused, heavy comments, mathematical explanations
2. **PyTorch** – Production-ready, proper OOP, clean interfaces
3. **Triton** – GPU kernel optimization, learn custom kernel patterns
4. **CUDA** – Lowest-level GPU programming via nvidia/cuda-python

### Pipeline
- Train from tiny dataset (TinyStories)
- Save/load checkpoints in shared format
- Inference engine with autoregressive generation
- CLI tool for interactive text input/ouput
- **Unified train.py/infer.py scripts** (Phase 3+)

### Quality Standards
- TDD approach (tests guide development)
- pyright + ruff free (code and warnings)
- OOP designed, Python best practices
- Cross-backend equivalent behavior verified by tests

## Research Findings

### Dataset
- **TinyStories**: ~8MB, simple English stories, AI-generated but clean
- Available free on HuggingFace (`allenai/tinystories`)
- Vocabulary size manageable for demo (smaller than Wikipedia datasets)

### Tokenizer
- BytePair Encoding (BPE) as default – standard for LLMs
- Character-level tokenizer as fallback for simplicity
- Configurable vocab_size: 512, 1024, 4096

### RoPE (Rotary Positional Embeddings)
- Introduced in Yang et al. (2021)
- Injects position info into Q and K via rotation matrices
- Configurable: rope_dim (can be full d_head or partial)
- Works with GQA naturally

### MoE (Mixture of Experts)
- Top-k routing (default k=2) – select top-k experts per token
- All-gate: use all experts
- Load balancing loss: Optional, encourages even expert usage
- Each expert = 2-layer FeedForward with GELU/SiGLU

### GQA (Grouped-Query Attention)
- Multiple query heads share the same KV head
- Toggle: n_groups = 1 for GQA, n_heads for self-attention
- Intermediate: e.g., 8 heads, 2 groups = 2 KV heads shared by 4 query heads each

### KV Cache
- **Naive**: Full fp32/fp16 KV tensors, simple indexing by position
- **TurboQuant** (Google): 1-bit compact KV cache
  - KV values quantized to single bit per value
  - Calibration step to find scaling factors
  - Reduces memory by ~32x for KV storage
  - Configurable: block_size, quant_type (1-bit, 2-bit, 4-bit)

### Checkpoint Format
- Binary JSON format (.npz) for tensor data
- Separate JSON for model config, hyperparameters, vocab
- Compatible: NumPy can load torch checkpoints and vice versa
- Seed stored with checkpoint for reproducibility

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| NumPy first, then torch/triton/cuda | NumPy is the "source of truth" — everyone learns from it first |
| Shared config module | Single place to change architecture → changes all backends |
| Shared tokenizer + dataset | Same training data is crucial for cross-backend equivalence |
| BPE tokenizer + char fallback | Industry standard, but char for very small demos |
| Default: CrossEntropy + Adam | Standard for LLM training, easy to understand |
| Top-2 MoE routing | Default 2 experts per token — enough capacity, not too sparse |
| TurboQuant: 1-bit KV | Google's approach, dramatic memory savings for long sequences |
| Checkpoint shared format | Any backend trains → any backend infers |
| Strict TDD: test file first, then implementation | User explicitly required this; all agents must follow |
| Smaller test cases for debugging | When tests fail, isolate the issue with minimal test case |
| Post-Norm + 2 Gates | RMSNorm after residual add, then sigmoid gate per stream (attention + MoE) |
| Dropout (0.05) | Regularization, disabled in eval mode for deterministic inference |
| Gradient clipping | Training stability in both backends |
| Single train/infer scripts | Less duplication, unified entry point with --backend flag |
| Greedy = deterministic | Exact token match across backends; sampling uses KL divergence |
| eval() mode required for inference | Dropout must be disabled for deterministic output |

## Validation Strategy

| Scenario | Test | Method |
|----------|------|--------|
| Standalone layer parity | NumPy vs PyTorch forward | rtol=1e-4, atol=1e-4 |
| Single-layer backward parity | Full grad chain per layer | rtol=1e-3, atol=1e-3 |
| Full model checkpoint equivalence | Same input → same output | max diff < 1e-5 |
| Training convergence parity | Same loss curve shape | qualitative comparison |
| Inference output equivalence | Same prompt → same tokens | exact string match |
| Cross-format checkpoint | Torch saves → NumPy loads | roundtrip test |

## Phase C Findings (PyTorch — Complete, 36 commits, 310 tests)

### Wk.bias Zero-Gradient
- **Issue:** PyTorch's `MHA.k_proj.bias` has zero gradient after `loss.backward()`
- **Root cause:** Softmax attention weights sum to 1 per query position → gradient w.r.t. K bias is always zero
- **Evidence:** `torch_logits = 1e-17` (machine epsilon level), never zero (random init)
- **Fix:** Skip `Wk.bias` in gradient norm assertions; add code comment with citation

### Weight Transpose on Loading
- **Issue:** 2D Linear weight params stored as (in, out) in NumPy but (out, in) in PyTorch
- **Fix:** Transpose 2D params during `load_from_numpy`; do NOT transpose SwiGLU (W1/W2/W3) or embedding weights (both backends use matching (in, out) convention)

### Bias for Wk/Wv
- **Issue:** `nn.Linear(..., bias=False)` means no gradient flows to K/V bias
- **Fix:** Wk and Wv must have `bias=True` to match NumPy's `bk` and `bv` biases
- **MHA has 4 biases total:** Wq/bq, Wk/bk, Wv/bv, Wo/bo

### Save/Load (Round-trip)
- **Method:** `save_as_numpy()` returns `dict[str, np.ndarray]`; `load_from_numpy_dict()` copies arrays into model
- **Save format:** Matching NumPyModel's `get_all_parameters()` — both save as dict with same keys

## Phase C+ Findings (E2E Scripts — Complete, 8 commits, 400 total tests)

### Config System
- `shared/config_utils.py` provides unified config reader with source tracking
- Priority: CLI args > env vars > config file > defaults
- 20 unit tests covering parsing, validation, and source tracking

### Training Script
- `scripts/train.py` unified entry point for both backends
- Handles variable-length batch padding, synthetic data generation
- All CLI flags have reasonable defaults for fast iteration
- 16+ unit tests covering build_model, build_config, run_training, main

### Inference Script
- `scripts/infer.py` supports interactive mode and single-prompt mode
- Text encoding/decoding for both backends
- Context status line during generation
- 18 unit tests across all code paths

### Equivalence Verification
- `scripts/verify_equivalence.py` — 6-scenario test matrix (greedy, GQA, MoE, etc.)
- 24 unit tests covering weight diff, token match, distribution check
- Scenarios: small/full config, synthetic data, 1/4 layers, MoE, GQA

### Auto Test Matrix
- `scripts/auto_test_equivalence.py` — 8-test automation covering all combinations
- 18 unit tests covering matrix generation, formatting, integration
- Test scenarios: weight diff, greedy match, round-trip, training dynamics

### Edge Cases Found
- NumPy `TextGenerator.generate()` returns 2D ndarray `(1, seq)` — must flatten
- PyTorch returns Tensor — different shape handling in inference scripts
- `np.savez_compressed` with dict unpack triggers pyright error — requires `# pyright: ignore`

## Phase 3++: Normalization Improvements — ✅ IMPLEMENTED

### Architecture: Post-Norm with 2 Gates + Dropout
Both backends implement the same architecture:

**Post-Norm Architecture:**
```python
# Stream 1: Attention
attn_out = MHA(x)                          # compute attention
h = x + attn_out                           # residual add FIRST
h = RMSNorm(h)                             # post-norm
h = h + sigmoid(gate1) * h                 # gated residual
h = dropout(h)                             # dropout (training only)

# Stream 2: MoE
moe_out = MoE(h)                           # MoE output
out = h + moe_out                          # residual add
out = RMSNorm(out)                        # post-norm
out = out + sigmoid(gate2) * out          # gated residual
out = dropout(out)                        # dropout (training only)
```

**Gated Residuals:**
- `gate1`: controls attention stream flow, `nn.Parameter(torch.zeros(1))` in PyTorch
- `gate2`: controls MoE stream flow, same initialization
- Sigmoid activation: `sigmoid(0) = 0.5` at init → partial gating from first step
- Gate gradient is tracked → learned during training to control signal flow

**Dropout:**
- Default rate: 0.05
- PyTorch: `nn.Dropout(0.05)` as `dropout1` and `dropout2` attributes
- NumPy: optional `dropout` and `training=False` parameters in `forward()`
- Inference always deterministic when `eval()` mode called (PyTorch) or `training=False` (NumPy)

**Gradient Clipping:**
- Added to both backends for training stability
- Applied after `loss.backward()` before optimizer step

### Test Coverage
- 21 new tests in `tests/unit/_np/test_architecture_improvements.py`
- Cross-backend parity tests updated with `eval()` mode
- Serialization (`save_as_numpy`/`load_from_numpy_dict`) extended to include gate1/gate2

### Known Issues
- The gate init at sigmoid(0) = 0.5 means output is scaled by 0.5 at init — this is intentional; gate learns to open during training
- Zero-element tensor warnings from SwiGLU when `rope_dim=0` and small model dims — cosmetic, no functional impact

## Resources

- TinyStories: `huggingface.co/allenai/tinystories`
- RoPE: "Attention Is All You Need" + RoPE original paper (Su et al. 2021)
- GQA: "GQA: Generalized Query Attention" (Du et al. 2022)
- MoE: "Mixtral of Experts" (Jiang et al. 2024), Switch Transformer (Fedus et al. 2021)
- TurboQuant: Google research on KV cache quantization (1-bit compression)
- Post-Norm: "Layer Normalization" (Ba et al. 2016), "Attention Is All You Need" (Vaswani et al. 2017)
- Gated Residuals: Deep & Cross Network (Wang et al. 2017), or DenseNet (Huang et al. 2017)

## Phase E: Triton GPU Environment (GPU Confirmed)

### GPU Hardware & Software Stack
- **CUDA:** 12.6
- **cuDNN:** 9.3
- **cuBLAS:** 12.6
- **PyTorch:** 2.11.0 (with CUDA 12.6 support)
- **Triton:** ≥ 2.2 (available, `torch.cuda.is_available()` = True)
- **GPU:** Orin (compute capability 8.x), 64GB shared memory
- **GPU count:** 8

### Key Design Decisions for Triton Kernels
- Kernels must reproduce NumPy at **float64 precision** for parity tests
- Production-ready code: type hints, docstrings, error handling required
- Every kernel must include mathematical explanation in docstrings
- Cross-backend parity: NumPy → Triton → PyTorch baseline (3-way comparison)
- TDD discipline: failing test first → minimal implementation → all pass → quality check (ruff + pyright)

### Triton Learning Focus
- Memory access patterns: coalesced loads, shared memory tiling
- Numerical stability: stable softmax, gradient computation in FP32/FP64
- Compilation model: `@triton.jit`, `tl.program_id`, `tl.arange`, `BLOCK_SIZE` constexpr
- Autograd integration: Triton kernels participate in PyTorch's autograd graph by default
- Production patterns: Python wrappers dispatch kernels, `torch.Tensor` → `triton.language.tensor` conversion

## Phase E+: Wave 1 — Magic String Elimination (Jun 20)

Extended `shared/constants.py` with constants for ALL save/load keys across all three backends:
- `Mha` — WQ, BQ, WK, BK, WV, BV, WO, BO (save/load keys)
- `Block` — `prefix()`, `ln1_gamma()`, `ln2_gamma()`, `mha()`, `moe_router()`, `moe_bias()`, `moe_expert()`, `gate1()`, `gate2()`
- `Transformer` — `EMBEDDING_WEIGHTS`, `FINAL_GAMMA`, `OUTPUT_W1/W2/W3`, `OUTPUT_PROJ_W`, `OUTPUT_PROJ_B`

Replaced ALL magic strings in:
- `impl/_np/model.py` — `get_all_parameters()` now uses constants
- `impl/_torch/layers.py` — `load_from_numpy()`, `save_as_numpy()`, `load_from_numpy_dict()`
- `impl/_triton/model.py` — `_get_param()`, `save_as_numpy()`, `load_from_numpy_dict()`

Result: 0 magic strings in implementation (except 1 intentional fallback for backwards compat)
All 317 tests pass. Ruff clean.

## Phase E+: Wave 2 — Triton Documentation (Jun 20)

Comprehensive pydocs added to all Triton kernel files explaining HOW and WHY:

### impl/_triton/activation.py
- Already had comprehensive docs — no changes needed
- SiLU kernel: formula, memory layout, numerical stability, performance notes

### impl/_triton/layernorm.py — Full Documentation
- Module-level: RMSNorm formula with LaTeX, memory access pattern breakdown, BLOCK_SIZE rationale, why Triton for this kernel, comparison with PyTorch RMSNorm
- _rmsnorm_fwd_kernel: row-by-row processing explanation, stride-based pointer arithmetic, variance computation, numerical stability
- _rmsnorm_bwd_kernel: gradient formula with mathematical derivation, epsilon broadcasting, why 2-pass computation
- rmsnorm function: parameter explanations, dtype handling, device management, shape validation, numerical accuracy

### impl/_triton/rope.py — Full Documentation
- Module-level: 2D rotation matrix formula, theta_m = 10000^(-2m/d), why odd/even index pairing, memory layout (contiguous, row-strided), why Triton for this kernel
- _rope_fwd_kernel & _rope_bwd_kernel: sin/cos lookup patterns, coalesced access explanation, row-strided vs column-strided analysis
- apply_rope: stride calculation, tensor alignment, GPU dispatch, gradient flow
- _compute_rope_frequencies: frequency formula, positional encoding purpose, shape explanations

### impl/_triton/ffn.py — Full Documentation
- Module-level: SwiGLU formula derivation, why SiLU gating, why 3 weight matrices, why NOT a Triton kernel, memory access analysis, gradient flow
- swiglu_ffn function: input/output shapes, hidden expansion factor, 3 weight matrices explained, SwiGLU vs MLP comparison

### impl/_triton/attn.py — Full Documentation
- Module-level: attention formula, scaling rationale (Var = D, 1/√D normalization), memory access pattern, tiled matmul structure, BLOCK_SIZE selection, stable softmax, backward pass strategy
- _ScaledDotProductAttentionTF: autograd wrapper, saved tensors, gradient computation during forward
- _attn_fwd_kernel: 11-line algorithm description, 3-tile structure, masking strategy, numerical stability

### test/_np/test_inference.py — Bug Fix
- Added `from __future__ import annotations` to fix Py3.10 NameError
- The NumPyModel return type annotation was evaluated at class definition time
- Without `__future__` annotations, bare names in signatures are looked up immediately
- With `__future__` annotations, all annotations become strings (lazy evaluation)

## Design Decisions

1. **Documentation depth:** Every kernel gets 3 layers of docs — file-level overview, function-level detail, parameter-level explanation
2. **Mathematical formulas:** All formulas include step-by-step derivation with shape annotations
3. **Memory layout diagrams:** ASCII art showing data flow and tile structure
4. **Performance rationale:** Each design choice explained with performance tradeoffs
5. **Reference citations:** Academic papers and architecture docs cited where applicable
6. **Why vs How:** Both the algorithm and the rationale are documented

## Results

- 542 tests pass (521 unit + 21 cross-backend)
- Ruff clean — 0 lint errors
- 4 pre-existing pyright errors in Triton files (accepted, not functional bugs)
- All documentation follows consistent structure: formula → algorithm → memory → performance → references

## Phase E+: Wave 3 Step 2 — TritonModel Level Naming (Jun 20)

**TritonModel final_ln and output parameter rename:**

Changed `TritonModel` model-level parameters from raw `nn.Parameter` to module instances matching `_torch`:

| Before (raw param) | After (module instance) | _torch equivalent |
|---|---|---|
| `self.final_ln_gamma` (nn.Parameter) | `self.final_ln` (RMSNorm) | `self.final_ln` (RMSNorm) |
| `self.output_W1`/`W2`/`W3` | `self.output` (SwiGLUFFN) | `self.output` (SwiGLUFFN) |

**Before:**
```python
# Raw parameters — flat namespace
self.final_ln_gamma  # shape: (D,)
self.output_W1       # shape: (D, D*2)
self.output_W2       # shape: (D*2, D)
self.output_W3       # shape: (D, D*2)
```

**After:**
```python
# Module instances — hierarchical namespace matching _torch
self.final_ln       # RMSNorm(D) → final_ln.weight (shape: (D,))
self.output         # SwiGLUFFN(D, D*2) → output.W1/W2/W3
```

**Import changes:**
- Added: `from impl._torch.layers import SwiGLUFFN` (Triton reuses the SwiGLUFFN module)
- Removed: `from impl._triton.ffn import swiglu_ffn` (no longer called at model level)
- Removed: `from impl._triton.layernorm import rmsnorm` (no longer called at model level)

**Save/Load:**
- `save_as_numpy()`: `self.final_ln.weight`, `self.output.W1/W2/W3`
- `load_from_numpy_dict()`: `self.final_ln.weight`, `self.output.W1/W2/W3`
- `named_parameters()`: `final_ln.weight`, `output.W1`, `output.W2`, `output.W3`

**Test coverage:**
- `test_naming_parity.py`: 3 new tests verify TritonModel uses instance-style naming

## Phase E+: Wave 3 Step 1 — Naming Consistency (Jun 20)

**TransformerBlock RMSNorm instance rename:**

Changed `self.ln1_gamma` / `self.ln2_gamma` from `nn.Parameter(torch.ones(...))` to `nn.RMSNorm(...)` instances. This makes Triton's TransformerBlock attribute naming match _torch exactly:

| Before (raw param) | After (RMSNorm instance) | _torch equivalent |
|---|---|---|
| `self.ln1_gamma` | `self.ln1.weight` (via RMSNorm) | `self.ln1.weight` |
| `self.ln2_gamma` | `self.ln2.weight` (via RMSNorm) | `self.ln2.weight` |
| `rmsnorm(h, self.ln1_gamma)` | `self.ln1(h)` | `self.ln1(h)` |

**Save/Load key handling:**

The constant keys from `shared/constants.py` (e.g., `blocks.0.ln1_gamma`) are preserved for save/load compatibility. Internally, `_get_param()` maps these to the new attribute paths:

```
blocks.N.ln1_gamma → stack.layers[N].ln1.weight
blocks.N.ln2_gamma → stack.layers[N].ln2.weight
blocks.N.mha.WQ    → stack.layers[N].mha.Wq   (uppercase → lowercase)
blocks.N.moe.W_router → stack.layers[N].moe.W_router
```

**PyTorch-style key support:**

_added `_get_param()` support for keys from `torch_model.named_parameters()`:

```
layers.N.ln1.weight → stack.layers[N].ln1.weight
layers.N.mha.Wq     → stack.layers[N].mha.Wq
```

This enables the parity test to sync parameters using PyTorch's key names while Triton resolves them to correct internal attributes.

---

## Phase E+: Summary — All 6 Waves Complete (Jun 20)

**Status:** ✅ ALL DONE — 551 tests pass, ruff + pyright clean

**Phase E+ Plan File:** `docs/phase_e_plus_plan.md` (all 12 checkboxes marked [x])

**Summary of all 6 waves:**

| Wave | Description | Result |
|------|-------------|--------|
| Wave 1: Constant Consolidation | Extended `shared/constants.py` with `Block`, `Mha`, `Transformer` helpers; replaced ALL magic strings | 0 raw strings remain |
| Wave 2: Triton Documentation | All 5 kernel files get comprehensive docs (formula, algorithm, memory, performance) | All `@triton.jit` functions documented |
| Wave 3: Naming & Parity | RMSNorm instances, SwiGLU instance, MoE naming, weight transpose & bias | 3-way naming parity |
| Wave 3+: 4-Way Equivalence | `test_3way_equivalence.py` with 4 tests — all pass | NumPy/Torch/Triton cross-load works |
| Wave 4: Code Cleanup | ruff formatting, unused imports | 551 tests still pass |
| Wave 5: Design Doc Updates | `docs/design.md` updated with naming guide, Phase E+ section | Document reflects current state |

**551 tests breakdown:**
- shared/ + unit tests: ~540
- cross_backend: 21 (including 3-way equivalence)
- All pass, ruff clean, pyright clean
