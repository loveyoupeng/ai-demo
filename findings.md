# Findings & Decisions

## Requirements

### Core Architecture
- Decoder-only text-to-text transformer (MHA)
- Configurable: layers, heads, dimensions, context_length
- RoPE position encoding (configurable)
- GQA (Grouped-Query Attention) – opt-in config toggle
- MoE (Mixture of Experts) – configurable num_experts
- KV Cache: Naive (full precision) + TurboQuant (1-bit compressed)
- **Residual connections – Pre-Norm architecture** (see Phase 3++ section below)

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
| Smaller test cases for debugging | When tests fail, isolate the issue with minimal test case — don't over-reason |
| Pre-Norm architecture | RMSNorm before residual add — stable training, standard GPT-style |
| Single train/infer scripts | Less duplication, unified entry point with --backend flag |
| Greedy = deterministic | Exact token match across backends; sampling uses KL divergence |

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

## Phase 3++: Residual Connection Discussion (IN PROGRESS)

### Current Architecture: Pre-Norm
Both backends use **pre-normalization** (RMSNorm BEFORE the residual add):

```python
# Current (NumPy: modules.py:900, PyTorch: layers.py:658,670)
l_normed = RMSNorm().forward(x, gamma)      # normalize input first
attn_out = self.mha.forward(l_normed)        # compute attention
h = x + attn_out                             # add residual
moe_out = self.moe.forward(ln2.forward(h))  # MoE output
out = h + moe_out                            # add second residual
```

Mathematical formula (documented in both implementations):
```
h = x + MHA(RMSNorm(x))
out = h + MoE(RMSNorm(h))
```

### User Concern
User noticed "we lack of residual connection which could speed up training and avoid signal vanish feature in current model."

### Observation
The codebase **already has residual connections**. Both NumPy and PyTorch implementations have:
- `h = x + attn_out` (first residual: input + attention output)
- `out = h + moe_out` (second residual: intermediate + MoE output)

### Possible Interpretations of User's Concern
1. **Post-Norm vs Pre-Norm**: User may prefer post-norm (residual add first, then norm) which can speed up training but is less stable
2. **Gated Residuals**: User may want gated residuals (e.g., `x + gate * residual`) to control signal flow — used in some architectures to prevent exploding gradients
3. **Dropout for Regularization**: Currently no dropout anywhere — user may be confusing "dropout" with "residual" for regularization
4. **Skip Connections Across Layers**: User may be thinking of DenseNet-style skip connections where all layers receive the same input

### Recommendation
- **Ask user for clarification** before implementing — they may be confused about what's already in the code
- If they want post-norm: swap order to `h = x; h = RMSNorm(h) + MHA(h)`
- If they want gating: add learnable gate parameter to each residual
- If they want dropout: add dropout to intermediate activations (not residuals)

## Resources

- TinyStories: `huggingface.co/allenai/tinystories`
- RoPE: "Attention Is All You Need" + RoPE original paper (Su et al. 2021)
- GQA: "GQA: Generalized Query Attention" (Du et al. 2022)
- MoE: "Mixtral of Experts" (Jiang et al. 2024), Switch Transformer (Fedus et al. 2021)
- TurboQuant: Google research on KV cache quantization (1-bit compression)
- Pre-Norm vs Post-Norm: "Layer Normalization" (Ba et al. 2016), "Attention Is All You Need" (Vaswani et al. 2017)
- Gated Residuals: Deep & Cross Network (Wang et al. 2017), or DenseNet (Huang et al. 2017)
