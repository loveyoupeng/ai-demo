# Phase B: NumPy Implementation — Execution Plan

## Goal
Build a fully functional decoder-only transformer LLM in NumPy, fully tested, that trains on TinyStories and generates text. This is the **reference implementation** that all future backends (PyTorch, Triton, CUDA) will match.

**Principle:** Educational, every operation explained. Every matrix dimension annotated in comments.

---

## Architecture (NumPy-specific)

```
Input (tokens: [batch, seq_len])
    │
    ▼
┌─────────────────────┐
│  Token Embedding     │ shape: [B, S, D]
├─────────────────────┤
│  RMSNorm (ln1)       │ layer norm
├─────────────────────┤
│  RoPE                │ position encoding on Q, K
├─────────────────────┤
│  Stream 1: MHA       │ → [B, S, D]
│    ├── Q, K, V proj  │
│    ├── Multi-head     │
│    └── Output proj    │
├─────────────────────┤
│  Residual add        │ h = h + attn_out
├─────────────────────┤
│  Stream 2: MoE       │ → [B, S, D]
│    ├── Top-k routing  │
│    ├── Expert 1       │ SiLU(w1 @ x) * (w3 @ x)
│    └── Expert N       │ w2 @ gated_output
├─────────────────────┤
│  Residual add        │ h = h + moe_out
├─────────────────────┤
│  RMSNorm (final)     │
├─────────────────────┤
│  Output LM Head      │ → [B, S, V]
└─────────────────────┘
```

---

## Directory Structure

```
impl/
 └── numpy/
     ├── __init__.py           # Package init + public API exports
     └── utils/                # Small shared utilities
         ├── __init__.py
         ├── seed.py           # NumPy RNG helpers (seed, rand, etc.)
         └── tensors.py        # Array helper ops (reshape, concat, eye, etc.)

tests/
 └── unit/
     └── numpy/
         ├── __init__.py       # Package init
         └── test_modules.py   # Module import test
```

**Structure rationale:** 
- `impl/` top-level (as per design doc) with `numpy/` as one backend
- `utils/` subfolder for tiny helper functions that are reused across modules
- Test mirror structure: `tests/unit/numpy/` for all NumPy-specific tests
- Each test file follows the "one class per test file" pattern

---

## TDD Process (same as Phase A)

For each component, follow this strict cycle:

### Step 1: Write test file — ALL TESTS FAIL
```bash
# Write tests/unit/numpy/test_<component>.py — only tests, NO implementation
# Run: PYTHONPATH=shared PYTHONPATH=impl PYTHONPATH=impl/numpy uv run pytest tests/unit/numpy/test_<component>.py -v --timeout=10
# Expected: ALL tests fail (ModuleNotFoundError / ImportError)
```

### Step 2: Implement minimal code
```bash
# Write minimal implementation that makes ALL tests pass
# Run: PYTHONPATH=shared PYTHONPATH=impl PYTHONPATH=impl/numpy uv run pytest tests/unit/numpy/test_<component>.py -v --timeout=10
# Expected: ALL pass
```

### Step 3: Quality check
```bash
PYTHONPATH=shared ruff check impl/numpy/<component>.py && \
PYTHONPATH=shared pyright impl/numpy/<component>.py && \
PYTHONPATH=shared ruff check tests/unit/numpy/test_<component>.py && \
PYTHONPATH=shared pyright tests/unit/numpy/test_<component>.py
```

### Step 4: Commit
```bash
git add -A && git commit -m "b<stage>: <component> — <N> tests pass"
```

### Rules
- NEVER implement without failing tests first
- ONE component per commit (atomic)
- ONE test file per component (or related sub-components at most)
- Small, fast tests — no network, no disk I/O, pure NumPy
- Test with float32 (matching real training dtype). Comparisons use `rtol=1e-3` for layer unit tests.

---

## Stages (15 sub-phases, 20 commits)

### Phase B0: Project scaffolding

| Stage | Files | Gate | Test Count |
|-------|-------|------|------------|
| **B0.1** | `impl/`, `tests/unit/numpy/` directories, `__init__.py` files | pytest collects 0 tests | 0 |
| **B0.2** | `tests/unit/numpy/test_modules.py` — import the package | pytest discovers tests | 1 |

```python
# tests/unit/numpy/test_modules.py
from impl.numpy import NumPyModel, ModelConfig

def test_numPyModel_imports():
    assert NumPyModel is not None
```

---

### Phase B1: Basic layers (B1 in design doc)

### B1.1: Embedding layer

**What it does:** Maps token IDs to dense vectors.

```python
class Embedding:
    # Parameters: weight [vocab_size, embed_dim]
    # Forward: x → lookup(x, weight) → [batch, seq_len, embed_dim]
    # Forward takes: input_ids: np.ndarray[int32], weight: np.ndarray
    # Returns: np.ndarray[float32]
```

```python
# tests/unit/numpy/test_embedding.py
class TestEmbeddingForward:
    def test_output_shape(self):
        # input_ids: [batch=2, seq_len=4] with values in [0, vocab_size)
        # weight: [vocab_size=16, embed_dim=8]
        # output should be [2, 4, 8]

    def test_lookup_correctness(self):
        # Verify embedding[i] maps to the i-th row of weight

    def test_batch_handling(self):
        # Multiple sequences processed in parallel
```

**Gate:** `pytest` discovers → all fail → implement → all pass (3 tests) → ruff+pyright → commit

---

### B1.2: RMSNorm (LayerNorm-style normalization)

**What it does:** Normalizes each feature vector to unit variance, scales by learned gamma.

```python
class RMSNorm:
    # Parameters: gamma [embed_dim] (learned scale)
    # Forward: x → normalize(x, gamma) → [batch, seq_len, embed_dim]
    # Formula: RmsNorm(x) = x / Rms(x) * gamma
    #   Rms(x) = sqrt(mean(x^2)) + eps
```

```python
# tests/unit/numpy/test_rmsnorm.py
class TestRMSNormForward:
    def test_output_shape(self):
        # input [batch, seq_len, embed_dim], gamma [embed_dim]
        # output [batch, seq_len, embed_dim]

    def test_unit_variance(self):
        # After normalization, per-feature variance ≈ 1

    def test_identity_without_gamma(self):
        # With gamma=1, output ≈ normalized input

    def test_learned_scale(self):
        # gamma controls output magnitude

class TestRMSNormBackward:
    def test_gradient_shape(self):
        # Gradient w.r.t. input has same shape as input

    def test_gradient_correct(self):
        # Numerical gradient check (finite differences)
```

**Gate:** Forward tests (4) + backward tests (2) = 6 tests → 12 total

---

### B1.3: SiLU activation

**What it does:** Element-wise Swish/SiLU activation: f(x) = x * sigmoid(x)

```python
class SiLULayer:
    # Forward: x → x * sigmoid(x) → element-wise
    # Backward: d_out → element-wise chain rule
```

```python
# tests/unit/numpy/test_silu.py
class TestSiLULayer:
    def test_output_shape(self):
        # Same shape as input

    def test_output_range(self):
        # SiLU(x) ≈ x for large x, ≈ 0 for negative

    def test_output_at_zero(self):
        # SiLU(0) = 0 * 0.5 = 0

    def test_backward_matches_forward(self):
        # d/dx Sigmoid(x) * x + Sigmoid(x)
```

**Gate:** 4 tests → commit

---

### B1.4: SwiGLU feedforward

**What it does:** SwiGLU — a modern feedforward with gating:

```python
class SwiGLUFFN:
    # Parameters: w1 [embed_dim, ff_dim], w2 [ff_dim, embed_dim], w3 [embed_dim, ff_dim]
    # Forward: SiLU(w1 @ x) * (w3 @ x) @ w2
    # Gate mechanism: SiLU provides smooth gating
```

```python
# tests/unit/numpy/test_swiglu.py
class TestSwiGLUFFN:
    def test_output_shape(self):
        # [batch, seq_len, embed_dim] → [batch, seq_len, embed_dim]

    def test_gating_behavior(self):
        # w1 and w3 projections gated together

    def test_no_gradient_leakage(self):
        # Gradients flow through all 3 weight matrices

    def test_backward_correct(self):
        # Numerical gradient check across w1, w2, w3
```

**Gate:** 4 tests → commit

---

### Phase B2: RoPE Position Encoding

### B2.1: RoPE ( Rotary Positional Embedding )

**What it does:** Injects positional information into Q and K matrices via rotation.

```python
class RoPE:
    # Parameters: freqs (computed from config, no learnable params)
    # Forward: Q, K → apply rotation by position → rotated Q, K
    # Formula: q_m' = q_m * cos(mθ) - q_{m+r} * sin(mθ)
    #           q_{m+r}' = q_m * sin(mθ) + q_{m+r} * cos(mθ)
```

```python
# tests/unit/numpy/test_rope.py
class TestRoPE:
    def test_output_shape(self):
        # Same shape as input (Q, K unchanged shape)

    def test_rotates_by_position(self):
        # Position 0 and position 1 produce different rotations

    def test_no_gradient_leakage_qk(self):
        # Q and K gradients don't interfere

    def test_full_vs_partial(self):
        # rope_dim=0 (full) vs rope_dim<n (partial) behave correctly
```

**Gate:** 4 tests → commit

---

### Phase B3: MHA (B3 in design doc)

### B3.1: Multi-Head Attention (without GQA)

**What it does:** Standard scaled dot-product multi-head attention.

```python
class MultiHeadAttention:
    # Parameters: q_proj [embed_dim, n_heads*head_dim], k_proj [...], v_proj [...], o_proj [embed_dim, embed_dim]
    # Forward: X → Q,K,V projections → reshaped to (B, n_heads, S, head_dim) → softmax(QK^T/sqrt(d)) → concatenate → output_proj
    # Returns: [batch, seq_len, embed_dim]
```

```python
# tests/unit/numpy/test_mha.py
class TestMultiHeadAttentionForward:
    def test_output_shape(self):
        # X [B, S, D] → output [B, S, D]

    def test_attention_mechanism(self):
        # Softmax normalized across sequence dimension

    def test_gradient_flow(self):
        # All weight matrices get non-zero gradients

    def test_deterministic(self):
        # Same input + same seed → same output
```

**Gate:** 4 tests → commit

---

### B3.2: MHA with GQA (Grouped-Query Attention)

**What it does:** K and V are shared across groups of query heads.

```python
# Tests test: n_heads=8, n_groups=4 → Q has 8 heads, K/V have 4 heads (broadcast)
# Tests test: output shape still [B, S, D]
# Tests test: gradient flows correctly to shared K/V projections
```

```python
# tests/unit/numpy/test_gqa.py
class TestGroupedQueryAttention:
    def test_k_v_broadcast(self):
        # K and V from n_groups projections, broadcast to n_heads

    def test_output_shape(self):
        # Still [B, S, D]

    def test_gradient_correct(self):
        # K/V grads are summed across all query groups
```

**Gate:** 3 tests → commit

---

### Phase B4: MoE (B5 in design doc)

### B4.1: MoE Expert FFN + Top-K Routing

**What it does:** For each token, selects top-k experts, routes, aggregates.

```python
class MoE:
    # Parameters: router [embed_dim, n_experts], experts: list of SwiGLUFFN
    # Forward: x → compute router_scores = softmax(x @ router.T) → select top_k → weighted sum of expert outputs
```

```python
# tests/unit/numpy/test_moe.py
class TestMoEForward:
    def test_output_shape(self):
        # [B, S, D] → [B, S, D]

    def test_top_k_selection(self):
        # Only top-k experts get non-zero weights

    def test_expert_routing(self):
        # Different inputs get different expert combinations

    def test_gradient_correct(self):
        # Gradients flow to all selected experts
```

**Gate:** 3-4 tests → commit

---

### Phase B5: TransformerBlock (B6 in design doc)

### B5.1: Complete TransformerBlock with residual connections

**What it does:** Combines attention + MoE + LayerNorm + residual in one block.

```python
class TransformerBlock:
    # Components: RMSNorm(x), MultiHeadAttention(x), RMSNorm(h), MoE(h), ResidualAdd
    # Forward: h = x + MHA(RMSNorm(x)) + MoE(RMSNorm(x + MHA(x)))
    # Backward: compute gradients for all internal components
```

```python
# tests/unit/numpy/test_transformer_block.py
class TestTransformerBlockForward:
    def test_output_shape(self):
        # X [B, S, D] → output [B, S, D]

    def test_residual_connection(self):
        # Output contains original input (residual pass-through)

    def test_attention_and_moe(self):
        # Both streams contribute to output

    def test_gradient_chaining(self):
        # Gradients flow through all internal components (MHA → MoE → LN)
```

**Gate:** 4 tests → commit

---

### Phase B6: DecoderStack (B7 in design doc) + Full Model (B8)

### B6.1: DecoderStack — stack of TransformerBlocks

**What it does:** Chains n_layers of TransformerBlocks together.

```python
class DecoderStack:
    # Components: blocks[TransformerBlock] * n_layers
    # Forward: X → block_0 → block_1 → ... → block_{n-1}
```

```python
# tests/unit/numpy/test_decoder_stack.py
class TestDecoderStackForward:
    def test_output_shape(self):
        # X [B, S, D] → output [B, S, D]

    def test_gradient_chaining(self):
        # Gradients flow through all stacked layers

    def test_single_layer(self):
        # Works with n_layers=1
```

**Gate:** 3 tests → commit

---

### B6.2: Full NumPyModel (MHA + MoE enabled)

**What it does:** Complete transformer with embedding → blocks → layernorm → LM head.

```python
class NumPyModel:
    # Components: Embedding, DecoderStack (n_layers blocks), RMSNorm (final), SwiGLUFFN (output_proj)
    # Forward: tokens → embedding → stack → layernorm → output_proj → [B, S, V]
    # Backward: full gradient computation for all parameters
```

```python
# tests/unit/numpy/test_model.py
class TestNumPyModelForward:
    def test_output_shape(self):
        # input_tokens: [B, S] (int32)
        # output: logit_matrix [B, S, V] (float32)

    def test_with_embedding(self):
        # tokens[0] → embedding[0] → through blocks → output

    def test_gradient_existence(self):
        # All parameters have gradients after backward

    def test_small_model(self):
        # Works with minimal config (vocab=16, D=32, layers=1, heads=2)

class TestNumPyModelBackward:
    def test_gradient_shapes(self):
        # gradients match parameter shapes

    def test_numerical_gradient_check(self):
        # Compare backprop w/ finite-diff
        # Forward pass + backward → compare with (f(x+ε) - f(x-ε)) / (2ε)
        # tolerance: rtol=1e-3 for numerical precision
```

**Gate:** Forward (4) + Backward (2) = 6 tests → commit

---

### Phase B7: Loss + Optimizer (B11 in design doc)

### B7.1: Cross-Entropy Loss with masking

**What it does:** Computes per-token loss, handles padding mask.

```python
class CrossEntropyLoss:
    # Forward: logits [B, S, V], targets [B, S], mask [B, S]
    # Returns: scalar loss (mean over non-masked positions)

    # Computes: loss = -mean(mask * log_softmax(logits)[targets])
```

```python
# tests/unit/numpy/test_crossentropy.py
class TestCrossEntropyLossForward:
    def test_scalar_output(self):
        # loss is a scalar

    def test_uniform_logits(self):
        # With uniform logits, loss ≈ log(V) (max entropy)

    def test_masking(self):
        # Masked positions contribute zero to loss

    def test_perfect_predictions(self):
        # If logits are one-hot at target, loss ≈ 0
```

**Gate:** 4 tests → commit

---

### B7.2: AdamW Optimizer

**What it does:** AdamW with weight decay, bias correction.

```python
class AdamW:
    def step(params_dict, grads_dict, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    # Updates each parameter using AdamW update rule
    # Handles weight decay separately from gradient update
```

```python
# tests/unit/numpy/test_adamw.py
class TestAdamW:
    def test_updates_parameters(self):
        # After step, parameter values change

    def test_bias_correction(self):
        # Initial steps account for bias in moments

    def test_weight_decay(self):
        # Weight decay applies L2 regularization
```

**Gate:** 3 tests → commit

---

### Phase B8: Full Training Loop (B12 in design doc)

### B8.1: Training loop with batch iteration

**What it does:** Full training loop: data → batch → forward → loss → backward → step → log.

```python
def train_step(model, batch_input, batch_target, optimizer, loss_fn) -> float:
    # 1. forward = model(batch_input) → [B, S, V]
    # 2. loss = loss_fn(forward, batch_target) → scalar
    # 3. backward = model.backward(forward_grad) → gradients for all params
    # 4. optimizer.step(model.params, gradients)
    # 5. return loss.item()
```

**Tests focus on the orchestration:**
```python
# tests/unit/numpy/test_training_loop.py
class TestTrainingLoop:
    def test_training_reduses_loss(self):
        # Run several steps, loss should decrease

    def test_params_update(self):
        # Model parameters change after training steps

    def test_gradient_accumulation(self):
        # Accumulated gradients used correctly
```

**Gate:** 3 tests → commit

---

### Phase B9: KV Cache (B13, B14 in design doc)

### B9.1: Naive KV Cache

**What it does:** Stores K,V for each layer to avoid recomputing during autoregressive decoding.

```python
class NaiveKVCache:
    def __init__(max_length, n_layers, n_heads, head_dim)
    def update(k, v, pos)  # Store new K,V at position pos
    def get() → (k_cache, v_cache)  # Return all stored K,V
    def clear()
```

```python
# tests/unit/numpy/test_naive_kvcache.py
class TestNaiveKVCache:
    def test_output_shape(self):
        # k_cache, v_cache have correct shapes

    def test_positional_storage(self):
        # update at pos=0, pos=1, pos=2 → cache has 3 positions

    def test_incremental_growth(self):
        # Cache grows sequentially, positions not overwritten

    def test_clear_behavior(self):
        # After clear, cache is empty
```

**Gate:** 4 tests → commit

---

### B9.2: TurboQuant KV Cache

**What it does:** 1-bit compressed K,V storage with per-channel scaling.

```python
class TurboQuantKVCache:
    def __init__(max_length, n_layers, n_heads, head_dim, quant_type="1-bit")
    def update(k, v, pos)  # Quantize, store
    def get() → (dequantized_k, dequantized_v)
```

```python
# tests/unit/numpy/test_turboquant_kvcache.py
class TestTurboQuantKVCache:
    def test_compression_ratio(self):
        # 1-bit storage uses less memory

    def test_dequantization(self):
        # Dequantized values approximate original

    def test_quantization_accuracy(self):
        # Error bounded within expected range
```

**Gate:** 3-4 tests → commit

---

### Phase B10: Inference Engine (B15 in design doc)

### B10.1: Autoregressive inference

**What it does:** Greedy or sampled generation with KV cache.

```python
class NumPyModel:
    def generate(self, prompt_tokens, max_new_tokens, temperature=1.0, top_k=50) → list[int]
    # 1. encode prompt: forward(prompt_tokens)
    2. Take last token logits, sample/argmax
    3. Append to sequence, forward next token with cache
    4. Repeat until max_new_tokens or EOS
    5. Convert to list of token IDs
```

```python
# tests/unit/numpy/test_inference.py
class TestInference:
    def test_output_length(self):
        # Generated tokens have correct length

    def test_greedy_deterministic(self):
        # Same prompt → same output without randomness

    def test_temperature_sampling(self):
        # Higher temperature → more diverse outputs
```

**Gate:** 3 tests → commit

---

### B10.2: CLI interface

**What it does:** `uv run impl/numpy/cli.py --prompt "hello" --max_new_tokens 10`

```bash
# Simple argparse: --prompt, --max_new_tokens, --temperature, --model_dir, --checkpoint_name
# Loads from shared/checkpoint.py + uses NumPyModel
```

```python
# tests/unit/numpy/test_cli.py
# Tests CLI parsing only (no actual inference needed)
class TestCLI:
    def test_help_text(self):
        # CLI --help exits with code 0

    def test_prompt_parsing(self):
        # CLI --prompt correctly parsed
```

**Gate:** 2 tests → commit

---

### Phase B11: Integration tests (end-to-end)

### B11.1: Full pipeline test

**What it does:** Train a small model on synthetic data, verify loss decreases.

```python
# tests/unit/numpy/test_full_training.py
class TestFullTraining:
    def test_loss_decreases(self):
        # Use hardcoded small dataset (no TinyStories needed)
        # 50 steps of training on [16, 16] token sequences
        # Loss should clearly decrease

    def test_save_load(self):
        # Save model → load → same params → same forward pass

    def test_inference_after_training(self):
        # After training, generate text (no errors)
```

**Gate:** 3 tests → commit

---

### B11.2: Model config validation + integration with shared module

**What it does:** NumPyModel integrates with `shared/config.py` and `shared/constants.py`.

```python
# tests/unit/numpy/test_numpy_config_integration.py
class TestNumPyModelConfigIntegration:
    def test_builds_with_transformer_config(self):
        # NumPyModel(TransformerConfig(...)) works

    def test_all_params_initialized(self):
        # All config parameters produce model parameters

    def test_param_count_matches_constants(self):
        # NumPy model has expected number of parameters (from shared.constants)
        # Each parameter name from shared constants
```

**Gate:** 3 tests → commit

---

## Summary

| Stage | Commit Message | Test Count |
|-------|---------------|------------|
| B0.1 | `b0: project scaffolding` | 1 (import test) |
| B1.1 | `b1: NumPy Embedding layer — 3 tests` | 3 |
| B1.2 | `b2: RMSNorm — 6 tests` | 6 |
| B1.3 | `b3: SiLU activation — 4 tests` | 4 |
| B1.4 | `b4: SwiGLU FFN — 4 tests` | 4 |
| B2.1 | `b5: RoPE position encoding — 4 tests` | 4 |
| B3.1 | `b6: Multi-Head Attention (no GQA) — 4 tests` | 4 |
| B3.2 | `b7: Grouped-Query Attention — 3 tests` | 3 |
| B4.1 | `b8: MoE with top-k routing — 4 tests` | 4 |
| B5.1 | `b9: TransformerBlock — 4 tests` | 4 |
| B6.1 | `b10: DecoderStack — 3 tests` | 3 |
| B6.2 | `b11: NumPyModel full (forward+backward) — 6 tests` | 6 |
| B7.1 | `b12: Cross-Entropy Loss — 4 tests` | 4 |
| B7.2 | `b13: AdamW Optimizer — 3 tests` | 3 |
| B8.1 | `b14: Training loop — 3 tests` | 3 |
| B9.1 | `b15: Naive KV Cache — 4 tests` | 4 |
| B9.2 | `b16: TurboQuant KV Cache — 4 tests` | 4 |
| B10.1 | `b17: Autoregressive Inference — 3 tests` | 3 |
| B10.2 | `b18: CLI interface — 2 tests` | 2 |
| B11.1 | `b19: Full training pipeline — 3 tests` | 3 |
| B11.2 | `b20: NumPy model ↔ shared module integration — 3 tests` | 3 |

**Total: ~65-70 new tests, 21 commits**

---

## Phase B Final Gate

After all stages complete:

```bash
# 1. All unit tests pass
PYTHONPATH=shared PYTHONPATH=impl uv run pytest tests/unit/numpy/ -v --timeout=300

# 2. Ruff + pyright clean on NumPy code
PYTHONPATH=shared ruff check impl/numpy/ tests/unit/numpy/
PYTHONPATH=shared pyright impl/numpy/ tests/unit/numpy/

# 3. Small model trains on TinyStories (sanity check)
PYTHONPATH=shared PYTHONPATH=impl uv run impl/numpy/cli.py --prompt "The" --max_new_tokens 5 --model_dir test_model --n_layers 1 --embed_dim 32 --n_heads 2 --vocab_size 512

# 4. Checkpoint round-trip (NumPy)
PYTHONPATH=shared uv run impl/numpy/cli.py train --checkpoint_name tiny_demo --n_layers 1 --embed_dim 32 --n_heads 2

# 5. All existing tests still pass
PYTHONPATH=shared uv run pytest tests/unit/ -v --timeout=300
```

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Backward pass math is complex | High | Test each layer's backward independently before composing |
| Numerical instability in attention | Medium | Use `max(0, scores)` before softmax, use float64 for testing |
| Memory blowup with large tensors | Low | Use smallest possible configs (vocab=16, D=32) |
| Numerical gradient mismatches | Medium | Use finite-diff with ε=1e-5, compare with rtol=1e-3 |
| Test runs too slowly | Low | Use tiny configs (1 layer, 32 embed) in all tests |

---

## What NumPy Model Produces

After training, the model produces:

```
Input: "Once upon a time"
Output: "Once upon a time the cat sat on the mat. She was happy..." (generated text)

Checkpoint saved at: models/tiny_demo/
  ├── config.json   (TransformerConfig as JSON)
  └── model.npz     (numpy arrays for all parameters)
```
