# Phase C: PyTorch Implementation — Execution Plan

**Status:** ⏳ PLANNED but NOT EXECUTED — Execution ready
**Start Date:** — (not started)
**End Date:** —
**Progress:** 0/14 sub-phases, 0/20 commits, 0/65 tests

## Progress Summary

| Stage | Status | Tests | Description |
|-------|--------|-------|-------------|
| C0: Project scaffolding | ⏳ Not started | 0/1 | Directories and package initialization |
| C1: Basic layers | ⏳ Not started | 0/17 | Embedding, RMSNorm, SiLU, SwiGLU |
| C2: RoPE Position Encoding | ⏳ Not started | 0/4 | Rotary position embeddings |
| C3: MHA with GQA | ⏳ Not started | 0/6 | Multi-head attention |
| C4: MoE | ⏳ Not started | 0/4 | Expert routing |
| C5: TransformerBlock | ⏳ Not started | 0/4 | Attention + MoE + LN |
| C6: DecoderStack | ⏳ Not started | 0/3 | Stacked blocks |
| C7: Full TorchModel | ⏳ Not started | 0/6 | Forward + backward + parity |
| C8: Loss + Optimizer | ⏳ Not started | 0/4 | CrossEntropy + AdamW |
| C9: Training Loop | ⏳ Not started | 0/3 | Autograd training |
| C10: KV Cache | ⏳ Not started | 0/8 | Naive + TurboQuant |
| C11: Inference Engine | ⏳ Not started | 0/3 | Greedy + sampled |
| C12: CLI | ⏳ Not started | 0/2 | Argument parsing |
| C13: End-to-end + Parity | ⏳ Not started | 0/2 | Full pipeline test |
| C14: Cross-Backend Parity | ⏳ Not started | 0/10 | NumPy vs PyTorch |

---

## Goal

Build a **production-ready** decoder-only transformer in PyTorch, mirroring the NumPy reference implementation exactly. The key difference: `torch.autograd` replaces manual finite-difference gradients.

**Principle:** Every layer produces numerically identical output to its NumPy counterpart at `float64` precision.

---

## Architecture (PyTorch-specific)

```
Input (tokens: [batch, seq_len])
    │
    ▼
┌─────────────────────┐
│  Token Embedding     │ shape: [B, S, D]
├─────────────────────┤
│  RMSNorm (ln1)       │ nn.LayerNorm-style normalization
├─────────────────────┤
│  RoPE                │ position encoding on Q, K
├─────────────────────┤
│  Stream 1: MHA       │ → [B, S, D]
│    ├── Q, K, V proj  │ nn.Linear
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

**Exactly the same forward graph as NumPy** — only the runtime backend changes.

---

## Directory Structure

```
impl/
├── _np/            # Existing — NumPy implementation
└── _torch/         # NEW — PyTorch implementation
    ├── __init__.py           # Package init + public API exports
    │
tests/
├── unit/
│   ├── _np/                # Existing — NumPy tests
│   └── _torch/             # NEW — PyTorch-specific tests
│       └── __init__.py
└── cross_backend/         # NEW — Parity tests
    └── __init__.py
```

---

## TDD Process (same as Phase B)

```
Step 1: Write test file — ALL TESTS FAIL
  Write tests/unit/_torch/test_<component>.py — only tests, NO implementation
  Run: PYTHONPATH=shared PYTHONPATH=impl uv run pytest tests/unit/_torch/test_<component>.py -v --timeout=10
  Expected: ALL tests fail (ModuleNotFoundError / ImportError)

Step 2: Implement minimal code
  Write minimal implementation that makes ALL tests pass
  Run: ALL pass

Step 3: Quality check
  PYTHONPATH=shared ruff check impl/_torch/<component>.py tests/unit/_torch/test_<component>.py
  PYTHONPATH=shared pyright impl/_torch/<component>.py tests/unit/_torch/test_<component>.py

Step 4: Commit
  git add -A && git commit -m "c<stage>: <component> — <N> tests pass"
```

### Rules
- NEVER implement without failing tests first
- ONE component per commit (atomic)
- ONE test file per component
- Small, fast tests — pure PyTorch, no network
- **Test tolerance policy** (AGENTS.md tiered policy):
  - **Standalone layers** (tested in isolation): `rtol=1e-4, atol=1e-4` — e.g., RMSNorm, SiLU, SwiGLU, RoPE, MHA tested independently without chaining through multiple layers
  - **Component in single chain** (e.g., TransformerBlock with one level of gradient accumulation): `rtol=1e-3, atol=1e-3`
  - **Multi-layer chains** (DecoderStack, full model): `rtol=1e-2, atol=1e-2`
- All tests use `dtype=torch.float64` for numerical comparisons
- After training (where float32 is the actual dtype), `float64` test comparisons use `rtol=1e-2`

---

## Stages (14 sub-phases, 20+ commits)

### Phase C0: Project scaffolding

- [ ] **C0.1** | `impl/_torch/`, `tests/unit/_torch/`, `tests/cross_backend/` directories, `__init__.py` files | pytest collects 0 tests | 0
- [ ] **C0.2** | `tests/unit/_torch/test_modules.py` — import the package | pytest discovers tests | 1

```python
# tests/unit/_torch/test_modules.py
from impl._torch import TorchModel, ModelConfig

def test_torchModel_imports():
    assert TorchModel is not None
```

---

### Phase C1: Basic layers (layers in `modules.py`)

#### C1.1: Embedding layer

**What it does:** Maps token IDs to dense vectors.

```python
class Embedding(nn.Module):
    # Parameters: weight [vocab_size, embed_dim] — stored as nn.Parameter
    # Forward: x → lookup(x, weight) → [batch, seq_len, embed_dim]
    # Forward takes: input_ids: torch.Tensor[int32/64]
    # Returns: torch.Tensor[float32/64]
```

```python
# tests/unit/_torch/test_embedding.py
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

**Gate:** 3 tests → commit

---

#### C1.2: RMSNorm (LayerNorm-style normalization)

**What it does:** Normalizes each feature vector to unit variance, scales by learned gamma.

```python
class RMSNorm(nn.Module):
    # Parameters: gamma [embed_dim] (nn.Parameter)
    # Forward: x → x / sqrt(mean(x^2)) * gamma
    #   Rms(x) = sqrt(mean(x^2)) + eps
```

```python
# tests/unit/_torch/test_rmsnorm.py
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
        # Gradient check via autograd
```

**Gate:** Forward tests (4) + backward tests (2) = 6 tests → commit

---

#### C1.3: SiLU activation

**What it does:** Element-wise Swish/SiLU activation: f(x) = x * sigmoid(x)

```python
class SiLULayer(nn.Module):
    # Forward: x → x * sigmoid(x) → element-wise
    # Backward: handled automatically by autograd
```

```python
# tests/unit/_torch/test_silu.py
class TestSiLULayer:
    def test_output_shape(self):
        # Same shape as input

    def test_output_at_zero(self):
        # SiLU(0) = 0 * 0.5 = 0

    def test_output_range_large_positive(self):
        # SiLU(x) ≈ x for large x

    def test_output_range_negative(self):
        # SiLU(x) ≈ 0 for large negative x
```

**Gate:** 4 tests → commit

---

#### C1.4: SwiGLU feedforward

**What it does:** SwiGLU — a modern feedforward with gating:

```python
class SwiGLUFFN(nn.Module):
    # Parameters: w1 [embed_dim, ff_dim], w2 [ff_dim, embed_dim], w3 [embed_dim, ff_dim]
    # Forward: SiLU(w1 @ x) * (w3 @ x) @ w2
    # Gate mechanism: SiLU provides smooth gating
```

```python
# tests/unit/_torch/test_swiglu.py
class TestSwiGLUFFN:
    def test_output_shape(self):
        # [batch, seq_len, embed_dim] → [batch, seq_len, embed_dim]

    def test_gating_behavior(self):
        # w1 and w3 projections gated together

    def test_ff_dim_independence(self):
        # output size does not depend on ff_dim

    def test_gradient_existence(self):
        # All weights get non-zero gradients through autograd
```

**Gate:** 4 tests → commit

---

### Phase C2: RoPE Position Encoding

#### C2.1: RoPE (Rotary Positional Embedding)

**What it does:** Injects positional information into Q and K matrices via rotation.

```python
class RoPE(nn.Module):
    # No learnable parameters (weights computed from config)
    # Forward: Q, K → apply rotation by position → rotated Q, K
    # Formula: q_m' = q_m * cos(mθ) - q_{m+r} * sin(mθ)
```

```python
# tests/unit/_torch/test_rope.py
class TestRoPEForward:
    def test_output_shape(self):
        # Same shape as input (Q, K unchanged shape)
    
    def test_rotates_by_position(self):
        # Position 0 and position 1 produce different rotations
    
    def test_full_vs_partial(self):
        # rope_dim=0 (full) vs rope_dim<n (partial) behave correctly
```

**Gate:** 3-4 tests → commit

---

### Phase C3: MHA (Multi-Head Attention)

#### C3.1: Multi-Head Attention with GQA

**What it does:** Standard scaled dot-product attention with grouped-query attention.

```python
class MultiHeadAttention(nn.Module):
    # Parameters: q_proj [embed_dim, n_heads*head_dim], k_proj [...], v_proj [...], o_proj [embed_dim, embed_dim]
    # Forward: X → Q,K,V projections → reshaped → softmax(QK^T/sqrt(d)) → concatenate → output_proj
    # Returns: [batch, seq_len, embed_dim]
    # Supports: past_key_value for KV cache inference
```

```python
# tests/unit/_torch/test_attn.py
class TestMultiHeadAttentionForward:
    def test_output_shape(self):
        # X [B, S, D] → output [B, S, D]

    def test_attention_mechanism(self):
        # Softmax normalized across sequence dimension

    def test_gqa_support(self):
        # n_groups < n_heads: K/V shared across groups

    def test_gradient_flow(self):
        # All weight matrices get non-zero gradients

    def test_deterministic(self):
        # Same input + same seed → same output

    def test_kv_cache_input(self):
        # Accepts past_key_value for autoregressive decoding
```

**Gate:** 6 tests → commit

---

### Phase C4: MoE (Mixture of Experts)

#### C4.1: MoE Expert FFN + Top-K Routing

**What it does:** For each token, selects top-k experts, routes, aggregates.

```python
class MoE(nn.Module):
    # Parameters: router [embed_dim, n_experts], experts: nn.ModuleList of SwiGLUFFN
    # Forward: x → softmax(x @ router.T) → top-k → weighted sum of expert outputs
```

```python
# tests/unit/_torch/test_moe.py
class TestMoEForward:
    def test_output_shape(self):
        # [B, S, D] → [B, S, D]

    def test_top_k_selection(self):
        # Only top-k experts get non-zero weights

    def test_expert_routing(self):
        # Different inputs get different expert combinations

    def test_gradient_flow(self):
        # Gradients flow to all selected experts
```

**Gate:** 4 tests → commit

---

### Phase C5: TransformerBlock + DecoderStack

#### C5.1: Complete TransformerBlock

**What it does:** Combines attention + MoE + LayerNorm + residual in one block.

```python
class TransformerBlock(nn.Module):
    # Components: RMSNorm(x), MultiHeadAttention(x, kv_cache), MoE, ResidualAdd
    # Forward: h = x + MHA(RMSNorm(x)) + MoE(RMSNorm(x + MHA(x)))
    # Accepts: past_key_value for KV cache integration
    # Returns: [batch, seq_len, embed_dim]
```

```python
# tests/unit/_torch/test_transformer_block.py
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

**Gate:** 4 tests → commit (this stage can run parallel to Phase C6)

---

### Phase C6: DecoderStack

#### C6.1: DecoderStack — stack of TransformerBlocks

**What it does:** Chains n_layers of TransformerBlocks together.

```python
class DecoderStack(nn.Module):
    # Components: modules_list[TransformerBlock] * n_layers
    # Forward: X → block_0 → block_1 → ... → block_{n-1}
```

```python
# tests/unit/_torch/test_decoder_stack.py
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

### Phase C7: Full PyTorch Model (MHA + MoE enabled)

#### C7.1: Full TorchModel (forward + autograd backward)

**What it does:** Complete transformer with embedding → blocks → layernorm → output projection.

```python
class TorchModel(nn.Module):
    # Components: Embedding, DecoderStack (n_layers blocks), RMSNorm (final), SwiGLU (output_proj)
    # Forward: tokens → embedding → stack → layernorm → output_proj → [B, S, V]
    # Backward: loss.backward() → torch.autograd computes gradients
```

```python
# tests/unit/_torch/test_model.py
class TestTorchModelForward:
    def test_output_shape(self):
        # input_tokens: [B, S] (int64)
        # output: logit_matrix [B, S, V] (float32/64)

    def test_with_embedding(self):
        # tokens[0] → embedding[0] → through blocks → output

    def test_gradient_existence(self):
        # All parameters have gradients after backward

    def test_small_model(self):
        # Works with minimal config (vocab=16, D=32, layers=1, heads=2)

    def test_cross_backend_parity(self):
        # Same seed + same input → same output as NumPyModel (float64, rtol=1e-4)
        import torch
        import numpy as np
        from impl._torch import TorchModel
        from impl._np import NumPyModel
        # Build both models with same seed
        # Forward with float64 on both
        # Compare: rtol=1e-4, atol=1e-4

class TestTorchModelBackward:
    def test_backward_runs(self):
        # model backward() → autograd succeeds

    def test_all_params_have_grads(self):
        # All parameters have non-zero gradients after loss.backward()
```

**Gate:** 6 tests → commit (5 forward + 1 backward)

---

### Phase C8: Loss + Optimizer

#### C8.1: Cross-Entropy Loss wrapper

**What it does:** Wraps `torch.nn.functional.cross_entropy` with shift/mask support.

```python
class CrossEntropyLoss:
    # Forward: logits [B, S, V], targets [B, S], mask [B, S], ignore_index -100
    # Returns: scalar loss (mean over non-masked positions)
    
    # Computes: loss = -mean(mask * log_softmax(logits)[targets])
```

```python
# tests/unit/_torch/test_crossentropy.py
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

### Phase C9: Training Loop (autograd)

#### C9.1: Training loop with batch iteration

**What it does:** Full training loop: data → batch → forward → loss → backward → step → log.

```python
def train_step(model, batch_input, batch_target, optimizer, loss_fn) -> float:
    # 1. forward = model(batch_input) → [B, S, V]
    # 2. loss = loss_fn(forward, batch_target) → scalar
    # 3. loss.backward() → autograd computes gradients
    # 4. optimizer.step() → update parameters
    # 5. return loss.item()
```

**Tests focus on the orchestration:**

```python
# tests/unit/_torch/test_training_loop.py
class TestTrainingLoop:
    def test_training_reduces_loss(self):
        # Run several steps, loss should decrease

    def test_params_update(self):
        # Model parameters change after training steps

    def test_autograd_gradients(self):
        # After backward, model parameters have valid gradients
```

**Gate:** 3 tests → commit

---

### Phase C10: KV Cache + Inference

#### C10.1: TorchNaiveKVCache (parallel with C9)

**What it does:** Stores K,V for each layer — torch tensor version of NumPy KV cache.

```python
class TorchNaiveKVCache:
    def __init__(max_length, n_layers, n_heads, head_dim)
    def update(k, v, pos)    # Store new K,V at position pos
    def get() → (k_cache, v_cache)  # Return all stored K,V
    def clear()
```

```python
# tests/unit/_torch/test_naive_kvcache.py
class TestTorchNaiveKVCache:
    def test_output_shape(self):
        # k_cache, v_cache have correct shapes

    def test_positional_storage(self):
        # update at pos=0, pos=1, pos=2 → cache has 3 positions

    def test_incremental_growth(self):
        # Cache grows sequentially, positions not overwritten

    def test_clear_behavior(self):
        # After clear, cache is empty
```

**Gate:** 4 tests → commit — **can run in parallel with C10.2**

---

#### C10.2: TorchTurboQuantKVCache (parallel with C10.1)

**What it does:** 1-bit compressed K,V storage with per-channel scaling.

```python
class TorchTurboQuantKVCache:
    def __init__(max_length, n_layers, n_heads, head_dim, quant_type="1-bit")
    def update(k, v, pos)   # Quantize, store
    def get() → (dequantized_k, dequantized_v)
```

```python
# tests/unit/_torch/test_turboquant_kvcache.py
class TestTorchTurboQuantKVCache:
    def test_compression_ratio(self):
        # 1-bit storage uses less memory

    def test_dequantization(self):
        # Dequantized values approximate original

    def test_quantization_accuracy(self):
        # Error bounded within expected range
```

**Gate:** 3-4 tests → commit — **completes C10**

---

### Phase C11: Inference Engine

#### C11.1: Autoregressive inference

**What it does:** Greedy or sampled generation with KV cache.

```python
class TorchTextGenerator:
    def __init__(model, max_new_tokens, temperature, top_k)
    def generate(prompt_tokens) → list[int]  # full sequence
    def generate_greedy(prompt_tokens) → list[int]
    def generate_sampled(prompt_tokens, temperature) → list[int]
```

```python
# tests/unit/_torch/test_inference.py
class TestTorchInference:
    def test_output_length(self):
        # Generated tokens have correct length

    def test_greedy_deterministic(self):
        # Same prompt → same output without randomness

    def test_temperature_sampling(self):
        # Higher temperature → more diverse outputs
```

**Gate:** 3 tests → commit

---

### Phase C12: CLI

#### C12.1: CLI interface

**What it does:** `uv run impl/_torch/cli.py --prompt "hello" --max_new_tokens 10`

```python
# Simple argparse: --prompt, --max_new_tokens, --temperature, --model_dir, --checkpoint_name
# Uses TorchModel instead of NumPyModel
```

```python
# tests/unit/_torch/test_cli.py
class TestCLI:
    def test_help_text(self):
        # CLI --help exits with code 0

    def test_prompt_parsing(self):
        # CLI --prompt correctly parsed
```

**Gate:** 2 tests → commit

---

### Phase C13: End-to-end + Parity

#### C13.1: Full training pipeline test

**What it does:** Train a small model on synthetic data, verify loss decreases.

```python
# tests/unit/_torch/test_full_training.py
class TestFullTraining:
    def test_loss_decreases(self):
        # Use hardcoded small dataset (no TinyStories needed)
        # 50 steps of training on [16, 16] token sequences
        # Loss should clearly decrease

    def test_save_load(self):
        # Save model → load → same params → same forward pass
```

**Gate:** 2 tests → commit

---

### Phase C14: Cross-Backend Parity Tests

**What it does:** Test that every torch layer produces numerically identical output to its NumPy counterpart.

```python
# tests/cross_backend/test_layer_parity.py — Standalone layers: rtol=1e-4
# tests/cross_backend/test_single_chain.py   — 1-layer model: rtol=1e-3
# tests/cross_backend/test_multilayer_chain.py — 2+ layers: rtol=1e-2
# tests/cross_backend/test_checkpoint_roundtrip.py — checkpoint format (both read same .npz)
# tests/cross_backend/test_inference_equivalence.py — greedy generation: exact match
```

**Gate:** ~10 tests → commit

---

## Summary Table

| Stage | Status | Commit Message | Test Count | Description |
|-------|--------|---------------|------------|-------------|
| C0.1 | [ ] | `c0: project scaffolding` | 0 | Directories |
| C0.2 | [ ] | `c0: package imports` | 1 | Import tests |
| C1.1 | [ ] | `c1: PyTorch Embedding — 3 tests` | 3 | nn.Embedding or nn.Parameter lookup |
| C1.2 | [ ] | `c2: RMSNorm — 6 tests` | 6 | nn.Parameter gamma + forward/backward |
| C1.3 | [ ] | `c3: SiLU activation — 4 tests` | 4 | F.silu |
| C1.4 | [ ] | `c4: SwiGLU FFN — 4 tests` | 4 | nn.Linear + GELU/SiGLU |
| C2.1 | [ ] | `c5: RoPE position encoding — 4 tests` | 4 | Rotary position embeddings |
| C3.1 | [ ] | `c6: MHA with GQA — 6 tests` | 6 | nn.Linear projections + attention, KV cache support |
| C4.1 | [ ] | `c7: MoE top-k routing — 4 tests` | 4 | Softmax router + top-k experts |
| C5.1 | [ ] | `c8: TransformerBlock — 4 tests` | 4 | Attention + MoE + LN + residuals |
| C6.1 | [ ] | `c9: DecoderStack — 3 tests` | 3 | Chained blocks |
| C7.1 | [ ] | `c10: TorchModel full — 6 tests` | 6 | Forward + backward + cross-backend parity |
| C8.1 | [ ] | `c11: CrossEntropyLoss — 4 tests` | 4 | F.cross_entropy with shift |
| C9.1 | [ ] | `c12: Training loop — 3 tests` | 3 | autograd + optimizer.step |
| C10.1 | [ ] | `c13: Naive KV Cache — 4 tests` | 4 | Torch tensors cache |
| C10.2 | [ ] | `c14: TurboQuant KV Cache — 4 tests` | 4 | (parallel C10.1) |
| C11.1 | [ ] | `c15: Inference — 3 tests` | 3 | Greedy + sampled generation |
| C12.1 | [ ] | `c16: CLI interface — 2 tests` | 2 | argparse entry point |
| C13.1 | [ ] | `c17: Full training pipeline — 2 tests` | 2 | Train + save/load |
| C14.1 | [ ] | `c18: Cross-backend parity — 10 tests` | 10 | NumPy vs PyTorch |

**Total: ~65-70 new PyTorch tests, ~20 commits**

---

## Parallel Execution Map (same as Phase B)

```
Phase C1-C4: Basic layers (Embedding, RMSNorm, SiLU, SwiGLU)
         ↓
Phase C2: RoPE (parallel to C3-C4 since no deps)
         ↓
Phase C3: MHA (parallel to Phase C4: MoE — no dependencies)
         ↓
Phase C5-C6: TransformerBlock, DecoderStack (sequential, depend on C3+C4)
         ↓
Phase C7: Full TorchModel (depends on C6)
         ↓
Phase C8: CrossEntropyLoss (parallel to C7)
         ↓
Phase C9: Training Loop (depends on C7, C8)
    ←→ C10.1 + C10.2: KV Cache (parallel subagents)
         ↓
Phase C11: Inference (depends on C10)
         ↓
Phase C12: CLI (depends on C11)
         ↓
Phase C13: Full Pipeline (parallel with C14 parity)
         ↓
Phase C14: Cross-backend parity (parallel with C13)
```

**Dispatch plan (3 waves):**
1. **Wave 1:** C1+C2+C3+C4 (4 base layers + RoPE) — as many parallel subagents as needed
2. **Wave 2:** C5+C6+MHA+MoE all at once since they're all independent of each other
3. **Wave 3:** C7 (full model + loss) + C10.1 + C10.2 (both KV caches) + C8 (loss) — all parallel
4. **Wave 4:** C9 (training loop), C11 (inference), C13 (full pipeline), C14 (parity) — most of these are parallel

---

## Phase C Final Gate

After all stages complete:

```bash
# 1. All unit tests pass
PYTHONPATH=shared PYTHONPATH=impl uv run pytest tests/unit/_torch/ -v --timeout=300

# 2. Cross-backend parity passes
PYTHONPATH=shared PYTHONPATH=impl uv run pytest tests/cross_backend/ -v --timeout=300

# 3. Ruff + pyright clean on PyTorch code
PYTHONPATH=shared ruff check impl/_torch/ tests/unit/_torch/ tests/cross_backend/
PYTHONPATH=shared pyright impl/_torch/ tests/unit/_torch/ tests/cross_backend/

# 4. All existing tests still pass
PYTHONPATH=shared uv run pytest tests/unit/ -v --timeout=300

# 5. Tiny model trains and generates text
PYTHONPATH=shared PYTHONPATH=impl uv run impl/_torch/cli.py --prompt "The" --max_new_tokens 5
```

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| PyTorch random initialization !== NumPy initialization | High | Use same seed patterns; initialize weights identically (e.g., Kaiming uniform → match NumPy's default_rng behavior) |
| Float32 vs Float64 precision differences | Medium | All parity tests use float64; inference tests on float32 use rtol=1e-2 |
| Autograd numerical precision differs from finite-diff | Medium | Compare forward output (not gradients) in parity tests; compare backward on a subset of params |
| GQA implementation nuances | Medium | Match NumPy's exact reshaping/gather pattern |
| Test slowness with 2+ layer chains | Low | Use 1-layer models in parity tests; 2+ layers only in final parity test |
| Module organization (single file vs multiple) | Medium | Follow same `impl/_torch/` multi-file structure as `impl/_np/` |

---

## Key Differences From NumPy

| Aspect | NumPy | PyTorch |
|--------|-------|---------|
| Forward output dtype | float32 / float64 | Matches input dtype |
| Weight storage | instance attributes | `nn.Parameter` / `nn.Linear` |
| Backward pass | Finite-difference (expensive) | `loss.backward()` (fast) |
| Parameter dict | `get_all_parameters()` → dict | `model.state_dict()` |
| Optimizer | Manual AdamW step | `optimizer.step()` |
| Cross-entropy loss | Manual logsumexp | `F.cross_entropy` |
| KV cache | `np.ndarray` buffers | `torch.Tensor` buffers (same shape) |

---

*Plan ready for review. 14 sub-phases, ~20 atomic commits, ~65-70 tests across PyTorch + cross_backend.*
