# Phase 3++: Normalization & Architecture Improvements

**Status:** 🔲 Not Started
**Preceded by:** Phase 3+ (E2E Training/Inference) — Complete, 400 tests, ruff/pyright clean
**Goal:** Investigate and implement normalization/architecture improvements for faster training and better gradient flow

---

## 1. Background

### Current Architecture: Pre-Norm
Both NumPy and PyTorch implementations use **pre-normalization** (RMSNorm BEFORE the residual add):

```python
# Current flow (NumPy: modules.py:894-910, PyTorch: layers.py:643-672)
ln1_out = RMSNorm(x, gamma)        # (B, S, D) → (B, S, D)
attn_out = MHA(ln1_out)             # attention output
h = x + attn_out                    # residual: x + attention

ln2_out = RMSNorm(h, gamma)         # norm the intermediate result
moe_out = MoE(ln2_out)               # mixture of experts
out = h + moe_out                    # residual: intermediate + MoE
```

**Mathematical formula:**
```
h  = x + MHA(RMSNorm(x))
out = h + MoE(RMSNorm(h))
```

This is the standard **Pre-LayerNorm** (GPT-style) architecture.

### User Concern
User noted: *"we lack of residual connection which could speed up training and avoid signal vanish feature in current model"*

### Observation
The codebase **already has residual connections**. Both implementations have:
- `h = x + attn_out` (first residual: input + attention output)
- `out = h + moe_out` (second residual: intermediate + MoE output)

**Possible interpretations:**
1. **Post-Norm** — User may want residual add BEFORE norm (swapped order)
2. **Gated Residuals** — User may want learnable gates on residuals
3. **Dropout** — User may be confusing "dropout" with "residual" for regularization
4. **Skip Connections** — User may be thinking of DenseNet-style multi-layer skips

### Action Required
❗ **Clarify with user before implementing.** Do not proceed without explicit guidance.

---

## 2. Candidate Improvements

### Option A: Post-Norm (Residual First, Then Norm)
Swaps the order: compute residual, then normalize for next block.

```python
# Post-Norm variant
h = x + attn_out       # residual FIRST
h = RMSNorm(h, gamma)   # THEN normalize

moe_out = MoE(...)      # MoE on pre-norm'd h
out = h + moe_out       # residual
out = RMSNorm(out, gamma) # normalize output
```

**Pros:**
- Slightly faster training convergence in some architectures (e.g., BERT)
- Simpler gradient flow (direct path)

**Cons:**
- Less stable training at deeper layers (exploding gradients)
- Standard for encoder (BERT), not decoder (GPT) — **our goal is decoder**

**Change scope:**
- NumPy: Modify `TransformerBlock.forward()` (`modules.py:881-912`)
- PyTorch: Modify `TransformerBlock.forward()` (`layers.py:643-672`)
- Tests: Update residual gradient checks

### Option B: Gated Residuals
Add a learnable gate parameter to each residual: `x + gate * residual`

```python
self.gate1 = np.zeros(embed_dim)  # Learnable gate (init to zeros for gentle start)
h = x + self.gate1 * attn_out     # gated residual

self.gate2 = np.zeros(embed_dim)
out = h + self.gate2 * moe_out    # gated residual
```

**Pros:**
- Controls signal flow — prevents both exploding and vanishing gradients
- Used in successful architectures (Deep & Cross Network, ResNet variations)
- Smooth initialization (gate≈0 → starts near identity)

**Cons:**
- Adds 2 learnable parameters per layer (negligible overhead)
- Slightly more complex backward pass (gate gradient)

**Change scope:**
- NumPy: `modules.py:871-873` (add gate params), `forward()` method
- PyTorch: `layers.py` (add `nn.Parameter` gates), `forward()` method
- Tests: gate initialization, gradient norm on gate params

### Option C: Drop-In Dropout for Regularization
Not a residual improvement, but addresses regularization: currently **zero dropout** exists anywhere in the codebase.

```python
# Add dropout to specific activations (not to residual connections)
h = x + F.dropout(attn_out, p=0.1, training=self.training)
out = h + F.dropout(moe_out, p=0.1, training=self.training)
```

**Pros:**
- Reduces overfitting during training
- Standard LLM practice

**Cons:**
- Makes training non-deterministic (requires same seed for parity)
- Cross-backend parity becomes probabilistic (need statistical tolerance)

**Change scope:**
- NumPy/PyTorch: Add dropout after each residual (train mode only)
- Tests: Update parity checks with stochastic tolerance

---

## 3. Recommendation

**Primary goal:** "Speed up training and avoid signal vanishing."

**Recommended approach (in order of preference):**

1. **Gated Residuals (Option B)** — Best balance of signal control + stability + existing architecture compatibility
2. **Post-Norm (Option A)** — Simpler swap, but less suitable for decoder architectures
3. **Add Dropout (Option C)** — Complementary but introduces stochasticity

**Not recommended without user request:**
- DenseNet-style skip connections (adds significant complexity, minimal benefit for small models)
- Layer-wise LR decay (orthogonal to residual concern)

---

## 4. Implementation Plan (Once User Confirms)

### Step 1: Baseline Experiment (No Architectural Changes)
Before changing architecture, establish baseline:
1. Train current model → record loss curve, gradient norms, convergence steps
2. This ensures we can measure **any** improvement

### Step 2: Pick One Improvement
Based on user confirmation.

### Step 3: Write Failing Test First (TDD)
```python
# Example: gated residual gradient flow test
def test_residual_gradient_flow():
    """Gated residuals should maintain gradient norm > threshold."""
    model = TorchModel(n_layers=4, ...)
    x = torch.randn(1, 10, model.config.embed_dim)
    y = torch.randint(0, model.config.vocab_size, (1, 10))
    loss = loss_fn(model(x), y)
    loss.backward()
    
    # Check gradient norms per layer
    for i, name, p in get_layer_grads(model):
        if i >= 2 and "gate" in name:
            # Gate gradient should NOT vanish
            assert p.grad.norm() > 1e-6, f"Gate gradient vanished at layer {i}"
```

### Step 4: Implement in NumPy First
TDD approach: write test → implement → test passes.

### Step 5: Mirror in PyTorch
Implement parallel changes in PyTorch, ensuring parity.

### Step 6: Cross-Backend Parity Test
Verify both backends produce equivalent results:
- Same forward pass outputs (rtol=1e-3)
- Same gradient norms (rtol=1e-2)
- Same training loss curve

### Step 7: Update Scripts & Tests
- Update train.py to use new architecture
- Update verify_equivalence.py
- Update auto_test_equivalence.py matrix

---

## 5. TDD Approach

### Test-First Commit Pattern
Each sub-change follows:
1. **Write failing test** — captures desired behavior
2. **Implement minimum code** — makes test pass
3. **Verify all tests** — 400+ existing tests still pass
4. **Ruff + Pyright** — clean code
5. **Atomic commit** — one sub-change per commit

### Test Categories
| Test | Purpose | Tolerance |
|------|---------|-----------|
| `test_residual_direction` | Forward pass produces output != input | max_diff > 1e-6 |
| `test_residual_gradient_flow` | Gradients flow through residual in deep layers | grad_norm > 1e-6 |
| `test_no_vanishing_gradients` | No layer has near-zero gradient after 10 steps | grad_norm > 1e-4 |
| `test_pre_vs_post_norm` | Compare training curves (if user requests both) | qualitative |
| `test_gate_init_gentle` | Gate initialization starts near zero | gate_init < 1e-3 |
| `test_gate_grad_non_vanishing` | Gate gradient stays significant after training | avg_grad > 1e-4 |

---

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Post-Norm instability with deep layers | High | Start with 2 layers, gradually increase; monitor training curves |
| Gated residuals change gradient flow unpredictably | Medium | Initialize gate to zeros → starts as identity (no change) |
| Cross-backend parity breaks | Medium | Test both backends in lockstep; parity test runs after each commit |
| Training time increases | Low | Gated residuals add ~0.1ms per forward pass; negligible for small models |
| Dropout makes tests non-deterministic | High | If used, only enable for non-parity tests (e.g., train-only experiments) |

---

## 7. Phase 3++ Gate Checklist

Before Phase 3++ is considered complete:

- [ ] User has confirmed which architectural improvement to implement
- [ ] Baseline training metrics captured (loss curve, gradient norms, convergence rate)
- [ ] Failing test written BEFORE implementation (TDD mandate)
- [ ] Implementation in NumPy backend (`impl/_np/modules.py`)
- [ ] Implementation in PyTorch backend (`impl/_torch/layers.py`)
- [ ] Cross-backend parity test passes (both forward + backward)
- [ ] All 400+ existing tests still pass
- [ ] New tests added (minimum 10 tests for architectural changes)
- [ ] ruff + pyright clean on all modified files
- [ ] Documentation updated (code comments explain the change)
- [ ] Training improvement measured (quicker convergence or stable deeper layers)

---

## 8. Estimated Effort

| Component | Files | Estimates |
|-----------|-------|-----------|
| Baseline experiment | `scripts/train.py` + manual run | ~15 min |
| Failing tests | 2-3 test files | ~30 min |
| NumPy implementation | `impl/_np/modules.py` | ~20 min |
| PyTorch implementation | `impl/_torch/layers.py` | ~20 min |
| Cross-backend parity | `tests/cross_backend/` | ~15 min |
| Scripts + integration tests | 3-4 test files | ~30 min |
| **Total** | **~6 files + tests** | **~2 hours** |

---

## 9. Questions for User

1. **Are you aware that residual connections already exist?** (pre-norm: `x + MHA(...)` and `h + MoE(...)`)
2. **Do you want post-norm** (residual add first, then normalize) or **gated residuals** (learnable `gate * residual`)?
3. **Or were you thinking of dropout** — added regularization which is currently absent?
4. **What specific symptom** are you trying to fix? (slow convergence, gradient vanishing at layer 4+, unstable training?)
5. **Would you like to try both pre-norm and post-norm** and compare, or commit to one approach?

---

## 10. Next Steps

1. ❗ **User clarification required** — confirm which improvement to implement
2. Once confirmed: follow TDD → baseline → failing test → implementation → parity → merge
3. Small, focused steps — one architecture change at a time

---

## 11. Progress Tracking

| Sub-step | Status | Files | Gate |
|----------|--------|-------|------|
| 1. User clarification | ❓ Pending | — | User response on what improvement to implement |
| 2. Baseline metrics | 🔲 Not started | train.py (manual run) | Record loss curve, gradient norms |
| 3. Failing test | 🔲 Not started | test_*.py | Test fails before implementation |
| 4. NumPy impl | 🔲 Not started | modules.py | Failing test passes |
| 5. PyTorch impl | 🔲 Not started | layers.py | Test passes on both backends |
| 6. Cross-backend parity | 🔲 Not started | tests/cross_backend/ | Parity test passes |
| 7. Scripts update | 🔲 Not started | train.py, infer.py, test_*.py | All 400+ tests pass |

---

*Waiting for user clarification before proceeding. Residual connections already exist in pre-norm format — need to confirm whether user wants post-norm, gated residuals, or something else.*
