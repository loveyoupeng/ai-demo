# Phase 3++: Architecture Improvements — Post-Norm + Gated Residuals + Dropout

**Status:** 🔲 Not Started
**Preceded by:** Phase 3+ (E2E Training/Inference) — Complete, 400 tests, ruff/pyright clean
**Goal:** Implement Post-Norm, Gated Residuals, and Dropout together for production-ready training stability and faster convergence

---

## 1. Goal

Replace Pre-Norm architecture with Post-Norm + Gated Residuals + Dropout across both NumPy and PyTorch backends while:
- Maintaining cross-backend parity (inference modes)
- Adding gradient flow monitoring (training mode)
- Keeping all 400+ existing tests passing (may update thresholds)

### Why all 3 together?
| Improvement | What it does | Why combined? |
|-------------|-------------|---------------|
| Post-Norm | Residual add first, then normalize | Better gradient flow for deep networks |
| Gated Residuals | Learnable `gate * residual` for signal control | Stabilizes Post-Norm (post-norm can explode in deep nets without gates) |
| Dropout | Random dropout during training | Prevents overfitting on long training runs |

**This is a known production pattern:** Post-Norm + Gated Residuals handles depth stability; Dropout handles generalization. Together they form robust training.

---

## 2. Target Architecture

### Combined Formula
```
# First branch: Attention
attn_out = MHA(x)
h = x + attn_out                     # residual FIRST (post-norm)
h = ln(h)                            # normalize AFTER residual
h = h + ln(h) * gate1 * (1 - drop1) # gate + dropout

# Second branch: MoE
moe_out = MoE(h)
out = h + moe_out                    # residual FIRST (post-norm)
out = ln(out)                        # normalize AFTER residual
out = out + out * gate2 * (1 - drop2) # gate + dropout
```

### Key difference from current Pre-Norm
| | Pre-Norm (current) | Post-Norm (target) |
|---|---|---|
| Order | norm → residual | residual → norm |
| Formula | `h = x + MHA(ln(x))` | `h = x + attn → h = ln(h)` |
| Stability | More stable training | Faster convergence, needs gates |
| Best for | Decoders (GPT) | Encoders/deep networks (BERT) |
| Gradient flow | Gradient flows through norm → vanishes | Gradient flows directly → safer |

### Gate initialization
- Both gates initialized to **zero** → output = input (identity) at start
- Gates **learn to open** as training progresses
- Sigmoid activation ensures gate stays in [0, 1] range
- This provides **smooth training** — model starts as current architecture, learns to improve

### Dropout configuration
- Dropout applied to intermediate activations (not residual inputs)
- Rate: 0.05 (light, suitable for small models)
- Applied AFTER gating, BEFORE next layer
- **Disabled during inference** → deterministic behavior preserved

---

## 3. Implementation Plan

### Step 1: Write failing tests FIRST (TDD mandatory)

#### Test: `test_gradient_flow_post_norm`
```python
def test_gradient_flow_post_norm():
    """Post-Norm gated residuals should maintain gradient norm > threshold at layer 4."""
    model = TorchModel(n_layers=4, embed_dim=64, ...)
    x = torch.randn(1, 16, model.config.embed_dim, dtype=torch.float64)
    y = torch.randint(0, model.config.vocab_size, (1, 16))
    loss = loss_fn(model(x), y)
    loss.backward()
    
    # Post-Norm + gates → every layer should have non-vanishing gradient
    for i, name, p in get_layer_grads(model):
        if i >= 2:
            assert p.grad.norm() > 1e-6, f"Gradient vanished at layer {i}: {name}"
```

#### Test: `test_gate_initialization_identity`
```python
def test_gate_initialization_identity():
    """At init, gates = 0 → forward output ≈ input (identity behavior)."""
    model = TorchModel(n_layers=2, embed_dim=64)
    x = torch.randn(1, 16, model.config.embed_dim, dtype=torch.float64)
    
    with torch.no_grad():
        out = model(x)
    
    # Gates init to 0 → gated_residual ≈ input (before ln scaling changes things)
    # We check gate values, not output (ln changes output)
    for name, p in model.named_parameters():
        if 'gate' in name:
            assert p.abs().max() < 1e-4, f"Gate {name} not initialized to ~0"
```

#### Test: `test_post_norm_transformerblock`
```python
def test_post_norm_transformerblock_output_not_equal_input():
    """TransformerBlock output should differ from input (non-trivial residual)."""
    block = TransformerBlock(..., norm_type='post')
    x = torch.randn(1, 16, embed_dim, dtype=torch.float64, requires_grad=True)
    out = block(x)
    diff = (out - x).abs().max()
    assert diff > 1e-4, "Output should differ from input with Post-Norm"
```

#### Test: `test_pre_vs_post_norm_gradient_flow` (comparison test)
```python
def test_post_norm_has_better_gradient_flow():
    """Post-Norm gates should have stronger gradients at init than no gates."""
    model_with_gates = TorchModel(n_layers=4, embed_dim=128)
    model_without_gates = TorchModel(n_layers=4, embed_dim=128)
    
    x = torch.randn(1, 32, 128, dtype=torch.float64)
    y = torch.randint(0, 256, (1, 32))
    
    for model in [model_with_gates, model_without_gates]:
        x.grad = None
        loss = loss_fn(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            if 'gate' in name:
                assert p.grad.abs().max() > 1e-5, f"Gate gradient vanished: {name}"
```

#### Test: `test_dropout_disabled_inference_deterministic`
```python
def test_dropout_disabled_deterministic_inference():
    """With dropout disabled (eval/dropout p=0), inference should be deterministic."""
    model = TorchModel(n_layers=2, embed_dim=64)
    model.eval()  # Disable dropout
    x = torch.randn(1, 16, 64, dtype=torch.float64)
    
    out1 = model(x)
    out2 = model(x)
    assert (out1 == out2).all(), "Inference without dropout should be deterministic"
```

#### Test: `test_cross_backend_inference_parity_no_dropout`
```python
def test_inference_parity_without_dropout():
    """NumPy and PyTorch should produce equivalent outputs at inference (no dropout)."""
    # Train both, disable dropout, test same inputs
    np_model = NumPyModel(n_layers=2, ...)
    pt_model = TorchModel(n_layers=2, ...)
    
    # Load same initial weights
    pt_model.load_from_numpy(np_model.get_all_parameters())
    
    for x in sample_inputs(...):
        np_out = np_model(x, inference=True)  # dropout=0
        pt_out = pt_model(x, training=False)  # dropout=0
        diff = (np_out - pt_out).max()
        assert diff < 1e-4, f"Inference parity broken: max_diff={diff}"
```

#### Test: `test_gate_values_learn_during_training`
```python
def test_gates_learn_during_training():
    """Gates should change from initial zero values after training steps."""
    model = TorchModel(n_layers=2, embed_dim=128)
    initial_gates = {name: p.clone() for name, p in model.named_parameters() if 'gate' in name}
    
    # Train 10 steps with dropout disabled (deterministic)
    for _ in range(10):
        x = torch.randint(0, 256, (4, 64))
        y = torch.randint(0, 256, (4, 64))
        loss = loss_fn(model(x), y)
        loss.backward()
        # ... optimizer step ...
    
    for name, initial in initial_gates.items():
        p = dict(model.named_parameters())[name]
        assert (p != initial).any(), f"Gate {name} did not learn"
```

#### Test: `test_gated_residual_signal_control`
```python
def test_gated_residual_prevents_exploding_gradients():
    """Gated residuals should cap gradient norm, prevent explosion."""
    model_gated = TorchModel(n_layers=8, embed_dim=256, gate_init_scale=0.5)
    x = torch.randn(2, 64, 256, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, 256, (2, 64))
    
    for _ in range(50):
        x.grad = None
        loss = loss_fn(model_gated(x), y)
        loss.backward()
        max_grad = max(p.grad.abs().max() for p in model_gated.parameters() if p.grad is not None)
        assert max_grad < 100, f"Gradient exploded after 50 steps: {max_grad}"
```

### Step 2: Implement in NumPy first

**Files to modify:**
1. `impl/_np/feedforward.py` — Add `gate1`, `gate2` parameters
2. `impl/_np/modules.py` — Restructure TransformerBlock to Post-Norm + gates + dropout
3. `impl/_np/modules.py` — Update DecoderStack to pass dropout config
4. `impl/_np/model.py` — Add dropout mode flag (training/inference) to NumPyModel
5. `tests/unit/_np/` — Add/update tests

**Changes in TransformerBlock.forward():**
```python
def forward(self, x, dropout=0.0, training=False):
    # Branch 1: Attention
    attn_out = self.mha(x)  # No normalization first
    h = x + attn_out        # Residual FIRST
    
    # Post-norm
    h = RMSNorm().forward(h, self.ln1_gamma)
    
    # Gated + dropout
    h = h + h * self.gate1 * (np.random.rand(*h.shape) < (1 - dropout) if training else 1)
    
    # Branch 2: MoE
    moe_out = self.moe(h)
    
    # Residual FIRST
    out = h + moe_out
    
    # Post-norm
    out = RMSNorm().forward(out, self.ln2_gamma)
    
    # Gated + dropout
    out = out + out * self.gate2 * (np.random.rand(*out.shape) < (1 - dropout) if training else 1)
    
    return out
```

### Step 3: Mirror in PyTorch

**Files to modify:**
1. `impl/_torch/layers.py` — Add gate parameters (`nn.Parameter`)
2. `impl/_torch/layers.py` — Restructure TransformerBlock to Post-Norm + gates + dropout
3. `tests/unit/_torch/` — Add/update tests

**Changes in TransformerBlock:**
```python
def __init__(self, ...):
    # Existing
    self.gate1 = nn.Parameter(torch.zeros(1))  # Learnable gate
    self.gate2 = nn.Parameter(torch.zeros(1))  # Learnable gate
    self dropout1 = nn.Dropout(p)
    self.dropout2 = nn.Dropout(p)
    self.norm_type = 'post'  # or 'pre' for backward compatibility

def forward(self, x, dropout_prob=None):
    # Branch 1: Attention
    attn_out, _ = self.mha(x)
    h = x + attn_out
    
    # Post-norm
    h = self.ln1(h)
    
    # Gated + dropout
    h = h + h * self.gate1.sigmoid() * (1 - self.dropout1(h if self.training else 0))
    
    # Branch 2: MoE
    moe_out = self.moe(h)
    out = h + moe_out
    
    # Post-norm
    out = self.ln2(out)
    
    # Gated + dropout
    out = out + out * self.gate2.sigmoid() * (1 - self.dropout2(out if self.training else 0))
    
    return out
```

### Step 4: Cross-backend parity test after both implementations
### Step 5: Update train.py to use new training mode with dropout
### Step 6: Update verify_equivalence.py and auto_test_equivalence.py
### Step 7: Update all tests across backend files

---

## 4. Training Loop Changes

```python
# New training loop with dropout
def run_training(model, dataset, train_steps, batch_size, lr, seed, dropout_prob=0.05):
    model.train()  # Enable dropout in eval mode, this controls dropout

    losses = []
    for step in range(train_steps):
        batch = next(dataset)
        x, y = batch
        
        # Forward with dropout (only active during training)
        loss = model(x, y, training=True)
        
        # Backward (computes gate gradients too)
        loss.backward()
        
        # Clip gradients (prevent exploding)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 10 == 0:
            losses.append(loss.item())
            print(f"Step {step}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
    
    return losses  # Training loop returns loss curve
```

**Key additions:**
1. `model.train()` / `model.eval()` for dropout control
2. Gradient clipping to support post-norm stability
3. Gate gradient tracking

---

## 5. TDD Commit Plan

Each commit follows: **write failing test → implement → ruff+pyright → commit**

| Commit | Test | Implementation | Tolerance |
|--------|------|----------------|-----------|
| 1 | Gate init = 0 | Add gate params to NumPy | rtol=1e-4 |
| 2 | Cross-backend gate init parity | Add gate params to PyTorch | rtol=1e-4 |
| 3 | Gate learns after training | Gradient tracking test | — |
| 4 | Gate controls gradient norm | Forward pass test | max_grad < 100 |
| 5 | Post-Norm residual order | Swap order in NumPy | — |
| 6 | Post-Norm swap in PyTorch | Mirror changes | rtol=1e-4 |
| 7 | Gradient flow test | All layers have gradient | grad_norm > 1e-6 |
| 8 | Post-Norm pre/post test | Verify order changed | — |
| 9 | Dropout identity (eval mode) | Add dropout layers | — |
| 10 | Dropout non-deterministic (train) | Verify randomness disabled in eval | rtol=1e-4 |
| 11 | Cross-backend inference (no dropout) | Both modes with dropout=0 | rtol=1e-4, atol=1e-4 |
| 12 | Full training with all 3 features | Train 3-layer model | — |

---

## 6. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Post-Norm training divergence | High | Start with small model (64 embed, 2 layers); use gradient clipping |
| Gates stay at 0 (never learn) | Medium | Initialize with small positive scale (0.01); check during training |
| Dropout breaks cross-backend parity | High | Test inference with dropout=0; use tolerance for training comparisons |
| Gate parameters explode in deep networks | Medium | Gate init=0 + sigmoid activation caps at [0,1]; gradient clipping |
| Existing tests fail on new architecture | Medium | Update test tolerances; most tests check output ≠ input (still true) |
| Pre-Norm tests break | Medium | Add `norm_type='pre'` as fallback to maintain backward compatibility |

---

## 7. Phase 3++ Gate Checklist

Before Phase 3++ is considered complete:

- [ ] Failing test written for each sub-change (TDD mandate)
- [ ] Gates implemented in NumPy backend (`impl/_np/modules.py`)
- [ ] Gates implemented in PyTorch backend (`impl/_torch/layers.py`)
- [ ] Gate init to zero → identity at start verified
- [ ] Gate learns during training verified
- [ ] Post-Norm implemented in NumPy (`modules.py`)
- [ ] Post-Norm implemented in PyTorch (`layers.py`)
- [ ] Dropout implemented in PyTorch (`layers.py`)
- [ ] Dropout mode flag in NumPy (`model.py`)
- [ ] Cross-backend inference parity (dropout=0): rtol=1e-4, atol=1e-4
- [ ] All 400+ existing tests still pass (may update tolerances)
- [ ] New tests covering gates, post-norm, dropout added (minimum 15 new tests)
- [ ] ruff + pyright clean on all modified files
- [ ] Documentation updated (code comments explain post-norm/gate/dropout)
- [ ] Training loop updated with gradient clipping

---

## 8. Estimated Effort

| Component | Files | Estimates |
|-----------|-------|-----------|
| Gate params + tests | 4 files + tests | ~30 min |
| Post-Norm swap + tests | 4 files + tests | ~20 min |
| Dropout + tests | 4 files + tests | ~20 min |
| Cross-backend parity | 2 files | ~15 min |
| Train/infer scripts update | 3 files | ~20 min |
| Integration tests | 3 files | ~30 min |
| **Total** | **~20 files** | **~2-2.5 hours** |

---

## 9. Implementation Order

1. **Gates** (foundation) — simple params, affects both Post-Norm and Pre-Norm
2. **Post-Norm** (structure) — changes residual order, needs gates for stability
3. **Dropout** (regularization) — works with both, but best tested with Post-Norm

**Why this order:** Gates are needed regardless. Post-Norm needs gates. Dropout is independent but complements both.

---

## 10. Progress Tracking

| Step | Status | Files | Gate |
|------|--------|-------|------|
| 1. Gate tests | 🔲 Not started | test_gates.py | Test gate init = 0, gate learns |
| 2. Gate impl NumPy | 🔲 Not started | modules.py | Gate params added |
| 3. Gate impl PyTorch | 🔲 Not started | layers.py | Cross-backend parity |
| 4. Post-Norm tests | 🔲 Not started | test_post_norm.py | Residual order changed |
| 5. Post-Norm impl NumPy | 🔲 Not started | modules.py | Swap order |
| 6. Post-Norm impl PyTorch | 🔲 Not started | layers.py | Swap order |
| 7. Dropout tests | 🔲 Not started | test_dropout.py | Deterministic eval, stochastic train |
| 8. Dropout impl | 🔲 Not started | layers.py + model.py | Both backends |
| 9. Train loop update | 🔲 Not started | train.py, model.py | Gradient clipping |
| 10. Full integration | 🔲 Not started | All scripts | Training + inference works |

---

## 11. Backward Compatibility

To avoid breaking existing code that depends on Pre-Norm:

```python
class TransformerBlock:
    def __init__(self, ..., norm_type='post', dropout=0.05):
        # Default is post-norm with dropout for new code
        # But Pre-Norm can be used as fallback: TransformerBlock(norm_type='pre')
        pass
```

This lets users/test switch between Pre-Norm and Post-Norm without breaking existing tests that assume Pre-Norm.

---

*Plan ready. Awaiting confirmation to proceed with execution.*
