You are implementing a parity test for a Transformer project. Follow test-driven development: write the test first (red), fix it (green), then clean up (refactor).

## Project Context

Project: decoder-only transformer demo. NumPy is the core implementation, PyTorch is for parity/verification. All backward passes use manual computation (not autograd).

Current state: All 103 tests pass. The project is on the `main` branch.

## Your Task

Add 7 backward parity tests to `tests/parity/test_transformer.py`. All tests follow the exact same structure — only the NumPy key and PyTorch key differ.

## Existing Test Structure

The file `tests/parity/test_transformer.py` already has:

```python
class TestTransformerBackwardLmHeadParity:
    def test_backward_lm_head_parity(self):
        """Backward w.r.t. lm_head should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["lm_head"],
            pytorch_grads["lm_head"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def _sync_model_params(self, model_np, model_pt):
        """Sync all NumPy model params to PyTorch model."""
        # ... (already implemented correctly, DO NOT MODIFY)
```

## Tests to Add

Add exactly these 7 test methods to the `TestTransformerBackwardLmHeadParity` class. Each test varies only:
1. The test method name
2. The docstring
3. The NumPy key in the assert
4. The PyTorch key in the assert

All other code (seed, dimensions, setup, forward, backward) is identical.

### Test 1: blocks.0.ln1 gamma
```python
def test_backward_0_ln1_gamma_parity(self):
    """Backward w.r.t. blocks.0.ln1.gamma should match between NumPy and PyTorch."""
    # ... same setup as above ...
    np.testing.assert_allclose(
        numpy_grads["blocks.0.ln1.gamma"],
        pytorch_grads["blocks.0.ln1.weight"].detach().numpy(),
        rtol=1e-4, atol=1e-4,
    )
```

### Test 2: blocks.0.ln1 beta
```python
def test_backward_0_ln1_beta_parity(self):
    """Backward w.r.t. blocks.0.ln1.beta should match between NumPy and PyTorch."""
    np.testing.assert_allclose(
        numpy_grads["blocks.0.ln1.beta"],
        pytorch_grads["blocks.0.ln1.bias"].detach().numpy(),
        rtol=1e-4, atol=1e-4,
    )
```

### Test 3: blocks.0.ln2 gamma
```python
def test_backward_0_ln2_gamma_parity(self):
    """Backward w.r.t. blocks.0.ln2.gamma should match between NumPy and PyTorch."""
    np.testing.assert_allclose(
        numpy_grads["blocks.0.ln2.gamma"],
        pytorch_grads["blocks.0.ln2.weight"].detach().numpy(),
        rtol=1e-4, atol=1e-4,
    )
```

### Test 4: blocks.0.ln2 beta
```python
def test_backward_0_ln2_beta_parity(self):
    """Backward w.r.t. blocks.0.ln2.beta should match between NumPy and PyTorch."""
    np.testing.assert_allclose(
        numpy_grads["blocks.0.ln2.beta"],
        pytorch_grads["blocks.0.ln2.bias"].detach().numpy(),
        rtol=1e-4, atol=1e-4,
    )
```

### Test 5: blocks.0.mha W_q
```python
def test_backward_0_mha_Wq_parity(self):
    """Backward w.r.t. blocks.0.mha.W_q should match between NumPy and PyTorch."""
    np.testing.assert_allclose(
        numpy_grads["blocks.0.mha.W_q"],
        pytorch_grads["blocks.0.mha.qkv.W_q"].detach().numpy(),
        rtol=1e-4, atol=1e-4,
    )
```

### Test 6: blocks.0.mha W_k
```python
def test_backward_0_mha_Wk_parity(self):
    """Backward w.r.t. blocks.0.mha.W_k should match between NumPy and PyTorch."""
    np.testing.assert_allclose(
        numpy_grads["blocks.0.mha.W_k"],
        pytorch_grads["blocks.0.mha.qkv.W_k"].detach().numpy(),
        rtol=1e-4, atol=1e-4,
    )
```

### Test 7: blocks.0.moe expert.0 W1
```python
def test_backward_0_moe_expert_0_W1_parity(self):
    """Backward w.r.t. blocks.0.moe.expert.0.W1 should match between NumPy and PyTorch."""
    np.testing.assert_allclose(
        numpy_grads["blocks.0.moe.expert.0.W1"],
        pytorch_grads["blocks.0.moe.expert.0.w1"].detach().numpy(),
        rtol=1e-4, atol=1e-4,
    )
```

## Key Mapping Reference (NumPy → PyTorch backward key)

| Component | NumPy backward key | PyTorch backward key |
|-----------|-------------------|---------------------|
| LayerNorm gamma | `blocks.{i}.ln{1,2}.gamma` | `blocks.{i}.ln{1,2}.weight` |
| LayerNorm beta | `blocks.{i}.ln{1,2}.beta` | `blocks.{i}.ln{1,2}.bias` |
| MHA W_q | `blocks.{i}.mha.W_q` | `blocks.{i}.mha.qkv.W_q` |
| MHA W_k | `blocks.{i}.mha.W_k` | `blocks.{i}.mha.qkv.W_k` |
| MHA W_v | `blocks.{i}.mha.W_v` | `blocks.{i}.mha.qkv.W_v` |
| MHA W_o | `blocks.{i}.mha.W_o` | `blocks.{i}.mha.o.W_o` |
| MoE expert W1 | `blocks.{i}.moe.expert.{k}.W1` | `blocks.{i}.moe.expert.{k}.w1` |
| MoE expert b1 | `blocks.{i}.moe.expert.{k}.b1` | `blocks.{i}.moe.expert.{k}.b1` |
| MoE expert W2 | `blocks.{i}.moe.expert.{k}.W2` | `blocks.{i}.moe.expert.{k}.w2` |
| MoE expert b2 | `blocks.{i}.moe.expert.{k}.b2` | `blocks.{i}.moe.expert.{k}.b2` |

These key mappings exist because the NumPy and PyTorch implementations use different naming conventions in their backward methods. The test must compare the correct corresponding keys.

## Execution Steps

1. Read `tests/parity/test_transformer.py` to see the current file
2. Add all 7 test methods (copy the existing test as template, vary only the assertion keys)
3. Run `uv run pytest tests/parity/test_transformer.py -v --timeout=120` — expect all 8 tests (7 new + 1 existing) to pass
4. If any test fails, read the error, fix the key mapping, and re-run
5. Run `uv run pytest tests/ --timeout=120` — all 110 tests must pass
6. Commit with a descriptive message
7. Self-review: check for consistency, no copy-paste errors, follow existing code style

## Constraints

- Only modify `tests/parity/test_transformer.py`
- DO NOT modify `_sync_model_params` — it's already correct
- Use exact naming: method names, docstrings, and key strings as specified
- All tests must pass before considering task done
- Do not modify `src/` files
- The sync helper at lines that handle ln1/ln2 keys for gamma/beta — this is already correct

## Code Style

- Follow existing indentation (4 spaces)
- Follow existing import style (`from __future__ import annotations`, `import numpy as np`, `import torch`)
- Use `np.testing.assert_allclose` with `rtol=1e-4, atol=1e-4` exactly as existing test does
- Each test should be self-contained with its own imports and model creation
