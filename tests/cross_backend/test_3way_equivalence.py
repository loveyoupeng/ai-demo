"""3-way cross-backend training + inference equivalence test.

Acceptance criterion: any backend's trained model can be loaded by
another backend and produce identical outputs. All backends share the
Keys checkpoint scheme, so a checkpoint dict loads losslessly into any
track via ``load_from_numpy_dict``.

Tests:
  - Self-load (train on A → save → load into A → check self-consistency)
  - Cross-load (train on A → save → load into B → compare to A baseline)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from shared.config import TransformerConfig

# ── helpers ──────────────────────────────────────────────────────


def _cuda_isolated(func):
    """Decorator to clean CUDA state before/after a function."""

    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    return wrapper


def _cfg(seed: int = 42) -> TransformerConfig:
    """Small 1-layer MoE config shared by the equivalence tests."""
    return TransformerConfig.from_dict(
        {
            "vocab_size": 64,
            "embed_dim": 8,
            "n_layers": 1,
            "n_heads": 2,
            "n_experts": 2,
            "expert_dim": 8,
            "top_k": 1,
            "rope_dim": 0,
            "seed": seed,
        }
    )


def _train_and_save_ckpt_torch(torch_model, steps: int = 10) -> dict[str, np.ndarray]:
    """Run `steps` forward passes on GPU and return the checkpoint dict (Keys)."""
    model = torch_model.cuda()
    for _ in range(steps):
        x = torch.randint(0, 64, (2, 16), device="cuda", dtype=torch.long)
        model(x)
    return model.save_as_numpy()


def _produce_output(ckpt: dict[str, np.ndarray], model: Any) -> list[int]:
    """Load checkpoint into *model*, run greedy inference on a fixed prompt."""
    context_length = 16
    if isinstance(model, torch.nn.Module):
        m: Any = model
        if not next(m.parameters()).is_cuda:
            m = m.cuda()
        m.eval()
        m.load_from_numpy_dict(ckpt)
        prompt_tokens = [10, 11, 12, 13]
        generated = prompt_tokens.copy()
        for _ in range(8):
            seq = generated[-context_length:]
            x = torch.tensor([seq], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = m(x)
            generated.append(int(torch.argmax(out[0, -1]).item()))
    else:
        model.load_from_numpy_dict(ckpt)
        prompt_tokens = [10, 11, 12, 13]
        generated = prompt_tokens.copy()
        for _ in range(8):
            x = np.array([generated[-context_length:]], dtype=np.int32)
            logits = model.forward(x)
            generated.append(int(np.argmax(logits[0, -1])))
    return generated


# ── test ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@pytest.mark.gpu
class TestCrossBackendEquivalence:
    """Verify cross-backend checkpoint loading produces identical outputs."""

    @_cuda_isolated
    def test_torch_to_triton_self_consistency(self):
        """Triton model trained → loaded into TWO Torch copies → outputs match."""
        from impl._torch.layers import TorchModel
        from impl._triton.model import TritonModel

        triton_m = TritonModel(_cfg())
        ckpt = _train_and_save_ckpt_torch(triton_m, steps=10)

        out1 = _produce_output(ckpt, TorchModel(_cfg()))
        out2 = _produce_output(ckpt, TorchModel(_cfg()))
        assert out1 == out2, f"Self-consistency failed: {out1} vs {out2}"

    @_cuda_isolated
    def test_self_load_torch(self):
        """Train on Torch → load into Torch → same output."""
        from impl._torch.layers import TorchModel

        torch.manual_seed(42)
        torch_m = TorchModel(_cfg())
        ckpt = _train_and_save_ckpt_torch(torch_m, steps=10)

        out1 = _produce_output(ckpt, TorchModel(_cfg()))
        out2 = _produce_output(ckpt, TorchModel(_cfg()))
        assert out1 == out2, "Torch self-load failed"

    @_cuda_isolated
    def test_numpy_to_torch(self):
        """NumPy forward pass → load into Torch → greedy outputs match baseline."""
        from impl._np.model import NumPyModel
        from impl._torch.layers import TorchModel

        np.random.seed(42)
        np_m = NumPyModel(_cfg())
        ckpt = np_m.get_all_parameters()
        for _ in range(10):
            x = np.random.randint(0, 64, (2, 16), dtype=np.int32)
            np_m.forward(x)

        out1 = _produce_output(ckpt, NumPyModel(_cfg()))
        out2 = _produce_output(ckpt, TorchModel(_cfg()))
        assert out1 == out2, f"NumPy→Torch failed: {out1} vs {out2}"

    @_cuda_isolated
    def test_triton_to_numpy(self):
        """Triton model trained → load into NumPy → outputs match baseline."""
        from impl._np.model import NumPyModel
        from impl._triton.model import TritonModel

        triton_m = TritonModel(_cfg())
        ckpt = _train_and_save_ckpt_torch(triton_m, steps=10)

        out1 = _produce_output(ckpt, TritonModel(_cfg()))
        out2 = _produce_output(ckpt, NumPyModel(_cfg()))
        assert out1 == out2, f"Triton→NumPy failed: {out1} vs {out2}"
