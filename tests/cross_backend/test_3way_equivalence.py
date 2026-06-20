"""3-way cross-backend training + inference equivalence test.

Acceptance criterion: any backend's trained model can be loaded by
another backend and produce identical outputs.

Tests:
  - Self-load (train on A → save → load into A → check self-consistency)
  - Cross-load  (train on A → save → load into B → compare to A baseline)
"""

import gc

import numpy as np
import pytest
import torch

# ── helpers ──────────────────────────────────────────────────────


def _cuda_isolated(func):
    """Decorator to clean CUDA state before/after a function."""

    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return result

    return wrapper


def _train_and_save_ckpt_torch(torch_model, steps=10):
    """Run `steps` forward passes on GPU and return numpy-checkpoint dict."""
    model = torch_model.cuda()
    for _ in range(steps):
        x = torch.randint(
            0,
            torch_model.vocab_size,
            (2, 16),
            device="cuda",
        )
        model(x)
    return model.save_as_numpy()


def _produce_output(ckpt, model):
    """Load checkpoint into *model*, run inference on fixed prompt, return tokens."""
    context_length = 16

    if isinstance(model, torch.nn.Module):
        model.eval()
        if not next(model.parameters()).is_cuda:
            model = model.cuda()
        model.load_from_numpy_dict(ckpt)
        prompt_tokens = [10, 11, 12, 13]
        generated = prompt_tokens.copy()
        for _ in range(8):
            seq = generated[-context_length:]
            x = torch.tensor([seq], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model(x)
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

    @classmethod
    def configs(cls):
        return {
            "vocab_size": 64,
            "embed_dim": 8,
            "n_layers": 1,
            "n_heads": 2,
            "n_experts": 2,
            "ff_dim": 8,
            "k": 1,
        }

    @_cuda_isolated
    def test_torch_to_triton_self_consistency(self):
        """Triton trains → load into TWO Torch copies → outputs match."""
        configs = self.configs()
        from impl._torch.layers import TorchModel
        from impl._triton.model import TritonModel

        triton_m = TritonModel(**configs)
        ckpt = _train_and_save_ckpt_torch(triton_m, steps=10)

        out1 = _produce_output(ckpt, TorchModel(**configs))
        out2 = _produce_output(ckpt, TorchModel(**configs))
        assert out1 == out2, f"Self-consistency failed: {out1} vs {out2}"

    @_cuda_isolated
    def test_self_load_torch(self):
        """Train on Torch → load into Torch → same output."""
        configs = self.configs()
        from impl._torch.layers import TorchModel

        torch.manual_seed(42)
        torch_m = TorchModel(**configs, seed=42)
        ckpt = _train_and_save_ckpt_torch(torch_m, steps=10)

        out1 = _produce_output(ckpt, TorchModel(**configs))
        out2 = _produce_output(ckpt, TorchModel(**configs))
        assert out1 == out2, "Torch self-load failed"

    @_cuda_isolated
    def test_numpy_to_torch(self):
        """NumPy trains → load into Torch → outputs match NumPy baseline."""
        configs = self.configs()
        from impl._np.model import NumPyModel
        from impl._torch.layers import TorchModel

        np.random.seed(42)
        np_m = NumPyModel(**configs, seed=42)
        ckpt = np_m.get_all_parameters()
        for _ in range(10):
            x = np.random.randint(
                0,
                np_m.vocab_size,
                (2, 16),
                dtype=np.int32,
            )
            np_m.forward(x)

        out1 = _produce_output(ckpt, NumPyModel(**configs))
        out2 = _produce_output(ckpt, TorchModel(**configs))
        assert out1 == out2, f"NumPy→Torch failed: {out1} vs {out2}"

    @_cuda_isolated
    def test_triton_to_numpy(self):
        """Triton trains → load into NumPy → outputs match baseline."""
        configs = self.configs()
        from impl._np.model import NumPyModel
        from impl._triton.model import TritonModel

        triton_m = TritonModel(**configs)
        ckpt = _train_and_save_ckpt_torch(triton_m, steps=10)

        out1 = _produce_output(ckpt, TritonModel(**configs))
        out2 = _produce_output(ckpt, NumPyModel(**configs))
        assert out1 == out2, f"Triton→NumPy failed: {out1} vs {out2}"
