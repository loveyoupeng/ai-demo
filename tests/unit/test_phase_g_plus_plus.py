"""Cross-backend key-scheme and weight-interchange tests.

All four tracks (NumPy, PyTorch, Triton, CUDA) save and load parameters under
the single shared Keys scheme (``shared.constants``), so:

- key sets produced by ``get_all_parameters()`` / ``save_as_numpy()`` are
  identical across backends for the same model config;
- weights saved by one track load losslessly into any other track;
- round-trips (save → load → save) are exact.

Legacy key-mapping tests (old ``blocks.*`` → ``stack.layers.*`` normalization)
are gone: no mapping is needed anymore.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from impl._cuda.model import CUDAModel
from impl._np.model import NumPyModel
from impl._torch.layers import TorchModel
from impl._triton.model import TritonModel
from shared.config import TransformerConfig

# ── Model configs ─────────────────────────────────────────────────────────────

MODEL_CONFIG_1L = {
    "vocab_size": 64,
    "embed_dim": 32,
    "n_layers": 1,
    "n_heads": 4,
    "n_experts": 2,
    "ff_dim": 64,
    "k": 1,
    "rope_dim": 0,
    "seed": 42,
}

MODEL_CONFIG_2L = {
    "vocab_size": 128,
    "embed_dim": 64,
    "n_layers": 2,
    "n_heads": 4,
    "n_experts": 2,
    "ff_dim": 128,
    "k": 1,
    "rope_dim": 0,
    "seed": 42,
}

DENSE_CONFIG_1L = {
    "vocab_size": 64,
    "embed_dim": 32,
    "n_layers": 1,
    "n_heads": 4,
    "n_experts": 1,
    "ff_dim": 64,
    "k": 1,
    "rope_dim": 0,
    "seed": 42,
}


def _to_config(d: dict) -> TransformerConfig:
    """Map a test config dict onto the shared TransformerConfig.

    ``ff_dim`` → ``expert_dim``, ``k`` → ``top_k``; missing ``n_groups``
    defaults to ``n_heads`` (standard MHA).
    """
    mapped = {("expert_dim" if k == "ff_dim" else "top_k" if k == "k" else k): v for k, v in d.items()}
    return TransformerConfig(**mapped)


def create_torch_model(config: dict) -> TorchModel:
    """Create and return an initialized TorchModel."""
    return TorchModel(_to_config(config))


def create_numpy_model(config: dict) -> NumPyModel:
    """Create and return an initialized NumPyModel."""
    return NumPyModel(_to_config(config))


def create_triton_model(config: dict) -> TritonModel:
    """Create and return an initialized TritonModel."""
    return TritonModel(_to_config(config))


def create_cuda_model(config: dict) -> CUDAModel:
    """Create and return an initialized CUDAModel."""
    return CUDAModel(_to_config(config))


def weight_diff_params(params_a: dict, params_b: dict) -> float:
    """Compute max diff between two param dicts (numpy arrays or torch tensors)."""
    max_diff = 0.0
    for key in set(params_a) & set(params_b):
        a = params_a[key]
        b = params_b[key]
        a = a.cpu().detach().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
        b = b.cpu().detach().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        if a.shape == b.shape:
            max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    return max_diff


def train_torch_seed42(config: dict | None = None) -> dict[str, np.ndarray]:
    """Train a PyTorch model with a fixed seed and return its saved params."""
    mcfg = {**MODEL_CONFIG_1L}
    if config:
        mcfg.update(config)
    torch.manual_seed(mcfg["seed"])
    model = create_torch_model(mcfg)
    model.eval()

    ctx_len, train_steps, lr = 32, 2, 0.01
    tokens = torch.randint(0, mcfg["vocab_size"], (1, ctx_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.functional.cross_entropy

    for _ in range(train_steps):
        optimizer.zero_grad()
        logits = model(tokens)
        loss = loss_fn(logits.reshape(-1, mcfg["vocab_size"]), tokens.reshape(-1))
        loss.backward()
        optimizer.step()

    return model.save_as_numpy()


def train_numpy_seed42(config: dict | None = None) -> dict[str, np.ndarray]:
    """Train a NumPy model with a fixed seed and return its params."""
    mcfg = {**MODEL_CONFIG_1L}
    if config:
        mcfg.update(config)
    model = create_numpy_model(mcfg)
    np.random.seed(mcfg["seed"])

    ctx_len, train_steps, lr = 32, 2, 0.01
    for _ in range(train_steps):
        tokens = np.random.randint(0, mcfg["vocab_size"], (1, ctx_len), dtype=np.int32)
        model.forward(tokens)
        params = model.get_all_parameters()
        for param in params.values():
            param[:] -= lr * np.random.randn(*param.shape)

    return model.get_all_parameters()


# ─── Test Case 1: Key-scheme consistency across backends ─────────────────────


class TestKeySchemeConsistency:
    """All backends expose identical Keys-scheme parameter key sets."""

    def test_numpy_keys_match_keys_scheme(self):
        """NumPy key set equals the Keys-scheme key list for the config."""
        from shared.constants import all_param_keys

        cfg = MODEL_CONFIG_1L
        np_keys = set(create_numpy_model(cfg).get_all_parameters())
        expected = set(all_param_keys(cfg["n_layers"], has_moe=cfg["n_experts"] > 1, n_experts=cfg["n_experts"]))
        assert np_keys == expected, f"NumPy keys deviate from Keys scheme: {np_keys ^ expected}"

    def test_numpy_vs_triton_vs_cuda_key_sets(self):
        """NumPy, Triton and CUDA all save the exact same Keys-scheme keys."""
        cfg = MODEL_CONFIG_1L
        np_keys = set(create_numpy_model(cfg).get_all_parameters())
        triton_keys = set(create_triton_model(cfg).save_as_numpy())
        cuda_keys = set(create_cuda_model(cfg).get_all_parameters())

        assert triton_keys == np_keys, f"Triton keys deviate: {triton_keys ^ np_keys}"
        assert cuda_keys == np_keys, f"CUDA keys deviate: {cuda_keys ^ np_keys}"

    def test_dense_model_key_sets(self):
        """Dense (n_experts=1) models share keys across all backends too."""
        cfg = DENSE_CONFIG_1L
        np_keys = set(create_numpy_model(cfg).get_all_parameters())
        triton_keys = set(create_triton_model(cfg).save_as_numpy())
        cuda_keys = set(create_cuda_model(cfg).get_all_parameters())
        assert np_keys == triton_keys == cuda_keys

    def test_2_layer_moe_key_sets(self):
        """2-layer MoE model: all backends agree."""
        cfg = MODEL_CONFIG_2L
        np_keys = set(create_numpy_model(cfg).get_all_parameters())
        triton_keys = set(create_triton_model(cfg).save_as_numpy())
        cuda_keys = set(create_cuda_model(cfg).get_all_parameters())
        assert np_keys == triton_keys == cuda_keys


# ─── Test Case 2: Shared weights → identical outputs ──────────────────────────


class TestSharedWeightsOutputs:
    """Same weights loaded into different backends → same output for same input."""

    def test_torch_numpy_same_output(self):
        """NumPy weights loaded into Torch reproduce the NumPy forward output."""
        cfg = MODEL_CONFIG_1L
        np_model = create_numpy_model(cfg)
        torch_model = create_torch_model(cfg)

        # Load the NumPy reference weights into the Torch model
        torch_model.load_from_numpy_dict(np_model.get_all_parameters())
        torch_model.eval()

        input_tokens = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        np_out = np_model.forward(input_tokens)
        torch_out = torch_model(torch.from_numpy(input_tokens)).detach().numpy()

        max_diff = np.max(np.abs(np_out - torch_out))
        assert max_diff < 0.01, f"NumPy vs Torch output mismatch: max_diff={max_diff:.6f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_shared_weights_triton(self):
        """Triton model loaded with NumPy weights produces near-identical output."""
        cfg = MODEL_CONFIG_1L
        np_model = create_numpy_model(cfg)
        triton_model = create_triton_model(cfg)

        triton_model.load_from_numpy_dict(np_model.get_all_parameters())
        triton_model = triton_model.cuda().eval()

        input_tokens = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        np_out = np_model.forward(input_tokens)
        triton_out = triton_model(torch.from_numpy(input_tokens).to("cuda")).detach().cpu().numpy()

        # Triton SDPA kernel runs in fp32 → loose tier (1e-2)
        max_diff = np.max(np.abs(np_out - triton_out))
        assert max_diff < 1e-2, f"NumPy vs Triton output mismatch: max_diff={max_diff:.6f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_shared_weights_cuda(self):
        """CUDA model loaded with NumPy weights produces near-identical output."""
        cfg = MODEL_CONFIG_1L
        np_model = create_numpy_model(cfg)
        cuda_model = create_cuda_model(cfg)

        cuda_model.load_from_numpy_dict(np_model.get_all_parameters())

        input_tokens = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        np_out = np_model.forward(input_tokens)
        cuda_out = cuda_model.forward(torch.from_numpy(input_tokens).to("cuda")).detach().cpu().numpy()

        max_diff = np.max(np.abs(np_out - cuda_out))
        assert max_diff < 1e-2, f"NumPy vs CUDA output mismatch: max_diff={max_diff:.6f}"


# ─── Test Case 3: Self-Reproducibility ────────────────────────────────────────


class TestSelfReproducibility:
    """Same seed → identical weights for the same backend."""

    def test_numpy_self_reproducibility(self):
        """Two NumPy models with the same seed have identical weights."""
        mcfg = MODEL_CONFIG_1L
        np.random.seed(42)
        model_a = create_numpy_model(mcfg)
        np.random.seed(42)
        model_b = create_numpy_model(mcfg)

        params_a = model_a.get_all_parameters()
        params_b = model_b.get_all_parameters()
        max_diff = weight_diff_params(params_a, params_b)
        assert max_diff < 1e-10, f"Self-reproducibility failed: max_diff={max_diff}"

    def test_torch_self_reproducibility(self):
        """Two torch models with the same seed have identical weights."""
        torch.manual_seed(42)
        model_a = create_torch_model(MODEL_CONFIG_1L)
        torch.manual_seed(42)
        model_b = create_torch_model(MODEL_CONFIG_1L)

        params_a = {n: p.clone() for n, p in model_a.named_parameters()}
        params_b = {n: p.clone() for n, p in model_b.named_parameters()}
        for key in params_a:
            assert torch.allclose(params_a[key], params_b[key], atol=1e-10), f"Self-reproducibility failed for {key}"


# ─── Test Case 4: Round-Trip Equivalence ──────────────────────────────────────


class TestRoundTripEquivalence:
    """Round-trips (save → load → save) must be exact."""

    def test_torch_to_numpy_roundtrip(self):
        """torch save_as_numpy → NumPy load → params identical."""
        torch_model = create_torch_model(MODEL_CONFIG_1L)
        torch_model.eval()
        saved = torch_model.save_as_numpy()

        np_model = create_numpy_model(MODEL_CONFIG_1L)
        np_model.load_from_numpy_dict(saved)
        reloaded = np_model.get_all_parameters()

        max_diff = weight_diff_params(saved, reloaded)
        assert max_diff < 1e-10, f"Round-trip failed: diff={max_diff}"

    def test_numpy_to_torch_roundtrip(self):
        """NumPy get_all_parameters → torch load → save → identical."""
        np_model = create_numpy_model(MODEL_CONFIG_1L)
        saved = np_model.get_all_parameters()

        torch_model = create_torch_model(MODEL_CONFIG_1L)
        torch_model.eval()
        torch_model.load_from_numpy_dict(saved)
        reloaded = torch_model.save_as_numpy()

        max_diff = weight_diff_params(saved, reloaded)
        assert max_diff < 1e-10, f"Round-trip failed: diff={max_diff}"


# ─── Test Case 5: CUDA Key Mapping ────────────────────────────────────────────


class TestCUDAKeyMapping:
    """CUDA model parameters follow the shared Keys scheme."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_cuda_keys_exist(self):
        """CUDA get_all_parameters returns the full Keys-scheme key set."""
        cfg = MODEL_CONFIG_1L
        cuda_params = create_cuda_model(cfg).get_all_parameters()
        np_params = create_numpy_model(cfg).get_all_parameters()
        assert set(cuda_params) == set(np_params)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_cuda_weight_diff_with_torch(self):
        """CUDA model loaded with torch weights is identical (Keys scheme)."""
        cfg = MODEL_CONFIG_1L
        torch_model = create_torch_model(cfg)
        torch_model.eval()
        torch_params = torch_model.save_as_numpy()

        cuda_model = create_cuda_model(cfg)
        cuda_model.load_from_numpy_dict(torch_params)
        cuda_params = cuda_model.get_all_parameters()

        max_diff = weight_diff_params(torch_params, cuda_params)
        assert max_diff < 1e-6, f"CUDA load diff too high: {max_diff:.6f}"


# ─── Test Case 6: Training Dynamics ───────────────────────────────────────────


class TestTrainingDynamics:
    """Training steps reduce the loss on synthetic data."""

    def test_torch_train_decreasing(self):
        """A few PyTorch training steps reduce the cross-entropy loss."""
        model = create_torch_model(MODEL_CONFIG_1L)
        model.train()

        tokens = torch.randint(0, MODEL_CONFIG_1L["vocab_size"], (1, 32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        loss_fn = torch.nn.functional.cross_entropy

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(tokens)
            loss = loss_fn(logits.reshape(-1, MODEL_CONFIG_1L["vocab_size"]), tokens.reshape(-1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_triton_train_decreasing(self):
        """Triton train_step reduces the loss on synthetic data."""
        from impl._triton.training import train_step

        cfg = {**MODEL_CONFIG_1L, "rope_dim": 0}
        model = create_triton_model(cfg).cuda()
        model.train()

        tokens = torch.randint(0, cfg["vocab_size"], (1, 32), device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()

        losses = []
        for _ in range(5):
            losses.append(train_step(model, tokens, tokens, optimizer, loss_fn))

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
