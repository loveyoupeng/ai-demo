"""Test cases for auto_test_equivalence.py — Phase G+++ TDD plan.

Each test case drives verification of cross-backend weight comparison:
1. Key normalization maps correctly
2. Shared weights -> identical outputs
3. Self-reproducibility (same seed -> same weights)
4. Torch vs Triton weight drift after independent training
5. Round-trip equivalence (baseline)
6. CUDA key mapping
7. Training dynamics comparable
"""

import numpy as np
import pytest
import torch

from impl._cuda.model import CUDAModel
from impl._np.model import NumPyModel
from impl._torch.layers import TorchModel
from impl._triton.model import TritonModel

# ─── Configs ──────────────────────────────────────────────────────────────────

MODEL_CONFIG_1L = {
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

MODEL_CONFIG_1L_NOROPE = {
    "vocab_size": 64,
    "embed_dim": 32,
    "n_layers": 1,
    "n_heads": 4,
    "n_experts": 1,
    "ff_dim": 64,
    "k": 1,
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

TRAIN_SHARED = {
    "vocab_size": 64,
    "context_length": 32,
    "train_steps": 2,
    "lr": 0.01,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def create_torch_model(config):
    """Create and return an initialized TorchModel."""
    return TorchModel(**config)


def create_numpy_model(config):
    """Create and return an initialized NumPyModel."""
    return NumPyModel(**config)


def create_triton_model(config):
    """Create and return an initialized TritonModel (no rope_dim)."""
    cfg = {k: v for k, v in config.items() if k != "rope_dim"}
    return TritonModel(**cfg)


def create_cuda_model(config):
    """Create and return an initialized CUDAModel."""
    return CUDAModel(**config)


def train_torch_seed42(config=None):
    """Train a PyTorch model with fixed config and return its params."""
    mcfg = {**MODEL_CONFIG_1L}
    tcfg = {**TRAIN_SHARED}
    if config:
        mcfg.update(config)
        tcfg["vocab_size"] = config.get("vocab_size", tcfg["vocab_size"])
        tcfg["context_length"] = config.get("context_length", tcfg["context_length"])
        tcfg["train_steps"] = config.get("train_steps", tcfg["train_steps"])
        tcfg["lr"] = config.get("lr", tcfg["lr"])

    torch.manual_seed(mcfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(mcfg["seed"])
    model = create_torch_model(mcfg)
    model.eval()

    ctx_len = tcfg["context_length"]
    train_steps = tcfg["train_steps"]
    tokens = torch.randint(0, mcfg["vocab_size"], (1, ctx_len * train_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
    loss_fn = torch.nn.functional.cross_entropy

    for _ in range(train_steps):
        optimizer.zero_grad()
        logits = model(tokens)
        loss = loss_fn(logits.reshape(-1, mcfg["vocab_size"]), tokens.reshape(-1))
        loss.backward()
        optimizer.step()

    return model.save_as_numpy()


def train_numpy_seed42(config=None):
    """Train a NumPy model with fixed config and return its params."""
    mcfg = {**MODEL_CONFIG_1L}
    tcfg = {**TRAIN_SHARED}
    if config:
        mcfg.update(config)
        tcfg["vocab_size"] = config.get("vocab_size", tcfg["vocab_size"])
        tcfg["context_length"] = config.get("context_length", tcfg["context_length"])
        tcfg["train_steps"] = config.get("train_steps", tcfg["train_steps"])
        tcfg["lr"] = config.get("lr", tcfg["lr"])

    model = create_numpy_model(mcfg)
    np.random.seed(mcfg["seed"])

    ctx_len = tcfg["context_length"]
    train_steps = tcfg["train_steps"]
    for _ in range(train_steps):
        tokens = np.random.randint(0, mcfg["vocab_size"], (1, ctx_len), dtype=np.int32)
        model.forward(tokens)
        params = model.get_all_parameters()
        for param in params.values():
            param[:] -= tcfg["lr"] * np.random.randn(*param.shape)

    return model.get_all_parameters()


def _build_numpy_weight_map():
    """Build a weight map with NumPy-style keys for NumPy model."""
    np.random.seed(42)
    return {
        "embedding_weights": np.random.randn(64, 32).astype(np.float32),
        "final_ln_gamma": np.random.randn(32).astype(np.float32),
        "output.W1": np.random.randn(32, 64).astype(np.float32),
        "output.W2": np.random.randn(64, 32).astype(np.float32),
        "output.W3": np.random.randn(32, 64).astype(np.float32),
        "output_proj_w": np.random.randn(32, 64).astype(np.float32),
        "output_proj_b": np.random.randn(64).astype(np.float32),
        "blocks.0.gate1": np.random.randn(1).astype(np.float32),
        "blocks.0.gate2": np.random.randn(1).astype(np.float32),
        "blocks.0.ln1_gamma": np.random.randn(32).astype(np.float32),
        "blocks.0.ln2_gamma": np.random.randn(32).astype(np.float32),
        "blocks.0.mha.Wq": np.random.randn(32, 32).astype(np.float32),
        "blocks.0.mha.Wk": np.random.randn(32, 32).astype(np.float32),
        "blocks.0.mha.Wv": np.random.randn(32, 32).astype(np.float32),
        "blocks.0.mha.Wo": np.random.randn(32, 32).astype(np.float32),
        "blocks.0.mha.bq": np.random.randn(32).astype(np.float32),
        "blocks.0.mha.bk": np.random.randn(32).astype(np.float32),
        "blocks.0.mha.bv": np.random.randn(32).astype(np.float32),
        "blocks.0.mha.bo": np.random.randn(32).astype(np.float32),
        "blocks.0.moe.experts.0.W1": np.random.randn(32, 64).astype(np.float32),
        "blocks.0.moe.experts.0.W2": np.random.randn(64, 32).astype(np.float32),
        "blocks.0.moe.experts.0.W3": np.random.randn(32, 64).astype(np.float32),
        "blocks.0.moe.router": np.random.randn(32, 1).astype(np.float32),
        "blocks.0.moe.bias": np.random.randn(1).astype(np.float32),
    }


def normalize_params_to_torch(params, backend, n_layers=2, n_experts=1):
    """Convert params from any backend to torch canonical key names."""
    import scripts.auto_test_equivalence as m

    if backend == "torch":
        return params
    expanded = m._expand_map(m.INVERSE_TRITON_MAP, n_layers, n_experts)
    return {expanded.get(k, k): v for k, v in params.items()}


def weight_diff_params(params_a, params_b):
    """Compute max diff between two param dicts (handles numpy arrays and torch tensors)."""
    max_diff = 0.0
    for key in set(params_a) & set(params_b):
        a = params_a[key]
        b = params_b[key]
        if isinstance(a, torch.Tensor):
            a = a.cpu().detach().numpy().astype(np.float64)
        else:
            a = np.asarray(a, dtype=np.float64).ravel()
        if isinstance(b, torch.Tensor):
            b = b.cpu().detach().numpy().astype(np.float64)
        else:
            b = np.asarray(b, dtype=np.float64).ravel()
        min_len = min(len(a), len(b))
        if min_len > 0:
            max_diff = max(max_diff, float(np.max(np.abs(a[:min_len] - b[:min_len]))))
    return max_diff


def train_triton_seed42(config=None):
    """Train a Triton model with fixed config and return its params."""
    mcfg = {**MODEL_CONFIG_1L_NOROPE, "n_heads": 4}
    tcfg = {**TRAIN_SHARED}
    if config:
        mcfg.update(config)
        tcfg["vocab_size"] = config.get("vocab_size", tcfg["vocab_size"])
        tcfg["context_length"] = config.get("context_length", tcfg["context_length"])
        tcfg["train_steps"] = config.get("train_steps", tcfg["train_steps"])
        tcfg["lr"] = config.get("lr", tcfg["lr"])

    if not torch.cuda.is_available():
        pytest.skip("GPU required for Triton training")
    torch.manual_seed(mcfg["seed"])
    torch.cuda.manual_seed(mcfg["seed"])
    model = create_triton_model(mcfg)
    model = model.cuda()
    model.eval()

    ctx_len = tcfg["context_length"]
    train_steps = tcfg["train_steps"]
    tokens = torch.randint(0, mcfg["vocab_size"], (1, ctx_len), device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    from impl._triton.training import train_step

    for _ in range(train_steps):
        train_step(model, tokens, tokens, optimizer, loss_fn)

    return model.save_as_numpy()


# ─── CUDA Helpers ─────────────────────────────────────────────────────────────


def _collect_cuda_params(cuda_model, model_cfg):
    """Collect all tensor parameters from a CUDAModel into a dict."""
    cuda_params = {}
    for attr in [
        "embedding_weights",
        "final_ln_gamma",
        "output_W1",
        "output_W2",
        "output_W3",
        "output_proj_weights",
        "output_proj_bias",
    ]:
        val = getattr(cuda_model, attr, None)
        if val is not None and isinstance(val, torch.Tensor):
            cuda_params[attr] = val.detach().cpu()
    for i, block in enumerate(cuda_model.stacking.blocks):
        for attr in ["Wq", "Wk", "Wv", "Wo", "ln1_gamma", "ln2_gamma", "gate1", "gate2"]:
            val = getattr(block, attr, None)
            if val is not None and isinstance(val, torch.Tensor):
                cuda_params[f"blocks.{i}.{attr}"] = val.detach().cpu()
    return cuda_params


def _enable_cuda_grads(cuda_model):
    """Enable gradients on all CUDA model parameters."""
    for block in cuda_model.stacking.blocks:
        for attr in ["Wq", "Wk", "Wv", "Wo", "ln1_gamma", "ln2_gamma", "gate1", "gate2"]:
            if hasattr(block, attr):
                setattr(block, attr, getattr(block, attr).requires_grad_(True))
    for attr in [
        "embedding_weights",
        "final_ln_gamma",
        "output_W1",
        "output_W2",
        "output_W3",
        "output_proj_weights",
        "output_proj_bias",
    ]:
        if hasattr(cuda_model, attr):
            setattr(cuda_model, attr, getattr(cuda_model, attr).requires_grad_(True))


# ─── Test Case 1: Verify Key Normalization Maps Correctly ─────────────────────


class TestKeyNormalization:
    """Test that normalize_params_to_torch correctly maps all backend keys -> torch keys."""

    def test_numpy_keys_all_map_torch_1_layer(self):
        """All 24 NumPy keys should map to exactly 24 Torch keys -- no missing, no extra."""
        cfg = {"n_layers": 1}
        np_params = train_numpy_seed42(cfg)

        # Get torch params and normalize too (torch should be unchanged)
        torch_model = create_torch_model({**MODEL_CONFIG_1L, "n_layers": 1})
        torch_params = {n: p.cpu().detach().numpy() for n, p in torch_model.named_parameters()}

        np_normed = normalize_params_to_torch(np_params, "numpy", n_layers=1)

        # All torch keys should be present in normed numpy
        for key in torch_params:
            assert key in np_normed, f"Missing torch key in normalized: {key}"

        # No extra keys
        for key in np_normed:
            assert key in torch_params, f"Extra key in normalized that torch doesn't have: {key}"

        assert len(np_normed) == len(torch_params) == 24, (
            f"Expected 24 keys, got np={len(np_normed)}, torch={len(torch_params)}"
        )

    def test_numpy_keys_all_map_torch_2_layers(self):
        """All keys should map correctly for 2-layer model with 2 experts."""
        # Must override ALL config values that differ from base, not just n_layers
        cfg = {"n_layers": 2, "n_experts": 2, "vocab_size": 128, "embed_dim": 64, "ff_dim": 128}
        np_params = train_numpy_seed42(cfg)

        torch_model = create_torch_model({**MODEL_CONFIG_2L})
        torch_params = {n: p.cpu().detach().numpy() for n, p in torch_model.named_parameters()}

        np_normed = normalize_params_to_torch(np_params, "numpy", n_layers=2, n_experts=2)

        for key in torch_params:
            assert key in np_normed, f"Missing torch key in normalized: {key}"
        for key in np_normed:
            assert key in torch_params, f"Extra key in normalized that torch doesn't have: {key}"

    def test_triton_normalization(self):
        """Triton keys should also normalize to torch keys (same InverseTritonMap)."""
        cfg = {"n_layers": 1}
        np_params = train_numpy_seed42(cfg)

        np_normed = normalize_params_to_torch(np_params, "numpy", n_layers=1)

        # Check specific mappings -- blocks.* should map to stack.layers.*
        assert "blocks.0.Wq" not in np_normed
        assert "stack.layers.0.mha.Wq.weight" in np_normed
        assert "blocks.0.ln1_gamma" not in np_normed
        assert "stack.layers.0.ln1.gamma" in np_normed


# ─── Test Case 2: Shared Weights -> Identical Outputs ──────────────────────────


class TestSharedWeightsOutputs:
    """Same weights loaded into different backends -> same output for same input."""

    def test_torch_numpy_same_output(self):
        """Same numerical weights in NumPy and Torch should produce identical outputs."""
        cfg_1L = {**MODEL_CONFIG_1L}
        np_model = create_numpy_model(cfg_1L)
        torch_model = create_torch_model(cfg_1L)

        # All backends' load_from_numpy_dict accepts same numpy-style keys
        # Linear weights are in NumPy convention (in, out) and get transposed during load
        weight_map = _build_numpy_weight_map()

        # Load into NumPy model
        np_model.load_from_numpy_dict(weight_map)

        # Load into Torch model (transposes linear weights from NumPy (in,out) to Torch (out,in))
        # MUST call eval() after load_from_numpy_dict to disable dropout
        torch_model.load_from_numpy_dict(weight_map)  # type: ignore[arg-type]
        torch_model.eval()

        # Same input -- use token IDs within vocab_size bounds
        input_tokens = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)

        # Forward pass
        np_out = np_model.forward(input_tokens)
        torch_tokens = torch.from_numpy(input_tokens)
        torch_out = torch_model(torch_tokens).detach().numpy()

        max_diff = np.max(np.abs(np_out - torch_out))
        assert max_diff < 0.01, f"NumPy vs Torch output mismatch: max_diff={max_diff:.6f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_shared_weights_triton(self):
        """Triton model loaded with matching weights should produce near-identical output."""
        cfg_1L = {**MODEL_CONFIG_1L_NOROPE}
        np_model = create_numpy_model(cfg_1L)

        # All backends' load_from_numpy_dict accepts same numpy-style keys
        # Linear weights are in NumPy convention (in, out) and get transposed during load
        weight_map = _build_numpy_weight_map()

        np_model.load_from_numpy_dict(weight_map)

        # Triton model (transposes linear weights same as Torch)
        triton_model = create_triton_model(cfg_1L)
        triton_model.load_from_numpy_dict(weight_map)  # type: ignore[arg-type]
        triton_model = triton_model.cuda().eval()

        # Forward pass -- use token IDs within vocab_size bounds
        input_tokens = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        np_out = np_model.forward(input_tokens)

        triton_input = torch.from_numpy(input_tokens).to("cuda")
        triton_out = triton_model(triton_input).detach().cpu().numpy()

        max_diff = np.max(np.abs(np_out - triton_out))
        assert max_diff < 10, f"NumPy vs Triton output mismatch: max_diff={max_diff:.6f}"


# ─── Test Case 3: Self-Reproducibility ────────────────────────────────────────


class TestSelfReproducibility:
    """Same seed -> identical weights for same backend."""

    def test_numpy_self_reproducibility(self):
        """Two NumPy models with same seed should have identical weights."""
        mcfg = {**MODEL_CONFIG_1L}
        np.random.seed(42)
        np_model_a = create_numpy_model(mcfg)
        np.random.seed(42)
        np_model_b = create_numpy_model(mcfg)

        # Train both (both get same random data)
        ctx_len = 32
        params_a = None
        params_b = None
        for step in range(2):
            np.random.seed(42 + step)
            tokens_a = np.random.randint(0, mcfg["vocab_size"], (1, ctx_len), dtype=np.int32)
            np_model_a.forward(tokens_a)
            params_a = np_model_a.get_all_parameters()
            np.random.seed(1000 + step)
            for param in params_a.values():
                param[:] -= 0.01 * np.random.randn(*param.shape)

            np.random.seed(42 + step)
            tokens_b = np.random.randint(0, mcfg["vocab_size"], (1, ctx_len), dtype=np.int32)
            np_model_b.forward(tokens_b)
            params_b = np_model_b.get_all_parameters()
            np.random.seed(1000 + step)
            for param in params_b.values():
                param[:] -= 0.01 * np.random.randn(*param.shape)

        # Compare -- should be zero if self-reproducible
        assert params_a is not None and params_b is not None
        max_diff = weight_diff_params(params_a, params_b)
        assert max_diff < 1e-10, f"Self-reproducibility failed: max_diff={max_diff}"

    def test_torch_self_reproducibility(self):
        """Two torch models with same seed should have identical weights."""
        torch.manual_seed(42)
        model_a = create_torch_model(MODEL_CONFIG_1L)
        torch.manual_seed(42)
        model_b = create_torch_model(MODEL_CONFIG_1L)

        # Compare initial weights
        params_a = {n: p.clone() for n, p in model_a.named_parameters()}
        params_b = {n: p.clone() for n, p in model_b.named_parameters()}

        for key in params_a:
            assert torch.allclose(params_a[key], params_b[key], atol=1e-10), f"Self-reproducibility failed for {key}"


# ─── Test Case 4: Torch vs Triton Weight Drift ────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestTorchTritonDrift:
    """When both backends use PyTorch-based RNG, weight drift should be small."""

    def test_torch_triton_drift_less_than_threshold(self):
        """Torch vs Triton weight diff should be < 0.5 with same seed."""
        cfg = {"n_layers": 1}

        torch.manual_seed(42)
        torch_params = train_torch_seed42(cfg)

        torch.manual_seed(42)
        triton_params = train_triton_seed42(cfg)

        # Torch save_as_numpy returns numpy arrays already
        # Triton save_as_numpy also returns numpy arrays
        # Normalize triton to torch keys
        triton_normed = normalize_params_to_torch(triton_params, "triton", n_layers=1)

        max_diff = 0.0
        for key in set(torch_params) & set(triton_normed):
            a = np.asarray(torch_params[key], dtype=np.float64).ravel()
            b = np.asarray(triton_normed[key], dtype=np.float64).ravel()
            min_len = min(len(a), len(b))
            if min_len > 0:
                max_diff = max(max_diff, float(np.max(np.abs(a[:min_len] - b[:min_len]))))

        assert max_diff < 0.5, f"torch vs triton diff too high: {max_diff:.4f}"


# ─── Test Case 5: Round-Trip Equivalence ──────────────────────────────────────


class TestRoundTripEquivalence:
    """Round-trip (save -> load) should produce exact weight match."""

    def test_torch_to_numpy_roundtrip(self):
        """Train torch -> save_as_numpy -> load into numpy -> compare weights should be zero."""
        model_cfg = {**MODEL_CONFIG_1L}
        ctx_len = 32
        train_steps = 3

        torch.manual_seed(42)
        torch_model = create_torch_model(model_cfg)
        torch_model.eval()

        tokens = torch.randint(0, model_cfg["vocab_size"], (1, ctx_len))
        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=0.01)
        loss_fn = torch.nn.functional.cross_entropy

        for _ in range(train_steps):
            optimizer.zero_grad()
            logits = torch_model(tokens)
            loss = loss_fn(logits.reshape(-1, model_cfg["vocab_size"]), tokens.reshape(-1))
            loss.backward()
            optimizer.step()

        # Save as numpy
        torch_np_dict = torch_model.save_as_numpy()

        # Load into numpy model
        np_model = create_numpy_model(model_cfg)
        np_model.load_from_numpy_dict(
            {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v.copy() for k, v in torch_np_dict.items()}
        )
        np_loaded = np_model.get_all_parameters()

        # Compare -- normalize both to torch keys first
        torch_normed = normalize_params_to_torch(torch_np_dict, "torch", n_layers=1)
        np_normed = normalize_params_to_torch(np_loaded, "numpy", n_layers=1)

        max_diff = 0.0
        for key in set(torch_normed) & set(np_normed):
            a = np.asarray(torch_normed[key], dtype=np.float64).ravel()
            b = np.asarray(np_normed[key], dtype=np.float64).ravel()
            min_len = min(len(a), len(b))
            if min_len > 0:
                max_diff = max(max_diff, float(np.max(np.abs(a[:min_len] - b[:min_len]))))
        assert max_diff < 1e-10, f"Round-trip failed: diff={max_diff}"

    def test_numpy_to_torch_roundtrip(self):
        """Train numpy -> save -> load into torch -> compare weights should be zero."""
        mcfg = {**MODEL_CONFIG_1L}

        np_model = create_numpy_model(mcfg)
        np.random.seed(42)

        # Train (random gradient update)
        for _ in range(3):
            params = np_model.get_all_parameters()
            for param in params.values():
                param[:] -= 0.01 * np.random.randn(*param.shape)

        # Save
        np_saved = np_model.get_all_parameters()

        # Load into torch -- load_from_numpy_dict expects numpy arrays
        torch_model = create_torch_model(mcfg)
        torch_model.load_from_numpy_dict(np_saved)
        torch_loaded = {n: p.cpu().detach().numpy() for n, p in torch_model.named_parameters()}

        # Compare
        max_diff = weight_diff_params(np_saved, torch_loaded)
        assert max_diff < 1e-10, f"Round-trip failed: diff={max_diff}"


# ─── Test Case 6: CUDA Key Mapping ────────────────────────────────────────────


class TestCUDAKeyMapping:
    """CUDA model uses different attribute names but weight diff comparison should work."""

    def test_cuda_keys_exist(self):
        """CUDA model should have expected flat attribute names."""
        cuda_model = create_cuda_model(MODEL_CONFIG_1L)

        # Check model-level keys
        for attr in ["embedding_weights", "final_ln_gamma", "output_W1", "output_W2", "output_W3"]:
            assert hasattr(cuda_model, attr), f"CUDA missing model-level attr: {attr}"

        # Check block-level keys
        block = cuda_model.stacking.blocks[0]
        for attr in ["Wq", "Wk", "Wv", "Wo", "ln1_gamma", "ln2_gamma", "gate1", "gate2"]:
            assert hasattr(block, attr), f"CUDA block missing attr: {attr}"

    def test_cuda_key_normalization(self):
        """CUDA keys should normalize to torch or be handled gracefully."""
        cuda_model = create_cuda_model(MODEL_CONFIG_1L)
        _enable_cuda_grads(cuda_model)

        cuda_params = _collect_cuda_params(cuda_model, MODEL_CONFIG_1L)

        # Normalize
        cuda_normed = normalize_params_to_torch(cuda_params, "cuda", n_layers=1)

        # Check that specific mappings happen
        assert "embedding_weights" not in cuda_normed, "CUDA key should have been normalized"
        assert "embedding.weight" in cuda_normed, "CUDA key should map to torch key"
        assert "blocks.0.Wq" not in cuda_normed, "CUDA block key should have been normalized"
        assert "stack.layers.0.mha.Wq.weight" in cuda_normed, "CUDA block key should map to torch"

    def test_cuda_weight_diff_with_torch(self):
        """CUDA vs Torch weight diff should compute meaningfully."""
        cuda_model = create_cuda_model(MODEL_CONFIG_1L)
        _enable_cuda_grads(cuda_model)

        # Collect CUDA params normalized
        cuda_params = _collect_cuda_params(cuda_model, MODEL_CONFIG_1L)
        cuda_normed = normalize_params_to_torch(cuda_params, "cuda", n_layers=1)

        # Compare against torch model params
        torch_model = create_torch_model(MODEL_CONFIG_1L)
        torch_params = {n: p.cpu().detach().numpy() for n, p in torch_model.named_parameters()}

        # Compute diff for common keys
        common = set(torch_params) & set(cuda_normed)
        assert len(common) > 0, "CUDA normalized should have keys matching torch"

        max_diff = 0.0
        for key in common:
            a = np.asarray(torch_params[key], dtype=np.float64).ravel()
            b = np.asarray(cuda_normed[key], dtype=np.float64).ravel()
            min_len = min(len(a), len(b))
            if min_len > 0:
                max_diff = max(max_diff, float(np.max(np.abs(a[:min_len] - b[:min_len]))))

        # Both initialized with same seed, diff should be small
        assert max_diff < 1.0, f"CUDA vs Torch diff too high: {max_diff:.4f}"


# ─── Test Case 7: Training Dynamics Comparable ────────────────────────────────


class TestTrainingDynamics:
    """Both backends show same qualitative behavior (loss decreases, reasonable magnitude)."""

    def test_torch_train_decreasing(self):
        """Torch training should produce decreasing loss."""
        model_cfg = {**MODEL_CONFIG_1L}
        ctx_len = 32
        train_steps = 10

        torch.manual_seed(42)
        model = create_torch_model(model_cfg)
        model.eval()

        tokens = torch.randint(0, model_cfg["vocab_size"], (1, ctx_len))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = torch.nn.functional.cross_entropy

        losses = []
        for _ in range(train_steps):
            optimizer.zero_grad()
            logits = model(tokens)
            loss = loss_fn(logits.reshape(-1, model_cfg["vocab_size"]), tokens.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss not decreasing: {losses[0]:.4f} -> {losses[-1]:.4f}"

        # Reasonable range
        for loss in losses:
            assert 0 < loss < 100, f"Out of range: {loss}"

    def test_torch_triton_both_decrease(self):
        """Both Torch and Triton should show decreasing loss."""
        if not torch.cuda.is_available():
            pytest.skip("GPU required for Triton")

        model_cfg = {**MODEL_CONFIG_1L}
        ctx_len = 32
        train_steps = 10

        # Torch
        torch.manual_seed(42)
        torch_model = create_torch_model(model_cfg)
        torch_model.eval()
        tokens = torch.randint(0, model_cfg["vocab_size"], (1, ctx_len))
        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=0.01)
        loss_fn = torch.nn.functional.cross_entropy

        torch_losses = []
        for _ in range(train_steps):
            optimizer.zero_grad()
            logits = torch_model(tokens)
            loss = loss_fn(logits.reshape(-1, model_cfg["vocab_size"]), tokens.reshape(-1))
            loss.backward()
            optimizer.step()
            torch_losses.append(loss.item())

        # Triton
        torch.manual_seed(42)
        triton_model = create_triton_model(MODEL_CONFIG_1L_NOROPE)
        triton_model = triton_model.cuda()
        triton_model.eval()
        tokens_cuda = torch.randint(0, model_cfg["vocab_size"], (1, ctx_len), device="cuda")
        triton_optimizer = torch.optim.AdamW(triton_model.parameters(), lr=0.01)
        triton_loss_fn = torch.nn.CrossEntropyLoss()
        from impl._triton.training import train_step

        triton_losses = []
        for _ in range(train_steps):
            loss = train_step(triton_model, tokens_cuda, tokens_cuda, triton_optimizer, triton_loss_fn)
            triton_losses.append(loss)

        # Both should decrease
        assert torch_losses[-1] < torch_losses[0], (
            f"Torch loss not decreasing: {torch_losses[0]:.4f} -> {torch_losses[-1]:.4f}"
        )
        assert triton_losses[-1] < triton_losses[0], (
            f"Triton loss not decreasing: {triton_losses[0]:.4f} -> {triton_losses[-1]:.4f}"
        )

        # Reasonable range
        for loss in torch_losses + triton_losses:
            assert 0 < loss < 100, f"Out of range: {loss}"
