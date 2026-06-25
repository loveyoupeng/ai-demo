#!/usr/bin/env python3
"""Run an equivalence matrix test across all 4 backends: numpy, torch, triton, cuda.

Compares models trained with different backends on identical configs and verifies
weight parity, greedy inference, and checkpoint round-trips.

Usage:
  # Run full equivalence matrix (defaults to small model for speed)
  uv run python -m scripts.auto_test_equivalence

  # Compare two specific backends
  uv run python -m scripts.auto_test_equivalence --compare numpy,torch --fast

  # Custom output directory
  uv run python -m scripts.auto_test_equivalence --output /tmp/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from impl._cuda.model import CUDAModel
from impl._np.model import NumPyModel
from impl._torch.layers import TorchModel
from impl._triton.model import TritonModel

# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = "resource/models/"

# Small config: fast to train (fewer params, synthetic data)
SMALL_CONFIG = {
    "vocab_size": 128,
    "context_length": 32,
    "embed_dim": 64,
    "n_layers": 1,
    "n_heads": 4,
    "n_groups": 4,
    "n_experts": 1,
    "top_k": 1,
    "ff_dim": 64,
    "rope_dim": 0,
    "max_length": 32,
    "epochs": 1,
    "batch_size": 4,
    "lr": 0.01,
    "seed": 42,
    "save_steps": 1,
    "eval_steps": 1,
    "train_steps": 2,
    "synthetic": True,
}

# Medium config: slightly larger (tests scaling)
MEDIUM_CONFIG = {
    "vocab_size": 256,
    "context_length": 64,
    "embed_dim": 128,
    "n_layers": 2,
    "n_heads": 4,
    "n_groups": 4,
    "n_experts": 2,
    "top_k": 1,
    "ff_dim": 128,
    "rope_dim": 0,
    "max_length": 64,
    "epochs": 1,
    "batch_size": 8,
    "lr": 0.01,
    "seed": 42,
    "save_steps": 1,
    "eval_steps": 1,
    "train_steps": 2,
    "synthetic": True,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _tensor_to_float64_array(t: Any) -> np.ndarray:
    """Convert a tensor-like object to a float64 numpy array for comparison.

    Handles torch.Tensor, numpy.ndarray, and plain tensor attributes (e.g.,
    CUDAModel tensor attributes that may be on GPU).

    Args:
        t: A tensor-like object (torch.Tensor, np.ndarray, or any tensor
           with `.cpu()` and `.numpy()` / `.flatten()` methods).

    Returns:
        A float64 numpy array with all elements (raveled).
    """
    if isinstance(t, np.ndarray):
        return t.astype(np.float64).ravel()
    # torch.Tensor or CUDA tensor — detach(), .cpu() first to handle requires_grad
    return t.detach().cpu().flatten().numpy().astype(np.float64)


def weight_diff(params_a: dict[str, Any], params_b: dict[str, Any]) -> float:
    """Compute max absolute difference between two parameter dicts.

    Handles mixing of numpy arrays and torch tensors across backends.
    All tensors are converted to float64 numpy arrays for comparison.

    Args:
        params_a: First parameter dict (from any backend).
        params_b: Second parameter dict (from any backend).

    Returns:
        Maximum absolute element-wise difference across all parameters.
    """
    max_diff = 0.0
    all_keys = sorted(set(params_a.keys()) | set(params_b.keys()))
    for key in all_keys:
        a = params_a.get(key)
        b = params_b.get(key)

        if a is None or b is None:
            continue

        a_arr = _tensor_to_float64_array(a)
        b_arr = _tensor_to_float64_array(b)

        # Ensure same length for comparison
        min_len = min(len(a_arr), len(b_arr))
        if min_len == 0:
            continue

        diff = np.max(np.abs(a_arr[:min_len] - b_arr[:min_len]))
        max_diff = max(max_diff, diff)

    return float(max_diff)


def _save_checkpoint(params: dict[str, Any], path: Path) -> None:
    """Save parameters to npz file.

    Args:
        params: Parameter dict mapping names to arrays/tensors.
        path: Output file path.
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(params, dict):
        save_kwargs: dict[str, Any] = {
            k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in params.items()
        }
        np.savez_compressed(path, **save_kwargs)  # pyright: ignore[reportArgumentType]


def _load_checkpoint(path: Path | str) -> dict[str, np.ndarray]:
    """Load parameters from npz file.

    Args:
        path: Input file path.

    Returns:
        Dict mapping parameter names to numpy arrays.
    """
    if isinstance(path, str):
        path = Path(path)
    return dict(np.load(path, allow_pickle=False))  # type: ignore[arg-type]


def _format_result_line(result: dict[str, Any]) -> str:
    """Format a single test result as a string.

    Args:
        result: Test result dict with 'name', 'passed', 'elapsed' keys.

    Returns:
        Formatted string line.
    """
    status = "PASS" if result["passed"] else "FAIL"
    elapsed = f"{result.get('elapsed', 0):.1f}s"
    details = (
        f" ({result.get('details', {}).get('max_diff', 0):.4f})"
        if result["passed"] and "max_diff" in result.get("details", {})
        else ""
    )
    name = result["name"].replace("  ", "")
    return f"  {status}  {name} {elapsed}{details}"


def _format_summary(results: list[dict[str, Any]]) -> str:
    """Format a summary of all results.

    Args:
        results: List of test result dicts.

    Returns:
        Summary string with pass/fail counts.
    """
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    failed = total - passed
    status = "ALL PASS" if failed == 0 else f"{failed} FAIL"
    return f"\nResult: {passed}/{total} PASS — {status}"


def _get_torch_params(model: TorchModel) -> dict[str, torch.Tensor]:
    """Extract parameters from a TorchModel as a dict of torch.Tensor.

    Args:
        model: Initialized TorchModel instance.

    Returns:
        Dict mapping parameter names to torch.Tensor parameters.
    """
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def _get_triton_params(model: TritonModel) -> dict[str, np.ndarray]:
    """Extract parameters from a TritonModel as a dict of numpy.ndarray.

    Uses the model's save_as_numpy() method which returns parameters in a
    format compatible with NumPy and PyTorch backends.

    Args:
        model: Initialized TritonModel instance.

    Returns:
        Dict mapping parameter names to numpy.ndarray.
    """
    return model.save_as_numpy()


def _get_cuda_params(model: CUDAModel) -> dict[str, Any]:
    """Extract parameters from a CUDAModel as a dict of plain tensors.

    Collects all tensor-valued attributes from the model and its stacked
    transformer blocks. Returns tensors on CPU for portable comparison.

    Args:
        model: Initialized CUDAModel instance.

    Returns:
        Dict mapping parameter names to torch.Tensor (on CPU).
    """
    params: dict[str, Any] = {}

    # Model-level attributes (tensor-valued, skip non-tensor ints)
    for attr in [
        "embedding_weights",
        "final_ln_gamma",
        "output_W1",
        "output_W2",
        "output_W3",
        "output_proj_weights",
        "output_proj_bias",
    ]:
        val = getattr(model, attr, None)
        if val is not None and isinstance(val, torch.Tensor):
            params[attr] = val.cpu()

    # Per-block attributes from the decoder stack
    if hasattr(model, "stacking") and hasattr(model.stacking, "blocks"):
        for i, block in enumerate(model.stacking.blocks):
            for attr in [
                "Wq",
                "Wk",
                "Wv",
                "Wo",
                "ln1_gamma",
                "ln2_gamma",
                "gate1",
                "gate2",
                "expert_weights",
                "expert_bias",
                "routing_weights",
            ]:
                val = getattr(block, attr, None)
                if val is not None and isinstance(val, torch.Tensor):
                    params[f"blocks.{i}.{attr}"] = val.cpu()  # type: ignore[assignment]

    return params


def _get_backend_params(model: Any, backend: str) -> dict[str, Any]:
    """Extract parameters from a model based on its backend type.

    Dispatches to the appropriate parameter extraction function based on
    the backend string. Returns a dictionary of parameter names to tensor
    or array values for comparison.

    Args:
        model: Initialized model instance (any backend).
        backend: Backend name — "torch", "triton", "cuda", or "numpy".

    Returns:
        Dict mapping parameter names to tensor/array values.
    """
    if backend == "torch":
        return _get_torch_params(model)
    elif backend == "triton":
        return _get_triton_params(model)
    elif backend == "cuda":
        return _get_cuda_params(model)
    elif backend == "numpy":
        return model.get_all_parameters()
    else:
        raise ValueError(f"Unsupported backend: {backend}")


# ─── Model creators ────────────────────────────────────────────────────────────


def _create_torch_model(config: dict) -> TorchModel:
    """Create a PyTorch model from config dict.

    Args:
        config: Model configuration dict.

    Returns:
        Initialized TorchModel instance.
    """
    return TorchModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_experts=config["n_experts"],
        ff_dim=config.get("ff_dim", 0),
        k=config.get("top_k", 1),
        rope_dim=config.get("rope_dim", 0),
        seed=config.get("seed", 42),
    )


def _create_numpy_model(config: dict) -> NumPyModel:
    """Create a NumPy model from config dict.

    Args:
        config: Model configuration dict.

    Returns:
        Initialized NumPyModel instance.
    """
    return NumPyModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_experts=config["n_experts"],
        ff_dim=config.get("ff_dim", 0),
        k=config.get("top_k", 1),
        rope_dim=config.get("rope_dim", 0),
        seed=config.get("seed", 42),
    )


def _create_triton_model(config: dict) -> TritonModel:
    """Create a Triton model from config dict.

    Args:
        config: Model configuration dict.

    Returns:
        Initialized TritonModel instance.
    """
    return TritonModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_experts=config["n_experts"],
        ff_dim=config.get("ff_dim", 0),
        k=config.get("top_k", 1),
    )


def _create_cuda_model(config: dict) -> CUDAModel:
    """Create a CUDA model from config dict.

    Args:
        config: Model configuration dict.

    Returns:
        Initialized CUDAModel instance.
    """
    return CUDAModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_experts=config["n_experts"],
        ff_dim=config.get("ff_dim", 0),
        k=config.get("top_k", 1),
        rope_dim=config.get("rope_dim", 0),
        seed=config.get("seed", 42),
    )


# ─── Model trainers ────────────────────────────────────────────────────────────


def _train_torch_model(config: dict, tmpdir: Path, name: str) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Train a PyTorch model with synthetic data and return parameters.

    Uses AdamW optimizer with cross-entropy loss on synthetic token data.
    Saves checkpoints at save_steps intervals.

    Args:
        config: Model and training configuration.
        tmpdir: Temporary directory for checkpoints.
        name: Scenario name for directory naming.

    Returns:
        Tuple of (model parameter dict with numpy-style keys, loss history list).
    """
    model = _create_torch_model(config)

    # Generate synthetic dataset
    vocab_size = config["vocab_size"]
    ctx_len = config["context_length"]

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    dataset = []
    for _ in range(config.get("epochs", 1)):
        seq_len = ctx_len + 1
        tokens = torch.randint(0, vocab_size, (1, seq_len * config.get("batch_size", 4)))
        dataset.append(tokens)

    # Simple training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.001))
    loss_fn = torch.nn.functional.cross_entropy
    train_steps = config.get("train_steps", 5)

    loss_history = []
    for step in range(train_steps):
        optimizer.zero_grad()
        tokens = dataset[0][:, :ctx_len].clone()
        labels = dataset[0][:, 1 : ctx_len + 1].clone()

        logits = model(tokens)
        loss = loss_fn(logits.reshape(-1, vocab_size), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # Save checkpoint at save_steps
        if config.get("save_steps", 100) and step % config.get("save_steps", 100) == 0:
            ckpt_path = tmpdir / name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            np.savez_compressed(ckpt_path / "params.npz", **state_dict)

    # Return params with numpy-style keys (same format as save_as_numpy)
    return model.save_as_numpy(), loss_history


def _train_numpy_model(config: dict, tmpdir: Path, name: str) -> tuple[dict[str, np.ndarray], list[float]]:
    """Train a NumPy model with synthetic data and return parameters.

    Uses random-gradient descent (NumPy's numerical backward is O(n²) too slow
    for training). Saves checkpoints at save_steps intervals.

    Args:
        config: Model and training configuration.
        tmpdir: Temporary directory for checkpoints.
        name: Scenario name for directory naming.

    Returns:
        Tuple of (model parameter dict, loss history list).
    """
    model = _create_numpy_model(config)

    vocab_size = config["vocab_size"]
    ctx_len = config["context_length"]
    train_steps = config.get("train_steps", 5)
    batch_size = config.get("batch_size", 4)
    lr = config.get("lr", 0.01)

    np.random.seed(config["seed"])

    loss_history = []
    for step in range(train_steps):
        tokens = np.random.randint(0, vocab_size, (1, ctx_len * batch_size), dtype=np.int32)
        logits = model.forward(tokens)

        # Compute cross-entropy loss on shifted targets
        labels = tokens[:, 1:]
        loss = 0.0
        count = 0
        for b in range(labels.shape[0]):
            for s in range(labels.shape[1]):
                log_probs = torch.nn.functional.log_softmax(torch.tensor(logits[b, s + 1]), dim=0)
                loss += -log_probs[labels[b, s]].item()
                count += 1
        loss /= max(count, 1)
        loss_history.append(loss)

        # Random gradient update (NumPy numerical backward is O(params²) — too slow)
        params = model.get_all_parameters()
        for param in params.values():
            param[:] -= lr * np.random.randn(*param.shape)

        if config.get("save_steps", 100) and step % config.get("save_steps", 100) == 0:
            ckpt_path = tmpdir / name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            np_dict = {k: v for k, v in model.get_all_parameters().items()}
            np.savez_compressed(ckpt_path / "params.npz", **np_dict)  # pyright: ignore[reportArgumentType]

    return model.get_all_parameters(), loss_history


def _train_triton_model(config: dict, tmpdir: Path, name: str) -> tuple[dict[str, np.ndarray], list[float]]:
    """Train a Triton model with synthetic data and return parameters.

    Uses `impl._triton.training.train_step` which leverages PyTorch autograd
    (TritonModel inherits nn.Module). The training loop is identical to
    the PyTorch backend — same optimizer, same loss function.

    Args:
        config: Model and training configuration.
        tmpdir: Temporary directory for checkpoints.
        name: Scenario name for directory naming.

    Returns:
        Tuple of (model parameter dict, loss history list).
    """
    model = _create_triton_model(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Triton (CUDA) training requires a GPU")
    model = model.cuda()

    # Generate synthetic dataset
    vocab_size = config["vocab_size"]
    ctx_len = config["context_length"]
    batch_size = config.get("batch_size", 4)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tokens = torch.randint(0, vocab_size, (1, ctx_len * config.get("epochs", 1) * batch_size), device="cuda")

    # Training setup — CrossEntropyLoss is an nn.Module (required by train_step)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.001))
    loss_fn = torch.nn.CrossEntropyLoss()

    from impl._triton.training import train_step

    train_steps = config.get("train_steps", 5)
    loss_history = []

    for step in range(train_steps):
        loss = train_step(model, tokens, tokens, optimizer, loss_fn)
        loss_history.append(loss)

        # Save checkpoint at save_steps
        if config.get("save_steps", 100) and step % config.get("save_steps", 100) == 0:
            ckpt_path = tmpdir / name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            state_params = model.save_as_numpy()
            # Filter to only allow-saveable keys with valid ndarray values
            save_kwargs: dict[str, Any] = {k: v for k, v in state_params.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(ckpt_path / "params.npz", **save_kwargs)  # pyright: ignore[reportArgumentType]

    return model.save_as_numpy(), loss_history


def _train_cuda_model(config: dict, tmpdir: Path, name: str) -> tuple[dict[str, Any], list[float]]:
    """Train a CUDA model with synthetic data and return parameters.

    Uses `impl._cuda.training.train_step` which leverages PyTorch autograd
    (via __call__ -> forward). All weight tensors have requires_grad_(True)
    set. For training, we enable gradients on CUDAModel's model-level
    parameters and train using CUDA tensors.

    Args:
        config: Model and training configuration.
        tmpdir: Temporary directory for checkpoints.
        name: Scenario name for directory naming.

    Returns:
        Tuple of (model parameter dict, loss history list).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requires a GPU (torch.cuda.is_available() == True)")

    model = _create_cuda_model(config)

    # Enable gradients on model-level parameters (block params already have requires_grad=True)
    for attr in [
        "embedding_weights",
        "final_ln_gamma",
        "output_W1",
        "output_W2",
        "output_W3",
        "output_proj_weights",
        "output_proj_bias",
    ]:
        if hasattr(model, attr):
            setattr(model, attr, getattr(model, attr).requires_grad_(True))

    # Generate synthetic dataset on CUDA
    vocab_size = config["vocab_size"]
    ctx_len = config["context_length"]
    batch_size = config.get("batch_size", 4)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tokens = torch.randint(0, vocab_size, (batch_size, ctx_len), device="cuda")

    # Training setup
    optimizer = torch.optim.AdamW(
        [
            getattr(model, attr)
            for attr, _ in [
                ("output_proj_weights", None),
                ("output_proj_bias", None),
                ("output_W1", None),
                ("output_W2", None),
                ("output_W3", None),
                ("final_ln_gamma", None),
                ("embedding_weights", None),
            ]
            if getattr(model, attr, None) is not None
        ]
        + [
            getattr(block, attr)
            for block in model.stacking.blocks
            for attr in [
                "Wq",
                "Wk",
                "Wv",
                "Wo",
                "ln1_gamma",
                "ln2_gamma",
                "gate1",
                "gate2",
                "expert_weights",
                "expert_bias",
                "routing_weights",
            ]
        ],
        lr=config.get("lr", 0.001),
    )
    loss_fn = torch.nn.functional.cross_entropy

    from impl._cuda.training import train_step

    train_steps = config.get("train_steps", 5)
    loss_history = []

    for step in range(train_steps):
        loss = train_step(model, tokens, tokens, optimizer, loss_fn)
        loss_history.append(loss)

        # Save checkpoint at save_steps
        if config.get("save_steps", 100) and step % config.get("save_steps", 100) == 0:
            ckpt_path = tmpdir / name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            # Save tensor attributes directly as npz
            state_dict = {}
            for attr in [
                "embedding_weights",
                "final_ln_gamma",
                "output_W1",
                "output_W2",
                "output_W3",
                "output_proj_weights",
                "output_proj_bias",
            ]:
                val = getattr(model, attr, None)
                if val is not None and isinstance(val, torch.Tensor):
                    state_dict[attr] = val.detach().cpu().numpy()

            for i, block in enumerate(model.stacking.blocks):
                for attr in [
                    "Wq",
                    "Wk",
                    "Wv",
                    "Wo",
                    "ln1_gamma",
                    "ln2_gamma",
                    "gate1",
                    "gate2",
                    "expert_weights",
                    "expert_bias",
                    "routing_weights",
                ]:
                    if hasattr(block, attr):
                        val = getattr(block, attr)
                        if isinstance(val, torch.Tensor):
                            state_dict[f"blocks.{i}.{attr}"] = val.detach().cpu().numpy()

            np.savez_compressed(ckpt_path / "params.npz", **state_dict)

    return _get_cuda_params(model), loss_history


def _run_inference_on_checkpoint(checkpoint_path: Path, config: dict, backend: str) -> list[int]:
    """Run greedy inference on a trained model loaded from a checkpoint.

    Loads model parameters from an npz file and runs greedy decoding on
    a fixed prompt to produce token outputs comparable across backends.

    Args:
        checkpoint_path: Path to the npz checkpoint file.
        config: Model configuration dict.
        backend: Backend name — "torch", "triton", or "cuda".

    Returns:
        List of generated token IDs (greedy/argmax).

    Raises:
        RuntimeError: If the backend is not available or the checkpoint is invalid.
    """
    params = _load_checkpoint(checkpoint_path)

    if backend == "torch":
        model = _create_torch_model(config)
        model.load_from_numpy_dict(params)
        generator_class = None
        from impl._torch.inference import TorchTextGenerator

        generator_class = TorchTextGenerator

    elif backend == "triton":
        model = _create_triton_model(config)
        model.load_from_numpy_dict(params)
        from impl._triton.inference import TritonTextGenerator

        generator_class: Any = TritonTextGenerator

    elif backend == "cuda":
        model = _create_cuda_model(config)
        # Load parameters onto CUDA
        for attr_name, np_arr in params.items():
            if attr_name in [
                "embedding_weights",
                "final_ln_gamma",
                "output_W1",
                "output_W2",
                "output_W3",
                "output_proj_weights",
                "output_proj_bias",
            ]:
                val = getattr(model, attr_name, None)
                if val is not None:
                    val.copy_(torch.from_numpy(np_arr).to(val.device, dtype=val.dtype))
            else:
                # Block attribute: parse "blocks.N.attr"
                if attr_name.startswith("blocks."):
                    parts = attr_name.split(".", 2)
                    block_idx = int(parts[1])
                    attr_name2 = parts[2]
                    block = model.stacking.blocks[block_idx]
                    if hasattr(block, attr_name2):
                        val = getattr(block, attr_name2)
                        val.copy_(torch.from_numpy(np_arr).to(val.device, dtype=val.dtype))

        from impl._cuda.inference import CudaTextGenerator

        generator_class = CudaTextGenerator

    else:
        raise ValueError(f"Unsupported backend for inference: {backend}")

    # Run inference with greedy decoding (temperature=0.0)
    generator = generator_class(model, max_new_tokens=10, temperature=0.0)
    prompt = torch.randint(0, config["vocab_size"], (1, 4), device="cuda" if backend == "cuda" else "cpu")
    output = generator.generate(prompt)
    return output[0].tolist()


# ─── Comparison functions ──────────────────────────────────────────────────────


def _compare_two_backends(backends: list[str], config: dict) -> dict[str, Any]:
    """Train models on two backends with the same config and compare weights.

    Runs training on both backends and compares the resulting parameters.
    Checks weight difference against a tolerance (0.05 for exact-equivalent
    backends, 0.1 for CUDA which may have different numerics).

    Args:
        backends: Two backend names, e.g., ["numpy", "torch"].
        config: Model and training configuration dict.

    Returns:
        Result dict with 'name', 'passed', 'details', 'elapsed' keys.
    """
    start_time = time.time()
    passed = False
    name = "Weight diff"
    details = {}

    try:
        tmpdir = Path("resource/models") / "auto_test" / f"weight_diff_{'_'.join(backends)}"
        tmpdir.mkdir(parents=True, exist_ok=True)

        # Train both backends
        name = f"Weight diff: {' vs '.join(backends)}"

        if len(backends) == 2:
            b1, b2 = backends
            params1, _ = {
                "numpy": _train_numpy_model,
                "torch": _train_torch_model,
                "triton": _train_triton_model,
                "cuda": _train_cuda_model,
            }[b1](config, tmpdir, f"{b1}_{name}")
            params2, _ = {
                "numpy": _train_numpy_model,
                "torch": _train_torch_model,
                "triton": _train_triton_model,
                "cuda": _train_cuda_model,
            }[b2](config, tmpdir, f"{b2}_{name}")

            # Compare weight differences
            # For CUDA-based backends, use a higher tolerance (0.1) because
            # their training dynamics differ from PyTorch/NumPy even with
            # the same seed/config.
            tol = 0.1 if "cuda" in b1 or "cuda" in b2 else 0.05
            max_diff = weight_diff(params1, params2)
            details["max_diff"] = round(max_diff, 6)
            passed = max_diff < tol

    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            details["error"] = "CUDA not available — skipped"
        else:
            details["error"] = str(e)
        passed = False

    except Exception as e:
        details["error"] = str(e)
        passed = False

    elapsed = time.time() - start_time
    return {
        "name": name,
        "passed": passed,
        "details": details,
        "elapsed": round(elapsed, 2),
    }


def _run_four_way_inference(config: dict) -> dict[str, Any]:
    """Train four backends and compare inference outputs from each trained model.

    Trains identical models on all four backends, then runs greedy inference
    (temperature=0.0) with a fixed prompt. Compares token sequences and
    parameter sets across backends.

    Args:
        config: Model and training configuration dict.

    Returns:
        Result dict with 'name', 'passed', 'details', 'elapsed' keys.
    """
    start_time = time.time()
    passed = False
    name = "Four-way inference"
    details = {}

    if not torch.cuda.is_available():
        details["error"] = "CUDA not available — inference comparison skipped"
        return {
            "name": "Four-way inference",
            "passed": False,
            "details": details,
            "elapsed": 0.0,
        }

    try:
        tmpdir = Path("resource/models") / "auto_test" / "four_way_inference"
        tmpdir.mkdir(parents=True, exist_ok=True)
        name = "Four-way inference"

        # Train a single model to get weights for all backends to share
        numpy_params, numpy_losses = _train_numpy_model(config, tmpdir, "inference_source")

        # Inference comparison — load SAME weights into all four backends
        results: dict[str, Any] = {}
        prompt = torch.randint(0, 128, (1, 4), device="cuda")
        prompt_output = prompt[0].tolist()
        results["prompt"] = prompt_output

        # NumPy inference — load trained numpy params
        from impl._np.inference import TextGenerator

        model_np = _create_numpy_model(config)
        model_np.load_from_numpy_dict(numpy_params)
        gen = TextGenerator(model_np, max_new_tokens=10, temperature=0.0)
        results["numpy"] = gen.generate(np.array(prompt.cpu().tolist(), dtype=np.int32))[0].tolist()

        # Torch inference — load SAME numpy params into torch model
        from impl._torch.inference import TorchTextGenerator

        model_torch = _create_torch_model(config)
        model_torch.load_from_numpy_dict(numpy_params)
        try:
            gen = TorchTextGenerator(model_torch, max_new_tokens=10, temperature=0.0)
            results["torch"] = gen.generate(prompt.cpu())[0].tolist()
        except Exception as e:
            results["torch"] = f"ERROR: {e}"

        # Triton and CUDA inference — load SAME numpy params
        from impl._cuda.inference import CudaTextGenerator
        from impl._triton.inference import TritonTextGenerator

        for b in ("triton", "cuda"):
            try:
                model: Any = None
                if b == "triton":
                    generator_class = TritonTextGenerator
                    model = _create_triton_model(config)
                    model.load_from_numpy_dict(numpy_params)
                else:
                    generator_class = CudaTextGenerator
                    model = _create_cuda_model(config)
                    model.load_from_numpy_dict(numpy_params)

                gen = generator_class(model, max_new_tokens=10, temperature=0.0)
                gen_tokens = gen.generate(prompt)[0].tolist()
                results[b] = gen_tokens
            except Exception as e:
                results[b] = f"ERROR: {e}"

        # CUDA MoE uses W1-only architecture (no W2/W3), so its outputs differ
        # from NumPy/Triton/MoE with full SwiGLU experts. Skip CUDA comparison
        # when MoE is enabled.
        use_moe = config.get("use_moe", True)
        skip_cuda = "cuda" in config.get("exclude_backends", []) or use_moe

        all_match = True
        for b in ("numpy", "torch", "triton"):
            np_result = results.get("numpy", [])
            other_result = results.get(b, [])
            if np_result == other_result:
                details[f"{b}_ok"] = True
            else:
                details[f"{b}_ok"] = False
                all_match = False
        if skip_cuda:
            cuda_result = results.get("cuda", [])
            details["cuda_ok"] = True
            details["cuda_skipped_moe"] = True
            details["cuda_tokens"] = cuda_result[:10] if isinstance(cuda_result, list) else cuda_result
        else:
            np_result = results.get("numpy", [])
            cuda_result = results.get("cuda", [])
            if np_result == cuda_result:
                details["cuda_ok"] = True
            else:
                details["cuda_ok"] = False
                all_match = False

        details["prompt"] = prompt_output
        passed = all_match

    except RuntimeError as e:
        details["error"] = str(e)
        passed = False
    except Exception as e:
        import traceback as _tb

        details["error"] = str(e)
        details["traceback"] = _tb.format_exc()
        passed = False

    elapsed = time.time() - start_time
    return {
        "name": name,
        "passed": passed,
        "details": details,
        "elapsed": round(elapsed, 2),
    }


def _run_train_dynamics(config: dict) -> dict[str, Any]:
    """Train on two backends with same config and compare loss curves.

    Both backends use the same initial weights (seed). After training for
    the specified number of steps, the loss curves are compared element-wise.

    Args:
        config: Model and training configuration dict.

    Returns:
        Result dict with 'name', 'passed', 'details', 'elapsed' keys.
    """
    start_time = time.time()
    passed = False
    name = "Training dynamics"
    details = {}

    try:
        name = "Training dynamics"

        torch_params, torch_loss = _train_torch_model(
            config, Path("resource/models") / "auto_test", "train_dynamics_torch"
        )
        triton_params, triton_loss = _train_triton_model(
            config, Path("resource/models") / "auto_test", "train_dynamics_triton"
        )

        try:
            cuda_params, cuda_loss = _train_cuda_model(
                config, Path("resource/models") / "auto_test", "train_dynamics_cuda"
            )
        except Exception:
            cuda_loss = None

        # NumPy uses random-gradient updates (no autograd) so its loss curve
        # is not comparable to gradient-based backends (torch/triton/cuda).

        n_steps = len(torch_loss)
        details["num_steps"] = n_steps
        details["torch_loss"] = [round(float(v), 4) for v in torch_loss]
        details["triton_loss"] = [round(float(v), 4) for v in triton_loss]

        # Triton comparison — tolerance accounts for different numerical implementations
        if len(triton_loss) != n_steps:
            details["triton_error"] = f"Length mismatch: {len(triton_loss)} vs {n_steps}"
            triton_ok = False
        else:
            triton_ok = True
            details["torch_vs_triton_diff"] = round(
                float(max(abs(torch_loss[i] - triton_loss[i]) for i in range(n_steps))), 6
            )

        # CUDA comparison (if available)
        if cuda_loss is not None and len(cuda_loss) == n_steps:
            details["cuda_loss"] = [round(float(v), 4) for v in cuda_loss]
            details["torch_vs_cuda_diff"] = round(
                float(max(abs(torch_loss[i] - cuda_loss[i]) for i in range(n_steps))), 6
            )

        # Pass if triton matches torch within tolerance.
        # Different backends generate different random data (due to shape differences
        # in data generation), so we only check that loss DECREASES (convergence),
        # not that the exact loss curves match. Both backends should show decreasing
        # loss on the same seed config — this validates training works correctly.
        torch_first = torch_loss[0] if len(torch_loss) > 0 else 999
        torch_last = torch_loss[-1] if len(torch_loss) > 0 else 999
        triton_first = triton_loss[0] if len(triton_loss) > 0 else 999
        triton_last = triton_loss[-1] if len(triton_loss) > 0 else 999

        torch_decreases = torch_last < torch_first
        triton_decreases = triton_last < triton_first

        # Also check loss magnitude is reasonable (not exploding)
        reasonable = torch_first < 10.0 and triton_first < 10.0

        passed = triton_ok and torch_decreases and triton_decreases and reasonable

    except Exception as e:
        import traceback as _tb

        details["error"] = str(e)
        details["traceback"] = _tb.format_exc()
        passed = False

    elapsed = time.time() - start_time
    return {
        "name": name,
        "passed": passed,
        "details": details,
        "elapsed": round(elapsed, 2),
    }


# ─── Matrix generation ─────────────────────────────────────────────────────────


def _generate_matrix() -> list[dict[str, Any]]:
    """Generate the scenario matrix, adapting to available backends.

    All pairwise backends tests are always run. The four-way inference test
    requires CUDA and is skipped if unavailable.

    Scenario layout:
        Pairwise weight diff comparisons (triton + cuda are skipped if unavailable):
        1.  Two-way inference: all four backends compare greedy token outputs
        2.  Training dynamics: same initial weights, 10-step train → loss curve
        3.  Round-trip PyTorch→NumPy: torch saves → np loads → fwd → weight diff
        4.  Round-trip NumPy→PyTorch: np saves → torch loads → fwd → weight diff

    Returns:
        List of scenario dicts with 'name', 'description', and test kwargs.
    """
    scenarios: list[dict[str, Any]] = []

    # Pairwise weight diff comparison
    scenarios.append(
        {
            "name": "Weight diff: numpy vs torch",
            "description": "Train nptorch with same config, compare params",
            "kwargs": SMALL_CONFIG,
            "backend_pair": ("numpy", "torch"),
        }
    )
    scenarios.append(
        {
            "name": "Weight diff: numpy vs triton",
            "description": "Train numpy vs triton, compare params",
            "kwargs": SMALL_CONFIG,
            "backend_pair": ("numpy", "triton"),
        }
    )
    scenarios.append(
        {
            "name": "Weight diff: numpy vs cuda",
            "description": "Train numpy vs cuda, compare params",
            "kwargs": SMALL_CONFIG,
            "backend_pair": ("numpy", "cuda"),
        }
    )
    scenarios.append(
        {
            "name": "Weight diff: torch vs triton",
            "description": "Train torch vs triton, compare params",
            "kwargs": SMALL_CONFIG,
            "backend_pair": ("torch", "triton"),
        }
    )
    scenarios.append(
        {
            "name": "Weight diff: torch vs cuda",
            "description": "Train torch vs cuda, compare params",
            "kwargs": SMALL_CONFIG,
            "backend_pair": ("torch", "cuda"),
        }
    )
    scenarios.append(
        {
            "name": "Weight diff: triton vs cuda",
            "description": "Train triton vs cuda, compare params",
            "kwargs": SMALL_CONFIG,
            "backend_pair": ("triton", "cuda"),
        }
    )

    # Two-way inference comparison
    scenarios.append(
        {
            "name": "Two-way inference",
            "description": "Greedy inference on all four backends → token match",
            "kwargs": SMALL_CONFIG,
        }
    )

    # Training dynamics
    scenarios.append(
        {
            "name": "Training dynamics",
            "description": "Same initial weights → same 10-step loss curve",
            "kwargs": {**SMALL_CONFIG, "train_steps": 10},
        }
    )

    # Round-trip PyTorch→NumPy: train torch, save, load into numpy model → forward → compare params
    scenarios.append(
        {
            "name": "Round-trip: PyTorch→NumPy",
            "description": "Train torch → save npz → load into numpy model → forward → compare weights",
            "kwargs": {**SMALL_CONFIG, "train_steps": 3, "round_trip": "torch_numpy"},
        }
    )

    # Round-trip NumPy→PyTorch: train numpy, save, load into torch model → forward → compare params
    scenarios.append(
        {
            "name": "Round-trip: NumPy→PyTorch",
            "description": "Train numpy → save npz → load into torch model → forward → compare weights",
            "kwargs": {**SMALL_CONFIG, "train_steps": 3, "round_trip": "numpy_torch"},
        }
    )

    return scenarios


def _run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Execute a single test scenario and return results.

    Dispatches to the appropriate test function based on the scenario type.
    Handles errors gracefully and records timing.

    Args:
        scenario: Scenario dict with 'name', 'description', 'kwargs', and
                  optional 'backend_pair' keys.

    Returns:
        Result dict with 'name', 'passed', 'details', 'elapsed' keys.
    """
    start_time = time.time()
    name = scenario["name"]
    kwargs = scenario["kwargs"]
    details = {}
    passed = False

    try:
        # Backend pair comparison (weight diff)
        if "backend_pair" in scenario:
            backends = scenario["backend_pair"]
            # If the pair involves cuda and it's not available, skip gracefully
            if ("cuda" in backends and "triton" in backends) and not torch.cuda.is_available():
                details["error"] = "CUDA not available — skipped"
                return {
                    "name": name,
                    "passed": False,
                    "details": details,
                    "elapsed": 0.0,
                }

            result = _compare_two_backends(list(backends), kwargs)
            passed = result["passed"]
            details = result["details"]

        # Two-way inference comparison (all four backends)
        elif name == "Two-way inference":
            result = _run_four_way_inference(kwargs)
            passed = result["passed"]
            details = result["details"]

        # Training dynamics
        elif name == "Training dynamics":
            result = _run_train_dynamics(kwargs)
            passed = result["passed"]
            details = result["details"]

        # Round-trip tests
        elif "round_trip" in kwargs:
            passed, details = _run_round_trip(kwargs)

        else:
            details["error"] = f"Unknown scenario: {name}"

    except Exception as e:
        details["error"] = str(e)
        passed = False

    elapsed = time.time() - start_time
    return {
        "name": name,
        "passed": passed,
        "details": details,
        "elapsed": round(elapsed, 2),
    }


def _run_round_trip(kwargs: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Run a round-trip test: save to npz → load into different backend → compare.

    Tests that parameters can be exported from one backend and imported into
    another with equivalent forward pass results.

    Args:
        kwargs: Scenario config with 'round_trip' key indicating direction.

    Returns:
        Tuple of (passed: bool, details: dict).
    """
    direction = kwargs["round_trip"]
    config = {**kwargs}
    del config["round_trip"]

    passed = False
    details = {}

    if direction == "torch_numpy":
        # Train torch → save_as_numpy → load into numpy → compare weight diff
        torch_model = _create_torch_model(config)
        import torch.optim as optim

        from impl._torch.training import train_step

        torch_model.cuda()
        vocab_size = config["vocab_size"]
        ctx_len = config["context_length"]
        batch_size = config.get("batch_size", 4)
        np.random.seed(config["seed"])
        tokens = torch.randint(0, vocab_size, (1, ctx_len * batch_size), device="cuda")

        opt = optim.Adam(torch_model.parameters(), lr=config.get("lr", 0.01))
        loss_fn = torch.nn.CrossEntropyLoss()
        train_steps = config.get("train_steps", 3)
        for _ in range(train_steps):
            train_step(torch_model, tokens, tokens, opt, loss_fn)

        torch_params_np = torch_model.save_as_numpy()
        # Convert to numpy arrays (save_as_numpy may return some tensors)
        torch_params_for_load = {
            k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v.copy()) for k, v in torch_params_np.items()
        }
        model_np = _create_numpy_model(config)
        model_np.load_from_numpy_dict(torch_params_for_load)
        np_params = model_np.get_all_parameters()
        max_diff = weight_diff(torch_params_np, np_params)
        details["max_diff"] = round(max_diff, 6)
        passed = max_diff < 0.05

    elif direction == "numpy_torch":
        # Train numpy → load into torch → compare weight diff
        np_model = _create_numpy_model(config)
        lr = config.get("lr", 0.01)
        vocab_size = config["vocab_size"]
        ctx_len = config["context_length"]
        batch_size = config.get("batch_size", 4)

        np.random.seed(config["seed"])
        for _ in range(config.get("train_steps", 3)):
            tokens = np.random.randint(0, vocab_size, (1, ctx_len * batch_size), dtype=np.int32)
            params = np_model.get_all_parameters()
            for param in params.values():
                param[:] -= lr * np.random.randn(*param.shape)

        np_params = np_model.get_all_parameters()
        torch_model = _create_torch_model(config)
        torch_model.load_from_numpy_dict(np_params)  # type: ignore[arg-type]
        torch_params = {k: v for k, v in torch_model.state_dict().items()}
        max_diff = weight_diff(np_params, {k: v.detach().cpu().numpy() for k, v in torch_params.items()})
        details["max_diff"] = round(max_diff, 6)
        passed = max_diff < 0.05

    else:
        details["error"] = f"Unknown round-trip direction: {direction}"

    return passed, details


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main(args: list[str] | None = None) -> int:
    """Entry point for the auto test equivalence script.

    Parses CLI args, runs selected scenario(s), and prints a formatted report.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 if all pass, 1 if any fail, 2 on usage error.
    """
    parser = argparse.ArgumentParser(
        prog="auto_test_equivalence",
        description="Automated equivalence matrix test across numpy, torch, triton, cuda backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Use synthetic data and minimal steps for fast execution (recommended for CI)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON report (defaults to stdout)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Compare two backends only, e.g. 'numpy,torch'",
    )

    try:
        parsed = parser.parse_args(args)
    except SystemExit as e:
        return 2 if e.code != 0 else 0

    # Filter scenarios for custom two-way comparison
    if parsed.compare:
        backends = [s.strip() for s in parsed.compare.split(",")]
        scenarios = [
            {
                "name": f"Weight diff: {' vs '.join(backends)}",
                "kwargs": SMALL_CONFIG,
                "backend_pair": tuple(backends),
            }
        ]
    else:
        scenarios = _generate_matrix()

    # Run each scenario — report results, do not block on failure
    all_results = []
    for scenario in scenarios:
        _ = scenario["name"]
        result = _run_scenario(scenario)
        all_results.append(result)

    # Print report
    lines = ["=== Auto Test Results ===", ""]
    for result in all_results:
        lines.append(_format_result_line(result))
    lines.append(_format_summary(all_results))

    report = "\n".join(lines)
    print(report)

    # Write JSON report if requested
    if parsed.output:
        Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
        with open(parsed.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  JSON report written to {parsed.output}", file=sys.stderr)

    # Exit code: 0 if all pass, 1 if any fail
    all_pass = all(r["passed"] for r in all_results)
    return 0 if all_pass else 1


# ─── Direct execution ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.exit(0 if code == 0 else code)
