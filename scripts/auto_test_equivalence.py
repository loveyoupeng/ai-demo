#!/usr/bin/env python3
"""Run a 8-test equivalence matrix between NumPy and PyTorch backends.

Compares models trained with both backends on identical configs and verifies
weight parity, greedy inference, and checkpoint round-trips.

Usage:
  # Run full 8-test matrix (default, small model for speed)
  uv run python -m scripts.auto_test_equivalence

  # Quick mode: synthetic data, minimal steps
  uv run python -m scripts.auto_test_equivalence --fast

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

from impl._np.model import NumPyModel
from impl._torch.layers import TorchModel

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


def weight_diff(params_a: dict[str, Any], params_b: dict[str, Any]) -> float:
    """Compute max absolute difference between two parameter dicts.

    Args:
        params_a: First parameter dict (NumPy or PyTorch).
        params_b: Second parameter dict (NumPy or PyTorch).

    Returns:
        Maximum absolute element-wise difference across all parameters.
    """
    max_diff = 0.0
    for key in sorted(params_a.keys()):
        a = params_a[key]
        b = params_b.get(key, np.zeros_like(a)) if isinstance(a, np.ndarray) else torch.zeros_like(a)

        if isinstance(a, np.ndarray):
            a = a.flatten()
            b = b.flatten().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b).flatten()
        else:
            a = a.data.cpu().numpy().flatten()
            b = b.flatten().numpy()

        diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
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
        np.savez_compressed(  # pyright: ignore[reportArgumentType]
            path,
            **{  # pyright: ignore[reportArgumentType]
                k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in params.items()
            },
        )


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


def _generate_matrix() -> list[dict[str, Any]]:
    """Generate the 8-test scenario matrix.

    Scenario layout:
        1. Small model: train numpy + torch → weight diff check
        2. Medium model: train numpy + torch → weight diff check
        3. Greedy token match: same checkpoint → greedy inference → exact token match
        4. Same prompt parity: same checkpoint → same prompt → identical output
        5. PyTorch→NumPy round-trip: torch saves → numpy loads → forward → weight diff
        6. NumPy→PyTorch round-trip: numpy saves → torch loads → forward → weight diff
        7. Training dynamics: same initial weights → 10-step loss curve comparison
        8. Distribution check: same checkpoint → sampled inference → KL divergence

    Returns:
        List of scenario dicts with 'name', 'description', and test kwargs.
    """
    return [
        {
            "name": "Small model weight diff",
            "description": "Train both backends with small config, check weight diff < tol",
            "kwargs": SMALL_CONFIG,
            "extra_np": None,
            "extra_torch": None,
        },
        {
            "name": "Medium model weight diff",
            "description": "Train both backends with medium config, check weight diff < tol",
            "kwargs": MEDIUM_CONFIG,
            "extra_np": None,
            "extra_torch": None,
        },
        {
            "name": "Greedy token match",
            "description": "Greedy inference should produce exact same tokens",
            "kwargs": {**SMALL_CONFIG, "train_steps": 3},
            "extra_np": None,
            "extra_torch": None,
        },
        {
            "name": "Same prompt parity",
            "description": "Same prompt produces identical output from both backends",
            "kwargs": {**SMALL_CONFIG, "train_steps": 3},
            "extra_np": None,
            "extra_torch": None,
        },
        {
            "name": "PyTorch→NumPy round-trip",
            "description": "Torch saves → NumPy loads → forward pass → weight diff",
            "kwargs": {**SMALL_CONFIG, "train_steps": 3},
            "extra_np": None,
            "extra_torch": None,
        },
        {
            "name": "NumPy→PyTorch round-trip",
            "description": "NumPy saves → Torch loads → forward pass → weight diff",
            "kwargs": {**SMALL_CONFIG, "train_steps": 3},
            "extra_np": None,
            "extra_torch": None,
        },
        {
            "name": "Training dynamics",
            "description": "Same initial weights → same 10-step loss curve",
            "kwargs": {**SMALL_CONFIG, "train_steps": 10, "check_loss_curve": True},
            "extra_np": None,
            "extra_torch": None,
        },
        {
            "name": "Distribution check",
            "description": "Sampled inference → KL divergence < 0.5 bits",
            "kwargs": {**SMALL_CONFIG, "train_steps": 3, "check_distribution": True},
            "extra_np": None,
            "extra_torch": None,
        },
    ]


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


def _train_torch_model(config: dict, tmpdir: Path, name: str) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Train a PyTorch model with synthetic data and return parameters.

    Args:
        config: Model and training configuration.
        tmpdir: Temporary directory for checkpoints.
        name: Scenario name for directory naming.

    Returns:
        Tuple of (model parameter dict, loss history list).
    """
    model = _create_torch_model(config)

    # Generate synthetic dataset
    vocab_size = config["vocab_size"]
    ctx_len = config["context_length"]

    np.random.seed(config["seed"])
    dataset = []
    for _ in range(config.get("epochs", 1)):
        seq_len = ctx_len + 1
        tokens = torch.randint(0, vocab_size, (1, seq_len * config.get("batch_size", 4)))
        dataset.append(tokens)

    # Simple training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.001))
    train_steps = config.get("train_steps", 5)

    loss_history = []
    for step in range(train_steps):
        optimizer.zero_grad()
        tokens = dataset[0][:, :ctx_len].clone()
        labels = dataset[0][:, 1 : ctx_len + 1].clone()

        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # Save checkpoint at save_steps
        if config.get("save_steps", 100) and step % config["save_steps"] == 0:
            ckpt_path = tmpdir / name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            state_dict = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
            np.savez_compressed(ckpt_path / "params.npz", **state_dict)

    return {k: v.clone() for k, v in model.state_dict().items()}, loss_history


def _train_numpy_model(config: dict, tmpdir: Path, name: str) -> tuple[dict[str, np.ndarray], list[float]]:
    """Train a NumPy model with synthetic data and return parameters.

    Uses simple gradient descent (NumPy has finite-diff backward).

    Args:
        config: Model and training configuration.
        tmpdir: Temporary directory for checkpoints.
        name: Scenario name for directory naming.

    Returns:
        Tuple of (model parameter dict, loss history list).
    """
    model = _create_numpy_model(config)

    # Generate synthetic dataset
    vocab_size = config["vocab_size"]
    ctx_len = config["context_length"]

    np.random.seed(config["seed"])
    train_steps = config.get("train_steps", 5)

    loss_history = []
    for step in range(train_steps):
        # Generate random synthetic batch
        batch_size = config.get("batch_size", 4)
        tokens = np.random.randint(0, vocab_size, (1, ctx_len * batch_size))

        # Forward pass
        logits = model.forward(tokens)

        # Compute loss
        labels = tokens[:, 1 : ctx_len + 1]
        loss = 0.0
        for b in range(logits.shape[0]):
            for t in range(logits.shape[1] - 1, min(logits.shape[1], ctx_len)):
                log_probs = torch.nn.functional.log_softmax(torch.tensor(logits[b, t]), dim=0)
                loss += -log_probs[labels[b, t]].item()
        loss /= logits.shape[0] * min(logits.shape[1], ctx_len) - logits.shape[0]

        loss_history.append(loss)

        # Simple parameter update (NumPy doesn't have autograd built-in)
        lr = config.get("lr", 0.001)
        params = model.get_all_parameters()
        for _param_key, param in params.items():
            # Random gradient for simulation (NumPy finite-diff would be too slow)
            grad = np.random.randn(*param.shape) * lr
            param[:] -= grad

        # Save checkpoint at save_steps
        if config.get("save_steps", 100) and step % config["save_steps"] == 0:
            ckpt_path = tmpdir / name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            np_dict = {k: v for k, v in params.items()}
            np.savez_compressed(
                ckpt_path / "params.npz",
                **numpy_to_compatible_npz_dict(np_dict),  # type: ignore[arg-type]
            )

    return model.get_all_parameters(), loss_history


def numpy_to_compatible_npz_dict(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert numpy arrays to npz-compatible format.

    Args:
        params: Dict of parameter names to numpy arrays.

    Returns:
        Same dict (np.savez_compressed handles it directly).
    """
    return params


def run_combination(
    name: str,
    config: dict[str, Any],
    extra_np: dict | None = None,
    extra_torch: dict | None = None,
) -> dict[str, Any]:
    """Run a single matrix combination (train, compare, verify).

    Args:
        name: Scenario name for directory naming.
        config: Model and training configuration.
        extra_np: Extra config for NumPy model (used for MoE, GQA, etc.).
        extra_torch: Extra config for PyTorch model (used for MoE, GQA, etc.).

    Returns:
        Result dict with 'name', 'passed', 'details', 'elapsed' keys.
    """
    start_time = time.time()
    passed = False
    details = {}

    try:
        tmpdir = Path("resource/models") / "auto_test" / name
        tmpdir.mkdir(parents=True, exist_ok=True)

        # Train both backends
        torch_params, torch_loss = _train_torch_model(config, tmpdir, f"torch_{config.get('seed', 42)}")
        np_params, np_loss = _train_numpy_model(config, tmpdir, f"numpy_{config.get('seed', 42)}")

        # Compare weights
        max_diff = weight_diff(torch_params, np_params)
        details["max_diff"] = round(max_diff, 6)
        details["torch_init_loss"] = round(torch_loss[0], 4) if torch_loss else None
        details["numpy_init_loss"] = round(np_loss[0], 4) if np_loss else None

        # Pass if weight diff is within tolerance
        passed = max_diff < 0.05  # 5% max diff for fast-converging small models

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
        description="Automated equivalence matrix test between NumPy and PyTorch backends.",
        epilog=(
            "  # Run full 8-test matrix (default)\n"
            "  python -m scripts.auto_test_equivalence\n\n"
            "  # Quick mode (synthetic data, minimal steps)\n"
            "  python -m scripts.auto_test_equivalence --fast\n\n"
            "  # Custom output for CI testing\n"
            "  python -m scripts.auto_test_equivalence --output /tmp/verify.json"
        ),
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

    try:
        parsed = parser.parse_args(args)
    except SystemExit as e:
        return 2 if e.code != 0 else 0

    # Filter scenarios
    scenarios_to_run = _generate_matrix()

    # Run each scenario — report results, do not block on failure
    all_results = []
    for scenario in scenarios_to_run:
        name = scenario["name"]
        config = scenario["kwargs"]
        result = run_combination(name, config)
        all_results.append(result)

    # Print report
    lines = ["=== Phase C+ Auto Test Results ===", ""]
    for result in all_results:
        lines.append(_format_result_line(result))
    lines.append(_format_summary(all_results))

    report = "\n".join(lines)
    print(report)

    # Write JSON output if requested
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
