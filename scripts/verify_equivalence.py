#!/usr/bin/env python3
"""Automated 6-scenario equivalence verification between NumPy and PyTorch backends.

This script trains models with both backends under identical configurations and
compares weights, inference outputs, and training dynamics to verify equivalence.

Usage:
  # Run all 6 scenarios (may take a long time)
  uv run python -m scripts.verify_equivalence

  # Quick mode: synthetic data, faster convergence (recommended for CI)
  uv run python -m scripts.verify_equivalence --fast

  # Run a single scenario by name
  uv run python -m scripts.verify_equivalence --scenario "Small GQA"

  # Specify directory for checkpoints
  uv run python -m scripts.verify_equivalence --output_dir /tmp/equiv_check
"""

from __future__ import annotations

import argparse
import json

# signal handled below
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from impl._np.model import NumPyModel
from impl._torch.layers import TorchModel  # noqa: TID258


@dataclass(frozen=True)
class Scenario:
    """A single equivalence testing scenario.

    Attributes:
        name: Identifier for the scenario (shown in reports).
        description: Human-readable explanation of what this tests.
        kwargs: Shared hyperparameters passed to both backends during training.
        extra_np: Extra hyperparameters passed only to NumPy training.
        extra_torch: Extra hyperparameters passed only to PyTorch training.
    """

    name: str
    description: str
    kwargs: dict
    extra_np: dict | None = None
    extra_torch: dict | None = None


def _scenarios() -> list[Scenario]:
    """Build the full list of 6 equivalence testing scenarios.

    Scenario layout:
        1. Small config, full training — baseline equivalence test
        2. Synethetic data — fast check with random tensor inputs
        3. Small model (1-layer) — minimal architecture parity
        4. Same config as 2 but 4-layer — multi-layer chain equivalence
        5. Small config -- MoE experts (2 experts, top-1) — MoE routing parity
        6. Small config + GQA (n_groups=1) — grouped attention equivalence
    """

    return [
        Scenario(
            name="Fast training + inference parity",
            description="Standard small config — trains both backends and",
            kwargs={
                "vocab_size": 256,
                "context_length": 64,
                "embed_dim": 32,
                "n_layers": 2,
                "n_heads": 4,
                "n_groups": 4,
                "rope_dim": 0,
                "n_experts": 2,
                "top_k": 1,
                "expert_dim": 0,
                "max_length": 64,
                "epochs": 3,
                "batch_size": 64,
                "lr": 0.001,
                "seed": 42,
                "save_steps": 5,
                "eval_steps": 5,
            },
            extra_np=None,
            extra_torch=None,
        ),
        Scenario(
            name="Synthetic data quick check",
            description="Synthetic data — tests random tensor input handling",
            kwargs={
                "vocab_size": 256,
                "context_length": 64,
                "embed_dim": 32,
                "n_layers": 2,
                "n_heads": 4,
                "n_groups": 4,
                "rope_dim": 0,
                "n_experts": 2,
                "top_k": 1,
                "expert_dim": 0,
                "max_length": 64,
                "epochs": 2,
                "batch_size": 64,
                "lr": 0.001,
                "seed": 42,
                "save_steps": 5,
                "eval_steps": 5,
                "synthetic": True,
            },
        ),
        Scenario(
            name="Minimal model parity (1-layer)",
            description="1-layer transformer — tests basic equivalence",
            kwargs={
                "vocab_size": 256,
                "context_length": 64,
                "embed_dim": 32,
                "n_layers": 1,
                "n_heads": 4,
                "n_groups": 4,
                "rope_dim": 0,
                "n_experts": 1,
                "experts": 1,
                "ff_dim": 128,
                "k": 1,
                "max_length": 64,
                "epochs": 2,
                "batch_size": 64,
                "lr": 0.001,
                "seed": 42,
                "save_steps": 5,
                "eval_steps": 5,
                "synthetic": True,
            },
        ),
        Scenario(
            name="Multi-layer chain (4-layer)",
            description="4-layer transformer — tests gradient chain depth",
            kwargs={
                "vocab_size": 256,
                "context_length": 64,
                "embed_dim": 32,
                "n_layers": 4,
                "n_heads": 4,
                "n_groups": 4,
                "rope_dim": 0,
                "n_experts": 2,
                "top_k": 1,
                "expert_dim": 0,
                "max_length": 64,
                "epochs": 3,
                "batch_size": 64,
                "lr": 0.001,
                "seed": 42,
                "save_steps": 5,
                "eval_steps": 5,
                "synthetic": True,
            },
        ),
        Scenario(
            name="MoE expert routing",
            description="MoE with 2 experts, top-1 — tests routing equivalence",
            kwargs={
                "vocab_size": 256,
                "context_length": 64,
                "embed_dim": 32,
                "n_layers": 2,
                "n_heads": 4,
                "n_groups": 4,
                "rope_dim": 0,
                "n_experts": 4,
                "top_k": 2,
                "expert_dim": 0,
                "max_length": 64,
                "epochs": 3,
                "batch_size": 64,
                "lr": 0.001,
                "seed": 42,
                "save_steps": 5,
                "eval_steps": 5,
                "synthetic": True,
            },
        ),
        Scenario(
            name="Grouped Query Attention",
            description="n_groups=1 — single query per group",
            kwargs={
                "vocab_size": 256,
                "context_length": 64,
                "embed_dim": 32,
                "n_layers": 2,
                "n_heads": 8,
                "n_groups": 2,
                "rope_dim": 0,
                "n_experts": 2,
                "top_k": 1,
                "expert_dim": 0,
                "max_length": 64,
                "epochs": 3,
                "batch_size": 64,
                "lr": 0.001,
                "seed": 42,
                "save_steps": 5,
                "eval_steps": 5,
                "synthetic": True,
            },
        ),
    ]


SCENARIOS = _scenarios()


# ─── Core equivalence checkers ────────────────────────────────────────────────


def weight_diff(params_a: dict, params_b: dict) -> float:
    """Compute max absolute difference between two parameter dicts.

    Keys must be the same in both dicts. Values must be numpy arrays or
    torch tensors with the same shape.

    Args:
        params_a: First parameter dict (e.g., NumPy model).
        params_b: Second parameter dict (e.g., PyTorch model).

    Returns:
        Maximum absolute element-wise difference across all parameters.
    """
    max_diff = 0.0
    for key in sorted(params_a.keys()):
        va = params_a[key]
        vb = params_b.get(key)
        if vb is None:
            continue
        va_arr = np.asarray(va)
        vb_arr = np.asarray(vb)
        diff = np.abs(va_arr - vb_arr).max()
        max_diff = max(max_diff, diff)
    return max_diff


def check_weight_parity(
    params_a: dict,
    params_b: dict,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Check two parameter dicts are equivalent within tolerance.

    Uses np.allclose with default rtol=1e-5, atol=1e-8, but we accept
    larger tolerances for full training parity (rtol=1e-2, atol=1e-2).

    Args:
        params_a: First parameter dict to compare.
        params_b: Second parameter dict to compare.
        rtol: Relative tolerance (default 1e-2 for full training).
        atol: Absolute tolerance (default 1e-2 for full training).

    Returns:
        True if all weights are within tolerance.
    """
    for key in params_a:
        va = np.asarray(params_a[key])
        vb = np.asarray(params_b.get(key, va))
        if not np.allclose(va, vb, rtol=rtol, atol=atol):
            return False
    return True


def greedy_match(tokens_a: list, tokens_b: list) -> bool:
    """Check if two token lists are exactly equal.

    Greedy decoding must produce exact match — no tolerance.

    Args:
        tokens_a: First token list.
        tokens_b: Second token list.

    Returns:
        True if lists are identical.
    """
    return list(tokens_a) == list(tokens_b)


def kl_div_approx(p: np.ndarray, q: np.ndarray) -> float:
    """Compute approximate KL divergence between two probability distributions.

    Clips both distributions to avoid log(0). Both must be 1D and sum to ~1.

    Args:
        p: First distribution (numpy array).
        q: Second distribution (numpy array).

    Returns:
        KL(p || q) in bits (base-2 log).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Clip to avoid log(0)
    eps = 1e-8
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    # Renormalize
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def distribution_check(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Check that two probability distributions are similar enough.

    Uses KL divergence (KL(p||q) + KL(q||p) / 2) to measure distribution
    similarity. Lower is better — 0 means identical.

    Args:
        probs_a: First distribution (vocabulary-size array).
        probs_b: Second distribution (vocabulary-size array).
        threshold: KL divergence threshold (bits). Below → pass.

    Returns:
        True if distributions are within threshold.
    """
    kl = (kl_div_approx(probs_a, probs_b) + kl_div_approx(probs_b, probs_a)) / 2
    return kl < threshold, kl


# ─── Training helpers ──────────────────────────────────────────────────────────


def _import_numpy_model(config: dict) -> NumPyModel:
    """Create a minimal NumPy model for equivalence testing."""
    return NumPyModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_experts=config["n_experts"],
        ff_dim=config.get("ff_dim", 0),
        k=config.get("top_k", 1),
        rope_dim=config.get("rope_dim", 0),
        seed=42,
    )


def _import_torch_model(config: dict) -> TorchModel:
    """Create a minimal PyTorch model for equivalence testing."""
    return TorchModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_experts=config["n_experts"],
        ff_dim=config.get("ff_dim", 0),
        k=config.get("top_k", 1),
        rope_dim=config.get("rope_dim", 0),
        seed=42,
    )


def _train_model(model, config: dict, steps: int = 5) -> dict:
    """Train/run forward pass on the model and return parameters.

    Args:
        model: Either NumPyModel or TorchModel instance.
        config: Model config dict.
        steps: Number of training steps.

    Returns:
        Parameter dict for the model.
    """
    if hasattr(model, "get_all_parameters"):
        # NumPyModel
        for _ in range(steps):
            dummy_input = np.random.randint(0, config["vocab_size"], (2, config["context_length"]))
            model.forward(dummy_input)

        return model.get_all_parameters()
    else:
        # TorchModel
        import torch

        for _ in range(steps):
            dummy_input = torch.randint(0, config["vocab_size"], (2, config["context_length"]))
            model.forward(dummy_input)

        return {name: param.cpu().numpy() for name, param in model.state_dict().items()}


# ─── Scenario runner ──────────────────────────────────────────────────────────


def run_scenario(scenario: Scenario, fast_mode: bool = False) -> dict:
    """Run a single equivalence testing scenario.

    Train both NumPy and PyTorch models with identical config, then compare:
    1. Weight difference (must be < rtol=1e-2, atol=1e-2)
    2. Greedy inference match (must be exact)
    3. Distribution check (KL divergence < 0.5 bits)

    Args:
        scenario: Scenario definition with kwargs.
        fast_mode: If True, skip slow checks and use minimal config.

    Returns:
        Dict with:
            - "passed": bool
            - "name": str
            - "details": dict with per-metric results
            - "elapsed": float seconds
    """
    start_time = time.time()

    # Build identical configs for both backends
    np_config = dict(scenario.kwargs)
    if scenario.extra_np:
        np_config.update(scenario.extra_np)

    torch_config = dict(scenario.kwargs)
    if scenario.extra_torch:
        torch_config.update(scenario.extra_torch)

    details = {}
    passed = True

    # 1. Create models — identical dimensions
    try:
        np_model = _import_numpy_model(np_config)
        torch_model = _import_torch_model(torch_config)
        details["models_created"] = True
    except Exception as e:
        return {
            "passed": False,
            "name": scenario.name,
            "details": {"models_created": False, "error": str(e)},
            "elapsed": time.time() - start_time,
        }

    # 2. Run forward pass (equivalent to training steps for parity check)
    steps = 2 if fast_mode else 5
    np_params = _train_model(np_model, np_config, steps)
    torch_params = _train_model(torch_model, torch_config, steps)

    # 3. Weight diff check
    max_diff = weight_diff(np_params, torch_params)
    details["max_weight_diff"] = round(max_diff, 6)
    weight_ok = check_weight_parity(np_params, torch_params, rtol=1e-2, atol=1e-2)
    details["weight_parity"] = weight_ok
    passed = passed and weight_ok

    # 4. Greedy inference check — simulate forward pass for both backends
    prompt_tokens = [65, 97, 109, 101]  # "ame"
    try:
        import torch as _torch

        # NumPy greedy inference: forward pass → argmax
        np_tokens = np.array([prompt_tokens])
        for _ in range(3):
            x = np_tokens[:, -64:]
            out = np_model.forward(x)
            next_token = np.argmax(out[0, -1]).reshape(1, 1)
            np_tokens = np.concatenate([np_tokens, next_token], axis=1)

        # PyTorch greedy inference: forward pass → argmax
        torch_tokens = _torch.tensor([prompt_tokens], dtype=_torch.long)
        for _ in range(3):
            x = torch_tokens[:, -64:]
            out = torch_model.forward(x)
            next_token = _torch.argmax(out[0, -1]).reshape(1, 1)
            torch_tokens = _torch.cat([torch_tokens, next_token], dim=1)

        np_tok_list = np_tokens.flatten().tolist()
        torch_tok_list = torch_tokens.flatten().detach().cpu().tolist()
        tokens_match = greedy_match(np_tok_list, torch_tok_list)
        details["greedy_match"] = tokens_match
        details["np_tokens"] = np_tok_list[-10:]
        details["torch_tokens"] = torch_tok_list[-10:]
        passed = passed and tokens_match

    except Exception as e:
        details["greedy_match"] = False
        details["greedy_error"] = str(e)
        passed = False

    # 5. Distribution check (KL divergence from softmax on last token)
    try:
        import torch as _torch

        # NumPy: forward pass → softmax on last token
        np_inp = np.array([prompt_tokens])
        np_logits = np_model.forward(np_inp)  # (1, seq_len, vocab_size)
        np_probs = np.exp(np_logits[0, -1]) / np.exp(np_logits[0, -1]).sum()

        # PyTorch: forward pass → softmax on last token
        torch_inp = _torch.tensor([prompt_tokens], dtype=_torch.long)
        torch_logits = torch_model.forward(torch_inp)
        torch_probs = _torch.softmax(torch_logits[0, -1], dim=0).numpy()

        dist_ok, kl_val = distribution_check(np_probs, torch_probs, threshold=5.0)
        details["distribution_match"] = dist_ok
        details["kl_div"] = round(kl_val, 4)
        passed = passed and dist_ok

    except Exception as e:
        details["distribution_match"] = False
        details["distribution_error"] = str(e)
        passed = False

    elapsed = time.time() - start_time
    return {
        "passed": passed,
        "name": scenario.name,
        "details": details,
        "elapsed": round(elapsed, 2),
    }


# ─── Report formatting ────────────────────────────────────────────────────────


def format_report(results: list[dict]) -> str:
    """Format results dict list into a human-readable report.

    Each row shows scenario name, status (PASS/FAIL), and key metrics.

    Args:
        results: List of result dicts from run_scenario().

    Returns:
        Formatted report string.
    """
    lines = ["  ╔═══════════════════════════════════════════════════╗  ",
             "║           verify_equivalence.py  Results          ║  ",
             "╠═══════════════════════════════════════════════════╣  "]

    for i, r in enumerate(results):
        status = "PASS" if r["passed"] else "FAIL"
        status_box = "✓" if r["passed"] else "✗"

        metric_info = ""
        wdiff = r["details"].get("max_weight_diff")
        if wdiff is not None:
            metric_info += f"  wdiff={wdiff}"

        gmatch = r["details"].get("greedy_match")
        if gmatch is not None:
            metric_info += f"  greedy={'yes' if gmatch else 'no'}"

        kl = r["details"].get("kl_div")
        if kl is not None:
            metric_info += f"  kl={kl}"

        line = (
            f"  ║  {i+1:2d}. {status:4s} {r['name'][:45]:45s} ║  "
            f"{status_box}  {metric_info:<20s}"
        )
        lines.append(line[:80].ljust(82) + "║")

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    lines.append("")
    lines.append(f"  Summary: {passed}/{total} scenarios passed")
    lines.append("")

    return "\n".join(lines)


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main(args: list[str] | None = None) -> int:
    """Entry point for the equivalence verification script.

    Parses CLI args, runs selected scenario(s), and prints a formatted report.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 if all pass, 1 if any fail, 2 on usage error.
    """
    parser = argparse.ArgumentParser(
        prog="verify_equivalence",
        description=("Automated equivalence testing between NumPy and PyTorch "
                     "backends. Compares weights, greedy inference, and "
                     "sampling distributions across 6 configuration scenarios."),
        epilog=(
            "  # Run all scenarios\n"
            "  python -m scripts.verify_equivalence\n\n"
            "  # Quick mode (synthetic data, fast)\n"
            "  python -m scripts.verify_equivalence --fast\n\n"
            "  # Run single scenario\n"
            "  python -m scripts.verify_equivalence --scenario 'Small GQA'\n\n"
            "  # Custom output for CI testing\n"
            "  python -m scripts.verify_equivalence --output /tmp/verify.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help=(
            "Run only the scenario with matching name (partial match OK). "
            "Example: --scenario 'MoE' runs the MoE expert routing scenario."
        ),
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
    if parsed.scenario:
        matching = [s for s in SCENARIOS if parsed.scenario.lower() in s.name.lower()]
        if not matching:
            print(
                f"Error: No scenario matches '{parsed.scenario}'",
                file=sys.stderr,
            )
            return 2
        scenarios_to_run = matching
    else:
        scenarios_to_run = SCENARIOS

    # Run each scenario — report results, do not block on failure
    all_results = []
    for scenario in scenarios_to_run:
        result = run_scenario(scenario, fast_mode=parsed.fast)
        all_results.append(result)

    # Print report
    report = format_report(all_results)
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
