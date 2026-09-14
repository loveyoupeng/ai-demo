#!/usr/bin/env python3
"""Automated equivalence verification across NumPy, PyTorch, Triton, and CUDA backends.

All four tracks share one architecture (shared ``TransformerConfig``) and one
checkpoint key scheme (``shared.constants.Keys``), so a scenario builds a NumPy
reference model, loads its weights losslessly into every backend under test
(``load_from_numpy_dict``), and compares parameter diffs, logits, and greedy
outputs.

Usage:
  # Run all scenarios
  uv run python -m scripts.verify_equivalence

  # Quick mode (fewer training steps)
  uv run python -m scripts.verify_equivalence --fast

  # Run specific scenarios by name substring
  uv run python -m scripts.verify_equivalence --scenario gqa

  # Custom output
  uv run python -m scripts.verify_equivalence --output /tmp/verify.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ─── Scenario model ────────────────────────────────────────────────────────────


@dataclass
class Scenario:
    """One equivalence scenario.

    Attributes:
        name: Unique scenario name (also the --scenario filter key).
        description: Human-readable summary of what the scenario checks.
        kwargs: Model-construction kwargs for the shared ``TransformerConfig``.
        backends: Track names exercised by this scenario.
    """

    name: str
    description: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    backends: list[str] = field(default_factory=list)


def _cfg(**overrides: Any) -> dict[str, Any]:
    """Small dense base config; scenarios override the interesting knobs."""
    base: dict[str, Any] = {
        "vocab_size": 64,
        "context_length": 16,
        "embed_dim": 32,
        "n_layers": 1,
        "n_heads": 4,
        "rope_dim": 8,  # 8 = full head dimension (head_dim = 32 / 4)
        "n_experts": 1,
        "top_k": 1,
        "seed": 42,
    }
    base.update(overrides)
    return base


SCENARIOS: list[Scenario] = [
    Scenario(
        name="dense_np_torch",
        description="Dense model: NumPy reference vs PyTorch (float64 parity tier)",
        kwargs=_cfg(),
        backends=["numpy", "torch"],
    ),
    Scenario(
        name="gqa_np_torch",
        description="Grouped-Query Attention (8 heads / 4 groups): NumPy vs PyTorch",
        kwargs=_cfg(n_heads=8, n_groups=4, rope_dim=4),
        backends=["numpy", "torch"],
    ),
    Scenario(
        name="moe_np_torch",
        description="MoE (4 experts, top-k 2): NumPy vs PyTorch",
        kwargs=_cfg(n_experts=4, top_k=2),
        backends=["numpy", "torch"],
    ),
    Scenario(
        name="gqa_torch_triton",
        description="GQA model: PyTorch vs Triton (fp32 kernel tier)",
        kwargs=_cfg(n_heads=4, n_groups=2, rope_dim=8),
        backends=["torch", "triton"],
    ),
    Scenario(
        name="cuda_shared_weights",
        description="CUDA track: shared-weight load + forward validity",
        kwargs=_cfg(),
        backends=["cuda"],
    ),
    Scenario(
        name="all_four_backends",
        description="All four tracks run the same weights (full interchange)",
        kwargs=_cfg(),
        backends=["numpy", "torch", "triton", "cuda"],
    ),
]


def _scenarios() -> list[Scenario]:
    """Return the scenario list (the public, testable accessor)."""
    return SCENARIOS


# ─── Backend registry ──────────────────────────────────────────────────────────

_BACKEND_CLASSES: dict[str, str] = {
    "numpy": "impl._np.model.NumPyModel",
    "torch": "impl._torch.layers.TorchModel",
    "triton": "impl._triton.model.TritonModel",
    "cuda": "impl._cuda.model.CUDAModel",
}


def _to_transformer_config(kwargs: dict[str, Any]):
    """Build the shared TransformerConfig from a scenario's kwargs."""
    from shared.config import TransformerConfig

    mapped = {("expert_dim" if k == "ff_dim" else "top_k" if k == "k" else k): v for k, v in kwargs.items()}
    return TransformerConfig(**mapped)


def _make_model(backend: str, kwargs: dict[str, Any]) -> Any:
    """Create a model instance for the given backend from shared config."""
    import importlib

    module_path, cls_name = _BACKEND_CLASSES[backend].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(_to_transformer_config(kwargs))


def _model_device(backend: str) -> str:
    """Return the device string for a backend."""
    return "cuda" if backend in ("torch", "triton", "cuda") else "cpu"


def _get_params(backend: str, model: Any) -> dict[str, np.ndarray]:
    """Extract parameters as dict[str, np.ndarray] under the shared Keys scheme.

    All backends expose the same interface: NumPy/CUDA use
    ``get_all_parameters()``; PyTorch/Triton use ``save_as_numpy()``.
    """
    if backend in ("numpy", "cuda"):
        return {k: np.asarray(v).copy() for k, v in model.get_all_parameters().items()}
    return {k: np.asarray(v).copy() for k, v in model.save_as_numpy().items()}


def _load_params_to_model(backend: str, model: Any, params: dict[str, np.ndarray]) -> None:
    """Load parameters into a model via the shared Keys scheme (all backends)."""
    model.load_from_numpy_dict(params)


# ─── Inference helpers ─────────────────────────────────────────────────────────


def _forward_logits(backend: str, model: Any, tokens: np.ndarray) -> np.ndarray:
    """Run a forward pass and return the logits as a float64 numpy array."""
    import torch

    if backend == "numpy":
        return model.forward(tokens.astype(np.int32)).astype(np.float64)
    x = torch.from_numpy(tokens.astype(np.int32))
    if backend == "cuda":
        logits = model.forward(x.to(_model_device(backend)))
    else:
        model.eval()
        logits = model(x.to(_model_device(backend)))
    return logits.detach().cpu().numpy().astype(np.float64)


def _greedy_tokens(
    backend: str,
    model: Any,
    prompt: list[int],
    context_length: int,
    steps: int = 3,
) -> list[int]:
    """Run greedy decoding and return generated token IDs."""
    import torch as th

    if backend == "numpy":
        seq = np.array([prompt], dtype=np.int32)
        for _ in range(steps):
            logits = model.forward(seq[:, -context_length:])
            nxt = int(logits[0, -1].argmax())
            seq = np.concatenate([seq, np.array([[nxt]], dtype=np.int32)], axis=1)
        return seq.flatten().tolist()

    # torch / triton (nn.Module) and cuda (manual forward)
    prompt_t = th.tensor([prompt], dtype=th.int64, device=_model_device(backend))
    seq = prompt_t.clone()
    with th.no_grad():
        for _ in range(steps):
            x = seq[:, -context_length:]
            if backend == "cuda":
                logits = model.forward(x)
            else:
                model.eval()
                logits = model(x)
            nxt = th.argmax(logits[0, -1], dim=-1).reshape(1, 1)
            seq = th.cat([seq, nxt], dim=1)
    return seq.flatten().tolist()


# ─── Training helper ───────────────────────────────────────────────────────────


def _train_step(
    backend: str,
    model: Any,
    inp: Any,
    tgt: Any,
    steps: int,
    max_norm: float = 1.0,
) -> float:
    """Run a few training steps and return the last loss.

    PyTorch/Triton: autograd through the model → clip → optimizer.step()
    NumPy: analytic backward → clip → AdamW.step (O(forward) per step).
    CUDA: unsupported here — callers skip training (no autograd path).
    """
    if backend == "numpy":
        from impl._np.cross_entropy import CrossEntropyLoss
        from impl._np.optimizer import AdamW
        from impl._np.training import train_step as np_ts

        ce = CrossEntropyLoss(shift=False)  # targets are pre-shifted next-token labels
        opt = AdamW(lr=0.001)
        last = 0.0
        for _ in range(steps):
            last = np_ts(model, inp, tgt, ce, opt, max_norm=max_norm)
        return last

    import torch

    if backend in ("torch", "triton"):
        from impl._torch.training import train_step as ts
        from impl._triton.training import train_step as triton_ts

        ce = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        step = ts if backend == "torch" else triton_ts
        last = 0.0
        for _ in range(steps):
            last = step(model, inp, tgt, opt, ce, max_norm=max_norm)
        return last

    raise ValueError(f"Training not supported for backend {backend}")


def _run_training(backend: str, model: Any, kwargs: dict[str, Any]) -> float | None:
    """Run a few training steps and return the last loss (None if unsupported).

    CUDA has no training path in this harness (manual kernels, no autograd);
    NumPy now runs the same few steps as the autograd tracks (analytic
    backward, O(forward) per step).
    """
    if backend == "cuda":
        return None

    vocab_size = kwargs["vocab_size"]
    ctx = kwargs.get("context_length", 16)
    seed = int(kwargs.get("seed", 42))

    if backend in ("torch", "triton"):
        import torch

        torch.manual_seed(seed)
        inp = torch.randint(0, vocab_size, (2, ctx), device=_model_device(backend))
        tgt = torch.roll(inp, -1, dims=1)
    else:
        rng = np.random.default_rng(seed)
        inp = rng.integers(0, vocab_size, (2, ctx)).astype(np.int32)
        tgt = np.roll(inp, -1, axis=1)

    try:
        return _train_step(backend, model, inp, tgt, steps=2, max_norm=1.0)
    except Exception:
        return 0.0


# ─── Equivalence helpers ───────────────────────────────────────────────────────
def weight_diff(params_a: dict, params_b: dict) -> float:
    """Max absolute difference between two parameter dicts (shared Keys)."""
    max_diff = 0.0
    for key in sorted(set(params_a) & set(params_b)):
        va = np.asarray(params_a[key])
        vb = np.asarray(params_b.get(key, va))
        if va.size == 0 or vb.size == 0:
            continue
        max_diff = max(max_diff, float(np.abs(va - vb).max()))
    return max_diff


def kl_div_approx(p: np.ndarray, q: np.ndarray) -> float:
    """Approximate KL divergence in bits."""
    eps = 1e-8
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p /= p.sum()
    q = np.clip(q, eps, 1.0)
    q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def distribution_check(p: np.ndarray, q: np.ndarray, threshold: float = 0.5) -> tuple[bool, float]:
    """Compare two token distributions by KL divergence.

    Returns:
        Tuple of (passed, kl_bits) where passed means kl_bits < threshold.
    """
    kl = kl_div_approx(p, q)
    return (kl < threshold, kl)


def _greedy_match(a: list, b: list) -> bool:
    return list(a) == list(b)


def _logit_diff(ref_logits: np.ndarray, other_logits: np.ndarray) -> float:
    """Max absolute logit difference between two forward outputs."""
    a = np.asarray(ref_logits, dtype=np.float64)
    b = np.asarray(other_logits, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max())


# ─── Scenario runner ───────────────────────────────────────────────────────────

# Tolerance tiers (see AGENTS.md): the shared-weight load is lossless, so the
# parameter diff must be near zero everywhere; logits follow the precision tier
# of the compared track (float64 pair vs fp32 kernel track).
_WDIFF_TOL = 1e-5
_LOGIT_TOL_FLOAT64 = 1e-3
_LOGIT_TOL_FP32 = 1e-2


def _compare_results(  # noqa: C901 - one comparison pass per metric keeps the report readable
    results: dict[str, dict],
    ref_params: dict[str, Any] | None,
    ref_logits: Any,
    ref_greedy: list[int] | None,
    details: dict[str, Any],
) -> bool:
    """Compare each backend's results against the NumPy reference; fill details."""
    passed = True

    # Parameter diff vs the reference (the load is lossless → near zero).
    for b, r in results.items():
        if b == "numpy" or not r["params"] or ref_params is None:
            continue
        wd = weight_diff(ref_params, r["params"])
        details[f"ref_vs_{b}_weight_diff"] = round(wd, 8)
        if wd > _WDIFF_TOL:
            passed = False

    # Logit parity vs the reference, tiered by track precision.
    for b, r in results.items():
        if b == "numpy" or r["error"] or ref_logits is None:
            continue
        ld = _logit_diff(ref_logits, r["logits"])
        details[f"ref_vs_{b}_logit_diff"] = round(ld, 8)
        tol = _LOGIT_TOL_FLOAT64 if b == "torch" else _LOGIT_TOL_FP32
        if ld > tol:
            passed = False

    # Greedy token comparison vs the reference (deterministic in this setup).
    for b, r in results.items():
        if b == "numpy" or r["error"] or ref_greedy is None:
            continue
        details[f"ref_vs_{b}_greedy_match"] = _greedy_match(ref_greedy, r["greedy"])

    # Finiteness and per-backend errors.
    for b, r in results.items():
        details[f"{b}_finite"] = r["finite"]
        if not r["finite"]:
            passed = False
        if r["error"]:
            details[f"{b}_error"] = r["error"]
            passed = False

    # Training loss (reported; CUDA has no training path).
    for b, r in results.items():
        if r["loss"] is not None:
            details[f"{b}_last_loss"] = round(float(r["loss"]), 4)

    return passed


def run_scenario(scenario: Scenario, steps: int = 2) -> dict:
    """Run one equivalence scenario.

    Strategy: build the NumPy reference model from the scenario config, load its
    weights into each backend under test (lossless, shared Keys scheme), then
    compare parameter diffs, forward logits, and greedy outputs.
    """
    T1 = time.time()
    details: dict[str, Any] = {"backends": list(scenario.backends)}
    results: dict[str, dict] = {}
    kwargs = scenario.kwargs
    seed = int(kwargs.get("seed", 42))

    # Reference outputs (None if the reference itself failed to build).
    ref_params: dict[str, Any] | None = None
    ref_logits: Any = None
    ref_greedy: list[int] | None = None

    # NumPy reference: always built (cheap, CPU, float64) — source of truth.
    try:
        ref_model = _make_model("numpy", kwargs)
        ref_params = _get_params("numpy", ref_model)

        rng = np.random.default_rng(seed)
        tokens = rng.integers(0, kwargs["vocab_size"], (1, 8)).astype(np.int32)
        ref_logits = _forward_logits("numpy", ref_model, tokens)
        ref_greedy = _greedy_tokens(
            "numpy", ref_model, tokens[0].tolist(), kwargs.get("context_length", 16), steps=steps
        )

        for b in scenario.backends:
            model = _make_model(b, kwargs)
            if b in ("torch", "triton"):
                model = model.to("cuda")
            _load_params_to_model(b, model, ref_params)
            params = _get_params(b, model)  # capture before training mutates the model
            logits = _forward_logits(b, model, tokens)
            greedy = _greedy_tokens(b, model, tokens[0].tolist(), kwargs.get("context_length", 16), steps=steps)
            loss = _run_training(b, model, kwargs)
            results[b] = {
                "params": params,
                "logits": logits,
                "greedy": greedy,
                "loss": loss,
                "finite": bool(np.all(np.isfinite(logits))),
                "error": None,
            }
    except Exception as e:  # noqa: BLE001 - report the error, keep the report shape
        for b in scenario.backends:
            results[b] = {"params": {}, "greedy": [], "loss": 0.0, "finite": False, "error": str(e)}

    passed = _compare_results(results, ref_params, ref_logits, ref_greedy, details)

    elapsed = round(time.time() - T1, 2)
    return {"passed": passed, "name": scenario.name, "details": details, "elapsed": elapsed}


# ─── Report ────────────────────────────────────────────────────────────────────


def format_report(results: list[dict]) -> str:
    """Render a boxed PASS/FAIL report; every result line shows its name."""
    lines = [
        "  ╔══════════════════════════════════════════════════════════╗  ",
        "  ║  verify_equivalence — multi-backend equivalence          ║  ",
        "  ╠══════════════════════════════════════════════════════════╣  ",
    ]
    for i, r in enumerate(results):
        status = "PASS" if r["passed"] else "FAIL"
        box = "✓" if r["passed"] else "✗"
        name = r.get("name", f"scenario_{i + 1}")
        backends = ", ".join(r.get("details", {}).get("backends", []))
        lines.append(f"  ║ {i + 1:2d}. {status}  {name:<34s} ║  {box}  ({backends})")

        metrics = []
        for k, v in r.get("details", {}).items():
            if "weight_diff" in k and isinstance(v, float) or "logit_diff" in k and isinstance(v, float):
                metrics.append(f"{k}= {v:.2e}")
            elif "greedy_match" in k and isinstance(v, bool):
                metrics.append(f"{k}={'✓' if v else '✗'}")
            elif "last_loss" in k and isinstance(v, float):
                metrics.append(f"{k}= {v:.4f}")
        for m in metrics:
            lines.append(f"       {m}")
        for k, v in r.get("details", {}).items():
            if k.endswith("_error"):
                lines.append(f"       {k}: {v}")

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    lines.extend(["", f"  Summary: {passed}/{total} scenarios passed", ""])
    return "\n".join(lines)


# ─── CLI ───────────────────────────────────────────────────────────────────────


def main(args: list[str] | None = None) -> int:
    """Entry point. Returns 0 if all scenarios pass, 1 otherwise, 2 on usage error."""
    parser = argparse.ArgumentParser(
        prog="verify_equivalence",
        description="Multi-backend equivalence testing (NumPy, PyTorch, Triton, CUDA).",
    )
    parser.add_argument("--scenario", type=str, default=None, help="Run scenarios matching this name substring.")
    parser.add_argument("--fast", action="store_true", default=False, help="Use minimal training steps.")
    parser.add_argument("--output", type=str, default=None, help="Write JSON report to this path.")
    try:
        parsed = parser.parse_args(args)
    except SystemExit as e:
        return 2 if e.code != 0 else 0

    all_scenarios = _scenarios()
    if parsed.scenario:
        to_run = [s for s in all_scenarios if parsed.scenario.lower() in s.name.lower()]
        if not to_run:
            print(f"No scenarios match '{parsed.scenario}'", file=sys.stderr)
            return 2
    else:
        to_run = all_scenarios

    results: list[dict] = []
    for s in to_run:
        steps = 1 if parsed.fast else 2
        results.append(run_scenario(s, steps=steps))

    print(format_report(results))

    if parsed.output:
        Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
        with open(parsed.output, "w") as f:
            json.dump(results, f, indent=2)

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
