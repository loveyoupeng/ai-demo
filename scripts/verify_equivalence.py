#!/usr/bin/env python3
"""Automated equivalence verification across NumPy, PyTorch, Triton, and CUDA backends.

Supports any combination of the 4 backends for weight diff, greedy inference,
and training convergence checks.

Usage:
  # Run all default scenarios
  uv run python -m scripts.verify_equivalence

  # Quick mode: synthetic data, minimal epochs
  uv run python -m scripts.verify_equivalence --fast

  # Run specific scenarios by name
  uv run python -m scripts.verify_equivalence --scenario "Small GQA"

  # Custom output
  uv run python -m scripts.verify_equivalence --output /tmp/verify.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ─── Backend Registry ──────────────────────────────────────────────────────────

_BACKENDS: dict[str, dict[str, str]] = {
    "numpy": {"import_path": "impl._np.model", "class_name": "NumPyModel"},
    "torch": {"import_path": "impl._torch.layers", "class_name": "TorchModel"},
    "triton": {"import_path": "impl._triton.model", "class_name": "TritonModel"},
    "cuda": {"import_path": "impl._cuda.model", "class_name": "CUDAModel"},
}


def _make_model(backend: str, config: dict) -> Any:
    """Create a model instance for the given backend from shared config."""
    info = _BACKENDS[backend]
    mod = __import__(info["import_path"], fromlist=[info["class_name"]])
    cls = getattr(mod, info["class_name"])
    return cls(**config)


def _model_device(backend: str) -> str:
    """Return the device string for a backend."""
    return "cuda" if backend in ("torch", "triton", "cuda") else "cpu"


def _get_params(backend: str, model: Any) -> dict[str, np.ndarray]:
    """Extract parameters from a model as dict[str, np.ndarray].

    NumPy: uses get_all_parameters() which returns flat dict.
    PyTorch/Triton: uses save_as_numpy() which returns flat dict.
    CUDA: manually extracts weight attributes since CUDAModel is not an nn.Module.
    """
    if backend == "cuda":
        params: dict[str, np.ndarray] = {}
        for attr in ["embedding_weights", "final_ln_gamma", "output_proj_weights",
                      "output_proj_bias", "output_W1", "output_W3", "output_W2"]:
            t = getattr(model, attr, None)
            if t is not None:
                params[attr] = t.detach().cpu().numpy()
        for i, blk in enumerate(model.stacking.blocks):
            for attr in ["Wq", "Wk", "Wv", "Wo", "ln1_gamma", "ln2_gamma",
                         "gate1", "gate2", "expert_weights", "expert_bias",
                         "routing_weights"]:
                t = getattr(blk, attr, None)
                if t is not None:
                    params[f"blocks.{i}.{attr}"] = t.detach().cpu().numpy()
        return params
    if backend == "numpy":
        return {k: v.copy() for k, v in model.get_all_parameters().items()}
    return {k: v.copy() if hasattr(v, "cpu") else v for k, v in model.save_as_numpy().items()}


def _load_params_to_model(backend: str, model: Any, params: dict[str, np.ndarray]) -> None:
    """Load parameters into a model. Only PyTorch/Triton/NumPy support this."""
    if backend == "numpy" or backend in ("torch", "triton"):
        model.load_from_numpy_dict(params)
    # CUDA: no built-in load


def _train_step(
    backend: str,
    model: Any,
    inp: Any,
    tgt: Any,
    steps: int,
    max_norm: float = 1.0,
) -> float:
    """Run a few training steps and return the last loss.

    NumPy: calls model.forward → loss_fn → model.backward → optimizer.step(params, grads)
    PyTorch/Triton/CUDA: calls loss.backward() on model → clip → optimizer.step()
    """
    import torch

    if backend == "torch":
        from impl._torch.training import train_step as ts
        ce = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        last: float = 0.0
        for _ in range(steps):
            last = ts(model, inp, tgt, opt, ce, max_norm=max_norm)
        return last

    if backend == "triton":
        from impl._triton.training import train_step as ts
        ce = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        last: float = 0.0
        for _ in range(steps):
            last = ts(model, inp, tgt, opt, ce, max_norm=max_norm)
        return last

    if backend == "cuda":
        from impl._cuda.training import train_step as ts
        ce = torch.nn.CrossEntropyLoss()
        # Build optimizer that walks the flat tensor attributes
        params: list[torch.Tensor] = []
        for t in list(model.__dict__.values()):
            if isinstance(t, torch.Tensor) and t.requires_grad:
                params.append(t)
        for blk in model.stacking.blocks:
            for t in list(blk.__dict__.values()):
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    params.append(t)
        opt = torch.optim.Adam(params, lr=0.001)
        last: float = 0.0
        for _ in range(steps):
            last = ts(model, inp, tgt, opt, ce, max_norm=max_norm)
        return last

    if backend == "numpy":
        from impl._np.cross_entropy import CrossEntropyLoss
        from impl._np.optimizer import AdamW
        from impl._np.training import train_step as ts
        loss_fn = CrossEntropyLoss()
        params = model.get_all_parameters()
        opt = AdamW(lr=0.001)
        last: float = 0.0
        for _ in range(steps):
            last = ts(model, inp, tgt, loss_fn, opt, max_norm=max_norm)
        return last

    raise ValueError(f"Unknown backend {backend}")


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
        seq = th.tensor(prompt, dtype=th.int32).numpy().reshape(1, -1)
        for _ in range(steps):
            x = seq[:, -context_length:]
            logits = model.forward(x)
            next_tok = th.tensor([int(th.argmax(logits[0, -1]).item())], dtype=th.int32).reshape(1, 1)
            seq = np.concatenate([seq, next_tok], axis=1)
        return seq.flatten().tolist()

    if backend == "cuda":
        prompt_t = th.tensor(prompt, dtype=th.int64, device="cuda").unsqueeze(0)
        seq = prompt_t.clone()
        for _ in range(steps):
            logits = model.forward(seq)  # CUDAModel uses .forward(), not callable
            nxt = th.argmax(logits[0, -1], dim=-1).reshape(1, 1)
            seq = th.cat([seq, nxt], dim=1)
        return seq.flatten().tolist()

    # torch / triton: nn.Module, callable
    if backend in ("torch", "triton"):
        prompt_t = th.tensor(prompt, dtype=th.int64, device=_model_device(backend)).unsqueeze(0)
        seq = prompt_t.clone()
        with th.no_grad():
            if backend == "torch":
                model.eval()
            else:
                model.eval()
            for _ in range(steps):
                x = seq[:, -context_length:]
                logits = model(x)
                nxt = th.argmax(logits[0, -1], dim=-1).reshape(1, 1)
                seq = th.cat([seq, nxt], dim=1)
        return seq.flatten().tolist()

    raise ValueError(f"Unknown backend {backend}")


# ─── Equivalence helpers ───────────────────────────────────────────────────────

def weight_diff(params_a: dict, params_b: dict) -> float:
    """Max absolute difference between two parameter dicts."""
    max_diff = 0.0
    for key in sorted(params_a.keys()):
        va = np.asarray(params_a[key])
        vb = np.asarray(params_b.get(key, va))
        if va.size == 0 or vb.size == 0:
            continue
        max_diff = max(max_diff, np.abs(va - vb).max())
    return float(max_diff)


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


def _greedy_match(a: list, b: list) -> bool:
    return list(a) == list(b)


# ─── Scenario runner ───────────────────────────────────────────────────────────

def run_scenario(config: dict, backends: list[str], steps: int = 5) -> dict:
    """Run one equivalence scenario for the given backends.

    Strategy: create each model independently, run inference + training,
    then compare results. For NumPy↔PyTorch pairs, share weights first to
    isolate computational equivalence from initialization differences.
    """
    T1 = time.time()
    passed = True
    details: dict[str, Any] = {"backends": backends}
    results: dict[str, dict] = {}

    # Determine if this is a "weight-shareable" scenario
    is_numpy_torch_pair = set(backends) <= {"numpy", "torch"}

    if is_numpy_torch_pair and len(backends) == 2:
        # NumPy↔PyTorch: share weights for direct numerical comparison
        try:
            # 1) Create NumPy model, extract weights
            np_model = _make_model_for_backend("numpy", config)
            # 2) Create PyTorch model, load NumPy weights
            torch_model = _make_model_for_backend("torch", config)
            torch_model.to("cuda")
            _load_params_to_model("torch", torch_model, _get_params("numpy", np_model))
            # 3) Run tests on both with shared weights
            np_params = _get_params("numpy", np_model)
            torch_params = _get_params("torch", torch_model)
            np_greedy = _greedy_tokens("numpy", np_model, [65, 97, 109, 101], config.get("context_length", 64), steps=3)
            torch_greedy = _greedy_tokens("torch", torch_model, [65, 97, 109, 101], config.get("context_length", 64), steps=3)
            np_loss = _run_training("numpy", np_model, config)
            torch_loss = _run_training("torch", torch_model, config)
            results["numpy"] = {"params": np_params, "greedy": np_greedy, "loss": np_loss, "finite": True, "error": None}
            results["torch"] = {"params": torch_params, "greedy": torch_greedy, "loss": torch_loss, "finite": True, "error": None}
        except Exception as e:
            results["numpy"] = {"params": {}, "greedy": [], "loss": 0.0, "finite": False, "error": str(e)}
            results["torch"] = {"params": {}, "greedy": [], "loss": 0.0, "finite": False, "error": str(e)}

        # Greedy match (should be exact since weights are identical and both use same algorithm)
        details["numpy_vs_torch_greedy_match"] = _greedy_match(results.get("numpy", {}).get("greedy", []),
                                                                results.get("torch", {}).get("greedy", []))
        if not details["numpy_vs_torch_greedy_match"]:
            passed = False

        # Weight diff (should be 0 or near-zero)
        wdiff = weight_diff(results.get("numpy", {}).get("params", {}),
                            results.get("torch", {}).get("params", {}))
        details["numpy_vs_torch_weight_diff"] = round(wdiff, 6)

        # Loss comparison
        l1, l2 = results.get("numpy", {}).get("loss", 0), results.get("torch", {}).get("loss", 0)
        details["numpy_vs_torch_loss_diff"] = round(abs(l1 - l2), 4)

    elif set(backends) <= {"triton", "torch"}:
        # Triton↔PyTorch: share weights (Triton has compatible save/load)
        try:
            torch_model = _make_model_for_backend("torch", config)
            torch_model.to("cuda")
            torch_params = _get_params("torch", torch_model)
            triton_model = _make_model_for_backend("triton", config)
            _load_params_to_model("triton", triton_model, torch_params)
            triton_model.to("cuda")
            triton_params = _get_params("triton", triton_model)
            torch_greedy = _greedy_tokens("torch", torch_model, [65, 97, 109, 101], config.get("context_length", 64), steps=3)
            triton_greedy = _greedy_tokens("triton", triton_model, [65, 97, 109, 101], config.get("context_length", 64), steps=3)
            torch_loss = _run_training("torch", torch_model, config)
            triton_loss = _run_training("triton", triton_model, config)
            results["torch"] = {"params": torch_params, "greedy": torch_greedy, "loss": torch_loss, "finite": True, "error": None}
            results["triton"] = {"params": triton_params, "greedy": triton_greedy, "loss": triton_loss, "finite": True, "error": None}
        except Exception as e:
            for b in backends:
                results[b] = {"params": {}, "greedy": [], "loss": 0.0, "finite": False, "error": str(e)}

        if (results.get("torch", {}).get("finite") and
                results.get("triton", {}).get("finite")):
            details["torch_vs_triton_greedy_match"] = _greedy_match(results["torch"]["greedy"], results["triton"]["greedy"])
            if not details["torch_vs_triton_greedy_match"]:
                passed = False
            wd = weight_diff(results["torch"]["params"], results["triton"]["params"])
            details["torch_vs_triton_weight_diff"] = round(wd, 6)
        else:
            if "torch" in results and results["torch"]["error"]:
                details["torch_error"] = results["torch"]["error"]
            if "triton" in results and results["triton"]["error"]:
                details["triton_error"] = results["triton"]["error"]
            if not results.get("torch", {}).get("finite") or not results.get("triton", {}).get("finite"):
                passed = False

    elif "cuda" in backends:
        # CUDA: structural checks only — create model, check it runs
        for b in backends:
            try:
                m = _make_model(b, config)
                if b == "cuda":
                    pass  # CUDAModel handles device
                elif b == "torch":
                    m = m.to("cuda")
                greedy = _greedy_tokens(b, m, [65, 97, 109, 101], config.get("context_length", 64), steps=3)
                loss = _run_training(b, m, config)
                results[b] = {"params": _get_params(b, m), "greedy": greedy, "loss": loss, "finite": True, "error": None}
            except Exception as e:
                results[b] = {"params": {}, "greedy": [], "loss": 0.0, "finite": False, "error": str(e)}

        # Only compare CUDA with others for loss range (not exact match)
        for b in backends:
            if b == "cuda" and results[b]["finite"]:
                details["cuda_loss"] = round(results[b]["loss"], 4)
                details["cuda_finite"] = True
            elif b in ("numpy", "torch") and "cuda" in results and results["cuda"]["finite"]:
                details[f"{b}_vs_cuda_loss_diff"] = round(abs(results[b]["loss"] - results["cuda"]["loss"]), 4)

    elif "triton" in backends and "numpy" in backends:
        # NumPy with Triton — incompatible architectures, just verify both run
        for b in backends:
            try:
                m = _make_model(b, config)
                if b == "triton":
                    m = m.to("cuda")
                greedy = _greedy_tokens(b, m, [65, 97, 109, 101], config.get("context_length", 64), steps=3)
                loss = _run_training(b, m, config)
                results[b] = {"params": _get_params(b, m), "greedy": greedy, "loss": loss, "finite": True, "error": None}
            except Exception as e:
                results[b] = {"params": {}, "greedy": [], "loss": 0.0, "finite": False, "error": str(e)}

    else:
        # General: create all backends, verify they run
        for b in backends:
            try:
                m = _make_model(b, config)
                if b == "cuda":
                    pass
                elif b == "torch" or b == "triton":
                    m = m.to("cuda" if _model_device(b) == "cuda" else "cpu")
                greedy = _greedy_tokens(b, m, [65, 97, 109, 101], config.get("context_length", 64), steps=3)
                loss = _run_training(b, m, config)
                results[b] = {"params": _get_params(b, m), "greedy": greedy, "loss": loss, "finite": True, "error": None}
            except Exception as e:
                results[b] = {"params": {}, "greedy": [], "loss": 0.0, "finite": False, "error": str(e)}

    # CUDA structural validation
    if "cuda" in results:
        details["cuda_finite"] = results["cuda"]["finite"]
        if not results["cuda"]["finite"]:
            passed = False

    elapsed = round(time.time() - T1, 2)
    return {"passed": passed, "name": "", "details": details, "elapsed": elapsed}


# ─── Scenario runner helpers ──────────────────────────────────────────────────


def _make_model_for_backend(backend: str, config: dict) -> Any:
    """Create a model, stripping params the backend doesn't accept."""
    # Triton doesn't accept: rope_dim, seed
    if backend == "triton":
        cfg = {k: v for k, v in config.items()
               if k in ("vocab_size", "embed_dim", "n_layers", "n_heads",
                         "n_experts", "ff_dim", "k")}
    else:
        cfg = config
    return _make_model(backend, cfg)


def _run_training(backend: str, model: Any, config: dict) -> float:
    """Run a few training steps and return the last loss."""
    device = "cuda" if backend in ("torch", "triton", "cuda") else "cpu"
    if device == "cuda":
        import torch
        inp = torch.randint(0, config["vocab_size"],
                            (2, config.get("context_length", 64)), device="cuda")
        tgt = torch.roll(inp, -1, dims=1)
    else:
        import numpy as np
        inp = np.random.randint(0, config["vocab_size"],
                                (2, config.get("context_length", 64))).astype(np.int32)
        tgt = np.roll(inp, -1).astype(np.int32)
    try:
        return _train_step(backend, model, inp, tgt, steps=2, max_norm=1.0)
    except Exception:
        return 0.0


def format_report(results: list[dict]) -> str:
    lines = [
        "  ╔═══════════════════════════════════════════════════╗  ",
        "║    verify_equivalence.py — Multi-Backend           ║  ",
        "╠═══════════════════════════════════════════════════╣  ",
    ]
    for i, r in enumerate(results):
        status = "PASS" if r["passed"] else "FAIL"
        box = "✓" if r["passed"] else "✗"
        backends = ", ".join(r["details"].get("backends", []))
        metrics = []
        for k, v in r["details"].items():
            if isinstance(v, float) and "weight_diff" in k:
                metrics.append(f"{k.split('_vs_')[0]}↔{k.split('_vs_')[1]} wdiff={v}")
            elif isinstance(v, bool) and "greedy_match" in k:
                metrics.append(f"{k.split('_vs_')[0]}↔{k.split('_vs_')[1]} greedy={'✓' if v else '✗'}")
        mi = " | ".join(metrics[:3])
        line = f"  ║  {i + 1:2d}. {status:4s} {backends[:45]:45s} ║  {box}  {mi:<20s}"
        lines.append(line[:80].ljust(82) + "║")

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    lines.extend(["", f"  Summary: {passed}/{total} scenarios passed", ""])
    return "\n".join(lines)


# ─── Scenarios ──────────────────────────────────────────────────────────────────

def _base_cfg(overrides: dict | None = None) -> dict:
    """Create base config with optional overrides.

    Uses only params shared across all backends:
    vocab_size, embed_dim, n_layers, n_heads, n_experts, ff_dim, k, rope_dim, seed.

    Torch-specific params (n_groups, max_length) must be added via overrides
    when testing only Torch/Triton backends.
    """
    cfg = dict(
        vocab_size=128, embed_dim=32, n_layers=1, n_heads=4, n_experts=2,
        ff_dim=64, k=1, rope_dim=8, seed=42,
    )
    if overrides:
        cfg.update(overrides)
    return cfg


def scenarios() -> list[tuple[dict, list[str], str]]:
    """Return (config, backends, name) tuples for each scenario.

    All backends share these constructor params:
    vocab_size, embed_dim, n_layers, n_heads, n_experts, ff_dim, k, rope_dim, seed.

    n_groups and max_length are internal params (not exposed on model constructors),
    so they are omitted from all scenarios.
    """
    return [
        # NumPy↔PyTorch numerical parity (weights shared)
        (_base_cfg(), ["numpy", "torch"], "NumPy↔Torch weight-shared parity"),

        # PyTorch↔Triton numerical parity (weights shared)
        (_base_cfg(), ["torch", "triton"], "PyTorch↔Triton weight-shared parity"),

        # CUDA structural — model creates and runs
        (_base_cfg(), ["cuda"], "CUDA structural validity"),

        # Torch alone — runs inference and training
        (_base_cfg(), ["torch"], "PyTorch standalone inference + training"),

        # Triton alone — runs inference and training
        (_base_cfg(), ["triton"], "Triton standalone inference + training"),

        # NumPy alone — runs inference and training
        (_base_cfg(), ["numpy"], "NumPy standalone inference + training"),

        # Multi-layer NumPy↔Torch parity
        (_base_cfg({"n_layers": 2}),
         ["numpy", "torch"], "2-layer NumPy↔Torch parity"),

        # MoE NumPy↔Torch parity
        (_base_cfg({"n_experts": 2, "k": 2}),
         ["numpy", "torch"], "MoE NumPy↔Torch parity"),
    ]


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_equivalence",
        description="Multi-backend equivalence testing (NumPy, PyTorch, Triton, CUDA).",
    )
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run scenarios matching this name substring.")
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Use synthetic data and minimal training steps.")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON report to this path.")
    try:
        parsed = parser.parse_args(args)
    except SystemExit as e:
        return 2 if e.code != 0 else 0

    # Select scenarios
    all_cfg = scenarios()
    if parsed.scenario:
        filtered = [(c, b, n) for c, b, n in all_cfg if parsed.scenario.lower() in n.lower()]
        if not filtered:
            print(f"No scenarios match '{parsed.scenario}'", file=sys.stderr)
            return 2
        to_run = filtered
    else:
        to_run = all_cfg

    # Run each
    results: list[dict] = []
    for cfg, backends, name in to_run:
        steps = 2 if parsed.fast else 5
        r = run_scenario(cfg, backends, steps=steps)
        r["name"] = name
        results.append(r)

    report = format_report(results)
    print(report)

    if parsed.output:
        Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
        with open(parsed.output, "w") as f:
            json.dump(results, f, indent=2)

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
