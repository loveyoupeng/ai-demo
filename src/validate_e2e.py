#!/usr/bin/env python3
"""
End-to-end cross-backend validation script.

Validates NumPy and PyTorch implementations produce equivalent results
through 4-way cross-check scenarios.

Run:
    uv run src/validate_e2e.py [--epochs 3]
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from backends.numpy.numpy_backend import NumPyBackend
from backends.pytorch.pytorch_backend import PyTorchBackend
from core.base_backend import BaseTransformerBackend
from inference import AutoregressiveGenerator
from loss import CrossEntropyLoss
from model.pytorch.transformer import PyTorchTransformer
from model.transformer import Transformer
from optimizer import Adam
from tokenizer.char_tokenizer import CharTokenizer
from trainer import Trainer


# ============================================================
# Configuration
# ============================================================

EPOCHS: int = 3
NUM_STEPS: int = 5
SEED: int = 42
BATCH_SIZE: int = 4
SEQ_LEN: int = 16
EMBED_DIM: int = 32
NUM_LAYERS: int = 2
NUM_HEADS: int = 4
NUM_EXPERTS: int = 4
MAX_SEQ_LEN: int = 64
LEARNING_RATE: float = 0.001
CLIP_VALUE: float = 1.0
INFERENCE_PROMPT: str = "to be"

TEXT = (
    "It is a truth universally acknowledged, that a single man in possession of a "
    "fortune, must be in want of a wife. However little known the feelings or views "
    "of such a man may be on his first entering a neighbourhood, this truth is so "
    "well fixed in the minds of the surrounding families, that he is considered as "
    "the rightful property of some one or other of their daughters. My dear Mr. "
    "Bennet, said his wife to him one day, have you heard that Netherfield Park is "
    "at last let? The house is virtually let. Mr. Bennet replied. You puzzle me "
    "exceedingly. If you mean any thing, I must believe you. I am not afraid of "
    "him; I shall see his coming in time; and I am sure you will not wish to keep "
    "him from us, when his character becomes known. This is an unverifiable claim, "
    "but there is no evidence against it, and the universe seems designed to test "
    "whether our implementation can distinguish truth from fiction. The quick brown "
    "fox jumps over the lazy dog, and the validator watches with keen attention. "
).strip()


# ============================================================
# Helpers
# ============================================================


def _set_seeds(seed: int) -> None:
    """Set random seeds for NumPy, PyTorch, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _create_text_batches(
    text: str, tokenizer: CharTokenizer, num_batches: int | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create (input, target) batch pairs from text using tokenizer."""
    text_array = tokenizer.encode(text[:256])
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    if num_batches is None:
        num_batches = max(NUM_STEPS, BATCH_SIZE)
    segment_size = len(text_array) // num_batches
    for i in range(max(num_batches, NUM_STEPS)):
        start = i * segment_size
        end = start + SEQ_LEN
        seg = text_array[start:end]
        if len(seg) < SEQ_LEN:
            continue
        x = seg[: SEQ_LEN - 1].reshape(1, -1)
        y = seg[1:].reshape(1, -1)
        xs.append(x)
        ys.append(y)
    return xs, ys


def _init_config(vocab_size: int) -> dict:
    """Config shared by both backends."""
    return {
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "num_experts": NUM_EXPERTS,
        "max_seq_len": MAX_SEQ_LEN,
    }


def _record_logits(
    backend: BaseTransformerBackend, x: np.ndarray, y: np.ndarray
) -> dict[str, object]:
    """Run forward pass, return logits and loss info."""
    logits, cache = backend.forward(x)
    loss, _ = CrossEntropyLoss().forward(logits, y)
    logits_flat = logits.flatten().astype(np.float64)
    h = hashlib.md5(logits_flat.tobytes()).hexdigest()
    return {"logits": logits, "loss": loss, "hash": h, "cache": cache}


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    max_val = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max_val)
    return e_x / (np.sum(e_x, axis=-1, keepdims=True) + 1e-12)


def _generate_response_pt(
    backend: object,
    tokenizer: CharTokenizer,
    prompt: str,
    temperature: float,
    num_tokens: int,
) -> str:
    """Generate text using PyTorch backend."""
    np.random.seed(SEED + 100)
    ids = tokenizer.encode(prompt).reshape(1, -1)
    generated_ids: list[int] = []
    forward_fn = backend.forward  # type: ignore[attr-defined]
    for _ in range(num_tokens):
        logits_out, _ = forward_fn(ids)
        next_token_logits = logits_out[:, -1, :]
        probs = _softmax_np(next_token_logits)
        next_id = int(np.random.choice(probs.shape[-1], p=probs[0]))
        next_id_arr = np.array([[next_id]], dtype=np.int32)
        ids = np.concatenate([ids, next_id_arr], axis=1)
        generated_ids.append(next_id)
    return tokenizer.decode(np.array(generated_ids, dtype=np.int32))


def _compare(
    label_a: str,
    label_b: str,
    val_a: object,
    val_b: object,
    tolerance: float = 1e-4,
) -> tuple[bool, str]:
    """Compare two values, return (match, reason)."""
    if isinstance(val_a, np.ndarray) and isinstance(val_b, np.ndarray):
        if val_a.shape != val_b.shape:
            return False, f"DIFFER shapes: {val_a.shape} vs {val_b.shape}"
        diff = np.abs(val_a - val_b)
        if np.max(diff) > tolerance:
            return False, f"max_diff={np.max(diff):.8f}"
        return True, f"max_diff={np.max(diff):.8f}"
    if isinstance(val_a, str) and isinstance(val_b, str):
        if val_a == val_b:
            return True, "MATCH"
        return False, f"DIFFER: {val_a[:40]} vs {val_b[:40]}"
    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
        if abs(float(val_a) - float(val_b)) < tolerance:
            return True, "MATCH"
        return False, f"DIFFER: {val_a:.8f} vs {val_b:.8f}"
    if isinstance(val_a, dict) and isinstance(val_b, dict):
        if set(val_a.keys()) != set(val_b.keys()):
            return False, f"DIFFER: keys {set(val_a.keys())} vs {set(val_b.keys())}"
        all_match = True
        reasons: list[str] = []
        for k in val_a:
            m, r = _compare(
                f"{label_a}.{k}", f"{label_b}.{k}", val_a[k], val_b[k], tolerance
            )
            if not m:
                all_match = False
                reasons.append(r)
        if all_match:
            return True, "MATCH"
        return False, "; ".join(reasons)
    return str(val_a) == str(val_b), f"DIFFER: {type(val_a)} vs {type(val_b)}"


def _pt_canonical(key: str) -> str:
    """Map PyTorch internal names to canonical NumPy-style names."""
    _CACHE = {
        "token_embedding.embedding.weight": "token_embedding.weights",
        "lm_head.weight": "lm_head",
    }
    if key in _CACHE:
        return _CACHE[key]
    if ".moe" in key:
        key = key.replace(".moe.experts.", ".moe.expert.")
        key = key.replace(".moe.router.w", ".moe.router.weights")
        key = key.replace(".w1", ".W1").replace(".w2", ".W2")
        return key
    for suffix, new in [
        (".ln1.weight", ".gamma"),
        (".ln1.bias", ".beta"),
        (".ln2.weight", ".gamma"),
        (".ln2.bias", ".beta"),
    ]:
        if key.endswith(suffix):
            return key[: -len(suffix)] + new
    return key


def _generate_response_np(
    model: Transformer,
    tokenizer: CharTokenizer,
    prompt: str,
    temperature: float,
    num_tokens: int,
) -> str:
    """Generate text using NumPy model."""
    np.random.seed(SEED + 100)
    gen = AutoregressiveGenerator(model, tokenizer, temperature=temperature)
    gen_ids = gen.generate(prompt, num_new_tokens=num_tokens)
    return tokenizer.decode(gen_ids)


def _train_step_loop(
    trainer: object, xs: list[np.ndarray], ys: list[np.ndarray]
) -> list[float]:
    """Train loop abstraction - avoids type checker issues."""
    losses: list[float] = []
    for ep in range(EPOCHS):
        for bi in range(NUM_STEPS):
            loss_val = trainer.train_step(xs[bi], ys[bi])  # type: ignore[arg-type]
            losses.append(loss_val)
    return losses


# ============================================================
# Training function
# ============================================================


def _train_numpy(tokenizer: CharTokenizer) -> tuple[NumPyBackend, list[float]]:
    """Train NumPy model, return (backend, losses)."""
    _set_seeds(SEED)
    backend = NumPyBackend(**_init_config(tokenizer.vocab_size))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(backend, optimizer, loss_fn, clip_value=CLIP_VALUE)
    xs, ys = _create_text_batches(TEXT, tokenizer)
    losses = _train_step_loop(trainer, xs, ys)
    return backend, losses


def _train_pytorch(tokenizer: CharTokenizer) -> tuple[PyTorchBackend, list[float]]:
    """Train PyTorch model, return (backend, losses)."""
    _set_seeds(SEED)
    backend = PyTorchBackend(**_init_config(tokenizer.vocab_size))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(backend, optimizer, loss_fn, clip_value=CLIP_VALUE)
    xs, ys = _create_text_batches(TEXT, tokenizer)
    losses = _train_step_loop(trainer, xs, ys)
    return backend, losses


# ============================================================
# Main Validation
# ============================================================


def run_scenario_1() -> dict:
    """Scenario 1: NumPy backend train -> inference."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: NumPy backend train -> inference")
    print("=" * 70)

    tokenizer = CharTokenizer(TEXT)
    backend, losses = _train_numpy(tokenizer)

    test_x, test_y = _create_text_batches(TEXT, tokenizer)
    eval_result = _record_logits(backend, test_x[0], test_y[0])

    np.random.seed(SEED + 200)
    gen = AutoregressiveGenerator(backend.model, tokenizer, temperature=0.8)
    gen_ids = gen.generate(INFERENCE_PROMPT, num_new_tokens=30)
    response = tokenizer.decode(gen_ids)

    return {
        "scenario": 1,
        "backend_type": "NumPy",
        "losses": losses,
        "eval_loss": eval_result["loss"],
        "eval_hash": eval_result["hash"],
        "response": response,
    }


def run_scenario_2() -> dict:
    """Scenario 2: PyTorch backend train -> inference."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: PyTorch backend train -> inference")
    print("=" * 70)

    tokenizer = CharTokenizer(TEXT)
    backend, losses = _train_pytorch(tokenizer)

    test_x, test_y = _create_text_batches(TEXT, tokenizer)
    eval_result = _record_logits(backend, test_x[0], test_y[0])

    np.random.seed(SEED + 200)
    response = _generate_response_pt(backend, tokenizer, INFERENCE_PROMPT, 0.8, 30)

    return {
        "scenario": 2,
        "backend_type": "PyTorch",
        "losses": losses,
        "eval_loss": eval_result["loss"],
        "eval_hash": eval_result["hash"],
        "response": response,
    }


def run_scenario_3() -> dict:
    """Scenario 3: PT train -> load into NumPy -> inference."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: PyTorch train -> load into NumPy -> inference")
    print("=" * 70)

    _set_seeds(SEED)
    tokenizer = CharTokenizer(TEXT)
    vocab_size = tokenizer.vocab_size

    backend = PyTorchBackend(**_init_config(vocab_size))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(backend, optimizer, loss_fn, clip_value=CLIP_VALUE)
    xs, ys = _create_text_batches(TEXT, tokenizer)
    pt_losses = _train_step_loop(trainer, xs, ys)

    print(f"  PyTorch training complete, final loss: {pt_losses[-1]:.6f}")

    pt_params = backend.get_params()

    np_model = Transformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_experts=NUM_EXPERTS,
        max_seq_len=MAX_SEQ_LEN,
    )
    np_model.set_params(pt_params)

    test_x, test_y = _create_text_batches(TEXT, tokenizer)
    pt_logits, _ = backend.forward(test_x[0])
    np_logits, _ = np_model.forward(test_x[0])

    logits_match, logits_reason = _compare(
        "pt_logits", "np_logits", pt_logits, np_logits, 1e-4
    )

    np.random.seed(SEED + 200)
    np_gen = _generate_response_np(np_model, tokenizer, INFERENCE_PROMPT, 0.8, 30)

    np.random.seed(SEED + 200)
    pt_gen = _generate_response_pt(backend, tokenizer, INFERENCE_PROMPT, 0.8, 30)

    return {
        "scenario": 3,
        "backend_type": "PT->NP cross-load",
        "pt_losses": pt_losses,
        "logits_match": logits_match,
        "logits_reason": logits_reason,
        "pt_response": pt_gen,
        "np_response": np_gen,
    }


def run_scenario_4() -> dict:
    """Scenario 4: NumPy train -> load into PyTorch -> inference."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: NumPy train -> load into PyTorch -> inference")
    print("=" * 70)

    _set_seeds(SEED + 50)
    tokenizer = CharTokenizer(TEXT)
    vocab_size = tokenizer.vocab_size

    np_backend = NumPyBackend(**_init_config(vocab_size))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(np_backend, optimizer, loss_fn, clip_value=CLIP_VALUE)
    xs, ys = _create_text_batches(TEXT, tokenizer)
    np_losses = _train_step_loop(trainer, xs, ys)

    print(f"  NumPy training complete, final loss: {np_losses[-1]:.6f}")

    np_params = np_backend.model.get_params()

    _set_seeds(SEED + 50)
    pt_model = PyTorchTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_experts=NUM_EXPERTS,
        max_seq_len=MAX_SEQ_LEN,
    )

    class _PtBackendW:
        """Minimal PyTorch backend wrapper for cross-loading."""

        def __init__(self, model):
            self.model = model

        def forward(self, x, mask=None, use_cache=False, cache_idx=None):
            tensor_input = torch.from_numpy(x).to(torch.int64)
            tensor_mask = torch.from_numpy(mask) if mask is not None else None
            logits_tensor, _ = self.model.forward(
                tensor_input,
                mask=tensor_mask,
                use_cache=use_cache,
                cache_idx=cache_idx,
            )
            return logits_tensor.detach().float().cpu().numpy().astype(np.float64), {}

        def set_params(self, params):
            pt_backend = PyTorchBackend.__new__(PyTorchBackend)
            pt_backend.model = self.model
            pt_backend.set_params(params)

    pt_backend = _PtBackendW(pt_model)
    pt_backend.set_params(np_params)

    test_x, test_y = _create_text_batches(TEXT, tokenizer)
    np_logits, _ = np_backend.model.forward(test_x[0])
    pt_logits, _ = pt_backend.forward(test_x[0])

    logits_match, logits_reason = _compare(
        "np_logits", "pt_logits", np_logits, pt_logits, 1e-4
    )

    np.random.seed(SEED + 200 + 50)
    np_gen = _generate_response_np(
        np_backend.model, tokenizer, INFERENCE_PROMPT, 0.8, 30
    )

    np.random.seed(SEED + 200 + 50)
    pt_gen = _generate_response_pt(pt_backend, tokenizer, INFERENCE_PROMPT, 0.8, 30)

    return {
        "scenario": 4,
        "backend_type": "NP->PT cross-load",
        "np_losses": np_losses,
        "logits_match": logits_match,
        "logits_reason": logits_reason,
        "np_response": np_gen,
        "pt_response": pt_gen,
    }


def _print_comparison(result: dict) -> None:
    """Print a nicely formatted comparison for a scenario."""
    scenario = result["scenario"]
    print(f"\n  Scenario {scenario} results:")

    if scenario == 1:
        print(f"    NumPy training completed ({len(result['losses'])} steps)")
        print(f"    Final loss: {result['losses'][-1]:.6f}")
        print(f"    Eval loss:  {result['eval_loss']:.6f}")
        print(f"    Eval hash:  {result['eval_hash']}")
        print(f"    Response:   {result['response'][:80]}...")

    elif scenario == 2:
        print(f"    PyTorch training completed ({len(result['losses'])} steps)")
        print(f"    Final loss: {result['losses'][-1]:.6f}")
        print(f"    Eval loss:  {result['eval_loss']:.6f}")
        print(f"    Eval hash:  {result['eval_hash']}")
        print(f"    Response:   {result['response'][:80]}...")

    elif scenario == 3:
        print(f"    PyTorch trained: final loss = {result['pt_losses'][-1]:.6f}")
        if result["logits_match"]:
            print(f"    Forward pass match: {result['logits_reason']}")
        else:
            print(f"    Forward mismatch: {result['logits_reason']}")

    elif scenario == 4:
        print(f"    NumPy trained: final loss = {result['np_losses'][-1]:.6f}")
        if result["logits_match"]:
            print(f"    Forward pass match: {result['logits_reason']}")
        else:
            print(f"    Forward mismatch: {result['logits_reason']}")


def _print_cross_scenario_summary(s1, s2, s3, s4) -> None:
    """Print cross-scenario comparisons."""
    print("\n" + "=" * 70)
    print("CROSS-SCENARIO SUMMARY")
    print("=" * 70)

    print("\n  [2 vs 1] PyTorch train vs NumPy train:")
    l2, l1 = s2["eval_loss"], s1["eval_loss"]
    m = abs(l2 - l1) < 1e-3
    print(f"    Loss parity: {'MATCH' if m else f'NO MATCH (diff={abs(l2 - l1):.6f})'}")

    h2, h1 = s2["eval_hash"], s1["eval_hash"]
    if h2 == h1:
        print("    Identical logits hash")
    else:
        print("    Different logits hash (expected for floating-point)")
        print(f"      NumPy:     {h1}")
        print(f"      PyTorch:   {h2}")

    print("\n  [3] PT train -> load into NP -> forward pass:")
    if s3["logits_match"]:
        print("    NumPy model with PT params matches PyTorch forward")
    else:
        print(f"    Forward mismatch: {s3['logits_reason']}")

    print("\n  [4] NP train -> load into PT -> forward pass:")
    if s4["logits_match"]:
        print("    PyTorch model with NumPy params matches NumPy forward")
    else:
        print(f"    Forward mismatch: {s4['logits_reason']}")

    print("\n" + "-" * 70)


def main() -> None:
    """Run all 4 end-to-end validation scenarios."""
    print("=" * 70)
    print("E2E CROSS-BACKEND VALIDATION")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Steps per epoch: {NUM_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Heads: {NUM_HEADS}")
    print(f"  Experts: {NUM_EXPERTS}")
    print(f"  Seed: {SEED}")
    print("=" * 70)

    s1 = run_scenario_1()
    s2 = run_scenario_2()
    s3 = run_scenario_3()
    s4 = run_scenario_4()

    for result in [s1, s2, s3, s4]:
        _print_comparison(result)

    _print_cross_scenario_summary(s1, s2, s3, s4)


if __name__ == "__main__":
    main()
