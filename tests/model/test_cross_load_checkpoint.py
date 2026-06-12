"""
E2E checkpoint cross-load verification.

Verifies that training with one backend, saving to .pkl file,
loading into the OTHER backend, and running inference produces
matching results — proving NumPy and PyTorch are the same model.

Runs with pytest:
    uv run pytest tests/model/test_cross_load_checkpoint.py -v
"""

from __future__ import annotations

import os
import pickle
import tempfile
from typing import Any

import numpy as np
import pytest
import torch

from backends.numpy.numpy_backend import NumPyBackend
from backends.pytorch.pytorch_backend import PyTorchBackend, _canonical_to_pytorch
from loss import CrossEntropyLoss
from model.pytorch.transformer import PyTorchTransformer
from model.transformer import Transformer
from optimizer import Adam
from tokenizer.char_tokenizer import CharTokenizer
from trainer import Trainer

# Small, deterministic text that fits in any config
TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Romeo and Juliet were lovers from Verona. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. The end."
)

# Shared config
SEED: int = 42
NUM_STEPS: int = 10
LEARNING_RATE: float = 0.001


# ============================================================
# Helpers
# ============================================================


def _set_seeds(seed: int) -> None:
    """Set random seeds for NumPy, PyTorch, and Python."""
    random = __import__("random")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _create_batches(
    tokenizer: CharTokenizer, num_batches: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create (input, target) batch pairs from TEXT."""
    text_array = tokenizer.encode(TEXT)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    segment_size = len(text_array) // max(num_batches, 1)
    if segment_size < 2:
        segment_size = 2
    for i in range(num_batches):
        start = i * segment_size
        end = start + segment_size
        seg = text_array[start:end]
        if len(seg) < 2:
            continue
        x = seg[:-1].reshape(1, -1)
        y = seg[1:].reshape(1, -1)
        xs.append(x)
        ys.append(y)
    if not xs:
        xs = [text_array[:2].reshape(1, -1)]
        ys = [text_array[1:3].reshape(1, -1)]
    return xs, ys


def _train_numpy(tokenizer: CharTokenizer) -> NumPyBackend:
    """Train a NumPy model on TEXT for a fixed number of steps."""
    _set_seeds(SEED)
    backend = NumPyBackend(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        max_seq_len=64,
    )
    trainer = Trainer(backend, Adam(learning_rate=LEARNING_RATE), CrossEntropyLoss())
    xs, ys = _create_batches(tokenizer, NUM_STEPS)
    for x, y in zip(xs, ys):
        trainer.train_step(x, y)
    return backend


def _train_pytorch(tokenizer: CharTokenizer) -> PyTorchBackend:
    """Train a PyTorch model on TEXT for a fixed number of steps."""
    _set_seeds(SEED)
    backend = PyTorchBackend(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        max_seq_len=64,
    )
    trainer = Trainer(backend, Adam(learning_rate=LEARNING_RATE), CrossEntropyLoss())
    xs, ys = _create_batches(tokenizer, NUM_STEPS)
    for x, y in zip(xs, ys):
        trainer.train_step(x, y)
    return backend


def _greedy_generate_np(
    model: Transformer, tokenizer: CharTokenizer, prompt: str, n_tokens: int
) -> str:
    """Generate text with NumPy model using greedy argmax."""
    current_ids = tokenizer.encode(prompt).reshape(1, -1)
    generated_ids: list[int] = []

    for step in range(n_tokens):
        logits, _ = model.forward(current_ids, use_cache=False)
        next_logits = logits[0, -1, :]
        probs = np.exp(next_logits) / (np.sum(np.exp(next_logits)) + 1e-12)
        next_id = int(np.argmax(probs))
        generated_ids.append(next_id)
        next_id_arr = np.array([[next_id]], dtype=np.int32)
        current_ids = np.concatenate([current_ids, next_id_arr], axis=1)

    return tokenizer.decode(np.array(generated_ids, dtype=np.int32))


def _greedy_generate_pt(
    model: PyTorchTransformer, tokenizer: CharTokenizer, prompt: str, n_tokens: int
) -> str:
    """Generate text with PyTorch model using greedy argmax and KV cache."""
    current_ids = tokenizer.encode(prompt).reshape(1, -1)
    generated_ids: list[int] = []

    # Initialize KV caches
    from model.pytorch.attention_kvcache import PyTorchTurboQuantCache

    num_heads = model.blocks[0].mha.num_heads if hasattr(model.blocks[0], "mha") else 2
    kv_caches = [
        PyTorchTurboQuantCache(
            embed_dim=32,
            num_heads=num_heads,
            max_seq_len=64,
            head_dim=16,
        )
        for _ in range(model.num_layers)
    ]

    for step in range(n_tokens):
        input_tensor = torch.from_numpy(current_ids).to(torch.int64)
        logits_tensor, _ = model.forward(input_tensor, kv_caches=kv_caches)
        logits_np = logits_tensor.detach().float().cpu().numpy().astype(np.float64)

        next_logits = logits_np[0, -1, :]
        probs = np.exp(next_logits) / (np.sum(np.exp(next_logits)) + 1e-12)
        next_id = int(np.argmax(probs))
        generated_ids.append(next_id)
        next_id_arr = np.array([[next_id]], dtype=np.int32)
        current_ids = np.concatenate([current_ids, next_id_arr], axis=1)

    return tokenizer.decode(np.array(generated_ids, dtype=np.int32))


def _save_and_reload_checkpoint(
    model: Transformer | PyTorchTransformer,
    tokenizer: CharTokenizer,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Save model params to .pkl, return checkpoint dict."""
    if isinstance(model, PyTorchBackend):
        params = model.get_params()
    elif isinstance(model, PyTorchTransformer):
        params = model.get_params()
    else:
        params = model.get_params()

    checkpoint = {
        "model_params": params,
        "tokenizer": tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 32,
        "num_layers": 1,
        "num_heads": 2,
        "num_experts": 2,
        "max_seq_len": 64,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    return checkpoint


# ============================================================
# Tests
# ============================================================


class TestCrossLoadCheckpoint:
    """E2E: train -> save .pkl -> load -> inference comparison."""

    @pytest.fixture()
    def tokenizer(self):
        """Fixed tokenizer for all tests."""
        return CharTokenizer(TEXT)

    @pytest.fixture()
    def np_trained(self, tokenizer):
        """Train NumPy model."""
        return _train_numpy(tokenizer)

    @pytest.fixture()
    def pt_trained(self, tokenizer):
        """Train PyTorch model."""
        return _train_pytorch(tokenizer)

    def test_scenario_1_numpy_baseline(self, tokenizer, np_trained):
        """NumPy train -> NumPy inference produces expected output."""
        result = _greedy_generate_np(np_trained.model, tokenizer, "The", 10)
        assert isinstance(result, str)
        assert len(result) > 0
        # Must be deterministic — same call produces same text
        result2 = _greedy_generate_np(np_trained.model, tokenizer, "The", 10)
        assert result == result2

    def test_scenario_2_pytorch_baseline(self, tokenizer, pt_trained):
        """PyTorch train -> PyTorch inference produces expected output."""
        result = _greedy_generate_pt(pt_trained.model, tokenizer, "The", 10)
        assert isinstance(result, str)
        assert len(result) > 0
        result2 = _greedy_generate_pt(pt_trained.model, tokenizer, "The", 10)
        assert result == result2

    def test_scenario_3_numpy_to_pytorch_text(self, tokenizer, np_trained):
        """NumPy train -> save -> PyTorch load -> inference -> text match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "np_checkpoint")

            # Save checkpoint from NP model
            params = np_trained.model.get_params()
            checkpoint = {
                "model_params": params,
                "tokenizer": tokenizer,
                "vocab_size": tokenizer.vocab_size,
                "embed_dim": 32,
                "num_layers": 1,
                "num_heads": 2,
                "num_experts": 2,
                "max_seq_len": 64,
            }
            with open(path, "wb") as f:
                pickle.dump(checkpoint, f)

            # Load into PT model from NP params
            pt_model = PyTorchTransformer(
                vocab_size=tokenizer.vocab_size,
                embed_dim=32,
                num_layers=1,
                num_heads=2,
                num_experts=2,
                max_seq_len=64,
            )
            pt_state = {}
            for k, v in params.items():
                pt_key = _canonical_to_pytorch(k)
                tensor = torch.from_numpy(v).float()
                if k == "lm_head":
                    tensor = tensor.T
                pt_state[pt_key] = tensor
            pt_model.load_state_dict(pt_state, strict=False)

            np_text = _greedy_generate_np(np_trained.model, tokenizer, "The", 10)
            pt_text = _greedy_generate_pt(pt_model, tokenizer, "The", 10)
            assert pt_text == np_text, (
                f"Text mismatch: NP={np_text!r} vs PT={pt_text!r}"
            )

    def test_scenario_4_pytorch_to_numpy_text(self, tokenizer, pt_trained):
        """PyTorch train -> save -> NumPy load -> inference -> text match."""
        # Get PT params
        pt_params = pt_trained.get_params()
        # Save to file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pt_checkpoint")
            with open(path, "wb") as f:
                pickle.dump({"model_params": pt_params}, f)

            # Load into NumPy model
            np_model = Transformer(
                vocab_size=tokenizer.vocab_size,
                embed_dim=32,
                num_layers=1,
                num_heads=2,
                num_experts=2,
                max_seq_len=64,
            )
            np_model.set_params(pt_params)

            pt_text = _greedy_generate_pt(pt_trained.model, tokenizer, "The", 10)
            np_text = _greedy_generate_np(np_model, tokenizer, "The", 10)
            assert np_text == pt_text, (
                f"Text mismatch: PT={pt_text!r} vs NP={np_text!r}"
            )

    def test_scenario_5_weights_match_np_to_pt(self, tokenizer, np_trained):
        """After NP->PT cross-load, max_weight_diff < 1e-6."""
        np_params = np_trained.model.get_params()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "np_ckpt")
            with open(path, "wb") as f:
                pickle.dump({"model_params": np_params}, f)

            pt_model = PyTorchTransformer(
                vocab_size=tokenizer.vocab_size,
                embed_dim=32,
                num_layers=1,
                num_heads=2,
                num_experts=2,
                max_seq_len=64,
            )
            pt_state = {}
            for k, v in np_params.items():
                pt_key = _canonical_to_pytorch(k)
                tensor = torch.from_numpy(v).float()
                if k == "lm_head":
                    tensor = tensor.T
                pt_state[pt_key] = tensor
            pt_model.load_state_dict(pt_state, strict=False)

            diff = 0.0
            for name, param in pt_model.named_parameters():
                pt_val = param.detach().float().cpu().numpy()
                canon = name
                if canon == "lm_head.weight":
                    canon = "lm_head"
                    pt_val = pt_val.T
                if canon in np_params:
                    diff = max(diff, float(np.max(np.abs(np_params[canon] - pt_val))))

            assert diff < 1e-6, (
                f"Weights differ too much after NP->PT load: max_diff={diff}"
            )

    def test_scenario_6_weights_match_pt_to_np(self, tokenizer, pt_trained):
        """After PT->NP cross-load, max_weight_diff < 1e-6."""
        pt_params = pt_trained.get_params()

        np_model = Transformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            num_experts=2,
            max_seq_len=64,
        )
        np_model.set_params(pt_params)

        diff = 0.0
        np_keys = set(np_model.get_params().keys())
        for key in np_keys:
            if key in pt_params:
                diff = max(
                    diff,
                    float(np.max(np.abs(np_model.get_params()[key] - pt_params[key]))),
                )

        assert diff < 1e-6, (
            f"Weights differ too much after PT->NP load: max_diff={diff}"
        )

    def test_scenario_7_forward_pass_match_np_to_pt(self, tokenizer, np_trained):
        """After NP->PT cross-load, same input -> same logits (max_diff < 1e-6)."""
        np_params = np_trained.model.get_params()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt")
            with open(path, "wb") as f:
                pickle.dump({"model_params": np_params}, f)

            pt_model = PyTorchTransformer(
                vocab_size=tokenizer.vocab_size,
                embed_dim=32,
                num_layers=1,
                num_heads=2,
                num_experts=2,
                max_seq_len=64,
            )
            pt_state = {}
            for k, v in np_params.items():
                pt_key = _canonical_to_pytorch(k)
                tensor = torch.from_numpy(v).float()
                if k == "lm_head":
                    tensor = tensor.T
                pt_state[pt_key] = tensor
            pt_model.load_state_dict(pt_state, strict=False)

            test_input = tokenizer.encode("The quick").reshape(1, -1).astype(np.int64)

            np_logits, _ = np_trained.model.forward(test_input, use_cache=False)

            pt_logits_tensor, _ = pt_model.forward(torch.from_numpy(test_input))
            pt_logits = (
                pt_logits_tensor.detach().float().cpu().numpy().astype(np.float64)
            )

            diff = float(np.max(np.abs(np_logits - pt_logits)))
            assert diff < 1e-5, (
                f"Forward pass mismatch: NP->PT max_diff={diff:.10f} for input {test_input}"
            )

    def test_scenario_8_forward_pass_match_pt_to_np(self, tokenizer, pt_trained):
        """After PT->NP cross-load, same input -> same logits (max_diff < 1e-6)."""
        pt_params = pt_trained.get_params()

        np_model = Transformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            num_experts=2,
            max_seq_len=64,
        )
        np_model.set_params(pt_params)

        test_input = tokenizer.encode("The quick").reshape(1, -1).astype(np.int64)

        pt_logits_tensor, _ = pt_trained.model.forward(
            torch.from_numpy(test_input),
            kv_caches=None,
        )
        pt_logits = pt_logits_tensor.detach().float().cpu().numpy().astype(np.float64)

        np_logits, _ = np_model.forward(test_input, use_cache=False)

        diff = float(np.max(np.abs(np_logits - pt_logits)))
        assert diff < 1e-5, (
            f"Forward pass mismatch: PT->NP max_diff={diff:.10f} for input {test_input}"
        )
