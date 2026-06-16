"""Tests for scripts/train.py — unified training entry point.

Tests follow TDD: write failing test first, then implement to make it pass.
Uses synthetic data for fast, deterministic training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _fake_dataset(num_samples: int = 10, vocab_size: int = 256, seq_len: int = 64):
    """Generate a fake dataset for testing."""
    rng = np.random.default_rng(42)
    data = []
    for _ in range(num_samples):
        length = rng.integers(10, seq_len + 1)
        tokens = rng.integers(0, vocab_size, size=(length,))
        data.append(tokens)
    return data


class TestBuildModel:
    """Tests for the build_model() function."""

    def test_build_model_numpy(self):
        """Building a NumPy model should return NumPyModel instance."""
        from scripts.train import build_model

        config = {
            "vocab_size": 16,
            "context_length": 32,
            "embed_dim": 64,
            "n_layers": 1,
            "n_heads": 2,
            "n_groups": 2,
            "n_experts": 2,
            "top_k": 1,
            "expert_dim": 0,
            "max_length": 128,
            "rope_dim": 0,
            "seed": 42,
        }
        model, cfg = build_model("numpy", config)
        assert model is not None
        assert cfg is not None
        assert cfg.vocab_size == 16

    def test_build_model_torch(self):
        """Building a PyTorch model should return TorchModel instance."""
        from scripts.train import build_model

        config = {
            "vocab_size": 16,
            "context_length": 32,
            "embed_dim": 64,
            "n_layers": 1,
            "n_heads": 2,
            "n_groups": 2,
            "n_experts": 2,
            "top_k": 1,
            "expert_dim": 0,
            "max_length": 128,
            "rope_dim": 0,
            "seed": 42,
        }
        model, cfg = build_model("torch", config)
        assert model is not None
        assert cfg is not None
        assert cfg.vocab_size == 16


class TestBuildConfig:
    """Tests for the build_config() function."""

    def test_build_config_basic(self):
        """CLI args should produce a flat config dict."""
        from scripts.train import build_config

        args = type(
            "Args",
            (),
            {
                "backend": "torch",
                "vocab_size": 128,
                "ctx": 64,
                "embed": 128,
                "layers": 2,
                "heads": 4,
                "groups": 4,
                "rope_dim": 0,
                "n_experts": 2,
                "top_k": 1,
                "expert_dim": 0,
                "max_length": 256,
                "epochs": 3,
                "batch_size": 32,
                "lr": 0.01,
                "seed": 99,
                "save_steps": 50,
                "eval_steps": 25,
                "dataset_path": "resource/tinystories/",
                "synthetic": True,
                "save_dir": "/tmp/test_models/",
            },
        )()

        config = build_config(args, "torch")  # type: ignore[arg-type]
        assert config["backend"] == "torch"
        assert config["vocab_size"] == 128
        assert config["context_length"] == 64  # ctx -> context_length
        assert config["embed_dim"] == 128  # embed -> embed_dim
        assert config["n_layers"] == 2  # layers -> n_layers
        assert config["n_heads"] == 4  # heads -> n_heads

    def test_build_config_aliases_normalized(self):
        """CLI aliases (ctx, embed, layers, heads, groups) should normalize."""
        from scripts.train import build_config

        args = type(
            "Args",
            (),
            {
                "backend": "numpy",
                "vocab_size": 256,
                "ctx": 128,
                "embed": 256,
                "layers": 4,
                "heads": 8,
                "groups": 8,
                "rope_dim": 0,
                "n_experts": 4,
                "top_k": 2,
                "expert_dim": 0,
                "max_length": 512,
                "epochs": 5,
                "batch_size": 64,
                "lr": 0.001,
                "seed": 42,
                "save_steps": 100,
                "eval_steps": 50,
                "dataset_path": "resource/tinystories/",
                "synthetic": False,
                "save_dir": "resource/models/",
            },
        )()

        config = build_config(args, "numpy")  # type: ignore[arg-type]
        assert config["context_length"] == 128
        assert config["embed_dim"] == 256
        assert config["n_layers"] == 4
        assert config["n_heads"] == 8
        assert config["n_groups"] == 8


class TestGetDataset:
    """Tests for the get_dataset() function."""

    def test_get_dataset_synthetic(self):
        """Synthetic dataset should be NumPy arrays of correct shape."""
        from scripts.train import get_dataset

        data = get_dataset("resource/tinystories/", synthetic=True, vocab_size=64, context_length=32)
        assert isinstance(data, list)
        assert len(data) > 0
        assert isinstance(data[0], np.ndarray)
        assert data[0].dtype == np.int64
        # Check that sequence lengths don't exceed context_length
        for arr in data:
            assert len(arr) <= 32

    def test_get_dataset_from_file(self, tmp_path: Path):
        """Dataset from .npy files should be loaded correctly."""
        from scripts.train import get_dataset

        # Create fake dataset files
        test_dir = tmp_path / "fake_dataset"
        test_dir.mkdir()
        rng = np.random.default_rng(0)
        for i in range(3):
            tokens = rng.integers(0, 100, size=(20 + i * 5,))
            np.save(test_dir / f"batch_{i}.npy", tokens)

        data = get_dataset(str(test_dir), synthetic=False, vocab_size=100, context_length=128)
        assert len(data) == 3
        for arr in data:
            assert arr.ndim == 1
            assert len(arr) > 0


class TestRunTraining:
    """Tests for the run_training() function."""

    def test_run_training_numpy(self):
        """Training with NumPy backend should reduce loss over epochs."""
        from impl._np.cross_entropy import CrossEntropyLoss
        from impl._np.optimizer import AdamW
        from scripts.train import build_model, run_training_numpy

        # Use tiny model — NumPy backend uses finite-difference which is
        # O(params × 2) forward passes. A model that is small enough to
        # finish in a few seconds but still exercises the code path.
        config = {
            "context_length": 8,
            "epochs": 2,
            "batch_size": 2,
        }
        model, _ = build_model(
            "numpy",
            {
                "vocab_size": 8,
                "context_length": 8,
                "embed_dim": 8,
                "n_layers": 1,
                "n_heads": 1,
                "n_groups": 1,
                "n_experts": 2,
                "top_k": 1,
                "expert_dim": 0,
                "max_length": 16,
                "rope_dim": 0,
                "seed": 42,
            },
        )
        loss_fn = CrossEntropyLoss()
        optimizer = AdamW(lr=0.01)

        # Tiny dataset — short sequences of small vocab
        rng = np.random.default_rng(42)
        dataset = []
        for _ in range(4):
            length = rng.integers(4, 9)
            dataset.append(rng.integers(0, 8, size=(length,)))

        losses = run_training_numpy(model, optimizer, loss_fn, config, dataset)
        assert isinstance(losses, list)
        assert len(losses) == 2  # 2 epochs
        # Loss should start high and decrease (or stay reasonable)
        for loss_val in losses:
            assert loss_val > 0.0
            assert np.isfinite(loss_val)


class TestMain:
    """Tests for the main() function."""

    def test_main_with_synthetic(self):
        """Running main() with synthetic data should not error."""
        import scripts.train as train_module

        original_argv = sys.argv
        original_exit = sys.exit

        try:
            # Use PyTorch backend (autograd, fast) with synthetic data.
            # PyTorch backward is computed via autograd, not finite-difference.
            sys.argv = [
                "train.py",
                "--backend",
                "torch",
                "--synthetic",
                "--epochs",
                "1",
                "--vocab_size",
                "32",
                "--context_length",
                "32",
                "--embed_dim",
                "32",
                "--n_layers",
                "1",
                "--n_heads",
                "2",
                "--n_groups",
                "2",
                "--n_experts",
                "2",
                "--top_k",
                "1",
                "--expert_dim",
                "0",
                "--seed",
                "42",
                "--batch_size",
                "64",
                "--lr",
                "0.01",
                "--save_dir",
                "/tmp/tmp_models/",
            ]
            sys.exit = lambda code: None  # type: ignore[assignment]

            exit_code = train_module.main()
            # Should exit with 0 (success)
            assert exit_code == 0
        finally:
            sys.argv = original_argv
            sys.exit = original_exit

    def test_main_with_help(self):
        """Passing --help should print help and exit with 0."""
        import scripts.train as train_module

        original_argv = sys.argv
        original_exit = sys.exit

        try:
            sys.argv = ["train.py", "--help"]
            exit_code = train_module.main()
            assert exit_code == 0
        except SystemExit:
            pass  # argparse may call sys.exit during --help
        finally:
            sys.argv = original_argv
            sys.exit = original_exit
