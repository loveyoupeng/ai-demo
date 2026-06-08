"""Cross-backend param sharing tests."""
from __future__ import annotations

import numpy as np

from backends.numpy.numpy_backend import NumPyBackend
from backends.pytorch.pytorch_backend import PyTorchBackend


def test_cross_backend_param_keys_match():
    """Parameters from NumPy and PyTorch backends must have identical keys."""
    np.random.seed(42)

    np_backend = NumPyBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )
    pt_backend = PyTorchBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )

    np_params = np_backend.get_params()
    pt_params = pt_backend.get_params()

    assert set(np_params.keys()) == set(pt_params.keys()), (
        f"Keys do not match.\n"
        f"Missing in PT: {set(np_params.keys()) - set(pt_params.keys())}\n"
        f"Extra in PT: {set(pt_params.keys()) - set(np_params.keys())}"
    )


def test_cross_backend_param_values_transfer():
    """Transferring params from NumPy to PyTorch backend preserves values."""
    np.random.seed(42)

    np_backend = NumPyBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )
    pt_backend = PyTorchBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )

    np_params = {k: v.copy() for k, v in np_backend.get_params().items()}
    pt_backend.set_params(np_params)
    pt_params = pt_backend.get_params()

    for key in np_params:
        np.testing.assert_allclose(
            np_params[key],
            pt_params[key],
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Value mismatch for {key}",
        )


def test_cross_backend_forward_parity():
    """Forward output from both backends matches when initialized with same params."""
    np.random.seed(42)

    np_backend = NumPyBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )
    pt_backend = PyTorchBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )

    # Transfer params
    np_params = {k: v.copy() for k, v in np_backend.get_params().items()}
    pt_backend.set_params(np_params)

    # Same input
    input_ids = np.array([[0, 1, 2]], dtype=np.int64)
    np_logits, _ = np_backend.forward(input_ids)
    pt_logits, _ = pt_backend.forward(input_ids)

    np.testing.assert_allclose(
        np_logits, pt_logits, rtol=1e-6, atol=1e-6,
        err_msg="Forward outputs differ between backends",
    )


def test_cross_backend_backward_parity():
    """Backward gradients from both backends match when initialized with same params."""
    np.random.seed(42)

    np_backend = NumPyBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )
    pt_backend = PyTorchBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )

    # Transfer params
    np_params = {k: v.copy() for k, v in np_backend.get_params().items()}
    pt_backend.set_params(np_params)

    # Same input
    input_ids = np.array([[0, 1, 2]], dtype=np.int64)
    np_logits, np_cache = np_backend.forward(input_ids)
    pt_logits, pt_cache = pt_backend.forward(input_ids)

    # Same gradient
    grad_logits = np.ones((1, 3, 5), dtype=np.float64)
    np_grads = np_backend.backward(grad_logits, np_cache)
    pt_grads = pt_backend.backward(grad_logits, pt_cache)

    for key in np_grads:
        np.testing.assert_allclose(
            np_grads[key],
            pt_grads[key],
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Backward gradient mismatch for {key}",
        )


def test_cross_backend_training_loop_parity():
    """Training with both backends starting from identical params produces parities."""
    np.random.seed(42)

    np_backend = NumPyBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )
    pt_backend = PyTorchBackend(
        vocab_size=5,
        embed_dim=4,
        num_layers=1,
        num_heads=1,
        num_experts=1,
    )

    # Transfer params
    np_params = {k: v.copy() for k, v in np_backend.get_params().items()}
    pt_backend.set_params(np_params)

    # Do 1 PyTorch training step
    from optimizer import Adam
    from loss import CrossEntropyLoss
    from trainer import Trainer

    input_ids = np.array([[0, 1, 2]], dtype=np.int64)
    target_ids = np.array([[1, 2, 0]], dtype=np.int64)
    loss_fn = CrossEntropyLoss()
    pt_optimizer = Adam(learning_rate=0.01)

    pt_trainer = Trainer(pt_backend, pt_optimizer, loss_fn)
    pt_loss = pt_trainer.train_step(input_ids, target_ids)

    # Do 1 NumPy training step with same optimizer/loss
    np_optimizer = Adam(learning_rate=0.01)
    np_trainer = Trainer(np_backend, np_optimizer, loss_fn)
    np_loss = np_trainer.train_step(input_ids, target_ids)

    # Losses should match (both start from same params)
    np.testing.assert_allclose(
        np_loss, pt_loss, rtol=1e-6, atol=1e-6,
        err_msg="Training losses differ",
    )
