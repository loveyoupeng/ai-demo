from __future__ import annotations

import numpy as np

from backends.numpy.numpy_backend import NumPyBackend
from loss import CrossEntropyLoss
from optimizer import SGD
from trainer import Trainer


def test_training_loop_reduces_loss():
    """
    Test that running multiple training steps on the NumPy backend
    with SGD actually reduces the loss over time.
    """
    np.random.seed(42)
    vocab_size = 10
    embed_dim = 8
    num_layers = 1
    num_heads = 1
    num_experts = 1
    max_seq_len = 16
    batch_size = 2
    seq_len = 5

    backend = NumPyBackend(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )

    optimizer = SGD(learning_rate=0.1)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(backend, optimizer, loss_fn)

    # Generate a fixed batch of data to overfit on
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    target_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Train for 50 steps (see task_plan.md Phase 5)
    losses = []
    for i in range(50):
        loss = trainer.train_step(input_ids, target_ids)
        losses.append(loss)

    # Loss should decrease
    assert losses[-1] < losses[0], (
        f"Loss did not decrease after training. "
        f"Initial loss: {losses[0]}. Final loss: {losses[-1]}"
    )

    # After 10 steps, the reduction should be noticeable (at least 10%)
    reduction = (losses[0] - losses[-1]) / losses[0]
    assert reduction > 0.1, f"Loss reduction too small after 10 steps: {reduction:.2%}"


def test_gradient_clipping():
    """
    Test that gradient clipping bounds the gradient norm at clip_value.
    Uses very large learning rate (1.0) to generate exploding gradients.

    We capture the raw gradients from train_step by monkeypatching optimizer.step().
    """
    np.random.seed(42)
    vocab_size = 10
    embed_dim = 16
    num_layers = 2
    num_heads = 2
    num_experts = 2
    max_seq_len = 32
    batch_size = 2
    seq_len = 10

    # Build two backends with same initial params
    backend1 = NumPyBackend(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )
    backend2 = NumPyBackend(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )

    # Copy initial params so both start identical
    params = backend1.get_params()
    backend2.set_params({k: v.copy() for k, v in params.items()})

    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    target_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    loss_fn = CrossEntropyLoss()

    # Without clipping
    captured_grads_no_clip = []

    def capture_no_clip(params, grads):
        captured_grads_no_clip.append({k: v.copy() for k, v in grads.items()})
        return None

    no_clip_trainer = Trainer(backend1, SGD(learning_rate=1.0), loss_fn)
    no_clip_trainer.optimizer.step = lambda p, g: capture_no_clip(p, g)
    no_clip_trainer.train_step(input_ids, target_ids)

    # With clipping
    captured_grads_clipped = []
    clip_value = 0.5

    def capture_clipped(params, grads):
        captured_grads_clipped.append({k: v.copy() for k, v in grads.items()})
        return None

    clipped_trainer = Trainer(
        backend2, SGD(learning_rate=1.0), loss_fn, clip_value=clip_value
    )
    clipped_trainer.optimizer.step = lambda p, g: capture_clipped(p, g)
    clipped_trainer.train_step(input_ids, target_ids)

    # Calculate gradient norms
    grad_norm_no_clip = np.sqrt(
        sum(np.sum(g**2) for g in captured_grads_no_clip[0].values())
    )
    grad_norm_clipped = np.sqrt(
        sum(np.sum(g**2) for g in captured_grads_clipped[0].values())
    )

    # Verify clipping was applied (clip_value < natural norm triggers clipping)
    if grad_norm_no_clip > clip_value:
        assert grad_norm_clipped <= clip_value + 1e-5, (
            f"Gradient norm {grad_norm_clipped:.4f} exceeds clip_value {clip_value}"
        )
        assert grad_norm_clipped < grad_norm_no_clip, (
            f"Clipped norm ({grad_norm_clipped:.2f}) should be < no-clip norm ({grad_norm_no_clip:.2f})"
        )
    else:
        # No clipping needed — norms should be identical
        np.testing.assert_allclose(grad_norm_no_clip, grad_norm_clipped, rtol=1e-10)


def test_no_gradient_clipping_when_clip_value_none():
    """
    When clip_value is None (default), gradients should NOT be modified.
    """
    np.random.seed(42)
    vocab_size = 10
    embed_dim = 8
    num_layers = 1
    num_heads = 1
    num_experts = 1
    max_seq_len = 16
    batch_size = 2
    seq_len = 5

    backend1 = NumPyBackend(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )
    backend2 = NumPyBackend(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )

    params = backend1.get_params()
    backend2.set_params({k: v.copy() for k, v in params.items()})

    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    target_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    loss_fn = CrossEntropyLoss()

    captured_grads = []

    def capture_all(p, g):
        captured_grads.append({k: v.copy() for k, v in g.items()})
        return None

    # One with clip_value=None, one with clip_value=1000 (effective no-clip)
    trainer1 = Trainer(backend1, SGD(learning_rate=0.01), loss_fn, clip_value=None)
    trainer1.optimizer.step = lambda p, g: capture_all(p, g)

    trainer2 = Trainer(backend2, SGD(learning_rate=0.01), loss_fn, clip_value=1000)
    trainer2.optimizer.step = lambda p, g: capture_all(p, g)

    trainer1.train_step(input_ids, target_ids)
    captured_without = captured_grads.pop()

    captured_grads.clear()
    trainer2.train_step(input_ids, target_ids)
    captured_with = captured_grads[0]

    # Gradients should be identical (clipping only applies when norm exceeds clip_value)
    for key in captured_without:
        np.testing.assert_allclose(
            captured_without[key], captured_with[key], rtol=1e-10
        )
