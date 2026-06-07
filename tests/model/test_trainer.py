from __future__ import annotations

import numpy as np
from model.transformer import Transformer
from optimizer import Adam
from loss import CrossEntropyLoss
from trainer import Trainer


def test_trainer_train_step():
    """
    Integration test for Trainer.train_step.
    Verifies that a single training step runs and produces a valid loss.
    """
    np.random.seed(42)
    vocab_size = 10
    embed_dim = 8
    num_layers = 1
    num_heads = 2
    num_experts = 2
    max_seq_len = 20
    batch_size = 2
    seq_len = 5

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )

    optimizer = Adam(learning_rate=0.01)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    # Dummy data
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    target_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Initial loss
    logits, _ = model.forward(input_ids)
    initial_loss, _ = loss_fn.forward(logits, target_ids)

    # Perform one training step
    loss = trainer.train_step(input_ids, target_ids)

    # Check if loss is a float and positive
    assert isinstance(loss, float)
    assert loss > 0


def test_trainer_parameter_update():
    """
    Verifies that all parameters are actually updated after a training step.
    """
    np.random.seed(42)
    vocab_size = 10
    embed_dim = 8
    num_layers = 1
    num_heads = 2
    num_experts = 2
    max_seq_len = 20
    batch_size = 2
    seq_len = 5

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )

    optimizer = Adam(learning_rate=0.01)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    # Dummy data
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    target_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Capture initial parameters
    initial_params = {k: v.copy() for k, v in model.get_params().items()}

    # Perform one training step
    trainer.train_step(input_ids, target_ids)

    # Check if parameters have changed
    updated_params = model.get_params()
    for k in initial_params:
        assert not np.array_equal(initial_params[k], updated_params[k]), (
            f"Parameter {k} did not change"
        )


def test_trainer_loss_reduction():
    """
    Verifies that loss decreases significantly on a very simple task (overfitting a single batch).
    """
    np.random.seed(42)
    vocab_size = 5
    embed_dim = 4
    num_layers = 1
    num_heads = 1
    num_experts = 1
    max_seq_len = 10

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )

    optimizer = Adam(learning_rate=0.1)  # Higher LR for faster convergence in test
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    input_ids = np.array([[0, 1, 2]], dtype=int)
    target_ids = np.array([[1, 2, 0]], dtype=int)

    # Perform multiple steps
    initial_loss = trainer.train_step(input_ids, target_ids)
    loss = initial_loss
    for _ in range(50):
        loss = trainer.train_step(input_ids, target_ids)

    # Check for significant reduction (at least 50%)
    reduction_ratio = initial_loss / loss if loss > 0 else float("inf")
    assert reduction_ratio > 1.5, (
        f"Loss did not decrease significantly. Initial: {initial_loss}, Final: {loss}, Ratio: {reduction_ratio:.2f}"
    )


def test_trainer_fit_small_batch():
    """
    Test that Trainer.fit runs on a small dummy data loader.
    """
    np.random.seed(42)
    vocab_size = 10
    embed_dim = 8
    num_layers = 1
    num_heads = 2
    num_experts = 2
    max_seq_len = 20
    batch_size = 2
    seq_len = 5

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )

    optimizer = Adam(learning_rate=0.01)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    # Dummy data loader
    class DummyDataLoader:
        def __init__(self, num_batches, batch_size, seq_len, vocab_size):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __iter__(self):
            for _ in range(self.num_batches):
                input_ids = np.random.randint(
                    0, self.vocab_size, size=(self.batch_size, self.seq_len)
                )
                target_ids = np.random.randint(
                    0, self.vocab_size, size=(self.batch_size, self.seq_len)
                )
                yield input_ids, target_ids

        def __len__(self):
            return self.num_batches

    data_loader = DummyDataLoader(
        num_batches=5, batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size
    )

    # Should run without error
    trainer.fit(data_loader, epochs=1)

    # Check history
    assert len(trainer.history["loss"]) == 1
