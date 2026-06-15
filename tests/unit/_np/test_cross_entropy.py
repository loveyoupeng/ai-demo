"""Tests for Cross-Entropy Loss module."""

import math

import numpy as np


class TestCrossEntropyLossForward:
    """Tests for CrossEntropyLoss.forward()."""

    def test_scalar_output(self):
        """Loss is a scalar (0-D array or Python float)."""
        from impl._np.cross_entropy import CrossEntropyLoss

        np.random.seed(42)
        logits = np.random.randn(2, 8, 16).astype(np.float32)
        targets = np.random.randint(0, 16, size=(2, 8)).astype(np.int64)

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn.forward(logits, targets)

        assert isinstance(loss, (float, np.floating)) or np.ndim(loss) == 0

    def test_uniform_logits(self):
        """With uniform logits, loss ~= log(V)."""
        from impl._np.cross_entropy import CrossEntropyLoss

        logits = np.zeros((1, 4, 8), dtype=np.float64)
        targets = np.array([[0, 1, 2, 3]], dtype=np.int64)

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn.forward(logits, targets)

        expected = math.log(8)
        assert abs(loss - expected) / expected < 1e-3

    def test_masking(self):
        """Masked positions contribute zero to loss."""
        from impl._np.cross_entropy import CrossEntropyLoss

        np.random.seed(7)
        logits = np.random.randn(1, 6, 8).astype(np.float64)
        targets = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float64)

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn.forward(logits, targets, mask=mask)

        # Manually compute: only positions 0, 2, 4 (where mask=1) contribute
        positions = [0, 2, 4]
        m_targets = targets[0, positions]  # (3,)
        m_logits = logits[0, positions]  # (3, 8)
        log_softmax = m_logits - np.log(np.sum(np.exp(m_logits), axis=-1, keepdims=True))
        per_position_losses = -log_softmax[np.arange(3), m_targets]  # (3,)
        expected = float(np.mean(per_position_losses))

        assert np.isclose(loss, expected, rtol=1e-4)

    def test_perfect_predictions(self):
        """If logits are one-hot at correct target, loss ~= 0."""
        from impl._np.cross_entropy import CrossEntropyLoss

        B, S, V = 1, 4, 8
        logits = np.full((B, S, V), -1e9, dtype=np.float64)
        targets = np.array([[0, 1, 2, 3]], dtype=np.int64)

        # Set correct token logits to a large value
        for b in range(B):
            for s in range(S):
                logits[b, s, targets[b, s]] = 1e9

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn.forward(logits, targets)

        assert loss < 1e-3
