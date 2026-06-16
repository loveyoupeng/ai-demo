"""C6.1: Tests for PyTorch Cross-Entropy Loss.

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import math

import torch


class TestCrossEntropyLossForward:
    """Test the CrossEntropyLoss nn.Module forward pass."""

    def test_scalar_output(self) -> None:
        """Loss is a scalar tensor (0-D)."""
        from impl._torch.cross_entropy import CrossEntropyLoss

        B, S, V = 2, 8, 16
        logits = torch.randn(B, S, V, dtype=torch.float64)
        targets = torch.randint(0, V, size=(B, S), dtype=torch.int64)

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn(logits, targets)

        assert loss.ndim == 0, "Loss must be scalar (0-D tensor)"
        assert loss.dtype == torch.float64
        assert torch.isfinite(loss)
        assert loss >= 0.0

    def test_uniform_logits(self) -> None:
        """With uniform logits, loss ~= log(V) (maximum entropy)."""
        from impl._torch.cross_entropy import CrossEntropyLoss

        V = 8
        logits = torch.zeros(1, 4, V, dtype=torch.float64)
        targets = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn(logits, targets)

        expected = math.log(V)
        assert abs(loss.item() - expected) / expected < 1e-3

    def test_masking(self) -> None:
        """Masked positions contribute zero to loss."""
        from impl._torch.cross_entropy import CrossEntropyLoss

        torch.manual_seed(7)
        logits = torch.randn(1, 6, 8, dtype=torch.float64)
        targets = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int64)
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn(logits, targets, mask)

        # Manually compute: only positions 0, 2, 4 (where mask=1) contribute
        positions = [0, 2, 4]
        m_targets = targets[0, positions]
        m_logits = logits[0, positions]
        log_softmax = m_logits - torch.logsumexp(m_logits, dim=-1, keepdim=True)
        per_position_losses = -log_softmax[torch.arange(3), m_targets]
        expected = per_position_losses.mean().item()

        assert torch.isclose(torch.tensor(loss.item()), torch.tensor(expected), rtol=1e-4)

    def test_perfect_predictions(self) -> None:
        """If logits are one-hot at correct target, loss ~= 0."""
        from impl._torch.cross_entropy import CrossEntropyLoss

        B, S, V = 1, 4, 8
        logits = torch.full((B, S, V), -1e9, dtype=torch.float64)
        targets = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)

        for b in range(B):
            for s in range(S):
                logits[b, s, targets[b, s]] = 1e9

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn(logits, targets)

        assert loss.item() < 1e-3

    def test_ignore_index(self) -> None:
        """Positions with ignore_index (-100) do not contribute to loss."""
        from impl._torch.cross_entropy import CrossEntropyLoss

        V = 8
        logits = torch.randn(1, 4, V, dtype=torch.float64)
        targets = torch.tensor([[0, -100, 2, -100]], dtype=torch.int64)

        loss_fn = CrossEntropyLoss(shift=False)
        loss = loss_fn(logits, targets)

        # Only positions 0 and 2 contribute
        m_logits = logits[0, [0, 2]]
        m_targets = targets[0, [0, 2]]
        log_softmax = m_logits - torch.logsumexp(m_logits, dim=-1, keepdim=True)
        per_position_losses = -log_softmax[torch.arange(2), m_targets]
        expected = per_position_losses.mean().item()

        assert torch.isclose(torch.tensor(loss.item()), torch.tensor(expected), rtol=1e-4)

    def test_shift(self) -> None:
        """With shift=True, logits[:, :-1] predicts targets[:, 1:]."""
        from impl._torch.cross_entropy import CrossEntropyLoss

        torch.manual_seed(42)
        V = 8
        S = 6
        logits = torch.randn(1, S, V, dtype=torch.float64)
        targets = torch.randint(0, V, size=(1, S), dtype=torch.int64)

        loss_fn = CrossEntropyLoss(shift=True)
        loss = loss_fn(logits, targets)

        # Manual: shift logits and targets, then compute cross-entropy
        shifted_logits = logits[:, :-1]  # (1, S-1, V)
        shifted_targets = targets[:, 1:]  # (1, S-1)

        log_softmax = shifted_logits - torch.logsumexp(shifted_logits, dim=-1, keepdim=True)
        per_pos = -log_softmax[0, torch.arange(S - 1), shifted_targets[0]]
        expected = per_pos.mean().item()

        assert torch.isclose(torch.tensor(loss.item()), torch.tensor(expected), rtol=1e-4)
