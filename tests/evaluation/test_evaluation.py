from __future__ import annotations

import numpy as np
import pytest
from evaluation import calculate_perplexity, calculate_accuracy


def test_calculate_perplexity():
    # If loss is 0, perplexity should be exp(0) = 1
    assert calculate_perplexity(0.0) == pytest.approx(1.0)

    # If loss is ln(2), perplexity should be 2
    assert calculate_perplexity(np.log(2)) == pytest.approx(2.0)

    # Test with a larger loss
    assert calculate_perplexity(2.3) == pytest.approx(np.exp(2.3))


def test_calculate_accuracy():
    # Shape: [Batch=2, Seq_Len=3, Vocab_Size=4]
    logits = np.array(
        [
            [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]],  # Predictions: 0, 1, 2
            [[0, 0, 0, 10], [10, 0, 0, 0], [0, 10, 0, 0]],  # Predictions: 3, 0, 1
        ]
    )

    # Correct targets
    targets = np.array([[0, 1, 2], [3, 0, 1]])
    assert calculate_accuracy(logits, targets) == 1.0

    # Some incorrect targets
    targets_mixed = np.array(
        [
            [0, 1, 1],  # One wrong at index (0, 2)
            [3, 0, 1],
        ]
    )
    # 5 correct out of 6 total
    assert calculate_accuracy(logits, targets_mixed) == pytest.approx(5 / 6)

    # All incorrect
    targets_wrong = np.array([[3, 2, 1], [1, 2, 3]])
    assert calculate_accuracy(logits, targets_wrong) == 0.0
