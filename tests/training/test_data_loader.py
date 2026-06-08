from __future__ import annotations

import numpy as np

from training.data_loader import TextDataLoader
from tokenizer.char_tokenizer import CharTokenizer


def test_data_loader_returns_valid_batches():
    np.random.seed(42)
    text = "the quick brown fox jumps over the lazy dog. " * 10
    tokenizer = CharTokenizer(text)
    loader = TextDataLoader(text, tokenizer, batch_size=2, seq_len=5)

    num_batches = 0
    for x, y in loader:
        num_batches += 1
        assert x.shape == (2, 5), f"Expected (2, 5), got {x.shape}"
        assert y.shape == (2, 5), f"Expected (2, 5), got {y.shape}"
        assert np.max(x) < tokenizer.vocab_size, "x value exceeds vocab size"
        assert np.max(y) < tokenizer.vocab_size, "y value exceeds vocab size"
        # y should be x shifted by 1 (not identical)
        assert not np.array_equal(x, y), "y should differ from x in next-token shifted data"

    assert num_batches > 0, "Data loader should produce at least one batch"


def test_data_loader_length():
    text = "a b c d e f g" * 50
    tokenizer = CharTokenizer(text)
    seq_len = 10
    batch_size = 2
    loader = TextDataLoader(text, tokenizer, batch_size=batch_size, seq_len=seq_len)

    expected_samples = len(tokenizer.encode(text)) - seq_len - 1
    expected_batches = expected_samples // batch_size
    assert len(loader) == expected_batches, (
        f"Expected {expected_batches} batches, got {len(loader)}"
    )


def test_data_loader_batch_count():
    np.random.seed(42)
    text = "hello world " * 100
    tokenizer = CharTokenizer(text)
    loader = TextDataLoader(text, tokenizer, batch_size=4, seq_len=8)

    count = sum(1 for _ in loader)
    expected = len(loader)
    assert count == expected, f"Iterator yielded {count} but __len__ says {expected}"
