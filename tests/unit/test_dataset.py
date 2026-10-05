"""Tests for shared.dataset — TinyStories dataset loading and batching.

Tests cover: load_tinystories(), TextDataset class.
All data loaded from local resource files with no external downloads.
"""

from __future__ import annotations


class _StubTokenizer:
    """Minimal TokenizerLike stub: deterministic byte-level encoding."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) % 256 for ch in text]


class TestLoadTinyStories:
    """Test loading TinyStories dataset."""

    def test_load_training_data(self):
        """Loading train split returns non-empty list of stories."""
        from shared.dataset import load_tinystories

        data = load_tinystories("train")
        assert isinstance(data, list)
        assert len(data) > 0
        # First item should be a non-empty string
        assert isinstance(data[0], str)
        assert len(data[0]) > 0

    def test_load_validation_data(self):
        """Loading validation split returns non-empty list."""
        from shared.dataset import load_tinystories

        data = load_tinystories("validation")
        assert isinstance(data, list)
        assert len(data) > 0

    def test_load_returns_strings(self):
        """All loaded stories should be strings."""
        from shared.dataset import load_tinystories

        data = load_tinystories("train")
        for story in data:
            assert isinstance(story, str)


class TestTextDataset:
    """Test the TextDataset wrapper class."""

    def test_dataset_initialization(self):
        """Dataset initializes by concatenating tokenized text."""
        from shared.dataset import TextDataset, load_tinystories

        stories = load_tinystories("train")[:100]  # small subset for speed
        tok = _StubTokenizer()
        ds = TextDataset(stories, tok, context_length=32, seed=42)
        # Should have concatenated token IDs
        assert len(ds.token_ids) > 0
        assert len(ds.token_ids) >= 32  # at least one full window

    def test_get_sequences(self):
        """get_sequences returns correct number of (input, target) pairs."""
        from shared.dataset import TextDataset, load_tinystories

        stories = load_tinystories("train")[:100]
        tok = _StubTokenizer()
        ds = TextDataset(stories, tok, context_length=16, seed=42)

        seqs = ds.get_sequences(5, batch_size=3)
        assert len(seqs) == 5
        for inp, tgt in seqs:
            assert len(inp) == 3  # batch_size
            assert len(tgt) == 3  # matches input count

    def test_get_sequences_context_length(self):
        """Each sample is exactly context_length tokens."""
        from shared.dataset import TextDataset, load_tinystories

        stories = load_tinystories("train")[:100]
        tok = _StubTokenizer()

        for ctx_len in [16, 32, 64, 128]:
            ds = TextDataset(stories, tok, context_length=ctx_len, seed=42)
            seqs = ds.get_sequences(10, batch_size=5)
            for inp, _tgt in seqs:
                for window in inp:
                    assert len(window) == ctx_len
                for target in _tgt:
                    assert len(target) == ctx_len

    def test_get_sequences_context_length_custom(self):
        """Context length 1 produces minimal windows."""
        from shared.dataset import TextDataset, load_tinystories

        stories = load_tinystories("train")[:100]
        tok = _StubTokenizer()
        ds = TextDataset(stories, tok, context_length=1, seed=42)
        seqs = ds.get_sequences(3, batch_size=2)
        assert len(seqs) == 3
        for inp, _tgt in seqs:
            assert len(inp) == 2
            for w in inp:
                assert len(w) == 1

    def test_text_dataset_skip_empty_stories(self):
        """Empty stories are skipped during initialization."""
        from shared.dataset import TextDataset

        stories = ["Once upon", "", "   ", "The cat sat"]
        tok = _StubTokenizer()
        ds = TextDataset(stories, tok, context_length=8, seed=42)
        # Should still have tokens from non-empty stories
        assert len(ds.token_ids) > 0
