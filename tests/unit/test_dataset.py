"""Tests for shared.dataset — TinyStories dataset loading and batching.

Tests cover: load_tinystories(), TextDataset class, get_dataloader_sequences().
The TinyStories dataset downloads on first use (~1 min).
"""



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
        from shared.tokenizer import create_tokenizer

        stories = load_tinystories("train")[:100]  # small subset for speed
        tok = create_tokenizer()
        ds = TextDataset(stories, tok, context_length=32, seed=42)
        # Should have concatenated token IDs
        assert len(ds.token_ids) > 0
        assert len(ds.token_ids) >= 32  # at least one full window

    def test_get_sequences(self):
        """get_sequences returns correct number of (input, target) pairs."""
        from shared.dataset import TextDataset, load_tinystories
        from shared.tokenizer import create_tokenizer

        stories = load_tinystories("train")[:100]
        tok = create_tokenizer()
        ds = TextDataset(stories, tok, context_length=16, seed=42)

        seqs = ds.get_sequences(5, batch_size=3)
        assert len(seqs) == 5
        for inp, tgt in seqs:
            assert len(inp) == 3  # batch_size
            assert len(tgt) == 3  # matches input count

    def test_get_sequences_context_length(self):
        """Each sample is exactly context_length tokens."""
        from shared.dataset import TextDataset, load_tinystories
        from shared.tokenizer import create_tokenizer

        stories = load_tinystories("train")[:100]
        tok = create_tokenizer()

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
        from shared.tokenizer import create_tokenizer

        stories = load_tinystories("train")[:100]
        tok = create_tokenizer()
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
        from shared.tokenizer import create_tokenizer

        stories = ["Once upon", "", "   ", "The cat sat"]
        tok = create_tokenizer()
        ds = TextDataset(stories, tok, context_length=8, seed=42)
        # Should still have tokens from non-empty stories
        assert len(ds.token_ids) > 0


class TestDataloaderSequences:
    """Test the get_dataloader_sequences helper function."""

    def test_batch_creation(self):
        """Should produce correctly shaped batches."""
        from shared.dataset import (
            TextDataset,
            get_dataloader_sequences,
            load_tinystories,
        )
        from shared.tokenizer import create_tokenizer

        stories = load_tinystories("train")[:50]
        tok = create_tokenizer()
        ds = TextDataset(stories, tok, context_length=32, seed=42)

        batches = get_dataloader_sequences(ds, batch_size=4, num_batches=2)
        assert len(batches) == 2

        input_batch, target_batch = batches[0]
        assert len(input_batch) == 4  # batch_size
        assert len(target_batch) == 4  # matches input count

    def test_batch_input_target_length_match(self):
        """Input and target sequences have same shape."""
        from shared.dataset import (
            TextDataset,
            get_dataloader_sequences,
            load_tinystories,
        )
        from shared.tokenizer import create_tokenizer

        stories = load_tinystories("train")[:50]
        tok = create_tokenizer()
        ds = TextDataset(stories, tok, context_length=64, seed=42)

        batches = get_dataloader_sequences(ds, batch_size=8, num_batches=1)
        input_batch, target_batch = batches[0]

        for inp_seq, tgt_seq in zip(input_batch, target_batch, strict=True):
            assert len(inp_seq) == len(tgt_seq)
            for _i, (inp_token, tgt_token) in enumerate(zip(inp_seq, tgt_seq, strict=True)):
                assert isinstance(inp_token, int)
                assert isinstance(tgt_token, int)

    def test_multiple_batches(self):
        """num_batches controls the number of returned batches."""
        from shared.dataset import (
            TextDataset,
            get_dataloader_sequences,
            load_tinystories,
        )
        from shared.tokenizer import create_tokenizer

        stories = load_tinystories("train")[:50]
        tok = create_tokenizer()
        ds = TextDataset(stories, tok, context_length=32, seed=42)

        batches = get_dataloader_sequences(ds, batch_size=2, num_batches=5)
        assert len(batches) == 5

    def test_batch_shape_matches_batch_size(self):
        """Each batch has exactly batch_size sequences."""
        from shared.dataset import (
            TextDataset,
            get_dataloader_sequences,
            load_tinystories,
        )
        from shared.tokenizer import create_tokenizer

        stories = load_tinystories("train")[:50]
        tok = create_tokenizer()
        ds = TextDataset(stories, tok, context_length=32, seed=42)

        for batch_size in [1, 4, 8, 16]:
            batches = get_dataloader_sequences(
                ds, batch_size=batch_size, num_batches=1
            )
            assert len(batches[0][0]) == batch_size  # input batch
            assert len(batches[0][1]) == batch_size  # target batch
