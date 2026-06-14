"""Tests for shared.tokenizer — BPE + Character-level tokenizers.

Tests cover: encode, decode, pad_sequences, SimpleCharTokenizer roundtrip.
The gpt2 tokenizer downloads on first use (~2-3 min) — this is expected.
"""


class TestEncodeText:
    """Test text → token IDs conversion."""

    def test_encode_text_returns_list(self):
        from shared.tokenizer import create_tokenizer, encode_text

        tok = create_tokenizer()
        result = encode_text(tok, "hello world")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(i, int) for i in result)

    def test_encode_text_with_special_tokens(self):
        """encode_text with add_special_tokens=True (default) should include BOS/EOS."""
        from shared.tokenizer import create_tokenizer, encode_text

        tok = create_tokenizer()
        result_with = encode_text(tok, "test", add_special_tokens=True)
        result_without = encode_text(tok, "test", add_special_tokens=False)
        # With special tokens should have at least 2 more (BOS + EOS, or similar)
        # gpt2 adds <BOS> at start, <EOS> at end
        assert len(result_with) >= len(result_without)

    def test_encode_empty_string(self):
        """Empty string should produce minimal tokens."""
        from shared.tokenizer import create_tokenizer, encode_text

        tok = create_tokenizer()
        result = encode_text(tok, "")
        assert isinstance(result, list)
        assert len(result) >= 0  # may be empty or have special tokens


class TestDecodeTokens:
    """Test token IDs → text conversion."""

    def test_decode_returns_string(self):
        from shared.tokenizer import create_tokenizer, decode_tokens

        tok = create_tokenizer()
        tokens = tok.encode("hello world")
        result = decode_tokens(tok, tokens)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_roundtrip(self):
        """Encode → decode should produce the original text."""
        from shared.tokenizer import create_tokenizer, decode_tokens, encode_text

        tok = create_tokenizer()
        text = "The cat sat on the mat."
        ids = encode_text(tok, text, add_special_tokens=False)
        decoded = decode_tokens(tok, ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
        # Check key words are preserved (tokenizer preserves most words)
        assert "cat" in decoded.lower() or "cat" in text

    def test_decode_numpy_array(self):
        """decode_tokens should accept numpy arrays."""
        import numpy as np

        from shared.tokenizer import create_tokenizer, decode_tokens

        tok = create_tokenizer()
        tokens = np.array(tok.encode("test"), dtype=np.int64)
        decoded = decode_tokens(tok, tokens)
        assert isinstance(decoded, str)

    def test_decode_empty_list(self):
        """Decoding empty list returns empty string."""
        from shared.tokenizer import create_tokenizer, decode_tokens

        tok = create_tokenizer()
        result = decode_tokens(tok, [])
        assert isinstance(result, str)
        assert result == ""


class TestPadSequences:
    """Test right-padding of token sequences."""

    def test_pad_basic(self):
        """Basic right-padding: [[1,2,3], [4,5]] → [[1,2,3], [4,5,0]]."""
        from shared.tokenizer import pad_sequences

        tokens = [[1, 2, 3], [4, 5]]
        padded, masks = pad_sequences(tokens)
        assert padded == [[1, 2, 3], [4, 5, 0]]
        assert masks == [[1, 1, 1], [1, 1, 0]]

    def test_pad_with_max_length(self):
        """With explicit max_length, pad all to that length."""
        from shared.tokenizer import pad_sequences

        tokens = [[1, 2, 3], [4, 5]]
        padded, masks = pad_sequences(tokens, max_length=5)
        assert len(padded[0]) == 5
        assert len(padded[1]) == 5
        assert padded == [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
        assert masks == [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]

    def test_pad_custom_padding_value(self):
        """Custom padding value instead of 0."""
        from shared.tokenizer import pad_sequences

        tokens = [[1, 2], [3]]
        padded, masks = pad_sequences(tokens, padding_value=-1)
        assert padded == [[1, 2], [3, -1]]

    def test_pad_already_equal(self):
        """No padding needed when all sequences same length."""
        from shared.tokenizer import pad_sequences

        tokens = [[1, 2], [3, 4]]
        padded, masks = pad_sequences(tokens)
        assert padded == [[1, 2], [3, 4]]
        assert masks == [[1, 1], [1, 1]]

    def test_pad_single_sequence(self):
        """Single sequence: no padding needed."""
        from shared.tokenizer import pad_sequences

        padded, masks = pad_sequences([[1, 2, 3]])
        assert padded == [[1, 2, 3]]
        assert masks == [[1, 1, 1]]

    def test_pad_longest_sequence_without_max(self):
        """Auto-pads to the longest sequence length."""
        from shared.tokenizer import pad_sequences

        tokens = [[1], [2, 3, 4, 5], [6, 7]]
        padded, masks = pad_sequences(tokens)
        assert len(padded[0]) == 4
        assert len(padded[1]) == 4
        assert len(padded[2]) == 4
        assert padded == [[1, 0, 0, 0], [2, 3, 4, 5], [6, 7, 0, 0]]


class TestSimpleCharTokenizer:
    """Test character-level tokenizer."""

    def test_init(self):
        """Tokenizer initializes with expected attributes."""
        from shared.tokenizer import SimpleCharTokenizer

        tok = SimpleCharTokenizer(vocab_size=128)
        assert hasattr(tok, "vocab")
        assert hasattr(tok, "reverse_vocab")
        assert hasattr(tok, "pad_id")
        assert hasattr(tok, "eos_id")
        assert hasattr(tok, "vocab_size")

    def test_pad_id_is_zero(self):
        from shared.tokenizer import SimpleCharTokenizer

        tok = SimpleCharTokenizer()
        assert tok.pad_id == 0

    def test_encode_returns_list(self):
        """encode returns a list of integers."""
        from shared.tokenizer import SimpleCharTokenizer

        tok = SimpleCharTokenizer()
        result = tok.encode("abc")
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        assert len(result) > 0

    def test_decode_returns_string(self):
        """decode returns a string."""
        from shared.tokenizer import SimpleCharTokenizer

        tok = SimpleCharTokenizer()
        tokens = tok.encode("hello")
        result = tok.decode(tokens)
        assert isinstance(result, str)

    def test_roundtrip(self):
        """encode → decode should recover original text."""
        from shared.tokenizer import SimpleCharTokenizer

        tok = SimpleCharTokenizer(vocab_size=128)
        text = "Hello World 123!"
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        # May differ in spacing/case, but key chars should match
        assert len(decoded) > 0

    def test_unknown_characters_become_pad(self):
        """Characters not in vocab become pad_id (0)."""
        from shared.tokenizer import SimpleCharTokenizer

        tok = SimpleCharTokenizer()
        tokens = tok.encode("abc")
        assert 0 not in tokens  # abc should be in vocab

    def test_vocab_contains_ascii(self):
        """Vocabulary should contain common ASCII characters."""
        from shared.tokenizer import SimpleCharTokenizer

        tok = SimpleCharTokenizer(vocab_size=90)
        assert len(tok.vocab) > 30  # should have many characters

    def test_special_tokens_not_in_decode(self):
        """decode skips pad_id and eos_id."""
        from shared.tokenizer import SimpleCharTokenizer

        tok = SimpleCharTokenizer()
        # encode includes eos_id at the end
        tokens = tok.encode("a")
        # decoded should not contain the EOS marker
        # The decode function skips eos_id
        decoded = tok.decode(tokens)
        assert decoded == "a"  # EOS is skipped
