import pytest
import numpy as np
from tokenizer.char_tokenizer import CharTokenizer


@pytest.mark.timeout(2)
def test_tokenizer_encoding_decoding():
    """
    Test that encoding a string and then decoding it returns the original string.
    """
    text = "hello world"
    tokenizer = CharTokenizer(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text
    assert isinstance(encoded, np.ndarray)
    assert encoded.dtype == np.int32


@pytest.mark.timeout(2)
def test_tokenizer_vocab_size():
    """
    Test that the vocabulary size matches the number of unique characters.
    """
    text = "aaaaabbbbbccccc"
    tokenizer = CharTokenizer(text)

    # Unique chars are 'a', 'b', 'c'
    assert tokenizer.vocab_size == 3
    assert len(tokenizer.chars) == 3


@pytest.mark.timeout(2)
def test_tokenizer_unknown_char():
    """
    Test that attempting to encode a character not in the vocabulary raises a KeyError.
    """
    tokenizer = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tokenizer.encode("d")


@pytest.mark.timeout(2)
def test_tokenizer_empty_text():
    """
    Test tokenizer behavior with empty text.
    """
    tokenizer = CharTokenizer("")
    assert tokenizer.vocab_size == 0
    assert len(tokenizer.encode("")) == 0
    assert tokenizer.decode(np.array([], dtype=np.int32)) == ""
