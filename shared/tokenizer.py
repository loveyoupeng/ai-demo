"""Tokenizers for the decoder-only transformer.

Provides GPT-2 BPE tokenization (primary) and a character-level
tokenizer for small-vocab demos. All backends (numpy, torch, triton, cuda)
import from this module.

Tokenization is the bridge between text input and model tensors:
  text (str) -> token IDs (list[int]) -> model input (tensor)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer  # type: ignore[import-untyped]


@dataclass
class TokenizeResult:
    """Result of tokenizing text.

    Attributes:
        input_ids: Sequence of token IDs (e.g., [15496, 995, 11] for "hello")
        attention_mask: Corresponding attention mask (e.g., [1, 1, 1])
    """

    input_ids: list[int]
    attention_mask: list[int]


def create_tokenizer(model_path: str = "gpt2") -> AutoTokenizer:
    """Create and configure a tokenizer.

    Uses GPT-2 tokenizer (~50k vocab) for the demo. The first call
    downloads the tokenizer model (~500KB) — this is expected and cached
    by transformers for subsequent runs.

    Args:
        model_path: HuggingFace model ID (default: "gpt2").

    Returns:
        An AutoTokenizer with padding configured.

    Example:
        >>> tok = create_tokenizer()
        >>> ids = tok.encode("The cat sat")
        >>> print(len(ids))  # number of tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_text(
    tokenizer: AutoTokenizer,
    text: str,
    add_special_tokens: bool = True,
) -> list[int]:
    """Encode text to token IDs.

    Converts a string to a sequence of integer token IDs that the
    model can process. Special tokens (BOS, EOS) can be included
    depending on training/inference needs.

    Args:
        tokenizer: The tokenizer to use.
        text: Input text string.
        add_special_tokens: Whether to include BOS/EOS tokens.
            - True:  adds special tokens (used during training typically)
            - False: raw text tokens only (used during inference)

    Returns:
        List of token IDs (integers).

    Example:
        >>> tok = create_tokenizer()
        >>> encode_text(tok, "hello world")
        [15496, 995]
    """
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)  # type: ignore[reportAttributeAccessIssue]


def decode_tokens(
    tokenizer: AutoTokenizer,
    token_ids: list[int] | np.ndarray,
    skip_special_tokens: bool = True,
) -> str:
    """Decode token IDs back to text.

    Converts a sequence of token IDs back into a human-readable string.
    This is the inverse of encode_text().

    Args:
        tokenizer: The tokenizer to use.
        token_ids: Sequence of token IDs (list or numpy array).
        skip_special_tokens: Skip BOS, EOS, PAD tokens from output.
            - True: clean text output (default for inference)
            - False: includes special token markers

    Returns:
        Decoded text string.

    Example:
        >>> tok = create_tokenizer()
        >>> ids = [15496, 995]
        >>> decode_tokens(tok, ids)
        'hello world'
    """
    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)  # type: ignore[reportAttributeAccessIssue]


def pad_sequences(
    token_lists: list[list[int]],
    padding_value: int = 0,
    max_length: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Right-pad token sequences to the same length.

    Pads sequences on the right (after real tokens) with a specified
    padding value. Also generates attention masks: 1 for real tokens,
    0 for padding tokens.

    Dimensions:
        Input:  list of n sequences, each of variable length
        Output: (padded[n, max_len], mask[n, max_len])

    Args:
        token_lists: List of token ID sequences of varying lengths.
        padding_value: Integer value for padding (default 0 = PAD).
        max_length: If specified, pad all sequences to this length.
                    Otherwise, pad to the longest sequence.

    Returns:
        Tuple of (padded_sequences, attention_masks) where:
          - padded_sequences: n sequences, each max_length integers
          - attention_masks:   n sequences, each max_length 0/1

    Example:
        >>> tokens = [[1, 2, 3], [4, 5]]
        >>> padded, masks = pad_sequences(tokens)
        >>> padded  # right-padded to length 3
        [[1, 2, 3], [4, 5, 0]]
        >>> masks   # 1=real token, 0=padding
        [[1, 1, 1], [1, 1, 0]]
    """
    if max_length is None:
        max_length = max(len(s) for s in token_lists)

    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for seq in token_lists:
        pad_len = max_length - len(seq)
        padded.append(seq + [padding_value] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)

    return padded, masks


def save_tokenizer(
    tokenizer: AutoTokenizer,
    save_dir: str | None = None,
) -> str:
    """Save tokenizer files to disk for later loading.

    Args:
        tokenizer: The tokenizer to save.
        save_dir: Directory to save files. If None, uses "./models/tokenizer".

    Returns:
        The directory path where files were saved.
    """
    from pathlib import Path

    path = Path(save_dir or "./models/tokenizer")
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(path))  # type: ignore[reportAttributeAccessIssue]
    return str(path)


class SimpleCharTokenizer:
    """Minimal character-level tokenizer for small vocab demos.

    Maps each unique character to an integer ID. Useful when vocab_size
    needs to be very small (< 128 tokens), such as during quick prototyping.

    Vocabulary includes: a-z, A-Z, 0-9, and common punctuation.

    Attributes:
        vocab: dict mapping char -> integer ID
        reverse_vocab: dict mapping integer ID -> char (inverse)
        pad_id: Reserved ID for padding (always 0)
        eos_id: Reserved ID for end-of-sequence (always 1)

    Example:
        >>> tok = SimpleCharTokenizer(vocab_size=90)
        >>> tokens = tok.encode("Hello!")
        >>> print(tokens)  # [id('H'), id('e'), id('l'), id('l'), id('o'), id('!'), eos_id]
    """

    PAD: str = "<PAD>"
    EOS: str = "<EOS>"

    def __init__(self, vocab_size: int = 128) -> None:
        """Initialize tokenizer with a fixed vocab size.

        Args:
            vocab_size: Total capacity including PAD and EOS tokens.
        """
        self.pad_id = 0
        self.eos_id = 1
        self.vocab: dict[str, int] = {
            self.PAD: self.pad_id,
            self.EOS: self.eos_id,
        }
        self.reverse_vocab: dict[int, str] = {
            self.pad_id: self.PAD,
            self.eos_id: self.EOS,
        }
        self.vocab_size = vocab_size
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build vocabulary from common ASCII characters.

        Populates the vocab dictionary with a-z, A-Z, 0-9, and common
        punctuation marks. Characters are added in reverse order so that
        the most common characters (a, b, c...) get lower IDs.
        """
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\""
        for ch in reversed(chars):
            if ch not in self.vocab and len(self.vocab) < self.vocab_size - 2:
                self.vocab[ch] = len(self.vocab)
                self.reverse_vocab[len(self.vocab) - 1] = ch

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Maps each character to its ID, appending EOS at the end.
        Characters not in the vocab map to pad_id (0).

        Args:
            text: Input text string.

        Returns:
            List of token IDs, ending with eos_id.

        Example:
            >>> tok = SimpleCharTokenizer()
            >>> tok.encode("ab")
            [id('a'), id('b'), eos_id]
        """
        return [self.vocab.get(ch, self.pad_id) for ch in text] + [self.eos_id]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text.

        Skips pad_id and eos_id in the output. Only characters present
        in the reverse vocabulary are decoded.

        Args:
            token_ids: List of token IDs (may include pad_id, eos_id).

        Returns:
            Decoded text string (no special tokens in output).

        Example:
            >>> tok = SimpleCharTokenizer()
            >>> tok.decode([id('a'), id('b'), eos_id])
            'ab'
        """
        result = []
        for tid in token_ids:
            if tid == self.pad_id or tid == self.eos_id:
                continue
            result.append(self.reverse_vocab.get(tid, ""))
        return "".join(result)
