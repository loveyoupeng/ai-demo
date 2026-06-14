"""Dataset loading and preprocessing for the decoder-only transformer.

Loads TinyStories from HuggingFace, tokenizes into sequences,
and generates training batches. All backends import from here.

Data Pipeline:
    TinyStories text -> tokenize -> concatenate into flat stream ->
    sliding window sampling -> yield (input, target) pairs -> batch
"""

from __future__ import annotations

import random
from typing import Protocol

from datasets import load_dataset  # type: ignore[import-untyped]


class TokenizerLike(Protocol):
    """Tokenizer protocol for TextDataset."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        ...


def load_tinystories(
    split: str = "train",
    num_stories: int | None = None,
) -> list[str]:
    """Load TinyStories dataset from HuggingFace.

    Downloads the dataset (~8MB for all stories, cached by datasets lib).
    For the demo, a cap of 10k training stories is applied automatically.

    Args:
        split: Dataset split - "train", "validation", or "test".
        num_stories: Max stories to load. Auto-capped to 10000 for train.

    Returns:
        List of story strings.

    Example:
        >>> stories = load_tinystories("train")
        >>> print(f"Loaded {len(stories)} stories")  # ~10000
        >>> print(stories[0][:60])
        'Once upon a time, there was a little girl named...'
    """
    limit = num_stories or (10000 if split == "train" else 50)
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    stories: list[str] = []
    for i, example in enumerate(ds):
        text = example.get("text", "")
        if text and text.strip():
            stories.append(text)
            if i >= limit:
                break
    return stories


class TextDataset:
    """Dataset wrapper: tokenizes stories and samples sliding windows.

    Flattens all tokenized text into one large stream, then samples
    random starting positions to create training windows. Each window
    has `context_length` tokens; the target is shifted by 1.

    Dimensions:
        token_ids: Flat list of all tokenized text (variable length)
        Window input:  [context_length] tokens
        Window target: [context_length] tokens (shifted by 1)

    Example (context_length=3):
        Token stream: [1, 2, 3, 4, 5, 6, 7, ...]
        Windows picked randomly from this stream:
          Start at 0: input=[1,2,3], target=[2,3,4]
          Start at 5: input=[6,7,8], target=[7,8,9]

    Attributes:
        tokenizer: Tokenizer instance with .encode() method
        context_length: Window size in tokens
        rng: Random number generator (seeded for reproducibility)
        token_ids: Flat stream of all tokenized story text
    """

    def __init__(
        self,
        text_data: list[str],
        tokenizer: TokenizerLike,
        context_length: int = 256,
        seed: int = 42,
    ) -> None:
        """Initialize dataset by concatenating all tokenized text.

        Args:
            text_data: List of text strings to train on.
            tokenizer: Tokenizer instance with .encode(text, add_special_tokens=False)
            context_length: Sliding window size for each training sample.
            seed: Random seed for reproducibility of sample selection.
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.rng = random.Random(seed)
        self.token_ids: list[int] = []

        # Concatenate all tokenized text into one flat stream
        for story in text_data:
            if story.strip():
                try:
                    ids = tokenizer.encode(story, add_special_tokens=False)
                    if ids:
                        self.token_ids.extend(ids)
                except Exception:
                    continue  # skip stories that fail to tokenize

    def get_sequences(
        self,
        num_samples: int,
        batch_size: int = 8,
    ) -> list[tuple[list[int], list[int]]]:
        """Generate random training sequences from the token stream.

        Samples `num_samples` random windows from the concatenated token
        stream. Each window has `context_length` tokens; the target is
        the same window shifted by 1 position (standard next-token prediction).

        Args:
            num_samples: Number of (input, target) pairs to generate.
            batch_size: Number of samples per batch (used for grouping).

        Returns:
            List of (input_ids, target_ids) tuples.
            Each element: ([context_length] ints, [context_length] ints)

        Raises:
            ValueError: If no token data is available.

        Example:
            >>> ds = TextDataset(stories, tokenizer)
            >>> seqs = ds.get_sequences(10, batch_size=5)
            >>> len(seqs)  # 10 samples
            10
            >>> len(seqs[0][0])  # window length
            256
        """
        if not self.token_ids:
            raise ValueError(
                "No token data available. "
                "Check that text_data is non-empty and the tokenizer works."
            )

        max_start = max(1, len(self.token_ids) - self.context_length - 1)
        sequences: list[tuple[list[int], list[int]]] = []

        for _ in range(num_samples):
            start = self.rng.randint(0, max_start)
            window = self.token_ids[start : start + self.context_length]

            # Skip if window is incomplete (not enough tokens remaining)
            if len(window) < self.context_length:
                continue

            # Target = window shifted by 1 (predict next token)
            target = window[1:]
            sequences.append((window, target))

        return sequences


def get_dataloader_sequences(
    dataset: TextDataset,
    batch_size: int = 32,
    num_batches: int = 100,
) -> list[tuple[list[list[int]], list[list[int]]]]:
    """Generate training batches from a TextDataset.

    Returns a list of batches suitable for passing to a model's forward pass.
    Each batch contains `batch_size` sequences.

    Dimensions:
        input_batch:  [batch_size, context_length] -> int32
        target_batch: [batch_size, context_length] -> int32
        Returns:      [num_batches, batch_size, context_length] -> int32

    Args:
        dataset: A TextDataset instance with tokenized text.
        batch_size: Number of sequences per batch.
        num_batches: Number of batches to generate.

    Returns:
        List of (input_batch, target_batch) tuples.

    Example:
        >>> ds = TextDataset(stories, tokenizer, context_length=64)
        >>> batches = get_dataloader_sequences(ds, batch_size=8, num_batches=3)
        >>> len(batches)               # 3 batches
        3
        >>> len(batches[0][0])         # batch_size
        8
        >>> len(batches[0][0][0])      # context_length
        64
    """
    batches: list[tuple[list[list[int]], list[list[int]]]] = []
    for _ in range(num_batches):
        seqs = dataset.get_sequences(batch_size)
        input_batch = [seq[0] for seq in seqs]
        target_batch = [seq[1] for seq in seqs]
        batches.append((input_batch, target_batch))
    return batches
