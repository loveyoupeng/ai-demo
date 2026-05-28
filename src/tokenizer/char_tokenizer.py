import numpy as np
from typing import List, Dict, Tuple

class CharTokenizer:
    """
    A simple character-level tokenizer for pedagogical purposes.
    Maps every unique character in a text to a unique integer.
    """

    def __init__(self, text: str = ""):
        """
        Initialize the tokenizer with a vocabulary derived from the provided text.
        
        Args:
            text: A string used to build the initial vocabulary.
        """
        # Extract unique characters and sort them for consistency
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Mapping: char -> index
        self.char_to_int: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        # Mapping: index -> char
        self.int_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> np.ndarray:
        """
        Convert a string of text into an array of integers.
        
        Args:
            text: The input string.
            
        Returns:
            A numpy array of integers representing the token IDs.
        """
        # [Seq_Len]
        return np.array([self.char_to_int[c] for c in text], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """
        Convert an array of integers back into a string of text.
        
        Args:
            ids: A numpy array of token IDs.
            
        Returns:
            The decoded string.
        """
        return "".join([self.int_to_char[i] for i in ids])

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size}, chars={self.chars})"
