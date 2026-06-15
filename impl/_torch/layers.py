"""Embedding layer — maps token IDs to dense vectors.

Maps integer token IDs [batch, seq_len] to embedding vectors [batch, seq_len, embed_dim]
by looking up rows of a learnable weight matrix [vocab_size, embed_dim].

This mirrors the NumPy implementation in impl/_np/modules.py which uses
a stateless forward function. The PyTorch version stores weight as an
nn.Parameter for autograd.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Token token IDs to dense embedding vectors.

    Parameters:
        vocab_size: Number of tokens in the vocabulary.
        embed_dim: Dimension of the embedding vector.
    """

    __slots__ = ("weight",)

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        # weight.shape = [vocab_size, embed_dim]
        # Initialized with Kaiming uniform (default nn init)
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for token IDs.

        Args:
            input_ids: Token indices [batch, seq_len] (int64 or int32).

        Returns:
            Embedding vectors [batch, seq_len, embed_dim].
        """
        return nn.functional.embedding(input_ids, self.weight)
