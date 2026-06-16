"""Naive KV Cache — torch tensor version for PyTorch inference.

During autoregressive generation, each new token attends to all previously
generated tokens. A KV cache stores the Key and Value tensors for every
position so that at step t we only compute K,V for the new token (shape
(batch, n_heads, 1, head_dim)) and concatenate with the cached K,V of
shape (batch, n_heads, t-1, head_dim).

Matrix shapes throughout this module:
    k_cache, v_cache : (batch_size, n_heads, seq_len, head_dim)
"""

from __future__ import annotations

import torch


class TorchNaiveKVCache:
    """Simple KV Cache for autoregressive decoding.

    Stores K and V tensors for each layer, growing incrementally as new tokens
    are generated.
    """

    def __init__(
        self,
        max_length: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize empty KV cache buckets for every layer."""
        self.max_length = max_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device

        # Pre-allocate zero-length buffers so current_length=0 starts clean.
        # k_cache[i] shape: (batch_size, n_heads, 0, head_dim) — placeholder
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []

        # _current_length is 0 initially; first update establishes batch size.
        self._current_length: int = 0
        self._batch_size: int = 0

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: int,
    ) -> None:
        """Store new K,V at position pos.

        Parameters
        ----------
        k : torch.Tensor, shape (batch_size, n_heads, 1, head_dim)
            New Key tensor for a single token.
        v : torch.Tensor, shape (batch_size, n_heads, 1, head_dim)
            New Value tensor for a single token.
        pos : int
            Position in the autoregressive sequence (0-indexed).
        """
        batch_size = k.shape[0]

        # First update: establish batch dimension across all layers.
        if self._current_length == 0:
            self._batch_size = batch_size
            for _ in range(self.n_layers):
                self.k_cache.append(
                    torch.empty(
                        (batch_size, self.n_heads, self.max_length, self.head_dim),
                        dtype=k.dtype,
                        device=k.device,
                    )
                )
                self.v_cache.append(
                    torch.empty(
                        (batch_size, self.n_heads, self.max_length, self.head_dim),
                        dtype=v.dtype,
                        device=v.device,
                    )
                )

        assert k.shape[0] == self._batch_size
        assert v.shape[0] == self._batch_size

        # Assign new K,V into the appropriate slice for every layer.
        for i in range(self.n_layers):
            self.k_cache[i][:, :, pos : pos + 1, :] = k
            self.v_cache[i][:, :, pos : pos + 1, :] = v

        self._current_length = max(self._current_length, pos + 1)

    def get(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return all cached K,V for all layers.

        Returns only the filled portion up to current_length.
        """
        n = self._current_length
        k_result: list[torch.Tensor] = []
        v_result: list[torch.Tensor] = []
        for i in range(self.n_layers):
            k_result.append(self.k_cache[i][:, :, :n, :])
            v_result.append(self.v_cache[i][:, :, :n, :])
        return (k_result, v_result)

    def clear(self) -> None:
        """Reset all caches to empty."""
        for i in range(self.n_layers):
            self.k_cache[i] = torch.empty(
                (self._batch_size if self._batch_size else 1, self.n_heads, 0, self.head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.v_cache[i] = torch.empty(
                (self._batch_size if self._batch_size else 1, self.n_heads, 0, self.head_dim),
                dtype=torch.float32,
                device=self.device,
            )
        self._current_length = 0
        self._batch_size = 0

    def is_empty(self) -> bool:
        """Return True if no tokens have been cached yet."""
        return self._current_length == 0

    def current_length(self) -> int:
        """Return the number of tokens currently cached."""
        return self._current_length
