"""Naive KV Cache for autoregressive decoder-only transformer inference.

During autoregressive generation, each new token attends to all previously
generated tokens. A KV cache stores the Key and Value tensors for every
position so that at step t we only compute K,V for the new token (shape
(batch, n_heads, 1, head_dim)) and concatenate with the cached K,V of
shape (batch, n_heads, t-1, head_dim).

Matrix shapes throughout this module:
    k_cache, v_cache : (batch_size, n_heads, seq_len, head_dim)
"""

from __future__ import annotations

import copy

import numpy as np


class NaiveKVCache:
    """Simple KV Cache for autoregressive decoding.

    Stores K and V tensors for each layer, growing incrementally as new tokens
    are generated. During inference, the model computes K,V for only the new
    token (not the full sequence), then concatenates cached K,V with new K,V.

    Parameters
    ----------
    max_length : int
        Maximum sequence length the cache can hold.
    n_layers : int
        Number of transformer layers.
    n_heads : int
        Number of attention heads.
    head_dim : int
        Dimension per attention head.
    device : str, default "cpu"
        Device string placeholder.

    Attributes
    ----------
    k_cache : list of np.ndarray
        Cached K tensors for each layer. Initial shape (batch, n_heads, 0, head_dim).
    v_cache : list of np.ndarray
        Cached V tensors for each layer. Initial shape (batch, n_heads, 0, head_dim).
    current_length : int
        Number of tokens currently cached (shared across all layers).
    max_length : int
        Maximum allowed sequence length.
    """

    def __init__(
        self,
        max_length: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize empty KV cache buckets for every layer.

        Parameters
        ----------
        max_length : int
            Maximum sequence length the cache can hold.
        n_layers : int
            Number of transformer layers (each has its own K,V cache).
        n_heads : int
            Number of attention heads.
        head_dim : int
            Dimension per attention head.
        device : str, default "cpu"
            Device string placeholder.
        """
        self.max_length = max_length  # scalar: max sequence length
        self.n_layers = n_layers  # scalar: number of layers
        self.n_heads = n_heads  # scalar: number of attention heads
        self.head_dim = head_dim  # scalar: dimension per head
        self.device = device  # scalar: device string

        # Pre-allocate cache buffers — zero-length along sequence dim so that
        # current_length=0 starts the cache in a clean state.
        # k_cache[i] shape: (batch_size=1, n_heads, 0, head_dim) — placeholder,
        #   actual batch size is determined at first update.
        self.k_cache: list[np.ndarray] = [
            np.empty((1, n_heads, 0, head_dim), dtype=np.float32) for _ in range(n_layers)
        ]
        self.v_cache: list[np.ndarray] = [
            np.empty((1, n_heads, 0, head_dim), dtype=np.float32) for _ in range(n_layers)
        ]
        self._current_length: int = 0  # scalar: tokens cached so far

    def update(
        self,
        k: np.ndarray,
        v: np.ndarray,
        pos: int,
    ) -> None:
        """Store new K,V at position pos.

        If the batch size is not yet known (cache is empty), the first call
        establishes the batch dimension for all layers. When positions are
        filled sequentially the cache grows; if a position already exists its
        values are overwritten (useful for beam search or re-Decoding).

        Parameters
        ----------
        k : np.ndarray, shape (batch_size, n_heads, 1, head_dim)
            New Key tensor for a single token.
        v : np.ndarray, shape (batch_size, n_heads, 1, head_dim)
            New Value tensor for a single token.
        pos : int
            Position in the autoregressive sequence (0-indexed). Must satisfy
            0 <= pos < max_length, and pos <= current_length (sequential fill).
        """
        batch_size = k.shape[0]  # scalar: number of samples in this batch

        # First update: establish batch dimension across all layers so that
        # every layer's cache shares the same batch size regardless of which
        # layer got updated first.
        if self._current_length == 0:
            # Allocate cache buffers for every layer with shape
            # (batch, n_heads, max_len, head_dim).
            for i in range(self.n_layers):
                self.k_cache[i] = np.empty(
                    (batch_size, self.n_heads, self.max_length, self.head_dim),
                    dtype=np.float32,
                )
                self.v_cache[i] = np.empty(
                    (batch_size, self.n_heads, self.max_length, self.head_dim),
                    dtype=np.float32,
                )

        # Sanity: ensure the incoming K,V match the established batch size.
        # This catches mismatched batch dimensions before silent corruption.
        assert k.shape[0] == batch_size, f"K batch mismatch: expected {batch_size}, got {k.shape[0]}"
        assert v.shape[0] == batch_size, f"V batch mismatch: expected {batch_size}, got {v.shape[0]}"

        # Assign new K,V into the appropriate slice for every layer.
        # k[:, :, pos:pos+1, :] is (batch, n_heads, 1, head_dim) — a single
        # token at position `pos`.
        for i in range(self.n_layers):
            self.k_cache[i][:, :, pos : pos + 1, :] = k
            self.v_cache[i][:, :, pos : pos + 1, :] = v

        # Update position tracker — the last written position defines how
        # much sequence has been produced (0-indexed, so length = pos + 1).
        self._current_length = pos + 1

    def get(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return all cached K,V for all layers.

        Returns only the filled portion of the cache up to `current_length`,
        returning views into the pre-allocated buffers.

        Returns
        -------
        k_cache : list of np.ndarray
            Each element shape (batch_size, n_heads, seq_len_so_far, head_dim).
        v_cache : list of np.ndarray
            Each element shape (batch_size, n_heads, seq_len_so_far, head_dim).
        """
        # Take the prefix (0..current_length) for each layer so that external
        # code receives exactly the sequence produced so far.
        n = self._current_length  # scalar: how many tokens cached
        k_result: list[np.ndarray] = []
        v_result: list[np.ndarray] = []
        for i in range(self.n_layers):
            k_result.append(self.k_cache[i][:, :, :n, :])
            v_result.append(self.v_cache[i][:, :, :n, :])
        assert len(k_result) == self.n_layers
        assert len(v_result) == self.n_layers
        return (k_result, v_result)

    def clear(self) -> None:
        """Reset all caches to empty.

        After `clear()` the cache has zero length and will re-establish the
        batch dimension on the next `update()` call.
        """
        # Re-initialize to zero-length buffers.
        for i in range(self.n_layers):
            self.k_cache[i] = np.empty((1, self.n_heads, 0, self.head_dim), dtype=np.float32)
            self.v_cache[i] = np.empty((1, self.n_heads, 0, self.head_dim), dtype=np.float32)
        self._current_length = 0

    def is_empty(self) -> bool:
        """Return True if no tokens have been cached yet."""
        return self._current_length == 0

    def current_length(self) -> int:
        """Return the number of tokens currently cached."""
        return self._current_length

    def clone(self) -> NaiveKVCache:
        """Return a deep copy of this cache.

        Useful when forking decoding branches (e.g. beam search).

        Returns
        -------
        cloned : NaiveKVCache
            Independent copy with the same parameters and content.
        """
        return copy.deepcopy(self)
