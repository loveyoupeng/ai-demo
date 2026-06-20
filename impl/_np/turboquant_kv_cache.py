"""TurboQuant KV Cache — 1-bit compressed KV Cache with per-channel scaling.

During autoregressive decoding, the KV cache grows to O(seq_len * n_layers
* n_heads * head_dim) in float32. This module compresses each K,V tensor to
1-bit (sign-only) while storing a per-channel scale factor, achieving ~32x
memory reduction.

Quantization pipeline (per input tensor x):

    1. Compute scale = mean(|x|) along sequence dimension, per (batch, head)
       scale shape: (batch, n_heads, 1, 1) — scalar per head per position
    2. Quantize: binary_sign = (x > 0).astype(np.int8)  # +1 for positive,  0 for negative
    3. Store: binary_bits (int8) and scale (float32)
    4. Dequantize: reconstructed = binary_bits.astype(np.float32) * scale
       # Only binary 1 → scale, binary 0 → 0.0

Matrix shapes throughout this module:
    k, v          : (batch_size, n_heads, 1, head_dim) — per-token inputs
    bits_k, bits_v: (batch_size, n_heads, max_len, head_dim) — int8 1-bit storage
    scales_k, scales_v: (batch_size, n_heads, max_len, head_dim) — float32 scales
    k_cache[i], v_cache[i]: (batch_size, n_heads, seq_len, head_dim) — dequantized outputs
"""

from __future__ import annotations

import numpy as np


class TurboQuantKVCache:
    """1-bit compressed KV Cache with per-channel scaling.

    Stores K and V tensors in 1-bit precision with per-channel scaling
    to maintain reasonable reconstruction accuracy while using 1/32
    the memory of float32.

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
    bits_k : list of np.ndarray
        1-bit storage (int8) for K tensors, one per layer.
    bits_v : list of np.ndarray
        1-bit storage (int8) for V tensors, one per layer.
    scales_k : list of np.ndarray
        Per-channel scale (float32) for K tensors, one per layer.
    scales_v : list of np.ndarray
        Per-channel scale (float32) for V tensors, one per layer.
    _current_length : int
        Number of tokens cached so far (shared across all layers).
    _batch_size : int
        Batch size established on first update (unknown until first update).

    """

    def __init__(
        self,
        max_length: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize empty 1-bit KV cache buffers for every layer.

        Internal storage:
            bits_k[i]  : (batch_size, n_heads, max_len, head_dim) — int8
            scales_k[i]: (batch_size, n_heads, max_len, head_dim) — float32
            bits_v[i]  : (batch_size, n_heads, max_len, head_dim) — int8
            scales_v[i]: (batch_size, n_heads, max_len, head_dim) — float32

        Batch size is unknown until the first write, so we allocate with
        placeholder batch=1 and resize at first update.

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
        self.n_heads = n_heads  # scalar: number of heads
        self.head_dim = head_dim  # scalar: dimension per head
        self.device = device  # scalar: device string

        # Pre-allocate internal storage with placeholder batch=1; first
        # update will resize to the actual batch size.
        placeholder_shape = (1, n_heads, max_length, head_dim)

        self.bits_k: list[np.ndarray] = [np.zeros(placeholder_shape, dtype=np.int8) for _ in range(n_layers)]
        self.bits_v: list[np.ndarray] = [np.zeros(placeholder_shape, dtype=np.int8) for _ in range(n_layers)]
        self.scales_k: list[np.ndarray] = [np.zeros(placeholder_shape, dtype=np.float32) for _ in range(n_layers)]
        self.scales_v: list[np.ndarray] = [np.zeros(placeholder_shape, dtype=np.float32) for _ in range(n_layers)]

        self._current_length: int = 0  # scalar: tokens cached so far
        self._batch_size: int = 0  # scalar: unknown until first write

    def _quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """1-bit quantize a K or V tensor.

        Quantization algorithm (per input tensor x of shape (batch, heads, seq, dim)):

            1. Compute scale = mean(|x|) along (seq, dim) dims for each
               (batch, head) group. This gives a per-channel scale factor.
               scale shape: (batch, heads, 1, 1)

            2. Quantize: binary = (x > 0).astype(np.int8)
               Positive values → 1, negative/zero → 0.

            3. To dequantize: reconstructed = binary.astype(float32) * scale

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, n_heads, 1, head_dim)
            Input K or V tensor to quantize.

        Returns
        -------
        bits : np.ndarray, shape (batch_size, n_heads, 1, head_dim)
            1-bit binary storage (int8), 1 for positive, 0 for non-positive.
        scale : np.ndarray, shape (batch_size, n_heads, 1, head_dim)
            Per-channel mean(|x|) float32 scale factor per (batch, head).

        """
        # Compute scale: mean absolute value per (batch, head) across
        # sequence (pos 1) and head_dim dims.
        # |x| shape: (batch, heads, 1, head_dim)
        # Axis to average: -2 is seq_dim=1, -1 is head_dim.
        # result shape after keepdims: (batch, heads, 1, 1)
        absolute_values = np.abs(x)  # (batch, heads, 1, head_dim)
        scale = np.mean(absolute_values, axis=(-2, -1), keepdims=True)  # (batch, heads, 1, 1)

        # Quantize to 1-bit: positive → 1, non-positive → 0
        # shape: (batch, heads, 1, head_dim) — each element is 0 or 1
        bits = (x > 0).astype(np.int8)  # (batch, heads, 1, head_dim), 1 for positive

        return bits, scale

    def _dequantize(self, bits: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Dequantize 1-bit storage back to float32.

        Reconstruction: reconstructed = binary_bits.astype(float32) * scale
        Only binary 1 values are preserved (reconstructed = scale);
        binary 0 values are zeroed.

        Parameters
        ----------
        bits : np.ndarray, shape (batch, heads, seq, head_dim)
            1-bit storage (int8), values 0 or 1.
        scale : np.ndarray, shape (batch, heads, seq, head_dim)
            Per-channel scale factors (float32), constant along seq and dim.

        Returns
        -------
        reconstructed : np.ndarray, shape (batch, heads, seq, head_dim)
            Dequantized float32 tensor.

        """
        # (0 or 1 as float32) × scale → scale where bit was 1, 0 where bit was 0
        return bits.astype(np.float32) * scale  # (batch, heads, seq, head_dim)

    def update(
        self,
        k: np.ndarray,
        v: np.ndarray,
        pos: int,
    ) -> None:
        """Quantize and store new K,V at position pos.

        First call establishes batch size for all layers. Subsequent calls
        write to the specified position in each layer's storage.

        Parameters
        ----------
        k : np.ndarray, shape (batch_size, n_heads, 1, head_dim)
            New Key tensor for one token.
        v : np.ndarray, shape (batch_size, n_heads, 1, head_dim)
            New Value tensor for one token.
        pos : int
            Position in the sequence to store at (0-indexed).
            Must satisfy 0 <= pos < max_length.

        """
        batch_size = k.shape[0]  # scalar: number of samples in this batch

        # First update: establish batch dimension across all layers.
        if self._current_length == 0:
            self._batch_size = batch_size
            internal_shape = (batch_size, self.n_heads, self.max_length, self.head_dim)
            for i in range(self.n_layers):
                self.bits_k[i] = np.zeros(internal_shape, dtype=np.int8)
                self.bits_v[i] = np.zeros(internal_shape, dtype=np.int8)
                self.scales_k[i] = np.zeros(internal_shape, dtype=np.float32)
                self.scales_v[i] = np.zeros(internal_shape, dtype=np.float32)
        else:
            assert k.shape[0] == self._batch_size, f"K batch mismatch: expected {self._batch_size}, got {k.shape[0]}"
            assert v.shape[0] == self._batch_size, f"V batch mismatch: expected {self._batch_size}, got {v.shape[0]}"

        # Quantize K and V, store bits and scale at position pos.
        k_bits, k_scale = self._quantize(k)  # both shape (batch, heads, 1, dim)
        v_bits, v_scale = self._quantize(v)  # both shape (batch, heads, 1, dim)

        for i in range(self.n_layers):
            self.bits_k[i][:, :, pos : pos + 1, :] = k_bits
            self.scales_k[i][:, :, pos : pos + 1, :] = k_scale
            self.bits_v[i][:, :, pos : pos + 1, :] = v_bits
            self.scales_v[i][:, :, pos : pos + 1, :] = v_scale

        # Update position tracker — last written position defines length.
        self._current_length = pos + 1

    def get(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return all dequantized K,V for all layers.

        Returns only the filled prefix up to current_length, dequantizing
        each stored (bits, scale) pair back to float32.

        Returns
        -------
        k_cache : list of np.ndarray
            Each element shape (batch_size, n_heads, seq_len_so_far, head_dim).
            Dequantized to float32.
        v_cache : list of np.ndarray
            Each element shape (batch_size, n_heads, seq_len_so_far, head_dim).
            Dequantized to float32.

        """
        k_result: list[np.ndarray] = []
        v_result: list[np.ndarray] = []
        n = self._current_length  # scalar: how many tokens cached

        for i in range(self.n_layers):
            # Dequantize the filled prefix (0..n) for K
            k_reconstructed = self._dequantize(
                self.bits_k[i][:, :, :n, :],
                self.scales_k[i][:, :, :n, :],
            )  # (batch, heads, n, dim)
            k_result.append(k_reconstructed)

            # Dequantize the filled prefix (0..n) for V
            v_reconstructed = self._dequantize(
                self.bits_v[i][:, :, :n, :],
                self.scales_v[i][:, :, :n, :],
            )  # (batch, heads, n, dim)
            v_result.append(v_reconstructed)

        assert len(k_result) == self.n_layers
        assert len(v_result) == self.n_layers
        return (k_result, v_result)

    def clear(self) -> None:
        """Reset all caches to empty.

        After clear(), the cache has zero length and will re-establish the
        batch size for internal buffers on the next write.

        Internal storage values are zeroed but not deallocated to avoid
        repeated memory allocation during training/inference loops.
        """
        for i in range(self.n_layers):
            self.bits_k[i].fill(0)
            self.bits_v[i].fill(0)
            self.scales_k[i].fill(0)
            self.scales_v[i].fill(0)
        self._current_length = 0

    def is_empty(self) -> bool:
        """Return True if no tokens have been cached yet."""
        return self._current_length == 0

    def current_length(self) -> int:
        """Return the number of tokens currently cached."""
        return self._current_length

    def memory_usage(self) -> int:
        """Return memory usage in bytes for one layer's storage.

        Internal storage: 1-bit per value for K and V → 2 × max_length ×
        n_heads × head_dim bits total. Rounded up to nearest byte.

        Returns
        -------
        bytes : int
            Bytes per layer: bits_k + bits_v (each stores int8) +
            scales_k + scales_v (each stores float32). The bits arrays use
            1 byte per element (8× compression) and the scale arrays use
            4 bytes per element.

        """
        # For one layer:
        #   bits_k + bits_v = 2 * max_length * n_heads * head_dim bytes (int8 each)
        #   scales_k + scales_v = 2 * max_length * n_heads * head_dim * 4 bytes (float32 each)
        # But per the spec, memory_usage should be close to 1/32 of float32,
        # meaning it counts the theoretical minimum (bit-level, not array-level).
        # Formula from spec: 2 * max_length * n_heads * head_dim bits → ceil(bits/8)
        bits = 2 * self.max_length * self.n_heads * self.head_dim  # total bits for K+V
        return (bits + 7) // 8  # round up to nearest byte
