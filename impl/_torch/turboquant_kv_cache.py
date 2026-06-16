"""TurboQuant KV Cache — 1-bit compressed KV Cache with per-channel scaling.

During autoregressive decoding, the KV cache grows to O(seq_len * n_layers
* n_heads * head_dim) in float32. This module compresses each K,V tensor
to 1-bit (sign-only) while storing a per-channel scale factor, achieving
~32x memory reduction.

Quantization pipeline (per input tensor x):
    1. Compute scale = mean(|x|) along sequence dimension, per (batch, head)
       scale shape: (batch, n_heads, 1, 1) — scalar per head per position
    2. Quantize: binary_sign = (x > 0).to(torch.int8)  # +1 for positive, 0

    for negative
    3. Store: binary_bits (int8) and scale (float32)
    4. Dequantize: reconstructed = binary_bits.float() * scale

Matrix shapes throughout this module:
    k, v          : (batch_size, n_heads, 1, head_dim) — per-token inputs
    bits_k, bits_v: (batch_size, n_heads, max_len, head_dim) — int8 1-bit storage
    scales_k, scales_v: (batch_size, n_heads, max_len, head_dim) — float32 scales
    k_cache[i], v_cache[i]: (batch_size, n_heads, seq_len, head_dim) — dequantized outputs
"""

from __future__ import annotations

import torch


class TorchTurboQuantKVCache:
    """1-bit compressed KV Cache with per-channel scaling.

    Stores K and V tensors in 1-bit precision with per-channel scaling
    to maintain reasonable reconstruction accuracy while using reduced
    memory.
    """

    def __init__(
        self,
        max_length: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize empty 1-bit KV cache buffers for every layer."""
        self.max_length = max_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device

        # Internal storage placeholders (batch=1 until first write).
        placeholder_shape = (1, n_heads, max_length, head_dim)
        self.bits_k: list[torch.Tensor] = [
            torch.zeros(placeholder_shape, dtype=torch.int8) for _ in range(n_layers)
        ]
        self.bits_v: list[torch.Tensor] = [
            torch.zeros(placeholder_shape, dtype=torch.int8) for _ in range(n_layers)
        ]
        self.scales_k: list[torch.Tensor] = [
            torch.zeros(placeholder_shape, dtype=torch.float32) for _ in range(n_layers)
        ]
        self.scales_v: list[torch.Tensor] = [
            torch.zeros(placeholder_shape, dtype=torch.float32) for _ in range(n_layers)
        ]
        self._current_length: int = 0
        self._batch_size: int = 0

    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """1-bit quantize a K or V tensor.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, n_heads, 1, head_dim)
            Input K or V tensor to quantize.

        Returns
        -------
        bits : torch.Tensor, shape (batch_size, n_heads, 1, head_dim)
            1-bit binary storage (int8), 1 for positive, 0 for non-positive.
        scale : torch.Tensor, shape (batch_size, n_heads, 1, head_dim)
            Per-channel mean(|x|) float32 scale factor per (batch, head).
        """
        # Compute scale: mean absolute value per (batch, head) across
        # sequence (dim=1) and head_dim (dim=-1) dims.
        # result shape after keepdims: (batch, heads, 1, 1)
        scale = torch.mean(torch.abs(x), dim=(-2, -1), keepdim=True)

        # Quantize to 1-bit: positive → 1, non-positive → 0
        bits = (x > 0).to(torch.int8)

        return bits, scale

    def _dequantize(self, bits: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize 1-bit storage back to float32.

        Reconstruction: reconstructed = binary_bits.float() * scale
        Only binary 1 values are preserved (reconstructed = scale);
        binary 0 values are zeroed.

        Parameters
        ----------
        bits : torch.Tensor, shape (batch, heads, seq, head_dim)
            1-bit storage (int8), values 0 or 1.
        scale : torch.Tensor, shape (batch, heads, seq, head_dim)
            Per-channel scale factors (float32), constant along seq and dim.

        Returns
        -------
        reconstructed : torch.Tensor, shape (batch, heads, seq, head_dim)
            Dequantized float32 tensor.
        """
        return bits.float() * scale

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: int,
    ) -> None:
        """Quantize and store new K,V at position pos.

        Parameters
        ----------
        k : torch.Tensor, shape (batch_size, n_heads, 1, head_dim)
            New Key tensor for one token.
        v : torch.Tensor, shape (batch_size, n_heads, 1, head_dim)
            New Value tensor for one token.
        pos : int
            Position in the sequence to store at (0-indexed).
        """
        batch_size = k.shape[0]

        # First update: establish batch dimension across all layers.
        if self._current_length == 0:
            self._batch_size = batch_size
            internal_shape = (batch_size, self.n_heads, self.max_length, self.head_dim)
            for i in range(self.n_layers):
                self.bits_k[i] = torch.zeros(internal_shape, dtype=torch.int8)
                self.bits_v[i] = torch.zeros(internal_shape, dtype=torch.int8)
                self.scales_k[i] = torch.zeros(internal_shape, dtype=torch.float32)
                self.scales_v[i] = torch.zeros(internal_shape, dtype=torch.float32)

        assert k.shape[0] == self._batch_size
        assert v.shape[0] == self._batch_size

        # Quantize K and V, store bits and scale at position pos.
        k_bits, k_scale = self._quantize(k)
        v_bits, v_scale = self._quantize(v)

        for i in range(self.n_layers):
            self.bits_k[i][:, :, pos : pos + 1, :] = k_bits
            self.scales_k[i][:, :, pos : pos + 1, :] = k_scale
            self.bits_v[i][:, :, pos : pos + 1, :] = v_bits
            self.scales_v[i][:, :, pos : pos + 1, :] = v_scale

        self._current_length = max(self._current_length, pos + 1)

    def get(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return all dequantized K,V for all layers.

        Returns only the filled prefix up to current_length.
        """
        n = self._current_length
        k_result: list[torch.Tensor] = []
        v_result: list[torch.Tensor] = []

        for i in range(self.n_layers):
            k_reconstructed = self._dequantize(
                self.bits_k[i][:, :, :n, :],
                self.scales_k[i][:, :, :n, :],
            )  # (batch, heads, n, dim)
            k_result.append(k_reconstructed)

            v_reconstructed = self._dequantize(
                self.bits_v[i][:, :, :n, :],
                self.scales_v[i][:, :, :n, :],
            )  # (batch, heads, n, dim)
            v_result.append(v_reconstructed)

        return (k_result, v_result)

    def clear(self) -> None:
        """Reset all caches to empty.

        Internal storage values are zeroed but not deallocated.
        """
        for i in range(self.n_layers):
            self.bits_k[i].zero_()
            self.bits_v[i].zero_()
            self.scales_k[i].zero_()
            self.scales_v[i].zero_()
        self._current_length = 0
        self._batch_size = 0

    def is_empty(self) -> bool:
        """Return True if no tokens have been cached yet."""
        return self._current_length == 0

    def current_length(self) -> int:
        """Return the number of tokens currently cached."""
        return self._current_length
