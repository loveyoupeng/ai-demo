from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BITS = 4
NUM_LEVELS = 1 << NUM_BITS  # 2^4 = 16

RESIDUAL_WINDOW = 128  # recent tokens stored in full precision


# ---------------------------------------------------------------------------
# PyQuantize — static/utility methods for rotation + codebook + dequantization
# ---------------------------------------------------------------------------


class PyQuantize:
    """
    TurboQuant quantization utilities.

    Implements the core idea from the TurboQuant paper (Google):
    1. Random orthogonal rotation (QR decomposition of random Gaussian matrix).
    2. Beta(0.5, 0.5) optimally-quantized codebook (equiprobable levels).
    3. 4-bit scalar quantization with per-channel norm scaling.
    4. On-the-fly dequantization during attention inference.

    This is a *learning/reference* implementation — not production code.
    """

    @staticmethod
    def get_random_rotation_matrix(
        dim: int, seed: int | None = None,
    ) -> torch.Tensor:
        """
        Compute a random orthogonal matrix via QR decomposition of a
        Gaussian random matrix.  Computed once at cache initialization.

        Args:
            dim: Feature dimension.
            seed: Optional RNG seed for reproducibility.

        Returns:
            rotation_matrix: [dim, dim] orthogonal matrix (Q from QR).
        """
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
            R = torch.randn(dim, dim, generator=rng)
        else:
            R = torch.randn(dim, dim)

        Q, _ = torch.linalg.qr(R)
        return Q  # [dim, dim] orthogonal

    @staticmethod
    def get_beta_codebook(
        num_levels: int = NUM_LEVELS,
    ) -> torch.Tensor:
        """
        Compute equiprobable quantization levels for a Beta(0.5, 0.5)
        (arcsine) distribution — the optimal codebook from TurboQuant.

        The Beta(0.5, 0.5) CDF uses the arcsine function:
            F(x) = (2/pi) * arcsin(sqrt(x))

        Inverting F gives the equiprobable levels:
            level_k = sin^2(pi * (k + 0.5) / (2 * num_levels))

        This clusters levels at the tails [-1, 1], which
        matches the heavy-tailed nature of attention K/V activations.

        Args:
            num_levels: Number of quantization levels (default 16 for 4-bit).

        Returns:
            codebook: [num_levels] sorted float32 tensor of level values.
        """
        k = torch.arange(0.5, num_levels, dtype=torch.float32)
        levels = torch.sin(math.pi * k / (2.0 * num_levels)) ** 2
        # Map from [0, 1] to [-1, 1]
        levels = 2.0 * levels - 1.0
        return levels  # [num_levels] in [-1, 1]

    @staticmethod
    def rotate(data: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Rotate activations using the pre-computed orthogonal matrix.

        Args:
            data: [*, dim] tensor.
            rotation_matrix: [dim, dim] orthogonal matrix.

        Returns:
            Rotated data with same shape.
        """
        return data @ rotation_matrix

    @staticmethod
    def dequantize(
        indices: torch.Tensor,
        norms: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize: map uint8 indices back to float32 via codebook look-up,
        then scale by per-channel norm.

        Args:
            indices: uint8 tensor of quantization indices.
            norms: float32 tensor of per-channel scaling factors.
            codebook: [num_levels] float32 tensor of level values.

        Returns:
            dequantized: float32 tensor with same shape as indices.
        """
        # indices may be multi-D (e.g., [N, D] or [N, H]).
        # codebook is 1D [num_levels]; use index_select for proper look-up.
        flat_indices = indices.reshape(-1).to(torch.int64)  # cast uint8 to int64
        lookup = torch.index_select(codebook, 0, flat_indices)  # [N*...]
        levels = lookup.reshape(indices.shape)
        # norms may be 1D (e.g. [N] for [N, D] indices) — expand to [N, 1]
        # for proper broadcasting with [N, D]
        if norms.dim() < levels.dim():
            norms = norms.view(
                *norms.shape, *([1] * (levels.dim() - norms.dim())),
            )
        return levels * norms  # broadcast norms

    @staticmethod
    def quantize(
        data: torch.Tensor,
        rotation_matrix: torch.Tensor,
        codebook: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full quantization pipeline:
        1. Rotate data with QR matrix.
        2. Compute per-channel l2 norm.
        3. Normalize each channel by its norm -> values in [-1, 1].
        4. Map to nearest codebook level via argmin.

        Args:
            data: [*, dim] tensor (e.g. [seq_len, head_dim] or
                  [batch, heads, seq_len, head_dim]).
            rotation_matrix: [dim, dim] orthogonal matrix.
            codebook: [num_levels] float32 level values.

        Returns:
            indices: uint8 tensor of same shape as data.
            norms: float32 tensor of per-channel norms, shape [*, 1] squeezed.
        """
        # 1. Rotate: [*, D] -> [*, D]
        rotated = data @ rotation_matrix

        # 2. Compute per-element magnitude: rotate each element, then norm
        # along last dim.  Shape: [*, 1] for proper broadcasting
        # with [*, D]
        norms = rotated.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [*, 1]

        # 3. Normalize: [*, D]
        normalized = rotated / norms  # values in ~[-1, 1]

        # 4. Quantize: find closest codebook level for EACH element
        # independently.
        # For position (i, j) in [N, D], find the nearest codebook level:
        #   level = argmin_k |normalized[i,j] - codebook[k]|
        # Result: one index per element -> [N, D]
        normalized_flat = normalized.reshape(-1)  # [N * D]
        codebook_flat = codebook.view(1, -1)  # [1, num_levels]
        diff = (normalized_flat[:, None] - codebook_flat).abs()  # [N*D, num_levels]
        indices_flat = diff.argmin(dim=-1).to(torch.uint8)  # [N*D]
        indices = indices_flat.reshape(normalized.shape)  # [N, D]

        return indices.reshape(normalized.shape), norms.squeeze(-1)


# ---------------------------------------------------------------------------
# PyTorchTurboQuantCache — KV cache manager
# ---------------------------------------------------------------------------


class PyTorchTurboQuantCache(nn.Module):
    """
    KV cache with TurboQuant compression.

    Stores recent tokens in full precision (residual window) at the
    tail of the sequence and compresses older tokens via 4-bit
    quantization with random rotation and Beta(0.5, 0.5) codebook
    look-up.

    Storage layout in get_kv():
        [0 .. residual_window-1]       = full precision residual tokens
        [residual_window .. size-1]    = dequantized from indices/norms

    Usage:
        >>> cache = PyTorchTurboQuantCache(...)
        >>> for input_seq in stream:
        ...     k, v = compute_kv(input_seq)
        ...     cache.append(k, v)
        ...     k_cached, v_cached = cache.get_kv()
    """

    # Explicit type annotations for nn.Module.register_buffer attributes.
    # pyright does not infer tensor types from register_buffer, so we
    # annotate them here to satisfy static analysis for subscripting
    # (e.g., `self.residual_k[b, pos_in_total, :, :]`).
    residual_k: torch.Tensor
    residual_v: torch.Tensor
    k_indices: torch.Tensor
    k_norms: torch.Tensor
    v_indices: torch.Tensor
    v_norms: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        num_bits: int = NUM_BITS,
        residual_window: int = RESIDUAL_WINDOW,
        batch_size: int = 1,
        auto_clear: bool = False,
    ):
        """
        Initialize the TurboQuant KV cache.

        Args:
            embed_dim: Total model embedding dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head (embed_dim // num_heads).
            max_seq_len: Maximum sequence length the cache can hold.
            num_bits: Number of bits for quantization (default 4 -> 16 levels).
            residual_window: Recent tokens to keep in full precision.
            batch_size: Number of parallel sequences (default 1).
            auto_clear: If True, compact the cache (shift oldest tokens
                out and reduce capacity) each time ``append()`` is called.
                The new buffer always holds up to ``max_seq_len`` tokens.
                If False the buffer is fixed at the initial ``max_seq_len``
                capacity and never shrinks.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._auto_clear = auto_clear
        self.max_seq_len = max_seq_len
        self.num_bits = num_bits
        self.num_levels = 1 << num_bits
        self.residual_window = residual_window
        self.batch_size = batch_size

        self._size = 0
        self._has_compressed = False

        self._size = 0
        self._has_compressed = False

        # Pre-compute rotation matrix (per head_dim) and codebook once.
        # Stored as plain tensors — not nn.Parameter — since they are fixed.
        self.rotation_matrix: torch.Tensor = PyQuantize.get_random_rotation_matrix(
            head_dim, seed=42,
        )  # [head_dim, head_dim] orthogonal
        self.codebook: torch.Tensor = PyQuantize.get_beta_codebook(
            num_levels=self.num_levels,
        )  # [num_levels] in [-1, 1]

        # Initial full-precision (residual) storage:
        # [batch, max_res, num_heads, head_dim]
        # max_res is either residual_window (normal) or max_seq_len when
        # auto_clear is enabled so the buffer can hold the entire sequence
        # in full precision for very short sequences.
        max_res = self.max_seq_len if self._auto_clear else max(1, residual_window)

        self.register_buffer(
            "residual_k",
            torch.zeros(
                batch_size,
                max_res,
                num_heads,
                head_dim,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "residual_v",
            torch.zeros(
                batch_size,
                max_res,
                num_heads,
                head_dim,
                dtype=torch.float32,
            ),
        )

        # Compressed storage (quantized, beyond residual window).
        max_compressed = max(
            0, self.max_seq_len - (max_res if self._auto_clear else residual_window),
        )

        if max_compressed > 0:
            self._has_compressed = True
            self.max_compressed_slots = max_compressed
        else:
            if not self._auto_clear:
                self._has_compressed = False
                self.max_compressed_slots = 1  # always create at least one slot
            else:
                # auto_clear: capacity may be zero if max_seq_len <= residual_window
                self._has_compressed = False
                self.max_compressed_slots = 0

        # k_indices / k_norms: [max_compressed, num_heads]
        if self.max_compressed_slots > 0:
            self.register_buffer(
                "k_indices",
                torch.zeros(
                    self.max_compressed_slots,
                    num_heads,
                    dtype=torch.uint8,
                ),
            )
            self.register_buffer(
                "k_norms",
                torch.zeros(
                    self.max_compressed_slots,
                    num_heads,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "v_indices",
                torch.zeros(
                    self.max_compressed_slots,
                    num_heads,
                    dtype=torch.uint8,
                ),
            )
            self.register_buffer(
                "v_norms",
                torch.zeros(
                    self.max_compressed_slots,
                    num_heads,
                    dtype=torch.float32,
                ),
            )

    @property
    def size(self) -> int:
        """Number of tokens currently stored in the cache."""
        return self._size

    def get_kv(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get K/V tensors from the cache, dequantizing compressed tokens
        on-the-fly.

        Returns:
            k: [batch, num_heads, size, head_dim]
            v: [batch, num_heads, size, head_dim]
        """
        assert self._size > 0, "Cache is empty"

        # --- Build full K/V from residual + dequantized compressed ---
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []

        # Inference-only: process batch 0 (standard autoregressive-decoder
        # pattern where only a single batch item is used).
        for b in range(1):
            tokens: list[torch.Tensor] = []
            v_tokens: list[torch.Tensor] = []

            # 1. Residual tokens (full precision, at the tail)
            # Stored at [0 .. min(size, max_res) - 1]
            n_residual = min(self._size, self.max_res)
            if n_residual > 0:
                # Direct attribute access - class-level annotations satisfy
                # pyright for register_buffer attributes
                res_k = self.residual_k
                res_v = self.residual_v
                tokens.append(
                    res_k[b, :n_residual]
                )  # [n_residual, num_heads, head_dim]
                v_tokens.append(
                    res_v[b, :n_residual]
                )  # [n_residual, num_heads, head_dim]

            # 2. Compressed tokens (dequantized on-the-fly, older than
            #    residual)
            n_compressed = self._size - n_residual
            if n_compressed > 0 and self._has_compressed:
                # Read only the written portion
                written = min(n_compressed, self.k_indices.shape[0])

                # Dequantize: indices [written, num_heads] + norms
                # [written, num_heads] -> dequantized scalars
                # [written, num_heads]
                # Direct attribute access - class-level annotations satisfy
                # pyright for register_buffer attributes
                indices = self.k_indices
                norms = self.k_norms
                deq_k = PyQuantize.dequantize(
                    indices[:written],
                    norms[:written],
                    self.codebook,
                )  # [written, num_heads]

                v_indices = self.v_indices
                v_norms = self.v_norms
                deq_v = PyQuantize.dequantize(
                    v_indices[:written],
                    v_norms[:written],
                    self.codebook,
                )  # [written, num_heads]

                # Expand to [written, num_heads, head_dim] by
                # broadcasting.  Each scalar is duplicated across
                # head_dim (standard quantization stores one scale
                # per channel, and dequantization broadcasts).
                k_expanded = deq_k.unsqueeze(-1).expand(
                    -1, -1, self.head_dim
                )  # [written, num_heads, head_dim]
                v_expanded = deq_v.unsqueeze(-1).expand(
                    -1, -1, self.head_dim
                )  # [written, num_heads, head_dim]

                # Order: compressed first (older tokens at head),
                # then residual (tail)
                tokens.append(k_expanded)
                v_tokens.append(v_expanded)

            # Concatenate all tokens for this batch: [size, num_heads,
            # head_dim]
            if tokens:
                full_k = torch.cat(tokens, dim=0)
                full_v = torch.cat(v_tokens, dim=0)
            else:
                full_k = torch.zeros(
                    0, self.num_heads, self.head_dim, dtype=torch.float32
                )
                full_v = torch.zeros(
                    0, self.num_heads, self.head_dim, dtype=torch.float32
                )

            k_list.append(full_k)
            v_list.append(full_v)

        # Stack: [batch, size, num_heads, head_dim] ->
        # [batch, num_heads, size, head_dim]
        k_out = torch.stack(
            k_list, dim=0,
        )  # [batch, size, num_heads, head_dim]
        v_out = torch.stack(
            v_list, dim=0,
        )  # [batch, size, num_heads, head_dim]

        k_out = k_out.transpose(1, 2)  # [batch, num_heads, size, head_dim]
        v_out = v_out.transpose(1, 2)

        return k_out, v_out

    def _dequantize_indices(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """
        Dequantize stored indices + norms back to float32 tokens.

        Args:
            indices: [compressed_size, num_heads] uint8 indices.
            norms: [compressed_size, num_heads] float32 norms.
            offset: Starting position in the cache.

        Returns:
            tokens: [compressed_size, num_heads, head_dim] float32 tensor.
        """
        deq = PyQuantize.dequantize(indices, norms, self.codebook)
        # deq: [compressed_size, num_heads] ->
        # [compressed_size, num_heads, head_dim]
        return deq.unsqueeze(-1).expand(-1, -1, self.head_dim)

    @property
    def max_res(self) -> int:
        """Maximum number of full-precision (residual) tokens currently stored."""
        if self._auto_clear:
            # With auto_clear the buffer may be smaller than
            # self.residual_window — it tracks the actual capacity.
            return min(self.max_seq_len, self.residual_window)
        return self.residual_window

    def _realloc_buffers(self, max_res: int, max_comp: int) -> None:
        """Reallocate storage buffers to a new capacity.

        When ``auto_clear`` is enabled the buffers may need to grow or
        shrink dynamically.  We move tensors to plain attributes so their
        shapes can change.

        Args:
            max_res: New full-precision (residual) capacity.
            max_comp: New compressed-slot capacity.
        """
        if self.batch_size <= 0:
            return

        self.residual_k = torch.zeros(
            self.batch_size, max_res, self.num_heads, self.head_dim,
            dtype=torch.float32,
        )
        self.residual_v = torch.zeros(
            self.batch_size, max_res, self.num_heads, self.head_dim,
            dtype=torch.float32,
        )
        if max_comp > 0:
            self.k_indices = torch.zeros(max_comp, self.num_heads, dtype=torch.uint8)
            self.k_norms = torch.zeros(max_comp, self.num_heads, dtype=torch.float32)
            self.v_indices = torch.zeros(max_comp, self.num_heads, dtype=torch.uint8)
            self.v_norms = torch.zeros(max_comp, self.num_heads, dtype=torch.float32)
        else:
            self.k_indices = torch.zeros((0, self.num_heads), dtype=torch.uint8)
            self.k_norms = torch.zeros((0, self.num_heads), dtype=torch.float32)
            self.v_indices = torch.zeros((0, self.num_heads), dtype=torch.uint8)
            self.v_norms = torch.zeros((0, self.num_heads), dtype=torch.float32)
        self._has_compressed = max_comp > 0
        self.max_compressed_slots = max_comp

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Append new K/V tokens to the cache.

        Handles:
        - Storing tokens in the residual window (full precision).
        - Quantizing and storing older tokens (compressed).
        - Context compression when ``auto_clear`` is True (new tokens are
          appended, then the oldest tokens are compacted so that the
          total number of cached tokens never exceeds ``max_seq_len``).

        Args:
            k: New key tokens — [batch, num_heads, seq_len, head_dim]
            v: New value tokens — [batch, num_heads, seq_len, head_dim]
        """
        batch, heads, seq_len, hd = k.shape
        assert hd == self.head_dim
        assert heads == self.num_heads

        if seq_len == 0:
            return

        # ---- Context-compression (auto_clear) ----
        # If adding these tokens would exceed max_seq_len we compact
        # (shift the oldest tokens out) so that the new tokens are kept.
        if self._auto_clear and self._size + seq_len > self.max_seq_len:
            excess = self._size + seq_len - self.max_seq_len

            # Compact: shift everything by ``excess`` in favour of new data.
            self._compact(excess)

        # ---- Normal append ----
        # Inference-only: process batch 0 following the standard
        # autoregressive-decoder pattern.  Each position is counted
        # once in ``_size``.
        for t in range(seq_len):
            if self._size >= self.max_seq_len:
                break  # Cache is full

            pos_in_total = self._size

            k_token = k[0, :, t, :]  # [num_heads, head_dim]
            v_token = v[0, :, t, :]  # [num_heads, head_dim]

            if pos_in_total < self.max_res:
                # Store full precision in residual array
                self.residual_k[0, pos_in_total, :, :].copy_(k_token)  # pyright: ignore[reportIndexIssue]
                self.residual_v[0, pos_in_total, :, :].copy_(v_token)  # pyright: ignore[reportIndexIssue]
            else:
                # Quantize and store in compressed arrays
                if not self._has_compressed:
                    break  # No compressed storage, skip remaining

                compressed_offset = pos_in_total - self.max_res

                # Rotate each head's head_dim vector by the rotation matrix
                k_rotated = k_token @ self.rotation_matrix  # [heads, head_dim]
                v_rotated = v_token @ self.rotation_matrix  # [heads, head_dim]

                # Compute per-head l2 norm: [num_heads]
                k_norm = k_rotated.norm(dim=-1).clamp(min=1e-8)
                v_norm = v_rotated.norm(dim=-1).clamp(min=1e-8)

                # Normalize by norm: [heads, head_dim]
                k_normalized = k_rotated / k_norm[:, None]
                v_normalized = v_rotated / v_norm[:, None]

                # Quantize each element: [heads * head_dim] -> indices
                k_flat = k_normalized.reshape(-1)
                v_flat = v_normalized.reshape(-1)

                # For each element, find closest codebook level
                k_codebook = self.codebook
                k_diff = k_flat[:, None] - k_codebook[None, :]
                k_idx = k_diff.abs().argmin(dim=-1).to(torch.uint8)

                v_codebook = self.codebook
                v_diff = v_flat[:, None] - v_codebook[None, :]
                v_idx = v_diff.abs().argmin(dim=-1).to(torch.uint8)

                # Store
                k_idx_s = k_idx[:heads]
                k_norm_s = k_norm[:heads]
                v_idx_s = v_idx[:heads]
                v_norm_s = v_norm[:heads]

                if compressed_offset < self.k_indices.shape[0]:
                    self.k_indices[compressed_offset].copy_(k_idx_s)  # pyright: ignore[reportIndexIssue]
                    self.k_norms[compressed_offset].copy_(k_norm_s)  # pyright: ignore[reportIndexIssue]
                    self.v_indices[compressed_offset].copy_(v_idx_s)  # pyright: ignore[reportIndexIssue]
                    self.v_norms[compressed_offset].copy_(v_norm_s)  # pyright: ignore[reportIndexIssue]

            self._size += 1

    def _compact(self, num_to_remove: int) -> None:
        """Shift cached data by ``num_to_remove`` in favour of newer tokens.

        Used internally for context compression.  After compacting, the
        oldest ``num_to_remove`` tokens are discarded and the remaining
        slots shift forward.

        Args:
            num_to_remove: Number of oldest tokens to drop.
        """
        if num_to_remove <= 0 or num_to_remove >= self._size:
            # Nothing to compact, or would empty the cache
            self._size = max(0, self._size - num_to_remove)
            # Zero out residual storage (keep shape)
            self.residual_k[:, : self.max_res].zero_()
            self.residual_v[:, : self.max_res].zero_()
            if self._has_compressed:
                self.k_indices.zero_()
                self.k_norms.zero_()
                self.v_indices.zero_()
                self.v_norms.zero_()
            return

        new_size = self._size - num_to_remove
        old_max_res = self.max_res
        old_max_comp = self.max_compressed_slots

        if num_to_remove >= old_max_res:
            # All residual data is removed — shift everything down
            shifted = self.residual_k[:, num_to_remove:old_max_res + num_to_remove][:, :new_size].clone()
            self.residual_k[:, :new_size].copy_(shifted)
            shifted_v = self.residual_v[:, num_to_remove:old_max_res + num_to_remove][:, :new_size].clone()
            self.residual_v[:, :new_size].copy_(shifted_v)
        else:
            # Some residual data survives
            shifted = self.residual_k[:, num_to_remove:old_max_res].clone()
            self.residual_k[:, :old_max_res - num_to_remove].copy_(shifted)
            shifted_v = self.residual_v[:, num_to_remove:old_max_res].clone()
            self.residual_v[:, :old_max_res - num_to_remove].copy_(shifted_v)

        # Compact compressed storage
        if self._has_compressed:
            old_comp_start = max(0, old_max_res - num_to_remove)
            # Number of compressed slots to keep
            old_comp = old_max_comp
            new_comp_slots = max(0, old_comp - num_to_remove)

            if old_comp > 0 and new_comp_slots > 0:
                # Shift compressed data
                self.k_indices[:new_comp_slots].copy_(
                    self.k_indices[old_comp_start:old_comp_start + new_comp_slots]
                )
                self.k_norms[:new_comp_slots].copy_(
                    self.k_norms[old_comp_start:old_comp_start + new_comp_slots]
                )
                self.v_indices[:new_comp_slots].copy_(
                    self.v_indices[old_comp_start:old_comp_start + new_comp_slots]
                )
                self.v_norms[:new_comp_slots].copy_(
                    self.v_norms[old_comp_start:old_comp_start + new_comp_slots]
                )

            # Zero out remaining compressed slots
            if new_comp_slots < old_max_comp:
                self.k_indices[new_comp_slots:].zero_()
                self.k_norms[new_comp_slots:].zero_()
                self.v_indices[new_comp_slots:].zero_()
                self.v_norms[new_comp_slots:].zero_()

        self._size = new_size

    def compact_cache(self) -> None:
        """Compact the cache, moving the oldest tokens out.

        When ``auto_clear`` is set the cache may dynamically grow and
        shrink.  This method shifts the oldest data out so that the cache
        always holds at most ``max_seq_len`` tokens.

        This is useful **after** a prompt has finished — during
        autoregressive decoding the newest tokens are at the tail and
        compacting pushes them further back, which means the next
        ``append()`` call will have more room for new tokens.

        In typical usage (short prompts, moderate sequence lengths), the
        residual window (``residual_window``) tokens are kept in full
        precision and everything older is compressed (4-bit).  After
        compaction the full-precision window slides forward.
        """
        if self._size > self.max_seq_len:
            self._compact(self._size - self.max_seq_len)

    def reset(self) -> None:
        """Clear the cache, resetting size to zero."""
        self._size = 0
        # Direct attribute access - class-level annotations satisfy pyright
        self.residual_k.zero_()
        self.residual_v.zero_()
        if self._has_compressed and self.max_compressed_slots > 0:
            self.k_indices.zero_()
            self.k_norms.zero_()
            self.v_indices.zero_()
            self.v_norms.zero_()
