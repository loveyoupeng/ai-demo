# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Triton track: Python wiring around the Triton attention/FFN kernels.

Mirrors the NumPy/PyTorch tracks operator for operator (same block layout,
same key scheme, same math) so the cross-backend parity tests can load
weights across tracks. The attention core runs on the Triton
``scaled_dot_product_attention`` kernel and the feed-forward on the Triton
``swiglu_ffn`` kernel; everything else is plain PyTorch wiring.

Shared-seam note: ``RoPE`` is imported from ``impl._torch.layers`` — the
triton track reuses the torch building blocks it does not kernelize
(a deliberate seam, documented; see docs).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from impl._torch.layers import RoPE
from impl._triton.attn import scaled_dot_product_attention
from impl._triton.ffn import swiglu_ffn
from shared.config import TransformerConfig
from shared.constants import Attn, Keys, LayerNorm, Mlp
from shared.result import Result  # shared-seam: torch building block reused by triton
from shared.torch_softmax import stable_softmax_t

logger = logging.getLogger(__name__)


def copy_in_place(target: torch.Tensor, value: np.ndarray | None, *, transpose: bool = False) -> None:
    """Copy one checkpoint array into ``target``'s storage in place.

    The triton track's constructor-injection primitive: a model built as
    ``TritonModel(config, params)`` is *born* with its trained weights, so
    every operator owns the storage it was constructed with and there is no
    second lifecycle (no post-hoc ``load_from_numpy_dict`` step to forget).

    Parameters
    ----------
    target : torch.Tensor
        The live parameter tensor (or ``nn.Linear`` weight) the constructor
        created; its storage is written through ``target.data``.
    value : np.ndarray | None
        The checkpoint array for this key (``shared.constants.Keys`` scheme,
        ``(in, out)`` layout). None is the training path: ``target`` keeps
        its seed initialization (e.g. RMSNorm gains stay ones).
    transpose : bool, keyword-only
        True when ``target`` holds the transposed ``(out, in)`` layout of
        ``nn.Linear`` (attention projections, MoE router, lm_head).

    Notes
    -----
    Nothing is rebound and no caller array is aliased — the usual checkpoint
    layout is (in, out) whereas ``nn.Linear`` is (out, in), so those weights
    are transposed (``.T``) on the way in, exactly like ``save_as_numpy`` /
    ``load_from_numpy_dict`` do through the registry's ``torch_transpose``.
    """
    if value is None:
        return
    array = np.ascontiguousarray(value.T if transpose else value)
    # PROD: copy_ casts to the target dtype, so a float64 checkpoint loads
    # into the float32 model without a separate conversion pass.
    with torch.no_grad():
        target.data.copy_(torch.from_numpy(array))


class TritonMultiHeadAttention(nn.Module):
    """Multi-head attention with GQA and RoPE, on the Triton SDPA kernel.

    Same math as ``impl._torch.layers.MultiHeadAttention``:

        q = x @ Wq   (B, S, H*hd)  → (B, H, S, hd)
        k = x @ Wk   (B, S, G*hd)  → (B, G, S, hd)
        v = x @ Wv   (B, S, G*hd)  → (B, G, S, hd)
        q, k ← RoPE(q), RoPE(k)
        if G < H: repeat k, v so head h uses group h % G  → (B, H, S, hd)
        ctx = triton_sdpa(q, k, v)      (B, H, S, hd)
        out = ctx @ Wo                  (B, S, D)

    The Triton kernel computes scaled dot-product attention (Q @ K^T /
    sqrt(hd) → softmax → @ V) without materializing the (S, S) score matrix
    in the same memory layout as a naive implementation (tiled).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_index: int = 0,
        q_proj: np.ndarray | None = None,
        k_proj: np.ndarray | None = None,
        v_proj: np.ndarray | None = None,
        o_proj: np.ndarray | None = None,
        kv_down_proj: np.ndarray | None = None,
        k_up_proj: np.ndarray | None = None,
        v_up_proj: np.ndarray | None = None,
    ) -> None:
        """Build the projections, injecting this layer's trained weights.

        Classic mode: q/k/v/o no-bias Linears. MLA mode (ADR 0009,
        config.mla_latent_dim > 0): k/v_proj are NOT created; the latent
        factorization kv_down (D→L), k_up (L→G·hd), v_up (L→G·hd) replaces
        them and the cache stores only the latent c.

        All optional arrays are checkpoint arrays in the ``(in, out)``
        layout of ``shared.constants.Keys``, copied IN PLACE over the
        ``nn.Linear`` storage — the ``(out, in)`` layout, hence the
        transpose. None (default) keeps the seeded init (the training path).
        """
        super().__init__()
        D, H, G = config.embed_dim, config.n_heads, config.kv_heads
        hd = config.head_dim
        self.n_heads = H
        self.n_groups = G
        self.head_dim = hd
        self.rope_dim = config.rope_dim
        self.swa_window = config.swa_window
        self.mla_latent_dim = config.mla_latent_dim if config.has_mla_at(layer_index) else 0

        # No-bias projections (Llama convention); nn.Linear stores (out, in)
        self.q_proj = nn.Linear(D, H * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, D, bias=False)
        if self.mla_latent_dim > 0:
            L = self.mla_latent_dim
            self.kv_down_proj: nn.Linear | None = nn.Linear(D, L, bias=False)
            self.k_up_proj: nn.Linear | None = nn.Linear(L, G * hd, bias=False)
            self.v_up_proj: nn.Linear | None = nn.Linear(L, G * hd, bias=False)
            self.k_proj: nn.Linear | None = None
            self.v_proj: nn.Linear | None = None
        else:
            self.k_proj = nn.Linear(D, G * hd, bias=False)
            self.v_proj = nn.Linear(D, G * hd, bias=False)
            self.kv_down_proj = None
            self.k_up_proj = None
            self.v_up_proj = None
        copy_in_place(self.q_proj.weight, q_proj, transpose=True)
        copy_in_place(self.o_proj.weight, o_proj, transpose=True)
        if self.k_proj is not None:
            copy_in_place(self.k_proj.weight, k_proj, transpose=True)
            copy_in_place(self.v_proj.weight, v_proj, transpose=True)  # type: ignore[union-attr]
        if self.kv_down_proj is not None:
            copy_in_place(self.kv_down_proj.weight, kv_down_proj, transpose=True)
            copy_in_place(self.k_up_proj.weight, k_up_proj, transpose=True)  # type: ignore[union-attr]
            copy_in_place(self.v_up_proj.weight, v_up_proj, transpose=True)  # type: ignore[union-attr]
        self.rope = RoPE()

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move projection weights to x's device/dtype (triton kernels need it)."""
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        modules = [
            m
            for m in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.o_proj,
                self.kv_down_proj,
                self.k_up_proj,
                self.v_up_proj,
            )
            if m is not None
        ]
        if dtype is None:
            for module in modules:
                module.to(device)
        else:
            for module in modules:
                module.to(device, dtype)

    def forward(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        """MHA forward. x: (B, S, D) → out: (B, S, D); position: this
        pass's first absolute token index."""
        self._move_to_device(x)
        B, S, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        # (B, S, D) → (B, S, {H,G}*hd)
        q = self.q_proj(x)
        if self.mla_latent_dim > 0:
            assert self.kv_down_proj is not None and self.k_up_proj is not None and self.v_up_proj is not None
            c = self.kv_down_proj(x)
            k = self.k_up_proj(c)
            v = self.v_up_proj(c)
        else:
            assert self.k_proj is not None and self.v_proj is not None
            k = self.k_proj(x)
            v = self.v_proj(x)

        # Split into heads: (B, S, {H,G}*hd) → (B, {H,G}, S, hd)
        q = q.view(B, S, H, hd).permute(0, 2, 1, 3)  # (B, H, S, hd)
        k = k.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)
        v = v.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)

        # RoPE on q and k (RoPE's contract shape is (B, S, H, D)).
        q = self.rope(q.permute(0, 2, 1, 3), position, rope_dim=self.rope_dim).permute(0, 2, 1, 3)
        k = self.rope(k.permute(0, 2, 1, 3), position, rope_dim=self.rope_dim).permute(0, 2, 1, 3)

        # GQA: broadcast each K/V group to its H // G query heads.
        if G != H:
            k = k.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
            v = v.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)

        # Triton SDPA kernel: (B, H, S, hd) → (B, H, S, hd)
        ctx = scaled_dot_product_attention(q, k, v, is_causal=True, window=self.swa_window)

        # Merge heads: (B, H, S, hd) → (B, S, H*hd) → (B, S, D)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, S, H * hd)
        return self.o_proj(ctx)  # (B, S, D)

    def forward_state(self, x: torch.Tensor, position: int = 0) -> Result[torch.Tensor, dict]:
        """Forward pass plus the raw K/V intermediates the cache needs.

        Mirrors ``impl._torch.layers.MultiHeadAttention.forward_state``:
        ``forward`` calls this and drops the state; the prefill path
        (via the block) requests it and backfills the per-layer cache —
        no second pass. Same ``Result`` contract (shared/result.py).

        ``.value`` = out (B, S, D); ``.ctx.display`` = the per-group,
        RoPE'd K and the un-rotated V:
            "k_group": (B, G, S, hd)  "v_group": (B, G, S, hd)
        """
        self._move_to_device(x)
        B, S, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        q = self.q_proj(x)  # (B, S, H*hd)
        if self.mla_latent_dim > 0:
            assert self.kv_down_proj is not None and self.k_up_proj is not None and self.v_up_proj is not None
            c = self.kv_down_proj(x)  # (B, S, L) — the latent
            k = self.k_up_proj(c)  # (B, S, G*hd)
            v = self.v_up_proj(c)  # (B, S, G*hd)
            c_group = c  # (B, S, L) — the cacheable latent
        else:
            assert self.k_proj is not None and self.v_proj is not None
            k = self.k_proj(x)  # (B, S, G*hd)
            v = self.v_proj(x)  # (B, S, G*hd)
            c = None
            c_group = None

        q = q.view(B, S, H, hd).permute(0, 2, 1, 3)  # (B, H, S, hd)
        k = k.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)
        v = v.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)

        q = self.rope(q.permute(0, 2, 1, 3), position, rope_dim=self.rope_dim).permute(0, 2, 1, 3)
        k = self.rope(k.permute(0, 2, 1, 3), position, rope_dim=self.rope_dim).permute(0, 2, 1, 3)

        k_group, v_group = k, v  # (B, G, S, hd) — the cacheable per-group form

        if G != H:
            k_full = k_group.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
            v_full = v_group.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
        else:
            k_full, v_full = k_group, v_group

        ctx = scaled_dot_product_attention(q, k_full, v_full, is_causal=True, window=self.swa_window)  # (B, H, S, hd)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, S, H * hd)
        out = self.o_proj(ctx)  # (B, S, D)
        return Result.capture(out, {}, {"k_group": k_group, "v_group": v_group, "c_group": c_group})

    def forward_step(self, x: torch.Tensor, position: int, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mla_latent_dim > 0:
            missing = [k for k in ("c",) if k not in cache]
            assert not missing, f"MLA forward_step needs a latent {{'c'}} cache (missing: {missing})"
        else:
            missing = [k for k in ("k", "v") if k not in cache]
            assert not missing, f"forward_step needs a naive {{'k','v'}} cache (missing keys: {missing})"
        """Process ONE new token against the cached K/V (per-token inference path).

        Mirrors ``impl._torch.layers.MultiHeadAttention.forward_step``: the
        token's K/V are *appended* to the cache (per-group — GQA keeps the
        cache H // G times smaller), and attention runs against the cached
        (B, G, t, hd) K/V with a single query row.

        x: (B, 1, D) the new token's embedding.
        position: the absolute token index (0-based; RoPE angles for p).
        cache: this layer's dict {"k": (B, G, t, hd), "v": (B, G, t, hd)},
            mutated in place (K/V appended).

        Returns: out (B, 1, D).
        """
        self._move_to_device(x)
        B = x.shape[0]
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        # One token's Q/K/V: (B, 1, D) → (B, ·, 1, hd)
        q = self.q_proj(x).view(B, 1, H, hd).permute(0, 2, 1, 3)  # (B, H, 1, hd)
        if self.mla_latent_dim > 0:
            assert self.kv_down_proj is not None and self.k_up_proj is not None and self.v_up_proj is not None
            c = self.kv_down_proj(x)  # (B, 1, L) — the latent write
            k_pre = self.k_up_proj(c)
            v_pre = self.v_up_proj(c)
        else:
            assert self.k_proj is not None and self.v_proj is not None
            c = None
            k_pre = self.k_proj(x)
            v_pre = self.v_proj(x)
        k = k_pre.view(B, 1, G, hd).permute(0, 2, 1, 3)  # (B, G, 1, hd)
        v = v_pre.view(B, 1, G, hd).permute(0, 2, 1, 3)  # (B, G, 1, hd)

        # RoPE with the token's absolute position (contract: (B, S, H, D))
        q = self.rope(q.permute(0, 2, 1, 3), position, rope_dim=self.rope_dim).permute(0, 2, 1, 3)
        k = self.rope(k.permute(0, 2, 1, 3), position, rope_dim=self.rope_dim).permute(0, 2, 1, 3)

        # Append the new K/V — or, in MLA mode, the latent c (B, L, t+1);
        # K/V are reconstructed from the whole cached latent.
        if self.mla_latent_dim > 0:
            assert self.k_up_proj is not None and self.v_up_proj is not None
            assert c is not None
            cache["c"] = torch.cat([cache["c"], c.transpose(1, 2)], dim=2)  # (B, L, t+1)
            c_cache = cache["c"].transpose(1, 2)  # (B, t+1, L)
            k = self.k_up_proj(c_cache).view(B, -1, G, hd).permute(0, 2, 1, 3)  # (B, G, t+1, hd)
            v = self.v_up_proj(c_cache).view(B, -1, G, hd).permute(0, 2, 1, 3)
            k = self.rope(k.permute(0, 2, 1, 3), 0, rope_dim=self.rope_dim).permute(0, 2, 1, 3)
        else:
            cache["k"] = torch.cat([cache["k"], k], dim=2)  # (B, G, t+1, hd)
            cache["v"] = torch.cat([cache["v"], v], dim=2)  # (B, G, t+1, hd)
            k, v = cache["k"], cache["v"]

        # GQA: broadcast each cached K/V group to its H // G query heads.
        k_full, v_full = k, v
        if G != H:
            k_full = k_full.repeat_interleave(H // G, dim=1)  # (B, H, t+1, hd)
            v_full = v_full.repeat_interleave(H // G, dim=1)  # (B, H, t+1, hd)

        # SWA (ADR 0008) decode: slice the last w+1 cache rows — O(w) work.
        if self.swa_window > 0:
            start = max(0, k_full.shape[2] - (self.swa_window + 1))
            k_full, v_full = k_full[:, :, start:], v_full[:, :, start:]

        # Single query row: causal masking is implicit (all visible cached
        # positions precede the query).
        ctx = scaled_dot_product_attention(q, k_full, v_full)  # (B, H, 1, hd)

        # Merge heads: (B, H, 1, hd) → (B, 1, H*hd) → (B, 1, D)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, 1, H * hd)
        return self.o_proj(ctx)  # (B, 1, D)

    def forward_chunk(self, x: torch.Tensor, position: int, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a CHUNK of c tokens against the cached K/V (verification path).

        Mirrors ``impl._torch.layers.MultiHeadAttention.forward_chunk`` (which
        mirrors ``impl._np.attention.MultiHeadAttention.forward_chunk``) — the
        speculative-decoding verification pass (ADR 0003): one call appends
        all c tokens' K/V and scores every chunk position through the Triton
        SDPA kernel. Naive cache only.

        x: (B, c, D) the chunk's embeddings, c >= 1.
        position: the absolute index of the chunk's FIRST token (0-based).
        cache: this layer's dict {"k": (B, G, t, hd), "v": (B, G, t, hd)};
            mutated in place — the chunk's K/V are appended before attention.

        Returns: out (B, c, D).

        Masking: the Triton kernel's causal mask is row-absolute
        (``k_idx <= q_idx``), which over-blocks a chunk whose queries sit at
        rows 0..c-1 while its keys extend to t+c. The exact fix mirrors the
        torch shape math: left-pad the queries with t masked dummy rows so
        row-absolute causality IS the block-causal mask (dummy rows attend
        to nothing — their whole row is -inf), run the kernel, and crop.
        """
        self._move_to_device(x)
        B, c, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        q = self.q_proj(x).view(B, c, H, hd).permute(0, 2, 1, 3)  # (B, H, c, hd)
        if self.mla_latent_dim > 0:
            assert self.kv_down_proj is not None and self.k_up_proj is not None and self.v_up_proj is not None
            c_lat = self.kv_down_proj(x)  # (B, c, L) — the latent write
            k = self.k_up_proj(c_lat)
            v = self.v_up_proj(c_lat)
        else:
            assert self.k_proj is not None and self.v_proj is not None
            c_lat = None
            k = self.k_proj(x)
            v = self.v_proj(x)
        k = k.view(B, c, G, hd).permute(0, 2, 1, 3)  # (B, G, c, hd)
        v = v.view(B, c, G, hd).permute(0, 2, 1, 3)  # (B, G, c, hd)

        # RoPE the whole chunk at once (RoPE's contract: (B, S, H, hd)).
        q = self.rope(q.permute(0, 2, 1, 3), position, rope_dim=self.rope_dim).permute(0, 2, 1, 3)
        k = self.rope(k.permute(0, 2, 1, 3), position, rope_dim=self.rope_dim).permute(0, 2, 1, 3)

        # Append the chunk's K/V — or, in MLA mode, the latent c (B, L, t+c);
        # K/V are reconstructed from the whole cached latent.
        if self.mla_latent_dim > 0:
            assert self.k_up_proj is not None and self.v_up_proj is not None
            assert c_lat is not None
            cache["c"] = torch.cat([cache["c"], c_lat.transpose(1, 2)], dim=2)  # (B, L, t+c)
            c_cache = cache["c"].transpose(1, 2)  # (B, t+c, L)
            k = self.k_up_proj(c_cache).view(B, -1, G, hd).permute(0, 2, 1, 3)  # (B, G, t+c, hd)
            v = self.v_up_proj(c_cache).view(B, -1, G, hd).permute(0, 2, 1, 3)
            k = self.rope(k.permute(0, 2, 1, 3), 0, rope_dim=self.rope_dim).permute(0, 2, 1, 3)
        else:
            cache["k"] = torch.cat([cache["k"], k], dim=2)  # (B, G, t+c, hd)
            cache["v"] = torch.cat([cache["v"], v], dim=2)  # (B, G, t+c, hd)
            k, v = cache["k"], cache["v"]

        # GQA: broadcast each cached K/V group to its H // G query heads.
        k_full, v_full = k, v
        if G != H:
            k_full = k_full.repeat_interleave(H // G, dim=1)  # (B, H, t+c, hd)
            v_full = v_full.repeat_interleave(H // G, dim=1)  # (B, H, t+c, hd)

        t = k_full.shape[2] - c
        # Left-pad the queries with t dummy rows (zeros → -inf scores after
        # the causal mask; softmax rows are uniform garbage but discarded).
        q_pad = torch.zeros((B, H, t + c, hd), device=x.device, dtype=q.dtype)
        q_pad[:, :, t:, :] = q
        ctx = scaled_dot_product_attention(
            q_pad, k_full, v_full, is_causal=True, window=self.swa_window
        )  # (B, H, t+c, hd)
        ctx = ctx[:, :, t:, :]  # (B, H, c, hd) — crop the dummy rows

        # Merge heads: (B, H, c, hd) → (B, c, H*hd) → project to (B, c, D)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, c, H * hd)
        return self.o_proj(ctx)  # (B, c, D)


class TritonGatedDeltaNet(nn.Module):
    """Gated DeltaNet mixer on the Triton track (ADR 0006/0007).

    Same math as ``impl._torch.layers.GatedDeltaNet`` / the NumPy reference
    (``impl/_np/deltanet.py``). PROD note: the recurrent scan is plain torch
    ops here — a fused Triton chunked-WY kernel is the production path; at
    teaching scale (ctx <= 128, D = 64) the loop is fine and keeps the
    recurrence readable. The projections themselves are cuBLAS matmuls,
    which run on tensor cores under BF16.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        layer_index: int | None = None,
        mixer_index: int | None = None,
        params: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        from impl._torch.layers import GatedDeltaNet

        self._impl = GatedDeltaNet(embed_dim, n_heads, layer_index=layer_index, mixer_index=mixer_index, params=params)
        # register the inner module's parameters under this module
        self.add_module("impl", self._impl)

    def forward_state(self, x: torch.Tensor, position: int = 0) -> Result[torch.Tensor, dict]:
        return self._impl.forward_state(x, position)

    def forward(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        return self._impl.forward(x, position)

    def forward_step(self, x: torch.Tensor, position: int, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._impl.forward_step(x, position, cache)

    def forward_chunk(self, x: torch.Tensor, position: int, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._impl.forward_chunk(x, position, cache)

    def _move_to_device(self, x: torch.Tensor) -> None:
        self._impl.to(device=x.device, dtype=x.dtype if x.dtype.is_floating_point else None)


class TritonSwiGLUFFN(nn.Module):
    """Dense SwiGLU feed-forward, computed by the triton ``swiglu_ffn`` kernel.

    gate = SiLU(x @ gate_proj); up = x @ up_proj; out = (gate * up) @ down_proj.
    Weights stored in (in, out) layout (matching the NumPy reference):
    gate_proj, up_proj: (D, FF); down_proj: (FF, D).
    """

    __slots__ = ("gate_proj", "up_proj", "down_proj")

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        gate_proj: np.ndarray | None = None,
        up_proj: np.ndarray | None = None,
        down_proj: np.ndarray | None = None,
    ) -> None:
        """Create the three (in, out) projection weights.

        gate_proj/up_proj/down_proj: optional checkpoint arrays injected AT
        CONSTRUCTION (copied in place over the seeded storage; the ``(in,
        out)`` layout already matches the parameter layout). None (default)
        keeps the seeded init — the training path.
        """
        super().__init__()
        self.gate_proj = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.up_proj = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.down_proj = nn.Parameter(torch.empty(ff_dim, embed_dim))
        self.reset_parameters()
        copy_in_place(self.gate_proj, gate_proj)
        copy_in_place(self.up_proj, up_proj)
        copy_in_place(self.down_proj, down_proj)

    def reset_parameters(self) -> None:
        """Initialize all weights with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))

    def _move_to_device(self, x: torch.Tensor) -> None:
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is None:
            for p in [self.gate_proj, self.up_proj, self.down_proj]:
                p.data = p.data.to(device)
        else:
            for p in [self.gate_proj, self.up_proj, self.down_proj]:
                p.data = p.data.to(device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward via the triton kernel. x: (..., D) → out: (..., D)."""
        self._move_to_device(x)
        # kernel arg order: (x, gate_w, up_w, down_w)
        return swiglu_ffn(x, self.gate_proj, self.up_proj, self.down_proj)


class TritonExpert(nn.Module):
    """One SwiGLU expert of the MoE (same weights as ``TritonSwiGLUFFN``)."""

    __slots__ = ("gate_proj", "up_proj", "down_proj")

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        gate_proj: np.ndarray | None = None,
        up_proj: np.ndarray | None = None,
        down_proj: np.ndarray | None = None,
    ) -> None:
        """Create the expert's three (in, out) projection weights.

        gate_proj/up_proj/down_proj: optional checkpoint arrays injected AT
        CONSTRUCTION (copied in place; the ``(in, out)`` layout already
        matches). None (default) keeps the seeded init — the training path;
        the MoE passes this expert's own ``Keys.moe_expert`` /
        ``Keys.moe_shared_expert`` slice.
        """
        super().__init__()
        self.gate_proj = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.up_proj = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.down_proj = nn.Parameter(torch.empty(ff_dim, embed_dim))
        self.reset_parameters()
        copy_in_place(self.gate_proj, gate_proj)
        copy_in_place(self.up_proj, up_proj)
        copy_in_place(self.down_proj, down_proj)

    def reset_parameters(self) -> None:
        """Initialize all weights with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expert forward via the triton kernel. x: (..., D) → out: (..., D)."""
        return swiglu_ffn(x, self.gate_proj, self.up_proj, self.down_proj)


class TritonMixtureOfExperts(nn.Module):
    """Mixture of Experts on triton FFN kernels.

    Same routing math as the NumPy/PyTorch tracks: router softmax over all
    experts, top-k mask, renormalize, weighted sum of expert outputs.
    Router: gate = nn.Linear(D, E, bias=False) (Mixtral convention).
    """

    def __init__(
        self,
        embed_dim: int,
        n_experts: int,
        ff_dim: int,
        top_k: int,
        n_shared_experts: int = 0,
        layer_index: int | None = None,
        params: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Create the router + expert FFNs, injecting this layer's trained weights.

        layer_index/params: the layer's position in the stack and the flat
        registry-keyed checkpoint. When both are given, every operator is
        born with its trained array — the router from ``Keys.moe_gate(i)``
        and each expert's three projections from ``Keys.moe_expert(i, j,
        proj)`` / ``Keys.moe_shared_expert(i, s, proj)`` — copied in place
        over the storage created here. None (default) keeps the seeded
        init — the training path.
        """
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_shared_experts = n_shared_experts

        def expert_slice(idx: int, *, shared: bool) -> dict[str, np.ndarray]:
            """This expert's trained projections — {} on the training path."""
            if params is None or layer_index is None:
                return {}
            key = Keys.moe_shared_expert if shared else Keys.moe_expert
            return {
                "gate_proj": params[key(layer_index, idx, Mlp.GATE_PROJ)],
                "up_proj": params[key(layer_index, idx, Mlp.UP_PROJ)],
                "down_proj": params[key(layer_index, idx, Mlp.DOWN_PROJ)],
            }

        self.gate = nn.Linear(embed_dim, n_experts, bias=False)
        # The router is nn.Linear (E, D) while the checkpoint keeps (D, E).
        copy_in_place(
            self.gate.weight,
            None if params is None or layer_index is None else params[Keys.moe_gate(layer_index)],
            transpose=True,
        )
        # Typed view for pyright; nn.ModuleList registers parameters for torch
        self.expert_list: list[TritonExpert] = [
            TritonExpert(embed_dim, ff_dim, **expert_slice(j, shared=False)) for j in range(n_experts)
        ]
        self.experts = nn.ModuleList(self.expert_list)
        # Shared experts (ADR 0002): ungated, always active — same Triton FFN
        # kernels, just not selected by the router.
        self.shared_expert_list: list[TritonExpert] = [
            TritonExpert(embed_dim, ff_dim, **expert_slice(s, shared=True)) for s in range(n_shared_experts)
        ]
        self.shared_experts = nn.ModuleList(self.shared_expert_list)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        for expert in self.expert_list + self.shared_expert_list:
            expert.reset_parameters()

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move router + expert weights to x's device/dtype."""
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is None:
            self.gate.to(device)
            for expert in self.expert_list + self.shared_expert_list:
                for p in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                    p.data = p.data.to(device)
        else:
            self.gate.to(device, dtype)
            for expert in self.expert_list + self.shared_expert_list:
                for p in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                    p.data = p.data.to(device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE forward. x: (B, S, D) → out: (B, S, D)."""
        self._move_to_device(x)
        E = self.n_experts

        # Router scores and softmax over all experts: (B, S, E)
        scores = self.gate(x)  # (B, S, E)
        probs = stable_softmax_t(scores, dim=-1)  # (B, S, E)

        # Top-k mask: keep the k largest probs per token, zero the rest.
        if self.top_k < E:
            kth_vals, _ = torch.topk(probs, self.top_k, dim=-1)  # (B, S, k)
            threshold = kth_vals[..., -1:].expand_as(probs)  # (B, S, E)
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
            probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-8)  # renormalize

        # Weighted sum of expert outputs (all experts computed; zeros masked).
        out = torch.zeros_like(x)  # (B, S, D)
        for expert_idx, expert in enumerate(self.experts):
            w = probs[..., expert_idx : expert_idx + 1]  # (B, S, 1)
            out = out + w * expert(x)  # (B, S, D)
        # Shared experts (ADR 0002): ungated additive branch, averaged.
        if self.shared_expert_list:
            shared_sum = torch.zeros_like(x)
            for shared in self.shared_experts:
                shared_sum = shared_sum + shared(x)  # (B, S, D)
            out = out + shared_sum / len(self.shared_expert_list)
        return out


class TritonTransformerBlock(nn.Module):
    """One decoder-only transformer block (LLaMA-style, pre-norm).

        h   = x + attention(rms_norm(x))       # (B, S, D)
        out = h + feed_forward(rms_norm(h))    # (B, S, D)

    ``feed_forward`` is the dense triton SwiGLU by default, or the MoE when
    ``config.has_moe()``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_index: int = 0,
        params: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Build the block's sublayers, injecting this layer's trained weights.

        layer_index/params: the layer's position in the stack and the flat
        registry-keyed checkpoint. When both are given, EVERY sublayer is
        born with its trained array — the two RMSNorm gains from
        ``Keys.ln(layer_index, ...)``, the four attention projections from
        ``Keys.attn(...)``, the FFN/MoE weights from ``Keys.ffn(...)`` /
        ``Keys.moe_*`` — each copied in place over the storage created
        here. None (default) is the training path: gammas start at ones
        (identity) and the projections at their seeded init.
        """
        super().__init__()
        self.config = config
        self.block_layer_index: int = layer_index
        D = config.embed_dim
        i = layer_index
        p = params  # short alias; None → the training path (all keys below unused)

        # nn.RMSNorm exposes its gain as ``weight`` (D,), init ones.
        # One pre-norm PER MIXER (ADR 0007) + the post-attention norm.
        self.mixer_norms = nn.ModuleList([nn.RMSNorm(D, eps=config.norm_eps) for _ in range(config.n_mixers())])
        self.post_attention_layernorm = nn.RMSNorm(D, eps=config.norm_eps)
        if p is not None:
            for m, norm in enumerate(self.mixer_norms):
                copy_in_place(norm.weight, p[Keys.mixer_ln(i, m)])  # type: ignore[arg-type]
            copy_in_place(self.post_attention_layernorm.weight, p[Keys.ln(i, LayerNorm.POST_ATTENTION)])
        # Mixing sublayers (ADR 0007): n_deltanet_sublayers Gated DeltaNet
        # mixers followed by ONE full RoPE attention mixer.
        mixers: list[TritonMultiHeadAttention | TritonGatedDeltaNet] = []
        for m in range(config.n_mixers()):
            if config.mixer_kind(m) == "d":
                mixers.append(
                    TritonGatedDeltaNet(
                        embed_dim=D,
                        n_heads=config.n_heads,
                        layer_index=i if p is not None else None,
                        mixer_index=m if p is not None else None,
                        params=p,
                    )
                )
            else:
                mixers.append(
                    TritonMultiHeadAttention(
                        config,
                        layer_index=i,
                        q_proj=None if p is None else p[Keys.attn(i, m, Attn.Q_PROJ)],
                        k_proj=None if p is None else p.get(Keys.attn(i, m, Attn.K_PROJ)),
                        v_proj=None if p is None else p.get(Keys.attn(i, m, Attn.V_PROJ)),
                        o_proj=None if p is None else p[Keys.attn(i, m, Attn.O_PROJ)],
                        kv_down_proj=None if p is None else p.get(Keys.attn(i, m, Attn.KV_DOWN_PROJ)),
                        k_up_proj=None if p is None else p.get(Keys.attn(i, m, Attn.K_UP_PROJ)),
                        v_up_proj=None if p is None else p.get(Keys.attn(i, m, Attn.V_UP_PROJ)),
                    )
                )
        self.mixers: list[TritonMultiHeadAttention | TritonGatedDeltaNet] = mixers
        self.self_attn = nn.ModuleList(mixers)
        if config.has_moe():
            self.mlp: TritonSwiGLUFFN | TritonMixtureOfExperts = TritonMixtureOfExperts(
                embed_dim=D,
                n_experts=config.n_experts,
                ff_dim=config.expert_dim,
                top_k=config.top_k,
                n_shared_experts=config.n_shared_experts,
                layer_index=i if p is not None else None,
                params=p,
            )
        else:
            self.mlp = TritonSwiGLUFFN(
                embed_dim=D,
                ff_dim=config.expert_dim,
                gate_proj=None if p is None else p[Keys.ffn(i, Mlp.GATE_PROJ)],
                up_proj=None if p is None else p[Keys.ffn(i, Mlp.UP_PROJ)],
                down_proj=None if p is None else p[Keys.ffn(i, Mlp.DOWN_PROJ)],
            )

    def forward(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        """Block forward. x: (B, S, D) → out: (B, S, D).

        h = x; for each mixer m: h = h + mixers[m](mixer_norms[m](h));
        out = h + mlp(post_attention_layernorm(h))."""
        self._move_to_device(x)
        h = x
        for norm, mixer in zip(self.mixer_norms, self.mixers, strict=True):
            h = h + mixer(norm(h), position)  # (B, S, D)
        ff_out = self.mlp(self.post_attention_layernorm(h))  # (B, S, D)
        return h + ff_out  # (B, S, D)

    def forward_state(self, x: torch.Tensor, position: int = 0) -> Result[torch.Tensor, list[dict]]:
        """Block forward plus the per-mixer cache state the prefill needs.

        Mirrors ``impl._torch.layers.TransformerBlock.forward_state``:
        ``forward`` calls this and drops the state; the prefill path
        requests it and backfills the per-mixer caches — no second pass.
        Same ``Result`` contract (shared/result.py).

        ``.value`` = out (B, S, D); ``.ctx.display`` = the list of per-mixer
        displays — {"k_group", "v_group"} for the attention mixer,
        {"s_final", "s_hist"} for each DeltaNet mixer.
        """
        h = x
        displays: list[dict[str, Any]] = []
        for norm, mixer in zip(self.mixer_norms, self.mixers, strict=True):
            mixer_res = mixer.forward_state(norm(h), position)
            h = h + mixer_res.value  # (B, S, D)
            displays.append(mixer_res.ctx.display)
        ff_out = self.mlp(self.post_attention_layernorm(h))  # (B, S, D)
        return Result.capture(h + ff_out, {}, displays)  # type: ignore[arg-type]

    def forward_step(self, x: torch.Tensor, position: int, cache: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Process ONE new token against this block's per-mixer caches.

        Mirrors ``impl._torch.layers.TransformerBlock.forward_step``: each
        mixer's cache entry (``cache[m]``) is updated in place.

        x: (B, 1, D) the new token's embedding at absolute ``position``.
        cache: this block's per-mixer cache list, mutated in place.

        Returns: out (B, 1, D).
        """
        self._move_to_device(x)
        h = x
        for m, (norm, mixer) in enumerate(zip(self.mixer_norms, self.mixers, strict=True)):
            h = h + mixer.forward_step(norm(h), position, cache[m])  # (B, 1, D)
        ff_out = self.mlp(self.post_attention_layernorm(h))  # (B, 1, D)
        return h + ff_out  # (B, 1, D)

    def forward_chunk(self, x: torch.Tensor, position: int, cache: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Process a CHUNK of c tokens through this block (KV-cached path).

        Mirrors ``impl._torch.layers.TransformerBlock.forward_chunk``: each
        mixer's chunk form runs against ``cache[m]``.

        x: (B, c, D) the chunk's vectors. position: the absolute index of
        the chunk's first token. cache: this block's per-mixer cache list,
        mutated in place.

        Returns: out (B, c, D).
        """
        self._move_to_device(x)
        h = x
        for m, (norm, mixer) in enumerate(zip(self.mixer_norms, self.mixers, strict=True)):
            h = h + mixer.forward_chunk(norm(h), position, cache[m])  # (B, c, D)
        ff_out = self.mlp(self.post_attention_layernorm(h))  # (B, c, D)
        return h + ff_out  # (B, c, D)

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move all block parameters to x's device/dtype."""
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        for norm in self.mixer_norms:
            if dtype is None:
                norm.to(device)
            else:
                norm.to(device, dtype)
        if dtype is None:
            self.post_attention_layernorm.to(device)
        else:
            self.post_attention_layernorm.to(device, dtype)
        for mixer in self.mixers:
            mixer._move_to_device(x)
        self.mlp._move_to_device(x)


class TritonDecoderStack(nn.Module):
    """Stack of TritonTransformerBlocks (the "body" of the decoder).

    out = block_{n-1}( ... block_1(block_0(x)) ... ), x: (B, S, D) → (B, S, D).
    """

    def __init__(self, config: TransformerConfig, params: Mapping[str, np.ndarray] | None = None) -> None:
        """Build ``n_layers`` blocks, injecting each layer's slice of ``params``.

        params: the flat registry-keyed checkpoint (the ``shared.constants.
        Keys`` scheme), already validated by the model constructor. Every
        block is born with its trained weights (``Keys.ln/attn/ffn/moe_*``
        sliced by its ``layer_index``). None (default) keeps the seeded
        init — the training path.
        """
        super().__init__()
        self.config = config
        # Typed view for pyright; nn.ModuleList registers parameters for torch
        self.blocks: list[TritonTransformerBlock] = [
            TritonTransformerBlock(config, layer_index=i, params=params) for i in range(config.n_layers)
        ]
        self.layers = nn.ModuleList(self.blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            for mixer in block.mixers:
                if isinstance(mixer, TritonGatedDeltaNet):
                    for lin in (
                        mixer._impl.q_proj,
                        mixer._impl.k_proj,
                        mixer._impl.v_proj,
                        mixer._impl.o_proj,
                        mixer._impl.a_proj,
                        mixer._impl.gate_proj,
                    ):
                        lin.reset_parameters()
                elif self.config.has_mla_at(block.block_layer_index):
                    assert (
                        mixer.kv_down_proj is not None and mixer.k_up_proj is not None and mixer.v_up_proj is not None
                    )
                    mixer.q_proj.reset_parameters()
                    mixer.o_proj.reset_parameters()
                    mixer.kv_down_proj.reset_parameters()
                    mixer.k_up_proj.reset_parameters()
                    mixer.v_up_proj.reset_parameters()
                else:
                    assert mixer.k_proj is not None and mixer.v_proj is not None
                    mixer.q_proj.reset_parameters()
                    mixer.k_proj.reset_parameters()
                    mixer.v_proj.reset_parameters()
                    mixer.o_proj.reset_parameters()
            block.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        """Forward through all blocks. x: (B, S, D) → out: (B, S, D)."""
        self._move_to_device(x)
        out = x
        for block in self.blocks:
            out = block(out, position)
        return out

    def forward_state(self, x: torch.Tensor, position: int = 0) -> Result[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Forward through all blocks, capturing each block's attention state.

        Mirrors ``impl._torch.layers.DecoderStack.forward_state``:
        ``forward`` drops the state; the model's prefill requests it and
        backfills the per-layer cache — no second pass. Same ``Result``
        contract (shared/result.py).

        ``.value`` = out (B, S, D); ``.ctx.record`` = {block
        index: {"k_group", "v_group"}}.
        """
        self._move_to_device(x)
        states: dict[int, dict[str, torch.Tensor]] = {}
        out = x
        for i, block in enumerate(self.blocks):
            block_res = block.forward_state(out, position)
            out = block_res.value
            states[i] = block_res.ctx.display  # {block index: k_group/v_group}
        return Result.capture(out, {}, {}, states)

    def forward_step(self, x: torch.Tensor, position: int, cache: list[list[dict[str, torch.Tensor]]]) -> torch.Tensor:
        """Process ONE new token through all blocks (KV-cached path).

        Mirrors ``impl._torch.layers.DecoderStack.forward_step``: each block
        appends the token's K/V to its cache entry (``cache[i]`` mutated in
        place) and attends against the cached tensors.

        x: (B, 1, D) the new token's embedding at absolute ``position``.
        cache: one dict per block, from the model's ``make_cache``.

        Returns: out (B, 1, D).
        """
        self._move_to_device(x)
        out = x
        for i, block in enumerate(self.blocks):
            out = block.forward_step(out, position, cache[i])
        return out

    def forward_chunk(self, x: torch.Tensor, position: int, cache: list[list[dict[str, torch.Tensor]]]) -> torch.Tensor:
        """Process a CHUNK of c tokens through all blocks (KV-cached path).

        Mirrors ``impl._torch.layers.DecoderStack.forward_chunk`` (which
        mirrors ``impl._np.stack.DecoderStack.forward_chunk``): each block
        appends the chunk's K/V to its cache entry (``cache[i]`` mutated
        in place) and scores all chunk positions in parallel.

        x: (B, c, D) the chunk's vectors. position: the absolute index of
        the chunk's first token. cache: one dict per block, from the
        model's ``make_cache``.

        Returns: out (B, c, D).
        """
        self._move_to_device(x)
        out = x
        for i, block in enumerate(self.blocks):
            out = block.forward_chunk(out, position, cache[i])
        return out

    def _move_to_device(self, x: torch.Tensor) -> None:
        for block in self.blocks:
            block._move_to_device(x)
