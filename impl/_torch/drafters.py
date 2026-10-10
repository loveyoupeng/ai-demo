"""Torch drafters — the MTP and DSpark draft models for the torch family.

Shape letters (CONTEXT.md → "Shape notation"): B = batch size (the
speculative engine is B=1), S = sequence length, D = embed_dim, V =
vocab_size, k = the drafter's block size.

The torch-family counterpart of ``impl/_np/drafters.py`` (the math
reference): the SAME ``shared.draft.Drafter`` protocol, the SAME sidecar
key scheme (``shared/draft.py`` — the single owner of the format), and
the SAME forward math, implemented as ``nn.Module`` graphs over torch
tensors so:

  * drafters train by distillation with autograd
    (``scripts/train_drafters.py``), and
  * the same class serves the torch, triton, and cuda tracks
    (the one-implementation rule for the torch family; the only allowed
    cross-track import is triton → torch).

Round-trip guarantee (ADR 0003): a sidecar saved by the NumPy track
loads here and drafts the SAME tokens, because loading runs through
``expected_drafter_params`` — keys flagged ``torch_transpose``
(``in_proj`` and the attention projections ``q/k/v/o``: stored
(in, out) in the npz, transposed on load into an ``nn.Linear``-backed
module) are flipped exactly there. FFN weights (``gate/up/down_proj``)
are stored (in, out) as used — no transpose, so they stay plain
parameters.

**MTP** (multi-token prediction, DeepSeek-V3 style — sequential drafting):
one transformer block conditioned on the target's final-norm hidden at
the anchor, drafting one token per block step with its own tiny KV cache
(each step fed the previously drafted token). See the NumPy module for
the full walkthrough; the math is identical.

**DSpark** (arXiv 2607.05147 — semi-autoregressive parallel drafting):
non-causal parallel backbone over the block + causal sequential
refinement, conditioned on the anchor hidden replicated to every block
position and paired with learned block-position embeddings. One drafter
forward drafts the whole block; stateless across rounds.

Both drafters SHARE the target's embedding matrix and lm_head — handed in
at construction as (V, D) and (D, V) tensors, referenced and
gradient-tracked but never copied into the sidecar (DeepSeek-V3 MTP
design: no duplicated V×D weights). Confidence-scheduled verification is
the engine's job (``shared/spec_engine.py`` — the torch-family
production engine, the counterpart of ``impl/_np/spec.py``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from impl._torch.layers import RMSNorm
from shared.config import TransformerConfig
from shared.draft import DSHARK, MTP, DrafterMeta, expected_drafter_params


def _stable_probs(logits: torch.Tensor) -> torch.Tensor:
    """Stable softmax over the last axis: logits (..., V) → probs (..., V)."""
    z = logits - torch.max(logits, dim=-1, keepdim=True).values
    e = torch.exp(z)
    return e / torch.sum(e, dim=-1, keepdim=True)


def _rope_apply(x: torch.Tensor, positions: torch.Tensor, rope_dim: int = 0) -> torch.Tensor:
    """RoPE over (B, S, heads, D): rotate even/odd pairs by pos·theta_k.

    Same math as ``impl._torch.layers.RoPE`` (theta_k = 10000^(-2k/d),
    positions (S,) → angles (S, pair_dim), broadcast over batch and heads),
    inlined as a function so importing this module never consumes the
    global RNG. ``rope_dim`` 0 rotates every dimension; 0 < rope_dim < D
    rotates the first rope_dim and passes the rest through (the main
    config's rule).
    """
    d = x.shape[-1]
    if rope_dim > 0 and rope_dim < d:
        x_rot, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    else:
        x_rot, x_pass = x, None
    B, S, H = x_rot.shape[0], x_rot.shape[1], x_rot.shape[2]
    pair_dim = x_rot.shape[-1] // 2
    positions = positions.to(x.device)
    # theta_k = 10000^(-2k / d) — (pair_dim,)
    freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, device=x.device, dtype=torch.float32) * 2.0 / x_rot.shape[-1]))
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # (S, pair_dim)
    cos, sin = torch.cos(angles), torch.sin(angles)  # (S, pair_dim)
    x_flat = x_rot.reshape(B, S, H, pair_dim, 2)  # last dim = (even, odd) pair
    x_even, x_odd = x_flat[..., 0], x_flat[..., 1]  # (B, S, H, pair_dim)
    cos_b, sin_b = cos[:, None, :], sin[:, None, :]  # (S, 1, pair_dim) → broadcast
    y_even = x_even * cos_b - x_odd * sin_b  # (B, S, H, pair_dim)
    y_odd = x_even * sin_b + x_odd * cos_b  # (B, S, H, pair_dim)
    rotated = torch.stack([y_even, y_odd], dim=-1).reshape(x_rot.shape)  # (B, S, H, d_rot)
    if x_pass is not None:
        return torch.cat([rotated, x_pass], dim=-1)  # (B, S, H, D)
    return rotated


class _DrafterBlock(nn.Module):
    """One transformer block of a drafter, over sidecar-flavored keys.

    A thin parameter owner reusing the track's real operator math (RMSNorm,
    GQA attention, SwiGLU) so the forward is the same code the target runs
    — the sidecar key scheme maps onto these attributes. ``causal`` selects
    the attention mask for the DSpark backbone (parallel → non-causal; seq
    module → causal).

    Shapes: forward x (B, S, D) → out (B, S, D); forward_step x (B, 1, D)
    → out (B, 1, D).
    """

    def __init__(self, D: int, H: int, G: int, ff_dim: int, rope_dim: int, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.n_heads, self.n_groups, self.head_dim = H, G, D // H
        self.rope_dim = rope_dim
        hd = self.head_dim
        self.input_layernorm = RMSNorm(D)
        self.post_attention_layernorm = RMSNorm(D)
        # (out, in) nn.Linear modules — the sidecar's (in, out) arrays are
        # transposed on load (torch_transpose flags in shared/draft.py).
        self.self_attn_q = nn.Linear(D, H * hd, bias=False)
        self.self_attn_k = nn.Linear(D, G * hd, bias=False)
        self.self_attn_v = nn.Linear(D, G * hd, bias=False)
        self.self_attn_o = nn.Linear(H * hd, D, bias=False)
        # FFN: stored (in, out) as used — plain parameters, no transpose.
        self.mlp_gate = nn.Parameter(torch.empty(D, ff_dim))
        self.mlp_up = nn.Parameter(torch.empty(D, ff_dim))
        self.mlp_down = nn.Parameter(torch.empty(ff_dim, D))
        # Kaiming uniform init, matching the track's SwiGLUFFN convention —
        # fresh (untrained) drafters start finite; sidecar loading overwrites.
        nn.init.kaiming_uniform_(self.mlp_gate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_up, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_down, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Standard pre-norm block forward with the configured mask.

        x: (B, S, D) → out: (B, S, D).
        """
        B, S, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        ln1 = self.input_layernorm(x)  # (B, S, D)
        q = self.self_attn_q(ln1).view(B, S, H, hd).permute(0, 2, 1, 3)  # (B, H, S, hd)
        k = self.self_attn_k(ln1).view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)
        v = self.self_attn_v(ln1).view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)
        # RoPE contract: (B, S, heads, hd) with positions (S,).
        q = _rope_apply(q.permute(0, 2, 1, 3), positions, self.rope_dim).permute(0, 2, 1, 3)  # (B, H, S, hd)
        k = _rope_apply(k.permute(0, 2, 1, 3), positions, self.rope_dim).permute(0, 2, 1, 3)  # (B, G, S, hd)
        if G != H:
            k = k.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
            v = v.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
        # Scale and softmax are SDPA's (both match the NumPy reference); only
        # the mask differs by module. Torch's bool attn_mask is an ALLOW mask
        # (True = attend) — the opposite of the NumPy reference's
        # True=masked convention.
        if self.causal:
            allow = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()  # True where j <= i
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)  # (B, H, S, hd)
        else:
            ctx = F.scaled_dot_product_attention(q, k, v)  # (B, H, S, hd) — bidirectional
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, S, H * hd)  # (B, S, H*hd)
        h = x + self.self_attn_o(ctx)  # (B, S, D)
        ln2 = self.post_attention_layernorm(h)  # (B, S, D)
        gate = F.silu(ln2 @ self.mlp_gate)  # (B, S, FF)
        up = ln2 @ self.mlp_up  # (B, S, FF)
        return h + (gate * up) @ self.mlp_down  # (B, S, D)

    def forward_step(self, x: torch.Tensor, position: int, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        """One-token step with the drafter's own tiny KV cache (MTP path).

        Mirrors the NumPy drafter block's ``forward_step``: the token's
        RoPE'd K and un-rotated V are appended to ``cache`` (per-group,
        (B, G, t, hd)) before attention; a single query row attends to the
        whole cache (causal masking is implicit — every cached position
        precedes it).

        x: (B, 1, D) → out: (B, 1, D).
        """
        B = x.shape[0]
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        ln1 = self.input_layernorm(x)  # (B, 1, D)
        q = self.self_attn_q(ln1).view(B, 1, H, hd).permute(0, 2, 1, 3)  # (B, H, 1, hd)
        k = self.self_attn_k(ln1).view(B, 1, G, hd).permute(0, 2, 1, 3)  # (B, G, 1, hd)
        v = self.self_attn_v(ln1).view(B, 1, G, hd).permute(0, 2, 1, 3)  # (B, G, 1, hd)
        positions = torch.tensor([position], device=x.device, dtype=torch.long)
        q = _rope_apply(q.permute(0, 2, 1, 3), positions, self.rope_dim).permute(0, 2, 1, 3)  # (B, H, 1, hd)
        k = _rope_apply(k.permute(0, 2, 1, 3), positions, self.rope_dim).permute(0, 2, 1, 3)  # (B, G, 1, hd)
        # Append the new K/V to the cache (per-group; K already RoPE'd).
        cache["k"] = torch.cat([cache["k"], k], dim=2)  # (B, G, t+1, hd)
        cache["v"] = torch.cat([cache["v"], v], dim=2)  # (B, G, t+1, hd)
        k_full, v_full = cache["k"], cache["v"]
        if G != H:
            k_full = k_full.repeat_interleave(H // G, dim=1)  # (B, H, t+1, hd)
            v_full = v_full.repeat_interleave(H // G, dim=1)  # (B, H, t+1, hd)
        # Single query row: no mask needed (all cached positions are past).
        ctx = F.scaled_dot_product_attention(q, k_full, v_full)  # (B, H, 1, hd)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, 1, H * hd)  # (B, 1, H*hd)
        h = x + self.self_attn_o(ctx)  # (B, 1, D)
        ln2 = self.post_attention_layernorm(h)  # (B, 1, D)
        gate = F.silu(ln2 @ self.mlp_gate)  # (B, 1, FF)
        up = ln2 @ self.mlp_up  # (B, 1, FF)
        return h + (gate * up) @ self.mlp_down  # (B, 1, D)

    # ── sidecar I/O ────────────────────────────────────────────────────────
    def params(self, prefix: str) -> dict[str, torch.Tensor]:
        """The sidecar key → tensor map for this block (see shared/draft.py)."""
        return {
            f"{prefix}.input_layernorm.weight": self.input_layernorm.weight,
            f"{prefix}.self_attn.q_proj.weight": self.self_attn_q.weight,
            f"{prefix}.self_attn.k_proj.weight": self.self_attn_k.weight,
            f"{prefix}.self_attn.v_proj.weight": self.self_attn_v.weight,
            f"{prefix}.self_attn.o_proj.weight": self.self_attn_o.weight,
            f"{prefix}.post_attention_layernorm.weight": self.post_attention_layernorm.weight,
            f"{prefix}.mlp.gate_proj.weight": self.mlp_gate,
            f"{prefix}.mlp.up_proj.weight": self.mlp_up,
            f"{prefix}.mlp.down_proj.weight": self.mlp_down,
        }


def _load_into_binding(
    meta: DrafterMeta,
    params: dict[str, torch.Tensor | np.ndarray],
    binding: dict[str, torch.Tensor],
) -> None:
    """Copy sidecar arrays into their owning tensors, transposing the keys
    ``expected_drafter_params`` flags as ``torch_transpose`` — the single
    load path every torch drafter shares, and the reason a NumPy-saved
    sidecar lands bit-exact here."""
    for key, shape, transpose in expected_drafter_params(meta):
        arr = params[key]
        loaded = torch.from_numpy(np.ascontiguousarray(arr)) if isinstance(arr, np.ndarray) else torch.as_tensor(arr)
        if tuple(loaded.shape) != shape:
            raise ValueError(f"drafter param {key!r} has shape {tuple(loaded.shape)}, expected {shape}")
        if transpose:
            loaded = loaded.T  # npz (in, out) → nn.Linear (out, in)
        target = binding[key]
        loaded = loaded.to(device=target.device, dtype=target.dtype)
        if tuple(loaded.shape) != tuple(target.shape):
            raise ValueError(
                f"drafter param {key!r} loaded to shape {tuple(loaded.shape)}, module shape {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(loaded)


class TorchMTPDrafter(nn.Module):
    """Sequential MTP drafter (DeepSeek-V3 style; the DSpark paper's baseline).

    One transformer block, conditioned on the target's final hidden state,
    drafts one token per step with its own tiny KV cache: ``draft`` runs k
    steps, each feeding the previously drafted token back in — the
    sequential dependency the parallel drafters approximate.
    """

    family = MTP

    def __init__(
        self,
        meta: DrafterMeta,
        embedding_weight: torch.Tensor,  # (V, D) — the TARGET's embedding (shared)
        lm_head_weight: torch.Tensor,  # (D, V) — the TARGET's lm_head (shared)
        seed: int = 0,
    ) -> None:
        super().__init__()
        D, H = meta.embed_dim, meta.n_heads
        G = meta.n_groups if meta.n_groups is not None else meta.n_heads
        self.meta = meta
        self.block_size = meta.block_size
        # Shared with the target — referenced (autograd-tracked) but NOT
        # registered as drafter parameters, so sidecars never duplicate them.
        self.embedding_weight = embedding_weight
        self.lm_head_weight = lm_head_weight
        torch.manual_seed(seed)  # deterministic init (same convention as TorchModel)
        self.norm = RMSNorm(D)
        self.in_proj = nn.Linear(2 * D, D, bias=False)
        self.block = _DrafterBlock(D, H, G, meta.ff_dim, meta.rope_dim, causal=True)
        self.out_norm = RMSNorm(D)
        self.reset()

    def reset(self) -> None:
        """Fresh sequence: empty the drafter's KV cache."""
        ref = self.norm.weight
        G, hd = self.block.n_groups, self.block.head_dim
        self._cache: dict[str, torch.Tensor] = {
            "k": torch.zeros(1, G, 0, hd, device=ref.device, dtype=ref.dtype),
            "v": torch.zeros(1, G, 0, hd, device=ref.device, dtype=ref.dtype),
        }

    def in_proj_forward(self, h_norm: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """x = in_proj(concat(h_norm, emb(token))): (B, s, D), (B, s) → (B, s, D)."""
        emb = self.embedding_weight[token_ids]  # (B, s, D)
        cat = torch.cat([h_norm, emb], dim=-1)  # (B, s, 2D)
        return self.in_proj(cat)  # (B, s, D)

    def draft(
        self, anchor_token: torch.Tensor, anchor_hidden: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Propose up to k tokens after the anchor, sequentially.

        anchor_token: (B,) ints — the last verified token.
        anchor_hidden: (B, D) — the target's final-norm hidden at the anchor.
        Returns (tokens (B, k'), probs — k' tensors (B, V), the drafter's own
        per-position distributions). k' = min(k, block_size).
        """
        k = min(k, self.block_size)
        B = anchor_hidden.shape[0]
        # The distillation path moves the drafter across devices/batches
        # (.to(device), batch-64 anchors); rebuild the tiny cache when its
        # shape/device no longer matches this call.
        if self._cache["k"].shape[0] != B or self._cache["k"].device != anchor_hidden.device:
            G, hd = self.block.n_groups, self.block.head_dim
            self._cache = {
                "k": torch.zeros(B, G, 0, hd, device=anchor_hidden.device, dtype=self.norm.weight.dtype),
                "v": torch.zeros(B, G, 0, hd, device=anchor_hidden.device, dtype=self.norm.weight.dtype),
            }
        h_norm = self.norm(anchor_hidden.reshape(B, 1, self.meta.embed_dim))  # (B, 1, D)
        anchor_pos = self._cache["k"].shape[2]  # absolute position of the anchor in the drafter's stream
        # Seed the stream: the anchor token's projected vector is step 0.
        x = self.in_proj_forward(h_norm, anchor_token.reshape(B, 1))
        self.block.forward_step(x, anchor_pos, self._cache)  # append the anchor's K/V
        tokens: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        prev = anchor_token  # (B,)
        for j in range(k):
            x = self.in_proj_forward(h_norm, prev.reshape(B, 1))
            x = self.block.forward_step(x, anchor_pos + 1 + j, self._cache)  # (B, 1, D)
            logits = self.out_norm(x) @ self.lm_head_weight  # (B, 1, V)
            probs = _stable_probs(logits[:, 0, :])  # (B, V)
            prev = torch.argmax(probs, dim=-1)  # (B,) greedy proposal
            tokens.append(prev)
            probs_list.append(probs)
        self._last_len = 1 + k  # anchor step + k drafts written to the cache
        return torch.stack(tokens, dim=1), probs_list  # (B, k), k × (B, V)

    def rollback(self, keep: int) -> None:
        """After a rejection: keep the anchor + first ``keep`` drafted steps
        in the drafter's KV cache, drop the rest (the next draft continues
        from the new anchor)."""
        n = 1 + keep
        self._cache["k"] = self._cache["k"][:, :, :n]
        self._cache["v"] = self._cache["v"][:, :, :n]

    # ── sidecar I/O ────────────────────────────────────────────────────────
    def _sidecar_binding(self) -> dict[str, torch.Tensor]:
        """The sidecar key → owning tensor map (transposes are applied on
        load against THESE; the numpy export below is its read-only copy)."""
        p: dict[str, torch.Tensor] = {
            "mtp.norm.weight": self.norm.weight,
            "mtp.in_proj.weight": self.in_proj.weight,
            "mtp.out_norm.weight": self.out_norm.weight,
        }
        p.update(self.block.params("mtp.block"))
        return p

    def get_all_parameters(self) -> dict[str, np.ndarray]:
        """The sidecar key → array map in npz layout (NumPy arrays, like the
        NumPy track — ``save_drafter`` and ``train_drafters`` both consume
        np arrays). Keys flagged torch_transpose export as (in, out) —
        the inverse of the load transposition."""
        flags = {key: t for key, _shape, t in expected_drafter_params(self.meta)}
        return {k: (v.T if flags[k] else v).detach().cpu().numpy() for k, v in self._sidecar_binding().items()}

    def load_from_numpy_dict(self, params: dict[str, torch.Tensor | np.ndarray]) -> None:
        """Copy sidecar arrays into the modules (transposes applied)."""
        _load_into_binding(self.meta, params, self._sidecar_binding())


class TorchDSparkDrafter(nn.Module):
    """Semi-autoregressive parallel drafter (arXiv 2607.05147, teaching scale).

    The whole draft block in ONE forward: a non-causal parallel backbone
    predicts all k positions at once from the anchor hidden state + learned
    block-position embeddings, then a lightweight causal sequential module
    refines the block for intra-token dependency (the semi-AR coupling
    that mitigates suffix decay). Stateless across rounds — rollback is
    a no-op.
    """

    family = DSHARK

    def __init__(
        self,
        meta: DrafterMeta,
        embedding_weight: torch.Tensor,  # (V, D) — the TARGET's embedding (shared)
        lm_head_weight: torch.Tensor,  # (D, V) — the TARGET's lm_head (shared)
        seed: int = 0,
    ) -> None:
        super().__init__()
        D, H = meta.embed_dim, meta.n_heads
        G = meta.n_groups if meta.n_groups is not None else meta.n_heads
        self.meta = meta
        self.block_size = meta.block_size
        self.embedding_weight = embedding_weight  # referenced, not registered
        self.lm_head_weight = lm_head_weight  # referenced, not registered
        torch.manual_seed(seed)  # deterministic init (same convention as TorchModel)
        self.norm = RMSNorm(D)
        self.in_proj = nn.Linear(2 * D, D, bias=False)
        self.pos_emb = nn.Parameter(torch.empty(meta.block_size, D))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.parallel = _DrafterBlock(D, H, G, meta.ff_dim, meta.rope_dim, causal=False)
        self.seq = _DrafterBlock(D, H, G, meta.ff_dim, meta.rope_dim, causal=True)
        self.out_norm = RMSNorm(D)

    def reset(self) -> None:
        """Stateless drafter — nothing to clear."""

    def draft(
        self, anchor_token: torch.Tensor, anchor_hidden: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Propose up to k tokens after the anchor in ONE forward pass.

        Same contract as ``TorchMTPDrafter.draft``; the whole block is
        predicted in parallel (non-causal backbone) then refined causally.
        """
        k = min(k, self.block_size)
        B, D = anchor_hidden.shape
        h_norm = self.norm(anchor_hidden.reshape(B, 1, D))  # (B, 1, D)
        # Replicate the anchor hidden to all k block positions, each paired
        # with its own learned block-position embedding.
        h_rep = h_norm.expand(B, k, D)  # (B, k, D) — no copy
        pos = self.pos_emb[:k].reshape(1, k, D).expand(B, k, D)  # (B, k, D)
        x = self.in_proj(torch.cat([h_rep, pos], dim=-1))  # (B, k, 2D) @ (2D, D) → (B, k, D)
        positions = torch.arange(k, device=x.device)  # block-local positions (RoPE over the block)
        p = self.parallel(x, positions)  # (B, k, D) — NON-causal
        s = self.seq(p, positions)  # (B, k, D) — causal refinement
        logits = self.out_norm(s) @ self.lm_head_weight  # (B, k, V)
        probs = _stable_probs(logits)  # (B, k, V)
        tokens = torch.argmax(logits, dim=-1)  # (B, k) greedy proposal
        self._last_len = k
        return tokens, [probs[:, j, :] for j in range(k)]  # k × (B, V)

    def rollback(self, keep: int) -> None:
        """Stateless — nothing to roll back."""

    # ── sidecar I/O ────────────────────────────────────────────────────────
    def _sidecar_binding(self) -> dict[str, torch.Tensor]:
        """The sidecar key → owning tensor map (transposes are applied on
        load against THESE; the numpy export below is its read-only copy)."""
        p: dict[str, torch.Tensor] = {
            "dspark.norm.weight": self.norm.weight,
            "dspark.in_proj.weight": self.in_proj.weight,
            "dspark.pos_emb.weight": self.pos_emb,
            "dspark.out_norm.weight": self.out_norm.weight,
        }
        p.update(self.parallel.params("dspark.parallel"))
        p.update(self.seq.params("dspark.seq"))
        return p

    def get_all_parameters(self) -> dict[str, np.ndarray]:
        """The sidecar key → array map in npz layout (NumPy arrays, like the
        NumPy track). Keys flagged torch_transpose export as (in, out) —
        the inverse of the load transposition."""
        flags = {key: t for key, _shape, t in expected_drafter_params(self.meta)}
        return {k: (v.T if flags[k] else v).detach().cpu().numpy() for k, v in self._sidecar_binding().items()}

    def load_from_numpy_dict(self, params: dict[str, torch.Tensor | np.ndarray]) -> None:
        """Copy sidecar arrays into the modules (transposes applied)."""
        _load_into_binding(self.meta, params, self._sidecar_binding())


def drafter_from_sidecar(
    meta: DrafterMeta,
    params: dict[str, torch.Tensor | np.ndarray],
    embedding_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> TorchMTPDrafter | TorchDSparkDrafter:
    """Build a torch drafter from a validated sidecar + the target's shared
    embedding/lm_head. Any track's sidecar loads here (round-trip guarantee)."""
    if meta.family == MTP:
        d: TorchMTPDrafter | TorchDSparkDrafter = TorchMTPDrafter(meta, embedding_weight, lm_head_weight)
    elif meta.family == DSHARK:
        d = TorchDSparkDrafter(meta, embedding_weight, lm_head_weight)
    else:
        raise ValueError(f"no torch drafter for family {meta.family!r}")
    d.load_from_numpy_dict(params)
    # Align the drafter with the target it runs beside: the shared
    # embedding/lm_head arrive in the target's dtype/device (float64 on the
    # torch record path, float32+CUDA on triton/cuda), while the sidecar
    # arrays load as float32 CPU — cast the whole drafter to the target.
    ref = embedding_weight
    d = d.to(dtype=ref.dtype, device=ref.device)
    return d


def make_drafter_meta(family: str, target_config: TransformerConfig, block_size: int, ff_dim: int) -> DrafterMeta:
    """DrafterMeta for a new drafter matching the target's D/V."""
    return DrafterMeta(
        family=family,
        block_size=block_size,
        embed_dim=target_config.embed_dim,
        n_heads=max(1, target_config.n_heads // 2),  # half the target's heads (tiny drafter)
        n_groups=None,
        ff_dim=ff_dim,
        rope_dim=target_config.rope_dim,
        vocab_size=target_config.vocab_size,
    )
