# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""TransformerBlock — CUDA assembly of all CUDA primitives.

Assembles:
  - RMSNorm     → impl/_cuda/layernorm.rmsnorm    (warp-reduction kernel)
  - MHA         → impl/_cuda/attention.scaled_dot_product_attention (CUDA softmax/weighted-sum)
  - SwiGLU FFN  → impl/_cuda/ffn.swiglu_ffn       (CUDA SiLU kernel inside SwiGLU)
  - RoPE        → impl/_cuda/rope.apply_rope       (CUDA rotary embedding kernel)
  - MoE         → torch routing (same math as the other tracks) + per-expert
                 CUDA SwiGLU FFN

Architecture (LLaMA-style pre-norm, identical to the NumPy/PyTorch tracks):
    Input:  x [B, S, D]
    │
    ├─ Stream 1: Attention ──────────────────────────────────────────────
    │   1. xn = RMSNorm(x, input_ln)                    # (B, S, D)
    │   2. q = xn @ Wq; k = xn @ Wk; v = xn @ Wv        # (B, ·, S, hd)
    │   3. q, k ← RoPE(q), RoPE(k)                      # CUDA rope kernel
    │   4. if G < H: repeat k, v to H heads (GQA)
    │   5. attn = SDPA(q, k, v) @ Wo                    # (B, S, D)
    │   6. h = x + attn                                  # residual
    │
    ├─ Stream 2: Feed-forward (dense SwiGLU or MoE) ─────────────────────
    │   7. hn = RMSNorm(h, post_ln)                     # (B, S, D)
    │   8. ff = SwiGLU(hn) | MoE(hn)                    # (B, S, D)
    │   9. out = h + ff                                  # residual
    │
    Output: out [B, S, D]

Reference
---------
Vaswani et al. "Attention Is All You Need" (2017)
https://arxiv.org/abs/1706.03762

Shazeer, "GLU Variants Improve Transformer" (2020)
https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from impl._cuda.attention import scaled_dot_product_attention as cuda_sdp_attention
from impl._cuda.ffn import swiglu_ffn
from impl._cuda.layernorm import rmsnorm
from impl._cuda.rope import apply_rope
from shared.config import TransformerConfig
from shared.constants import Attn, Keys, LayerNorm, Mlp
from shared.result import Result
from shared.torch_softmax import stable_softmax_t

logger = logging.getLogger(__name__)

# ── Weight initialization helpers ──────────────────────────────────────────────


def _init_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Xavier/uniform initialization for attention/FFN weights.

    Returns a (rows, cols) float32 tensor with values drawn from
    U(-limit, limit) where limit = sqrt(6 / (rows + cols)).
    """
    bound = math.sqrt(6.0 / (rows + cols))
    tensor = torch.empty(rows, cols, dtype=torch.float32)
    torch.nn.init.uniform_(tensor, -bound, bound, generator=torch.Generator().manual_seed(seed))
    return tensor


def _init_zeros(shape: tuple[int, ...]) -> torch.Tensor:
    """Initialize a tensor to zeros.

    Returns
    -------
    torch.Tensor
        Zeros tensor of the given shape, float32.
    """
    return torch.zeros(shape)


def _copy_in_place(tensor: torch.Tensor, value: np.ndarray) -> None:
    """Copy a checkpoint array over the storage ``tensor`` already owns.

    ``tensor.data.copy_`` writes IN PLACE: the tensor keeps its identity,
    its ``requires_grad`` flag and its leaf-ness, so a model born with
    trained weights stays trainable and the caller's arrays are never
    aliased. Every CUDA projection is stored in the checkpoint's ``(in, out)``
    layout, so the copy never transposes.

    Parameters
    ----------
    tensor : torch.Tensor
        Destination — the storage the constructor created (or a view of it,
        for one slice of the stacked MoE experts).
    value : np.ndarray
        Source array of the matching shape (cast to ``tensor.dtype``).
    """
    tensor.data.copy_(torch.from_numpy(np.asarray(value)).to(tensor.dtype))


# ── CUDA block assembly ────────────────────────────────────────────────────────


class CuGatedDeltaNet:
    """Gated DeltaNet sublayer for the CUDA track (ADR 0006).

    Plain-tensor owner (like ``CuTransformerBlock`` — not an nn.Module), but
    the math delegates to the torch track's ``GatedDeltaNet`` implementation
    (cuBLAS projections + a readable scan loop). PROD note: a fused NVRTC
    delta-rule kernel is the production path; at teaching scale the loop is
    fine and keeps the recurrence inspectable. Cache contract identical to
    the NumPy track: {"s": (B,H,hd,hd), "s_hist": [...]}.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        seed: int = 0,
        layer_index: int | None = None,
        mixer_index: int | None = None,
        params: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        from impl._torch.layers import GatedDeltaNet

        self._impl = GatedDeltaNet(embed_dim, n_heads, layer_index=layer_index, mixer_index=mixer_index, params=params)
        # mirror the plain-tensor attribute surface the block/stack code reads
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        # expose tensors as plain attributes for _param_tensors/_cuda_to_device walks
        self.q_proj: torch.Tensor = self._impl.q_proj.weight
        self.k_proj: torch.Tensor = self._impl.k_proj.weight
        self.v_proj: torch.Tensor = self._impl.v_proj.weight
        self.o_proj: torch.Tensor = self._impl.o_proj.weight
        self.a_proj: torch.Tensor = self._impl.a_proj.weight
        self.gate_proj: torch.Tensor = self._impl.gate_proj.weight
        self.dt_bias: torch.Tensor = self._impl.dt_bias
        self.norm_gamma: torch.Tensor = self._impl.norm.weight

    def _to_input(self, x: torch.Tensor) -> None:
        self._impl.to(device=x.device, dtype=x.dtype if x.dtype.is_floating_point else None)

    def forward(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        self._to_input(x)
        return self._impl.forward(x, position)

    def forward_state(self, x: torch.Tensor, position: int = 0) -> Result:
        self._to_input(x)
        return self._impl.forward_state(x, position)

    def forward_step(self, x: torch.Tensor, position: int, cache: dict) -> torch.Tensor:
        self._to_input(x)
        return self._impl.forward_step(x, position, cache)

    def forward_chunk(self, x: torch.Tensor, position: int, cache: dict) -> torch.Tensor:
        self._to_input(x)
        return self._impl.forward_chunk(x, position, cache)


class CuTransformerBlock:
    """CUDA TransformerBlock — assembly of all CUDA primitives.

    Mirrors ``impl._np.block.TransformerBlock``: same pre-norm layout,
    same GQA, same RoPE, same MoE routing math — with the heavy compute
    delegated to CUDA kernels.

    It does NOT inherit from nn.Module — weights are stored as plain tensors
    for parity checking against the NumPy implementation (easy to extract
    raw arrays for comparison).

    Attributes
    ----------
    input_layernorm_gamma : torch.Tensor, shape (D,)
        RMSNorm gain before attention (LLaMA naming).
    post_attention_layernorm_gamma : torch.Tensor, shape (D,)
        RMSNorm gain before the feed-forward.
    q_proj / k_proj / v_proj / o_proj : torch.Tensor
        Attention projections (in, out) layout: (D, H·hd), (D, G·hd),
        (D, G·hd), (H·hd, D).
    Dense FFN: gate_proj (D, FF), up_proj (D, FF), down_proj (FF, D).
    MoE: router (D, E); experts stacked as (E, D, FF), (E, D, FF), (E, FF, D).
    """

    # ------------------------------------------------------------------ init
    def __init__(  # noqa: C901 — kind-aware deltanet branch added one path
        self,
        config: TransformerConfig,
        seed: int = 0,
        layer_index: int = 0,
        params: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Build the block's tensors, injecting this layer's trained weights.

        layer_index/params: this layer's position in the stack and the flat
        registry-keyed checkpoint (the shared ``Keys`` scheme). When params
        is given, EVERY tensor is born with its trained array — the two
        RMSNorm gains from ``Keys.ln(layer_index, ...)``, the four attention
        projections from ``Keys.attn(...)``, and the FFN/MoE weights from
        ``Keys.ffn(...)`` / ``Keys.moe_gate(...)`` / ``Keys.moe_expert(...)``
        / ``Keys.moe_shared_expert(...)`` — each copied in place over the
        seed-initialized storage (``requires_grad`` survives; see
        ``_copy_in_place``). None (default) is the training path: the gammas
        stay at ones (identity) and every weight at its seeded init.

        ``seed`` is the per-layer init seed (unchanged); ``layer_index``
        only selects which checkpoint slice this block owns.
        """
        self.config = config
        D, H, G = config.embed_dim, config.n_heads, config.kv_heads
        hd, FF, E = config.head_dim, config.expert_dim, config.n_experts
        i = layer_index
        # Short alias; None → the training path. (``p`` is taken by the
        # per-tensor ``requires_grad_`` loops below.)
        ckpt = params
        self.n_heads = H
        self.n_groups = G
        self.n_experts = E
        self.head_dim = hd
        self.rope_dim = config.rope_dim

        # RMSNorm gains (D,) — one pre-norm PER MIXER (ADR 0007) + the
        # post-attention norm; identity at init.
        self.mixer_norm_gammas: list[torch.Tensor] = []
        for _m in range(config.n_mixers()):
            g = torch.ones(D, dtype=torch.float32)
            g.requires_grad_(True)
            self.mixer_norm_gammas.append(g)
        self.post_attention_layernorm_gamma = torch.ones(D, dtype=torch.float32)
        self.post_attention_layernorm_gamma.requires_grad_(True)

        # ── Mixing sublayers (ADR 0007): n_deltanet_sublayers Gated DeltaNet
        # mixers + ONE full-attention mixer (plain projections, no bias) ──
        self.mixers: list = []  # CuGatedDeltaNet or None (None = attention mixer)
        self.attn_projs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for m in range(config.n_mixers()):
            if config.mixer_kind(m) == "d":
                # torch nn.Linear init draws from the GLOBAL RNG — reseed it per
                # mixer so two same-seed blocks stay bit-identical (the CUDA track
                # seeds everything else through explicit generators).
                torch.manual_seed(seed + 1 + m)
                self.mixers.append(
                    CuGatedDeltaNet(
                        D,
                        H,
                        seed=seed + 1 + m,
                        layer_index=i if ckpt is not None else None,
                        mixer_index=m if ckpt is not None else None,
                        params=ckpt,
                    )
                )
                self.attn_projs.append(
                    (self.mixers[-1].q_proj, self.mixers[-1].k_proj, self.mixers[-1].v_proj, self.mixers[-1].o_proj)
                )
            else:
                self.mixers.append(None)
                q = _init_weight(D, H * hd, seed=seed + 2 + 4 * m)
                k = _init_weight(D, G * hd, seed=seed + 3 + 4 * m)
                v = _init_weight(D, G * hd, seed=seed + 4 + 4 * m)
                o = _init_weight(H * hd, D, seed=seed + 5 + 4 * m)
                for p in (q, k, v, o):
                    p.requires_grad_(True)
                self.attn_projs.append((q, k, v, o))

        if config.has_moe():
            # ── MoE: router + SwiGLU experts ───────────────────────────
            # Router: (D, E), no bias (Mixtral convention)
            self.router = _init_weight(D, E, seed=seed + 8)
            self.router.requires_grad_(True)
            # Experts: stacked SwiGLU projections
            bound = math.sqrt(6.0 / (D + FF))
            gen = torch.Generator().manual_seed(seed + 7)
            self.expert_gate_proj = torch.empty(E, D, FF, dtype=torch.float32)
            torch.nn.init.uniform_(self.expert_gate_proj, -bound, bound, generator=gen)
            self.expert_up_proj = torch.empty(E, D, FF, dtype=torch.float32)
            torch.nn.init.uniform_(self.expert_up_proj, -bound, bound, generator=gen)
            self.expert_down_proj = torch.empty(E, FF, D, dtype=torch.float32)
            torch.nn.init.uniform_(self.expert_down_proj, -bound, bound, generator=gen)
            for p in (self.expert_gate_proj, self.expert_up_proj, self.expert_down_proj):
                p.requires_grad_(True)
            # Shared experts (ADR 0002): stacked SwiGLU projections, always
            # active and ungated (not routed).
            N_S = config.n_shared_experts
            gen_sh = torch.Generator().manual_seed(seed + 12)
            self.shared_gate_proj = torch.empty(N_S, D, FF, dtype=torch.float32)
            torch.nn.init.uniform_(self.shared_gate_proj, -bound, bound, generator=gen_sh)
            self.shared_up_proj = torch.empty(N_S, D, FF, dtype=torch.float32)
            torch.nn.init.uniform_(self.shared_up_proj, -bound, bound, generator=gen_sh)
            self.shared_down_proj = torch.empty(N_S, FF, D, dtype=torch.float32)
            torch.nn.init.uniform_(self.shared_down_proj, -bound, bound, generator=gen_sh)
            for p in (self.shared_gate_proj, self.shared_up_proj, self.shared_down_proj):
                p.requires_grad_(True)
        else:
            # ── Dense SwiGLU feed-forward ──────────────────────────────
            self.gate_proj = _init_weight(D, FF, seed=seed + 9)
            self.up_proj = _init_weight(D, FF, seed=seed + 10)
            self.down_proj = _init_weight(FF, D, seed=seed + 11)
            for p in (self.gate_proj, self.up_proj, self.down_proj):
                p.requires_grad_(True)

        # ── Trained-slice injection (inference path) ───────────────────
        # One key vocabulary (shared.constants.Keys) end to end: the same
        # keys the model's save/load traverse. Layouts agree (CUDA stores
        # every projection as (in, out)), so each copy is a direct write
        # over storage this constructor just created.
        if ckpt is not None:
            for m, g in enumerate(self.mixer_norm_gammas):
                _copy_in_place(g, ckpt[Keys.mixer_ln(i, m)])
            _copy_in_place(self.post_attention_layernorm_gamma, ckpt[Keys.ln(i, LayerNorm.POST_ATTENTION)])
            for m in range(config.n_mixers()):
                if self.mixers[m] is not None:
                    # DeltaNet mixer: the sublayer object already injected its own
                    # slice at construction; the q/k/v/o aliases share that storage.
                    continue
                q, k, v, o = self.attn_projs[m]
                _copy_in_place(q, ckpt[Keys.attn(i, m, Attn.Q_PROJ)])
                _copy_in_place(k, ckpt[Keys.attn(i, m, Attn.K_PROJ)])
                _copy_in_place(v, ckpt[Keys.attn(i, m, Attn.V_PROJ)])
                _copy_in_place(o, ckpt[Keys.attn(i, m, Attn.O_PROJ)])
            if config.has_moe():
                _copy_in_place(self.router, ckpt[Keys.moe_gate(i)])
                for e in range(E):
                    # Views of the stacked (E, ·, ·) storages — the copy
                    # lands in the base tensor that holds requires_grad.
                    _copy_in_place(self.expert_gate_proj[e], ckpt[Keys.moe_expert(i, e, Mlp.GATE_PROJ)])
                    _copy_in_place(self.expert_up_proj[e], ckpt[Keys.moe_expert(i, e, Mlp.UP_PROJ)])
                    _copy_in_place(self.expert_down_proj[e], ckpt[Keys.moe_expert(i, e, Mlp.DOWN_PROJ)])
                for s in range(config.n_shared_experts):
                    _copy_in_place(self.shared_gate_proj[s], ckpt[Keys.moe_shared_expert(i, s, Mlp.GATE_PROJ)])
                    _copy_in_place(self.shared_up_proj[s], ckpt[Keys.moe_shared_expert(i, s, Mlp.UP_PROJ)])
                    _copy_in_place(self.shared_down_proj[s], ckpt[Keys.moe_shared_expert(i, s, Mlp.DOWN_PROJ)])
            else:
                _copy_in_place(self.gate_proj, ckpt[Keys.ffn(i, Mlp.GATE_PROJ)])
                _copy_in_place(self.up_proj, ckpt[Keys.ffn(i, Mlp.UP_PROJ)])
                _copy_in_place(self.down_proj, ckpt[Keys.ffn(i, Mlp.DOWN_PROJ)])

    # ---------------------------------------------------------------- forward
    def _ffn(self, h: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Feed-forward (dense SwiGLU or MoE) on ``h`` (already normed)."""
        if self.config.has_moe():
            return self._moe_forward(h, device)
        return swiglu_ffn(h, self.gate_proj.to(device), self.up_proj.to(device), self.down_proj.to(device))

    def _attn_forward(self, xn: torch.Tensor, position: int) -> torch.Tensor:
        """Full-attention mixer's full-sequence forward (CUDA SDPA path)."""
        B, S, _D = xn.shape
        device = xn.device
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        Wq, Wk, Wv, Wo = (t.to(device) for t in self.attn_projs[-1])
        q = xn @ Wq  # (B, S, H·hd)
        k = xn @ Wk  # (B, S, G·hd)
        v = xn @ Wv  # (B, S, G·hd)
        q = q.view(B, S, H, hd)
        k = k.view(B, S, G, hd)
        v = v.view(B, S, G, hd).transpose(1, 2).contiguous()  # (B, G, S, hd)
        q = apply_rope(q, position, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, H, S, hd)
        k = apply_rope(k, position, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, G, S, hd)
        if G != H:
            k = k.repeat_interleave(H // G, dim=1)
            v = v.repeat_interleave(H // G, dim=1)
        attn = cuda_sdp_attention(q, k, v, is_causal=True)  # (B, H, S, hd)
        return attn.transpose(1, 2).contiguous().view(B, S, H * hd) @ Wo  # (B, S, D)

    def _attn_forward_state(self, xn: torch.Tensor, position: int) -> tuple[torch.Tensor, dict]:
        """Full-attention mixer's forward + the cacheable per-group K/V."""
        B, S, _D = xn.shape
        device = xn.device
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        Wq, Wk, Wv, Wo = (t.to(device) for t in self.attn_projs[-1])
        q = xn @ Wq
        k = xn @ Wk
        v = xn @ Wv
        q = q.view(B, S, H, hd)
        k = k.view(B, S, G, hd)
        v = v.view(B, S, G, hd).transpose(1, 2).contiguous()  # (B, G, S, hd)
        q = apply_rope(q, position, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, H, S, hd)
        k = apply_rope(k, position, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, G, S, hd)
        k_group, v_group = k, v  # (B, G, S, hd) — the cacheable per-group form
        if G != H:
            k_full = k_group.repeat_interleave(H // G, dim=1)
            v_full = v_group.repeat_interleave(H // G, dim=1)
        else:
            k_full, v_full = k_group, v_group
        attn = cuda_sdp_attention(q, k_full, v_full, is_causal=True)  # (B, H, S, hd)
        attn_out = attn.transpose(1, 2).contiguous().view(B, S, H * hd) @ Wo  # (B, S, D)
        return attn_out, {"k_group": k_group, "v_group": v_group}

    def _attn_step(self, xn: torch.Tensor, position: int, cache: dict) -> torch.Tensor:
        """Full-attention mixer's one-token step against the cached K/V."""
        B = xn.shape[0]
        device = xn.device
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        Wq, Wk, Wv, Wo = (t.to(device) for t in self.attn_projs[-1])
        q = (xn @ Wq).view(B, 1, H, hd)
        k = (xn @ Wk).view(B, 1, G, hd)
        v = (xn @ Wv).view(B, 1, G, hd).transpose(1, 2).contiguous()  # (B, G, 1, hd)
        q = apply_rope(q, position, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, H, 1, hd)
        k = apply_rope(k, position, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, G, 1, hd)
        cache["k"] = torch.cat([cache["k"], k], dim=2)  # (B, G, t+1, hd)
        cache["v"] = torch.cat([cache["v"], v], dim=2)
        k_full, v_full = cache["k"], cache["v"]
        if G != H:
            k_full = k_full.repeat_interleave(H // G, dim=1)
            v_full = v_full.repeat_interleave(H // G, dim=1)
        attn = cuda_sdp_attention(q, k_full, v_full)  # (B, H, 1, hd) — all cached precede the query
        return attn.transpose(1, 2).contiguous().view(B, 1, H * hd) @ Wo  # (B, 1, D)

    def _attn_chunk(self, xn: torch.Tensor, position: int, cache: dict) -> torch.Tensor:
        """Full-attention mixer's chunk pass (verification path)."""
        B, c, _D = xn.shape
        device = xn.device
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        Wq, Wk, Wv, Wo = (t.to(device) for t in self.attn_projs[-1])
        q = xn @ Wq
        k = xn @ Wk
        v = xn @ Wv
        q = q.view(B, c, H, hd)
        k = k.view(B, c, G, hd)
        v = v.view(B, c, G, hd).transpose(1, 2).contiguous()  # (B, G, c, hd)
        q = apply_rope(q, position, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, H, c, hd)
        k = apply_rope(k, position, rope_dim=self.rope_dim).transpose(1, 2).contiguous()  # (B, G, c, hd)
        cache["k"] = torch.cat([cache["k"], k], dim=2)  # (B, G, t+c, hd)
        cache["v"] = torch.cat([cache["v"], v], dim=2)
        k_full, v_full = cache["k"], cache["v"]
        if G != H:
            k_full = k_full.repeat_interleave(H // G, dim=1)
            v_full = v_full.repeat_interleave(H // G, dim=1)
        t = k_full.shape[2] - c
        # Left-pad the queries with t dummy rows so the kernel's row-absolute
        # causal mask (key j <= row i) IS the block-causal (c, t+c) mask.
        q_pad = torch.zeros((B, H, t + c, hd), device=device, dtype=q.dtype)
        q_pad[:, :, t:, :] = q
        attn = cuda_sdp_attention(q_pad, k_full, v_full, is_causal=True)  # (B, H, t+c, hd)
        attn = attn[:, :, t:, :]  # (B, H, c, hd) — crop the dummy rows
        return attn.transpose(1, 2).contiguous().view(B, c, H * hd) @ Wo  # (B, c, D)

    def forward(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        """Forward pass through the TransformerBlock (ADR 0007 mixer chain).

        Pre-norm layout (identical math to the other tracks):

            h = x;  for m: h = h + mixer_m(RMSNorm_m(h))
            out = h + FFN(RMSNorm(h))               # FFN = SwiGLU | MoE
        """
        h = x
        for m in range(self.config.n_mixers()):
            device = x.device
            xn = rmsnorm(h, self.mixer_norm_gammas[m].to(device), eps=self.config.norm_eps)
            mixer = self.mixers[m]
            if mixer is not None:
                h = h + mixer.forward(xn, position).to(device)
            else:
                h = h + self._attn_forward(xn, position)
        hn = rmsnorm(h, self.post_attention_layernorm_gamma.to(x.device), eps=self.config.norm_eps)
        return h + self._ffn(hn, x.device)

    def forward_state(self, x: torch.Tensor, position: int = 0) -> Result[torch.Tensor, list[dict]]:
        """Block forward plus the per-mixer cache state the prefill needs.

        Mirrors ``impl._torch.layers.TransformerBlock.forward_state`` (which
        mirrors ``impl._np.block.TransformerBlock.forward_state``): the
        prefill path requests the state and backfills the per-mixer caches —
        no second pass. Same ``Result`` contract (shared/result.py).

        ``.value`` = out (B, S, D); ``.ctx.display`` = the list of per-mixer
        displays — {"k_group", "v_group"} for the attention mixer,
        {"s_final", "s_hist"} for each DeltaNet mixer.
        """
        h = x
        displays: list[dict[str, Any]] = []
        for m in range(self.config.n_mixers()):
            device = x.device
            xn = rmsnorm(h, self.mixer_norm_gammas[m].to(device), eps=self.config.norm_eps)
            mixer = self.mixers[m]
            if mixer is not None:
                res = mixer.forward_state(xn, position)
                h = h + res.value.to(device)
                displays.append(dict(res.ctx.display))
            else:
                attn_out, kv = self._attn_forward_state(xn, position)
                h = h + attn_out
                displays.append(kv)
        hn = rmsnorm(h, self.post_attention_layernorm_gamma.to(x.device), eps=self.config.norm_eps)
        out = h + self._ffn(hn, x.device)
        return Result.capture(out, {}, displays)  # type: ignore[arg-type]

    def forward_step(self, x: torch.Tensor, position: int, cache: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Process ONE new token against this block's per-mixer caches.

        Mirrors ``impl._torch.layers.TransformerBlock.forward_step``: each
        mixer's cache entry (``cache[m]``) is updated in place (the attention
        mixer appends the token's K/V; DeltaNet mixers advance the recurrent
        S state).

        x: (B, 1, D) the new token's embedding at absolute ``position``.
        cache: this block's per-mixer cache list, mutated in place.

        Returns: out (B, 1, D).
        """
        h = x
        for m in range(self.config.n_mixers()):
            device = x.device
            xn = rmsnorm(h, self.mixer_norm_gammas[m].to(device), eps=self.config.norm_eps)
            mixer = self.mixers[m]
            if mixer is not None:
                h = h + mixer.forward_step(xn, position, cache[m]).to(device)
            else:
                missing = [k for k in ("k", "v") if k not in cache[m]]
                assert not missing, f"forward_step needs a naive {{'k','v'}} cache (missing keys: {missing})"
                h = h + self._attn_step(xn, position, cache[m])
        hn = rmsnorm(h, self.post_attention_layernorm_gamma.to(x.device), eps=self.config.norm_eps)
        return h + self._ffn(hn, x.device)

    def forward_chunk(self, x: torch.Tensor, position: int, cache: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Process a CHUNK of c tokens through this block (KV-cached path).

        Mirrors ``impl._torch.layers.TransformerBlock.forward_chunk`` (which
        mirrors ``impl._np.block.TransformerBlock.forward_chunk``): each
        mixer's chunk form runs against ``cache[m]`` — the attention mixer
        appends the whole chunk's K/V in one pass and scores every chunk
        position through the CUDA SDPA path in parallel (causal within the
        chunk); DeltaNet mixers scan sequentially.

        x: (B, c, D) the chunk's vectors. position: the absolute index of
        the chunk's first token. cache: this block's per-mixer cache list,
        mutated in place.

        Returns: out (B, c, D).
        """
        h = x
        for m in range(self.config.n_mixers()):
            device = x.device
            xn = rmsnorm(h, self.mixer_norm_gammas[m].to(device), eps=self.config.norm_eps)
            mixer = self.mixers[m]
            if mixer is not None:
                h = h + mixer.forward_chunk(xn, position, cache[m]).to(device)
            else:
                h = h + self._attn_chunk(xn, position, cache[m])
        hn = rmsnorm(h, self.post_attention_layernorm_gamma.to(x.device), eps=self.config.norm_eps)
        return h + self._ffn(hn, x.device)

    def _moe_forward(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """MoE forward: torch routing (identical to the other tracks) + CUDA SwiGLU experts.

        scores = x @ router            (B, S, E)
        probs  = softmax(scores)       (B, S, E)
        top-k mask + renormalize       (B, S, E)
        out    = Σ_j probs_j · SwiGLU_j(x)   (B, S, D)

        Every expert is computed and multiplied by its (possibly zero)
        weight — the math is identical to the gathered-token variant.
        """
        E = self.n_experts
        router = self.router.to(device)
        eg = self.expert_gate_proj.to(device)
        eu = self.expert_up_proj.to(device)
        ed = self.expert_down_proj.to(device)

        # Router scores and softmax over all experts: (B, S, E)
        scores = x @ router  # (B, S, E)
        probs = stable_softmax_t(scores, dim=-1)  # (B, S, E)

        # Top-k mask: keep the k largest probs per token, zero the rest.
        if self.config.top_k < E:
            kth_vals, _ = torch.topk(probs, self.config.top_k, dim=-1)  # (B, S, k)
            threshold = kth_vals[..., -1:].expand_as(probs)  # (B, S, E)
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
            probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-8)  # renormalize

        # Weighted sum of expert outputs (CUDA SwiGLU per expert).
        out = torch.zeros_like(x)  # (B, S, D)
        for expert_idx in range(E):
            w = probs[..., expert_idx : expert_idx + 1]  # (B, S, 1)
            expert_out = swiglu_ffn(x, eg[expert_idx], eu[expert_idx], ed[expert_idx])  # (B, S, D)
            out = out + w * expert_out

        # Shared experts (ADR 0002): ungated additive branch, averaged.
        N_S = self.config.n_shared_experts
        if N_S > 0:
            sg = self.shared_gate_proj.to(device)
            su = self.shared_up_proj.to(device)
            sd = self.shared_down_proj.to(device)
            shared_sum = torch.zeros_like(x)
            for s in range(N_S):
                shared_sum = shared_sum + swiglu_ffn(x, sg[s], su[s], sd[s])  # (B, S, D)
            out = out + shared_sum / N_S
        return out
