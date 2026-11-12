# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Transformer block for the NumPy reference implementation.

One LLaMA-style pre-norm decoder block: RMSNorm → attention → residual,
RMSNorm → feed-forward (dense SwiGLU or MoE) → residual. Owns the forward
and the analytic backward (which composes the per-operator backwards).

**Intuition:** each block is a two-step conversation — the tokens first
talk to one another (attention collects context across the sequence),
then the model per token processes what it just heard (the FFN). The
residual skips are the memory: outputs are added, never overwritten, so
early token information always survives. Pre-norm keeps each sublayer's
input in a scale that dot-products can reason about.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from impl._np.attention import MultiHeadAttention
from impl._np.deltanet import GatedDeltaNet
from impl._np.ffn import SwiGLUFFN
from impl._np.layernorm import RMSNorm
from impl._np.moe import MixtureOfExperts
from shared.config import TransformerConfig
from shared.constants import Attn, Keys, LayerNorm, Mlp


class TransformerBlock:
    """One decoder-only transformer block (LLaMA-style, pre-norm).

    The canonical block (see LLaMA-2/3, HuggingFace ``LlamaDecoderLayer``):
    RMSNorm *before* each sublayer, with a plain additive residual:

    Shape letters (CONTEXT.md → "Shape notation"): B = batch size (parallel
    sequences; 1 = one generation), S = sequence length (this pass), D =
    embed_dim (model width), H = n_heads, G = n_groups, V = vocab_size,
    E = n_experts, FF = expert_dim, t = KV-cache depth, k = draft length.
        h_0 = x
        h_{m+1} = h_m + mixer_m(rms_norm_m(h_m))     for m = 0 .. M-1
        out     = h_M + feed_forward(rms_norm(h_M))  # (B, S, D)

    ``feed_forward`` is the dense SwiGLU by default, or the MoE when the
    config enables it. Pre-norm (normalize before the sublayer) keeps the
    residual stream clean and trains stably without careful warmup; the
    attention and FFN each see normalized inputs.

    Sublayers (all shapes (B, S, D) in / out). Every block runs M = n_mixers()
    mixing sublayers (ADR 0007: ``n_deltanet_sublayers`` Gated DeltaNet
    linear-attention mixers followed by ONE full RoPE attention mixer), each
    with its OWN pre-norm; then the FFN/MoE with the post-attention norm:
        mixer_norms[m] → mixers[m] → residual add        (m = 0 .. M-1)
        post_attention_layernorm → SwiGLUFFN | MixtureOfExperts → residual add

    Backward (chain rule, reverse of the forward)
    ---------------------------------------------
    Given dout (B, S, D), with h_0 = x and out = h_M + ff_out:

        d_ff_out = dout
        d_h_M   = dout + <ln2 backward>
        for m = M-1 .. 0 (reverse):
            d_h_m, mixer_grads[m] = mixers[m].backward(d_h_{m+1}, ln_m_out, position)
            d_h_m += d_h_{m+1}              (residual)
            d_ln_m_out, d_gamma_m         = mixer_norms[m].backward(d_h_m, h_m)
            d_h_m += d_ln_m_out             (residual into previous stream)
        dx = d_h_0

    The returned gradient dict uses the block's attribute paths as keys
    ("mixers.0.deltanet.q_proj", "mlp.gate_proj", ...) so the model can map
    them onto the shared Keys scheme in one place.
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
        born with its trained array — the post-attention RMSNorm gain from
        ``Keys.ln(layer_index, POST_ATTENTION)``, each mixer's pre-norm gain
        from ``Keys.mixer_ln(i, m)``, each mixer's projections from
        ``Keys.deltanet(i, m, ...)`` / ``Keys.attn(i, m, ...)``, the FFN/MoE
        weights from ``Keys.ffn(...)`` / ``Keys.moe_*`` — each copied in
        place over the seed-initialized storage. None (default) is the
        training path: gammas start at ones (identity) and the projections
        at their seeded init.

        The mapping is the same ``shared.constants.Keys`` vocabulary the
        block's ``backward`` gradient dict and the model's save/load use,
        so there is one key language end to end.
        """
        self.config = config
        D, H, G = config.embed_dim, config.n_heads, config.kv_heads
        seed = config.seed
        i = layer_index
        p = params  # short alias; None → the training path (all keys below unused)

        self.post_attention_layernorm = RMSNorm(
            D, eps=config.norm_eps, gamma=None if p is None else p[Keys.ln(i, LayerNorm.POST_ATTENTION)]
        )
        # Mixing sublayers (ADR 0007): every block runs n_deltanet_sublayers
        # Gated DeltaNet mixers followed by ONE full RoPE attention mixer,
        # each with its OWN pre-norm. Both mixer kinds expose the same
        # forward/step/chunk interface; only the cache dict shape differs
        # ({"k","v"} vs {"s","s_hist"}), which make_cache picks per mixer.
        self.mixer_norms: list[RMSNorm] = []
        self.mixers: list[MultiHeadAttention | GatedDeltaNet] = []
        for m in range(config.n_mixers()):
            self.mixer_norms.append(
                RMSNorm(D, eps=config.norm_eps, gamma=None if p is None else p[Keys.mixer_ln(i, m)])
            )
            if config.mixer_kind(m) == "d":
                self.mixers.append(
                    GatedDeltaNet(
                        embed_dim=D,
                        n_heads=H,
                        seed=seed + 1 + m,
                        layer_index=i if p is not None else None,
                        mixer_index=m if p is not None else None,
                        params=p,
                    )
                )
            else:
                # Full-attention mixer: SWA (ADR 0008) and MLA (ADR 0009) are
                # config-level knobs the operator reads directly.
                self.mixers.append(
                    MultiHeadAttention(
                        embed_dim=D,
                        n_heads=H,
                        n_groups=G,
                        rope_dim=config.rope_dim,
                        seed=seed + 1 + m,
                        swa_window=config.swa_window,
                        q_proj=None if p is None else p.get(Keys.attn(i, m, Attn.Q_PROJ)),
                        # ADR 0011: MLA only on even blocks of each group.
                        mla_latent_dim=config.mla_latent_dim if config.has_mla_at(i) else 0,
                        k_proj=None if p is None else p.get(Keys.attn(i, m, Attn.K_PROJ)),
                        v_proj=None if p is None else p.get(Keys.attn(i, m, Attn.V_PROJ)),
                        o_proj=None if p is None else p[Keys.attn(i, m, Attn.O_PROJ)],
                        kv_down_proj=None if p is None else p.get(Keys.attn(i, m, Attn.KV_DOWN_PROJ)),
                        k_up_proj=None if p is None else p.get(Keys.attn(i, m, Attn.K_UP_PROJ)),
                        v_up_proj=None if p is None else p.get(Keys.attn(i, m, Attn.V_UP_PROJ)),
                    )
                )
        if config.has_moe():
            self.mlp: SwiGLUFFN | MixtureOfExperts = MixtureOfExperts(
                embed_dim=D,
                n_experts=config.n_experts,
                ff_dim=config.expert_dim,
                top_k=config.top_k,
                seed=seed + 2,
                n_shared_experts=config.n_shared_experts,
                layer_index=i if p is not None else None,
                params=p,
            )
        else:
            self.mlp = SwiGLUFFN(
                embed_dim=D,
                ff_dim=config.expert_dim,
                seed=seed + 2,
                gate_proj=None if p is None else p[Keys.ffn(i, Mlp.GATE_PROJ)],
                up_proj=None if p is None else p[Keys.ffn(i, Mlp.UP_PROJ)],
                down_proj=None if p is None else p[Keys.ffn(i, Mlp.DOWN_PROJ)],
            )

    def forward(self, x: np.ndarray, position: int = 0, record: dict | None = None) -> np.ndarray:
        """Block forward: h=x; for m: h += mixer_m(norm_m(h)); out = h + mlp(ln2(h)).

        x: (B, S, D) → out: (B, S, D).

        Invariant/contract: the block NEVER normalizes the stream itself —
        only the copies (normed inputs) are normalized; every residual is a
        plain add, so ∂out/∂x carries the identity path (the gradient
        highway).
        position: the absolute index of this pass's first token (0-based)
            — threaded to attention/RoPE; a windowed prefill passes its
            offset so decode steps stay aligned.
        record: optional dict; when given, it is filled with the block's
            intermediate state (the per-mixer norm outputs and mixer states,
            the per-mixer residual streams h[m], ln2/FFN state, out) — the
            same state the analytic backward recomputes, captured once by
            the operators themselves. The math (and its bit-level result)
            is identical either way.
        """
        h = x  # (B, S, D) — the residual stream
        hs = [x]  # hs[m] = stream entering mixer m
        ln_outs: list[np.ndarray] = []
        mixer_states: list[dict] = []
        mixer_outs: list[np.ndarray] = []
        for _m, (norm, mixer) in enumerate(zip(self.mixer_norms, self.mixers, strict=True)):
            ln_out = norm.forward(h)  # (B, S, D)
            mixer_out, mixer_state = mixer.forward_state(ln_out, position).unwrap_with()  # (B, S, D)
            h = h + mixer_out  # (B, S, D) residual add
            hs.append(h)
            ln_outs.append(ln_out)
            mixer_states.append(mixer_state)
            mixer_outs.append(mixer_out)

        # Feed-forward (dense or MoE) with pre-norm and residual.
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, S, D)
        ff_out, ff_state = self.mlp.forward_state(ln2_out).unwrap_with()  # (B, S, D)
        out = h + ff_out  # (B, S, D)
        if record is not None:
            record.update(
                {
                    "x": x,
                    "hs": hs,
                    "ln_outs": ln_outs,
                    "mixers": mixer_states,
                    "mixer_outs": mixer_outs,
                    "ln2_out": ln2_out,
                    "ff": ff_state,
                    "mlp_out": ff_out,
                    "out": out,
                }
            )
        return out

    def backward(self, dout: np.ndarray, x: np.ndarray, position: int = 0) -> tuple[np.ndarray, dict]:
        """Analytic backward (derivation in the class docstring).

        dout: (B, S, D) upstream gradient.
        x: (B, S, D) the forward input (recomputes the intermediates).
        position: the first-token index used in forward.

        Returns: (dx, dparams) with attribute-path keys (see class docstring).
        """
        # --- Recompute forward intermediates (same math as forward) ---
        M = len(self.mixers)
        hs = [x]
        ln_outs: list[np.ndarray] = []
        for m in range(M):
            ln_outs.append(self.mixer_norms[m].forward(hs[m]))  # (B, S, D)
            hs.append(hs[m] + self.mixers[m].forward(ln_outs[m], position))  # (B, S, D)
        h = hs[M]
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, S, D)
        _ff_out = self.mlp.forward(ln2_out)  # (B, S, D)

        # --- Backward: out = h + ff_out ---
        d_ff_out = dout  # (B, S, D)
        d_ln2_out, mlp_grads = self.mlp.backward(d_ff_out, ln2_out)  # (B, S, D)
        d_h, d_gamma2 = self.post_attention_layernorm.backward(d_ln2_out, h)  # (B, S, D), (D,)
        d_h = dout + d_h  # residual: out = h + ff_out, so dout reaches h directly

        # --- Mixing sublayers, reverse order: h_{m+1} = h_m + mixer_m(ln_m(h_m)) ---
        dparams: dict = {"post_attention_layernorm.gamma": d_gamma2}
        for m in range(M - 1, -1, -1):
            d_ln_out, mixer_grads = self.mixers[m].backward(d_h, ln_outs[m], position)  # (B, S, D)
            d_h_m, d_gamma_m = self.mixer_norms[m].backward(d_ln_out, hs[m])  # (B, S, D), (D,)
            d_h = d_h + d_h_m  # residual: both the next-mixer input and the norm path feed h_m
            dparams[f"mixer_norms.{m}.gamma"] = d_gamma_m
            if isinstance(self.mixers[m], GatedDeltaNet):
                for name in ("q_proj", "k_proj", "v_proj", "o_proj", "a_proj", "dt_bias", "gate_proj", "norm"):
                    dparams[f"mixers.{m}.deltanet.{name}"] = mixer_grads[name]
            else:
                # The attention mixer's grad dict carries exactly the active
                # projection set: q/o always, plus k/v in classic mode or
                # kv_down/k_up/v_up in MLA mode (ADR 0009).
                for name in mixer_grads:
                    dparams[f"mixers.{m}.self_attn.{name}"] = mixer_grads[name]
        dx = d_h  # (B, S, D) — the stream entering mixer 0 is x itself

        if isinstance(self.mlp, MixtureOfExperts):
            dparams["mlp.gate"] = mlp_grads["gate"]
            dparams["mlp.experts"] = mlp_grads["experts"]  # list of {gate_proj, up_proj, down_proj}
            dparams["mlp.shared_experts"] = mlp_grads["shared_experts"]  # same shape, shared experts (ADR 0002)
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                dparams[f"mlp.{name}"] = mlp_grads[name]
        return dx, dparams

    def forward_step(
        self, x: np.ndarray, position: int, cache: list[dict], quantize: bool = False, record: dict | None = None
    ) -> np.ndarray:
        """Process ONE new token (per-token inference path, KV-cached mixers).

        x: (B, 1, D) the new token's vector — B sequences, ONE new position
            each (S = 1; the token's own embedding from the lookup).
        position: the absolute token index of the new token (for RoPE;
            0-based; must equal the caches' current depth or the positions
            desynchronize).
        cache: this block's per-MIXER cache LIST (one dict per mixer,
            ``cache[m]``; see MultiHeadAttention.forward_step /
            GatedDeltaNet.forward_step). Mutated in place; shape depends on
            quantize for the K/V mixer. Contract: the caches must hold
            exactly the positions 0..position-1 before the call.
        quantize: if True, append the new K/V to the attention mixer's cache
            in 1-bit TurboQuant form and dequantize the full cached tensor
            before attention; if False, append the full-precision K/V
            (default). DeltaNet mixers carry the recurrent S state — no K/V,
            no TurboQuant.
        record: optional dict filled with the step's intermediates (same keys
            as ``forward``) — see that method.

        Returns
        -------
        np.ndarray, shape (B, 1, D)
            The block output for the new token. Contract: equal to the
            matching slice of ``forward`` over the full sequence (the exact
            KV-step guarantee); the caches end holding positions
            0..position.
        """
        h = x  # (B, 1, D)
        hs = [x]
        ln_outs: list[np.ndarray] = []
        mixer_states: list[dict] = []
        mixer_outs: list[np.ndarray] = []
        for m, (norm, mixer) in enumerate(zip(self.mixer_norms, self.mixers, strict=True)):
            ln_out = norm.forward(h)  # (B, 1, D)
            mixer_state: dict | None = {} if record is not None else None
            if isinstance(mixer, GatedDeltaNet):
                mixer_out = mixer.forward_step(ln_out, position, cache[m], record=mixer_state)
            else:
                mixer_out = mixer.forward_step(ln_out, position, cache[m], quantize=quantize, record=mixer_state)
            h = h + mixer_out  # (B, 1, D)
            hs.append(h)
            ln_outs.append(ln_out)
            mixer_states.append(mixer_state if mixer_state is not None else {})
            mixer_outs.append(mixer_out)
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, 1, D)
        ff_out, ff_state = self.mlp.forward_state(ln2_out).unwrap_with()  # (B, 1, D)
        out = h + ff_out  # (B, 1, D)
        if record is not None:
            record.update(
                {
                    "x": x,
                    "hs": hs,
                    "ln_outs": ln_outs,
                    "mixers": mixer_states,
                    "mixer_outs": mixer_outs,
                    "ln2_out": ln2_out,
                    "ff": ff_state,
                    "mlp_out": ff_out,
                    "out": out,
                }
            )
        return out

    def forward_chunk(self, x: np.ndarray, position: int, cache: list[dict], record: dict | None = None) -> np.ndarray:
        """Process a CHUNK of c tokens through this block (KV-cached path).

        The speculative-decoding verification path (see ADR 0003): identical
        structure to ``forward_step`` but with each mixer's chunk form — the
        attention mixer appends the whole chunk's K/V in one pass and scores
        every chunk position in parallel (causal within the chunk); the
        DeltaNet mixers scan the chunk sequentially (S is a function of all
        preceding tokens). Naive cache only; the FFN is chunk-positionwise
        so it needs no chunk form.

        x: (B, c, D) the chunk's vectors — c = the draft block's length,
            each a D-wide embedding.
        position: the absolute index of the chunk's FIRST token; the chunk
            occupies positions position..position+c-1 (must equal the
            caches' current depth).
        cache: this block's per-mixer cache list, mutated in place.
        record: optional dict filled with the block's intermediates
            (same keys as ``forward_step``'s record).

        Returns
        -------
        np.ndarray, shape (B, c, D)
            The block output for every chunk position. Contract: equal to
            the matching c slices of ``forward`` over the full sequence
            (chunk parity); the caches end holding positions
            0..position+c-1 — the caller trims the rejected tail.
        """
        h = x  # (B, c, D)
        hs = [x]
        ln_outs: list[np.ndarray] = []
        mixer_states: list[dict] = []
        mixer_outs: list[np.ndarray] = []
        for m, (norm, mixer) in enumerate(zip(self.mixer_norms, self.mixers, strict=True)):
            ln_out = norm.forward(h)  # (B, c, D)
            mixer_state: dict | None = {} if record is not None else None
            mixer_out = mixer.forward_chunk(ln_out, position, cache[m], record=mixer_state)
            h = h + mixer_out  # (B, c, D)
            hs.append(h)
            ln_outs.append(ln_out)
            mixer_states.append(mixer_state if mixer_state is not None else {})
            mixer_outs.append(mixer_out)
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, c, D)
        ff_out, ff_state = self.mlp.forward_state(ln2_out).unwrap_with()  # (B, c, D)
        out = h + ff_out  # (B, c, D)
        if record is not None:
            record.update(
                {
                    "x": x,
                    "hs": hs,
                    "ln_outs": ln_outs,
                    "mixers": mixer_states,
                    "mixer_outs": mixer_outs,
                    "ln2_out": ln2_out,
                    "ff": ff_state,
                    "mlp_out": ff_out,
                    "out": out,
                }
            )
        return out
