"""Transformer block for the NumPy reference implementation.

One LLaMA-style pre-norm decoder block: RMSNorm → attention → residual,
RMSNorm → feed-forward (dense SwiGLU or MoE) → residual. Owns the forward
and the analytic backward (which composes the per-operator backwards).
"""

from __future__ import annotations

import numpy as np

from impl._np.attention import MultiHeadAttention
from impl._np.ffn import SwiGLUFFN
from impl._np.layernorm import RMSNorm
from impl._np.moe import MixtureOfExperts
from shared.config import TransformerConfig


class TransformerBlock:
    """One decoder-only transformer block (LLaMA-style, pre-norm).

    The canonical block (see LLaMA-2/3, HuggingFace ``LlamaDecoderLayer``):
    RMSNorm *before* each sublayer, with a plain additive residual:

        h   = x + attention(rms_norm(x))      # (B, S, D)
        out = h + feed_forward(rms_norm(h))   # (B, S, D)

    ``feed_forward`` is the dense SwiGLU by default, or the MoE when the
    config enables it. Pre-norm (normalize before the sublayer) keeps the
    residual stream clean and trains stably without careful warmup; the
    attention and FFN each see normalized inputs.

    Sublayers (all shapes (B, S, D) in / out):
        input_layernorm  → MultiHeadAttention → residual add
        post_attention_layernorm → SwiGLUFFN | MixtureOfExperts → residual add

    Backward (chain rule, reverse of the forward)
    ---------------------------------------------
    Given dout (B, S, D):

        d_ff_out = dout                  (out = h + ff_out)
        d_h      = dout + <ln2 backward> (the residual adds both paths)
        d_ln2_out, mlp_grads  = mlp.backward(d_ff_out, ln2_out)
        d_ln1_out, attn_grads = attn.backward(d_h, ln1_out)
        dx, ln1_grads         = ln1.backward(d_ln1_out, x)

    The returned gradient dict uses the block's attribute paths as keys
    ("input_layernorm.gamma", "self_attn.q_proj", "mlp.gate_proj", ...) so
    the model can map them onto the shared Keys scheme in one place.
    """

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        D, H, G = config.embed_dim, config.n_heads, config.kv_heads
        seed = config.seed

        self.input_layernorm = RMSNorm(D, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(D, eps=config.norm_eps)
        self.self_attn = MultiHeadAttention(embed_dim=D, n_heads=H, n_groups=G, rope_dim=config.rope_dim, seed=seed + 1)
        if config.has_moe():
            self.mlp: SwiGLUFFN | MixtureOfExperts = MixtureOfExperts(
                embed_dim=D,
                n_experts=config.n_experts,
                ff_dim=config.expert_dim,
                top_k=config.top_k,
                seed=seed + 2,
                n_shared_experts=config.n_shared_experts,
            )
        else:
            self.mlp = SwiGLUFFN(embed_dim=D, ff_dim=config.expert_dim, seed=seed + 2)

    def forward(self, x: np.ndarray, positions: np.ndarray | None = None, record: dict | None = None) -> np.ndarray:
        """Block forward. x: (B, S, D) → out: (B, S, D).

        record: optional dict; when given, it is filled with the block's
            intermediate state (ln1/ln2 outputs, the attention state, the
            FFN/MoE state, h, out) — the same state the analytic backward
            recomputes, captured once by the operators themselves. The
            math (and its bit-level result) is identical either way.
        """
        if positions is None:
            positions = np.arange(x.shape[1], dtype=np.int32)
        # Stream 1: attention with pre-norm and residual.
        ln1_out = self.input_layernorm.forward(x)  # (B, S, D)
        attn_out, attn_state = self.self_attn._forward_state(ln1_out, positions)  # (B, S, D)
        h = x + attn_out  # (B, S, D)

        # Stream 2: feed-forward (dense or MoE) with pre-norm and residual.
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, S, D)
        ff_out, ff_state = self.mlp._forward_state(ln2_out)  # (B, S, D)
        out = h + ff_out  # (B, S, D)
        if record is not None:
            record.update(
                {
                    "x": x,
                    "ln1_out": ln1_out,
                    "attn": attn_state,
                    "attn_out": attn_out,
                    "h": h,
                    "ln2_out": ln2_out,
                    "ff": ff_state,
                    "mlp_out": ff_out,
                    "out": out,
                }
            )
        return out

    def backward(self, dout: np.ndarray, x: np.ndarray, positions: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Analytic backward (derivation in the class docstring).

        dout: (B, S, D) upstream gradient.
        x: (B, S, D) the forward input (recomputes the intermediates).
        positions: the RoPE positions used in forward (None → arange(S)).

        Returns: (dx, dparams) with attribute-path keys (see class docstring).
        """
        if positions is None:
            positions = np.arange(x.shape[1], dtype=np.int32)

        # --- Recompute forward intermediates (same math as forward) ---
        ln1_out = self.input_layernorm.forward(x)  # (B, S, D)
        attn_out = self.self_attn.forward(ln1_out, positions)  # (B, S, D)
        h = x + attn_out  # (B, S, D)
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, S, D)
        _ff_out = self.mlp.forward(ln2_out)  # (B, S, D)

        # --- Backward: out = h + ff_out ---
        # The residual makes the upstream of the FFN output equal to dout.
        d_ff_out = dout  # (B, S, D)

        # --- FFN / MoE sublayer (input ln2_out) ---
        d_ln2_out, mlp_grads = self.mlp.backward(d_ff_out, ln2_out)  # (B, S, D)

        # --- Post-attention RMSNorm (input h, output ln2_out) ---
        d_h_ln2, d_gamma2 = self.post_attention_layernorm.backward(d_ln2_out, h)  # (B, S, D), (D,)

        # --- Residual: h = x + attn_out; dout reaches h directly too ---
        d_h = dout + d_h_ln2  # (B, S, D)

        # --- Attention sublayer (ln1_out is its input) ---
        d_ln1_out, attn_grads = self.self_attn.backward(d_h, ln1_out, positions)  # (B, S, D)

        # --- Input RMSNorm (input x, output ln1_out) ---
        dx_ln1, d_gamma1 = self.input_layernorm.backward(d_ln1_out, x)  # (B, S, D), (D,)

        # --- Residual: x feeds h directly, so all of d_h reaches x ---
        dx = d_h + dx_ln1  # (B, S, D)

        # --- Assemble the gradient dict with attribute-path keys ---
        dparams: dict = {
            "input_layernorm.gamma": d_gamma1,
            "post_attention_layernorm.gamma": d_gamma2,
        }
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            dparams[f"self_attn.{name}"] = attn_grads[name]
        if isinstance(self.mlp, MixtureOfExperts):
            dparams["mlp.gate"] = mlp_grads["gate"]
            dparams["mlp.experts"] = mlp_grads["experts"]  # list of {gate_proj, up_proj, down_proj}
            dparams["mlp.shared_experts"] = mlp_grads["shared_experts"]  # same shape, shared experts (ADR 0002)
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                dparams[f"mlp.{name}"] = mlp_grads[name]
        return dx, dparams

    def forward_step(
        self, x: np.ndarray, position: int, cache: dict, quantize: bool = False, record: dict | None = None
    ) -> np.ndarray:
        """Process ONE new token (per-token inference path, KV-cached attention).

        x: (B, 1, D) the new token's vector.
        position: the absolute token index (for RoPE; 0-based).
        cache: per-layer attention cache dict (see MultiHeadAttention.forward_step).
            Mutated in place; shape depends on quantize.
        quantize: if True, append the new K/V to the cache in 1-bit TurboQuant
            form and dequantize the full cached tensor before attention; if
            False, append the full-precision K/V (default).
        record: optional dict filled with the step's intermediates (same keys
            as ``forward``) — see that method.

        Returns: (B, 1, D) the block output for the new token.
        """
        ln1_out = self.input_layernorm.forward(x)  # (B, 1, D)
        attn_state: dict | None = {} if record is not None else None
        attn_out = self.self_attn.forward_step(ln1_out, position, cache, quantize=quantize, state=attn_state)
        h = x + attn_out  # (B, 1, D)
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, 1, D)
        ff_out, ff_state = self.mlp._forward_state(ln2_out)  # (B, 1, D)
        out = h + ff_out  # (B, 1, D)
        if record is not None:
            record.update(
                {
                    "x": x,
                    "ln1_out": ln1_out,
                    "attn": attn_state,
                    "attn_out": attn_out,
                    "h": h,
                    "ln2_out": ln2_out,
                    "ff": ff_state,
                    "mlp_out": ff_out,
                    "out": out,
                }
            )
        return out
