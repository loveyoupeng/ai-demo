# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Parameter key scheme for cross-backend checkpoints.

This module is the SINGLE SOURCE OF TRUTH for the flat-dict checkpoint keys
stored in ``ckpt.npz``. Every track (NumPy, PyTorch, Triton, CUDA) saves and
loads parameters under these keys, which is what makes a model trained on one
track runnable on any other.

Naming follows the industry convention used by HuggingFace Llama / Mixtral
models, so every key is directly searchable on the web:

    model.embed_tokens                                  (V, D)
    model.layers.{i}.mixers.{m}.input_layernorm.weight   (D,)     per-mixer pre-norm
    model.layers.{i}.mixers.{m}.self_attn.q_proj.weight (D, H*hd)  mixer m = full attention
    model.layers.{i}.mixers.{m}.self_attn.k_proj.weight (D, G*hd)
    model.layers.{i}.mixers.{m}.self_attn.v_proj.weight (D, G*hd)
    model.layers.{i}.mixers.{m}.self_attn.o_proj.weight (H*hd, D)
    model.layers.{i}.mixers.{m}.deltanet.q_proj.weight  (D, H*hd)  mixer m = Gated DeltaNet
    model.layers.{i}.mixers.{m}.deltanet.k_proj.weight  (D, H*hd)
    model.layers.{i}.mixers.{m}.deltanet.v_proj.weight  (D, H*hd)
    model.layers.{i}.mixers.{m}.deltanet.o_proj.weight  (H*hd, D)
    model.layers.{i}.mixers.{m}.deltanet.a_proj.weight  (D, H)
    model.layers.{i}.mixers.{m}.deltanet.dt_bias.weight (H,)
    model.layers.{i}.mixers.{m}.deltanet.gate_proj.weight (D, D)
    model.layers.{i}.mixers.{m}.deltanet.norm.weight   (D,)
    model.layers.{i}.post_attention_layernorm.weight    (D,)     norm before the FFN
    model.layers.{i}.mlp.gate_proj.weight               (D, FF)   dense SwiGLU
    model.layers.{i}.mlp.up_proj.weight                 (D, FF)
    model.layers.{i}.mlp.down_proj.weight               (FF, D)
    model.layers.{i}.mlp.gate.weight                    (D, E)    MoE router
    model.layers.{i}.mlp.experts.{j}.gate_proj.weight   (D, FF)   MoE experts
    model.layers.{i}.mlp.experts.{j}.up_proj.weight     (D, FF)
    model.layers.{i}.mlp.experts.{j}.down_proj.weight   (FF, D)
    model.norm.weight                                   (D,)      final RMSNorm
    model.lm_head.weight                                (D, V)

where D = embed_dim, V = vocab_size, H = n_heads, G = n_groups (K/V heads),
hd = head_dim, FF = expert_dim, E = n_experts, i = layer index, m = mixer
(mixing-sublayer) index within the block, j = expert index. Every block runs
``n_deltanet_sublayers`` DeltaNet mixers followed by one full-attention mixer
(ADR 0007), each mixer with its own pre-norm. The model uses no bias terms
(Llama convention), so every key ends in ``.weight``.
"""

from __future__ import annotations


class Attn:
    """Attention projection names (HuggingFace Llama convention)."""

    Q_PROJ: str = "q_proj"
    K_PROJ: str = "k_proj"
    V_PROJ: str = "v_proj"
    O_PROJ: str = "o_proj"
    # MLA projections (ADR 0009, active when config.mla_latent_dim > 0):
    # the K/V cache stores one latent c = x @ kv_down; K/V are rebuilt as
    # k = RoPE(c @ k_up), v = c @ v_up.
    KV_DOWN_PROJ: str = "kv_down_proj"
    K_UP_PROJ: str = "k_up_proj"
    V_UP_PROJ: str = "v_up_proj"


# All attention projections, in q/k/v/o order.
ATTN_PROJS: tuple[str, str, str, str] = (Attn.Q_PROJ, Attn.K_PROJ, Attn.V_PROJ, Attn.O_PROJ)

# MLA projection set (replaces k_proj/v_proj in the registry when active).
ATTN_MLA_PROJS: tuple[str, str, str] = (Attn.KV_DOWN_PROJ, Attn.K_UP_PROJ, Attn.V_UP_PROJ)


class Mlp:
    """Feed-forward / MoE projection names (HuggingFace Llama/Mixtral convention).

    SwiGLU naming: the *gate* projection is activated with SiLU and multiplies
    the *up* projection element-wise; the *down* projection maps back to the
    model width. In MoE layers, ``gate`` (without ``_proj``) is the router.
    """

    GATE_PROJ: str = "gate_proj"
    UP_PROJ: str = "up_proj"
    DOWN_PROJ: str = "down_proj"
    GATE: str = "gate"  # MoE router (distinct from gate_proj)


# All three SwiGLU projections, in gate/up/down order.
FFN_PROJS: tuple[str, str, str] = (Mlp.GATE_PROJ, Mlp.UP_PROJ, Mlp.DOWN_PROJ)


class LayerNorm:
    """Per-block normalization names (Llama convention)."""

    INPUT: str = "input_layernorm"  # pre-norm before a mixing sublayer (mixer m)
    POST_ATTENTION: str = "post_attention_layernorm"  # norm before feed-forward


class Mixer:
    """Per-mixer sub-namespace names (ADR 0007: a block's mixing sublayers)."""

    SELF_ATTN: str = "self_attn"  # full-attention mixer's projections
    DELTANET: str = "deltanet"  # Gated DeltaNet mixer's params


def mixer_ln_name(mixer_idx: int) -> str:
    """HF-ish per-mixer pre-norm attribute name: ``mixers.{m}.input_layernorm``."""
    return f"mixers.{mixer_idx}.{LayerNorm.INPUT}"


class DeltaNet:
    """Gated DeltaNet mixer parameter names (ADR 0006/0007), HF-ish convention.

    Per DeltaNet layer: q/k/v/o projections (same roles as attention),
    a per-head forget-gate projection ``a_proj`` (D → H) with a per-head
    bias ``dt_bias`` (H,), an output gate ``gate_proj`` (D → D), and an
    output RMSNorm gain ``norm`` (D,).
    """

    Q_PROJ: str = "q_proj"
    K_PROJ: str = "k_proj"
    V_PROJ: str = "v_proj"
    O_PROJ: str = "o_proj"
    A_PROJ: str = "a_proj"  # per-head forget-gate input projection (D → H)
    DT_BIAS: str = "dt_bias"  # per-head forget-gate bias (H,)
    GATE_PROJ: str = "gate_proj"  # output gate projection (D → D)
    NORM: str = "norm"  # output RMSNorm gain (D,)


# All DeltaNet projection-ish params, in q/k/v/o/a/gate order (dt_bias and
# norm are vectors, listed separately).
DELTANET_PROJS: tuple[str, str, str, str, str, str] = (
    DeltaNet.Q_PROJ,
    DeltaNet.K_PROJ,
    DeltaNet.V_PROJ,
    DeltaNet.O_PROJ,
    DeltaNet.A_PROJ,
    DeltaNet.GATE_PROJ,
)


class Keys:
    """Builders for the flat-dict checkpoint keys.

    Every save/load path in every track derives its keys from this class, so
    the key scheme has exactly one definition.
    """

    @staticmethod
    def embed() -> str:
        """Token embedding table key — shape (vocab_size, embed_dim)."""
        return "model.embed_tokens"

    @staticmethod
    def lm_head() -> str:
        """Language-model head key — shape (embed_dim, vocab_size)."""
        return "model.lm_head.weight"

    @staticmethod
    def final_norm() -> str:
        """Final RMSNorm gamma key — shape (embed_dim,)."""
        return "model.norm.weight"

    @staticmethod
    def layer(layer_idx: int) -> str:
        """Base path for one transformer block (layer)."""
        return f"model.layers.{layer_idx}"

    @staticmethod
    def attn(layer_idx: int, mixer_idx: int, proj: str) -> str:
        """Attention projection key of mixing sublayer ``mixer_idx``, e.g.
        ``Keys.attn(0, 3, Attn.Q_PROJ)`` →
        ``model.layers.0.mixers.3.self_attn.q_proj.weight``."""
        return f"model.layers.{layer_idx}.mixers.{mixer_idx}.{Mixer.SELF_ATTN}.{proj}.weight"

    @staticmethod
    def ln(layer_idx: int, which: str) -> str:
        """Per-block normalization key, e.g. ``Keys.ln(0, LayerNorm.POST_ATTENTION)``
        → ``model.layers.0.post_attention_layernorm.weight``."""
        return f"model.layers.{layer_idx}.{which}.weight"

    @staticmethod
    def mixer_ln(layer_idx: int, mixer_idx: int) -> str:
        """Per-mixer pre-norm key →
        ``model.layers.{i}.mixers.{m}.input_layernorm.weight``."""
        return f"model.layers.{layer_idx}.{mixer_ln_name(mixer_idx)}.weight"

    @staticmethod
    def ffn(layer_idx: int, proj: str) -> str:
        """Dense SwiGLU projection key, e.g. ``Keys.ffn(0, Mlp.GATE_PROJ)``
        → ``model.layers.0.mlp.gate_proj.weight``."""
        return f"model.layers.{layer_idx}.mlp.{proj}.weight"

    @staticmethod
    def moe_gate(layer_idx: int) -> str:
        """MoE router key → ``model.layers.{i}.mlp.gate.weight``."""
        return f"model.layers.{layer_idx}.mlp.{Mlp.GATE}.weight"

    @staticmethod
    def moe_expert(layer_idx: int, expert_idx: int, proj: str) -> str:
        """MoE expert SwiGLU projection key →
        ``model.layers.{i}.mlp.experts.{j}.gate_proj.weight``."""
        return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight"

    @staticmethod
    def moe_shared_expert(layer_idx: int, shared_idx: int, proj: str) -> str:
        """MoE shared-expert SwiGLU projection key →
        ``model.layers.{i}.mlp.shared_experts.{s}.gate_proj.weight``."""
        return f"model.layers.{layer_idx}.mlp.shared_experts.{shared_idx}.{proj}.weight"

    @staticmethod
    def deltanet(layer_idx: int, mixer_idx: int, param: str) -> str:
        """Gated DeltaNet parameter key of mixing sublayer ``mixer_idx``, e.g.
        ``Keys.deltanet(0, 1, DeltaNet.Q_PROJ)`` →
        ``model.layers.0.mixers.1.deltanet.q_proj.weight``."""
        return f"model.layers.{layer_idx}.mixers.{mixer_idx}.{Mixer.DELTANET}.{param}.weight"


# ── Generation / decoding ─────────────────────────────────────────────────────

REP_PENALTY: float = 1.3
"""Repetition-penalty factor for generation (Holtzman et al., CTL 2020).

Shared by every sampler in the repo (the server's ``_sample``, and the
NumPy/Torch record adapters' ``_pick``) so the four tracks stay
equivalent: a given (seed, temp, top_k) decodes identically everywhere.
1.3 is the standard production value; 1.0 disables the penalty (the hard
no-immediate-repeat block is separate and always on).
"""
