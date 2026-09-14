"""Parameter key scheme for cross-backend checkpoints.

This module is the SINGLE SOURCE OF TRUTH for the flat-dict checkpoint keys
stored in ``ckpt.npz``. Every track (NumPy, PyTorch, Triton, CUDA) saves and
loads parameters under these keys, which is what makes a model trained on one
track runnable on any other.

Naming follows the industry convention used by HuggingFace Llama / Mixtral
models, so every key is directly searchable on the web:

    model.embed_tokens                                  (V, D)
    model.layers.{i}.self_attn.q_proj.weight            (D, H*hd)
    model.layers.{i}.self_attn.k_proj.weight            (D, G*hd)
    model.layers.{i}.self_attn.v_proj.weight            (D, G*hd)
    model.layers.{i}.self_attn.o_proj.weight            (H*hd, D)
    model.layers.{i}.input_layernorm.weight             (D,)
    model.layers.{i}.post_attention_layernorm.weight    (D,)
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
hd = head_dim, FF = expert_dim, E = n_experts, i = layer index, j = expert
index. The model uses no bias terms (Llama convention), so every key ends in
``.weight``.
"""

from __future__ import annotations


class Attn:
    """Attention projection names (HuggingFace Llama convention)."""

    Q_PROJ: str = "q_proj"
    K_PROJ: str = "k_proj"
    V_PROJ: str = "v_proj"
    O_PROJ: str = "o_proj"


# All attention projections, in q/k/v/o order.
ATTN_PROJS: tuple[str, str, str, str] = (Attn.Q_PROJ, Attn.K_PROJ, Attn.V_PROJ, Attn.O_PROJ)


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

    INPUT: str = "input_layernorm"  # norm before attention
    POST_ATTENTION: str = "post_attention_layernorm"  # norm before feed-forward


class Keys:
    """Builders for the flat-dict checkpoint keys.

    Every save/load path in every track derives its keys from this class, so
    the key scheme has exactly one definition.
    """

    PREFIX: str = "model"

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
    def attn(layer_idx: int, proj: str) -> str:
        """Attention projection key, e.g. ``Keys.attn(0, Attn.Q_PROJ)``
        → ``model.layers.0.self_attn.q_proj.weight``."""
        return f"model.layers.{layer_idx}.self_attn.{proj}.weight"

    @staticmethod
    def ln(layer_idx: int, which: str) -> str:
        """Per-block normalization key, e.g. ``Keys.ln(0, LayerNorm.INPUT)``
        → ``model.layers.0.input_layernorm.weight``."""
        return f"model.layers.{layer_idx}.{which}.weight"

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


def all_param_keys(num_layers: int, has_moe: bool, n_experts: int) -> list[str]:
    """Every parameter key for a model with ``num_layers`` blocks.

    Dense layers contribute the three SwiGLU projections; MoE layers
    contribute the router plus the three projections per expert.
    """
    keys: list[str] = [Keys.embed()]
    for i in range(num_layers):
        keys.append(Keys.ln(i, LayerNorm.INPUT))
        keys.append(Keys.ln(i, LayerNorm.POST_ATTENTION))
        for proj in ATTN_PROJS:
            keys.append(Keys.attn(i, proj))
        if has_moe:
            keys.append(Keys.moe_gate(i))
            for j in range(n_experts):
                for proj in FFN_PROJS:
                    keys.append(Keys.moe_expert(i, j, proj))
        else:
            for proj in FFN_PROJS:
                keys.append(Keys.ffn(i, proj))
    keys.append(Keys.final_norm())
    keys.append(Keys.lm_head())
    return keys
