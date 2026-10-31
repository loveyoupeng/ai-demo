# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Test constants.py — the checkpoint key scheme (HF Llama/Mixtral naming).

Covers the name constants and the Keys builders.
"""

from shared.constants import (
    ATTN_PROJS,
    FFN_PROJS,
    Attn,
    Keys,
    LayerNorm,
    Mlp,
)

# --------------------------------------------------------------------------- #
#  Name constants                                                              #
# --------------------------------------------------------------------------- #


def test_attn_proj_names() -> None:
    assert (Attn.Q_PROJ, Attn.K_PROJ, Attn.V_PROJ, Attn.O_PROJ) == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert ATTN_PROJS == (Attn.Q_PROJ, Attn.K_PROJ, Attn.V_PROJ, Attn.O_PROJ)


def test_mlp_proj_names() -> None:
    assert (Mlp.GATE_PROJ, Mlp.UP_PROJ, Mlp.DOWN_PROJ) == ("gate_proj", "up_proj", "down_proj")
    assert Mlp.GATE == "gate"  # MoE router, distinct from gate_proj
    assert FFN_PROJS == (Mlp.GATE_PROJ, Mlp.UP_PROJ, Mlp.DOWN_PROJ)


def test_layernorm_names() -> None:
    assert LayerNorm.INPUT == "input_layernorm"
    assert LayerNorm.POST_ATTENTION == "post_attention_layernorm"


# --------------------------------------------------------------------------- #
#  Keys builders                                                               #
# --------------------------------------------------------------------------- #


def test_keys_embed() -> None:
    assert Keys.embed() == "model.embed_tokens"


def test_keys_lm_head() -> None:
    assert Keys.lm_head() == "model.lm_head.weight"


def test_keys_final_norm() -> None:
    assert Keys.final_norm() == "model.norm.weight"


def test_keys_layer() -> None:
    assert Keys.layer(0) == "model.layers.0"
    assert Keys.layer(3) == "model.layers.3"


def test_keys_attn() -> None:
    assert Keys.attn(0, Attn.Q_PROJ) == "model.layers.0.self_attn.q_proj.weight"
    assert Keys.attn(1, Attn.O_PROJ) == "model.layers.1.self_attn.o_proj.weight"


def test_keys_ln() -> None:
    assert Keys.ln(0, LayerNorm.INPUT) == "model.layers.0.input_layernorm.weight"
    assert Keys.ln(1, LayerNorm.POST_ATTENTION) == "model.layers.1.post_attention_layernorm.weight"


def test_keys_ffn() -> None:
    assert Keys.ffn(0, Mlp.GATE_PROJ) == "model.layers.0.mlp.gate_proj.weight"
    assert Keys.ffn(2, Mlp.DOWN_PROJ) == "model.layers.2.mlp.down_proj.weight"


def test_keys_moe_gate() -> None:
    assert Keys.moe_gate(0) == "model.layers.0.mlp.gate.weight"


def test_keys_moe_expert() -> None:
    assert Keys.moe_expert(0, 1, Mlp.UP_PROJ) == "model.layers.0.mlp.experts.1.up_proj.weight"
