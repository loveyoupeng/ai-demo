"""Test constants.py — the checkpoint key scheme (HF Llama/Mixtral naming).

Covers the name constants, the Keys builders, and all_param_keys().
"""

from shared.constants import (
    ATTN_PROJS,
    FFN_PROJS,
    Attn,
    Keys,
    LayerNorm,
    Mlp,
    all_param_keys,
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


# --------------------------------------------------------------------------- #
#  all_param_keys()                                                            #
# --------------------------------------------------------------------------- #


def test_all_param_keys_dense() -> None:
    keys = all_param_keys(num_layers=2, has_moe=False, n_experts=1)
    # model-level: embed, final norm, lm_head
    assert Keys.embed() in keys
    assert Keys.final_norm() in keys
    assert Keys.lm_head() in keys
    # per layer: 2 norms + 4 attn + 3 ffn
    assert len(keys) == 3 + 2 * (2 + 4 + 3)
    assert Keys.attn(1, Attn.K_PROJ) in keys
    assert Keys.ffn(1, Mlp.UP_PROJ) in keys
    # dense: no router, no expert keys
    assert Keys.moe_gate(0) not in keys
    assert Keys.moe_expert(0, 0, Mlp.GATE_PROJ) not in keys


def test_all_param_keys_moe() -> None:
    n_experts = 4
    keys = all_param_keys(num_layers=1, has_moe=True, n_experts=n_experts)
    # per layer: 2 norms + 4 attn + 1 router + 3*E expert projs
    assert len(keys) == 3 + (2 + 4 + 1 + 3 * n_experts)
    assert Keys.moe_gate(0) in keys
    for j in range(n_experts):
        for proj in FFN_PROJS:
            assert Keys.moe_expert(0, j, proj) in keys
    # MoE layer has no dense ffn keys
    assert Keys.ffn(0, Mlp.GATE_PROJ) not in keys


def test_all_param_keys_no_duplicates() -> None:
    keys = all_param_keys(num_layers=3, has_moe=True, n_experts=2)
    assert len(keys) == len(set(keys))


def test_all_param_keys_all_strings() -> None:
    keys = all_param_keys(num_layers=1, has_moe=False, n_experts=1)
    for key in keys:
        assert isinstance(key, str)
        assert key.startswith("model.")
