"""Test constants.py — Step 0: Only test class attribute existence and values.

NO helper functions tested yet. Only raw class attributes."""

from shared.constants import Attention, LayerNorm, MoE, Transformer

# --------------------------------------------------------------------------- #
#  Step 0a: Attention class attributes exist and are correct strings           #
# --------------------------------------------------------------------------- #


def test_attention_q_weight_is_string():
    assert isinstance(Attention.Q_WEIGHT, str)
    assert Attention.Q_WEIGHT == "q.weight"


def test_attention_k_weight_is_string():
    assert isinstance(Attention.K_WEIGHT, str)
    assert Attention.K_WEIGHT == "k.weight"


def test_attention_v_weight_is_string():
    assert isinstance(Attention.V_WEIGHT, str)
    assert Attention.V_WEIGHT == "v.weight"


def test_attention_o_weight_is_string():
    assert isinstance(Attention.O_WEIGHT, str)
    assert Attention.O_WEIGHT == "o.weight"


def test_attention_q_bias_is_string():
    assert isinstance(Attention.Q_BIAS, str)
    assert Attention.Q_BIAS == "q.bias"


def test_attention_k_bias_is_string():
    assert isinstance(Attention.K_BIAS, str)
    assert Attention.K_BIAS == "k.bias"


def test_attention_v_bias_is_string():
    assert isinstance(Attention.V_BIAS, str)
    assert Attention.V_BIAS == "v.bias"


def test_attention_o_bias_is_string():
    assert isinstance(Attention.O_BIAS, str)
    assert Attention.O_BIAS == "o.bias"


# --------------------------------------------------------------------------- #
#  Step 0b: LayerNorm class attributes exist and are correct strings           #
# --------------------------------------------------------------------------- #


def test_layernorm_gamma_is_string():
    assert isinstance(LayerNorm.LN_GAMMA, str)
    assert LayerNorm.LN_GAMMA == "ln_gamma"


def test_layernorm_bias_is_string():
    assert isinstance(LayerNorm.LN_BIAS, str)
    assert LayerNorm.LN_BIAS == "ln_bias"


# --------------------------------------------------------------------------- #
#  Step 0c: Transformer class attributes exist and are correct strings         #
# --------------------------------------------------------------------------- #


def test_transformer_embedding_is_string():
    assert isinstance(Transformer.EMBEDDING, str)
    assert Transformer.EMBEDDING == "embed"


def test_transformer_lm_head_weight_is_string():
    assert isinstance(Transformer.LM_HEAD_WEIGHT, str)
    assert Transformer.LM_HEAD_WEIGHT == "lm_head"


def test_transformer_lm_head_bias_is_string():
    assert isinstance(Transformer.LM_HEAD_BIAS, str)
    assert Transformer.LM_HEAD_BIAS == "lm_head_bias"


# --------------------------------------------------------------------------- #
#  Step 0d: MoE class attributes exist and are correct strings                 #
# --------------------------------------------------------------------------- #


def test_moe_w1_is_string():
    assert isinstance(MoE.W1, str)
    assert MoE.W1 == "w1"


def test_moe_w2_is_string():
    assert isinstance(MoE.W2, str)
    assert MoE.W2 == "w2"


def test_moe_w3_is_string():
    assert isinstance(MoE.W3, str)
    assert MoE.W3 == "w3"


# --------------------------------------------------------------------------- #
#  Step 1: block_param() helper                                                #
# --------------------------------------------------------------------------- #


def test_block_param_0_attn():
    from shared.constants import block_param

    result = block_param(0, "attn")
    assert isinstance(result, str)
    assert result == "blocks.0.attn"


def test_block_param_2_mlp():
    from shared.constants import block_param

    result = block_param(2, "mlp")
    assert result == "blocks.2.mlp"


# --------------------------------------------------------------------------- #
#  Step 2: attention_param() helper                                            #
# --------------------------------------------------------------------------- #


def test_attention_param_uses_constants():
    from shared.constants import Attention, attention_param

    result = attention_param(0, Attention.Q_WEIGHT)
    assert result == "blocks.0.attn.q.weight"


def test_attention_param_k():
    from shared.constants import Attention, attention_param

    result = attention_param(1, Attention.K_WEIGHT)
    assert result == "blocks.1.attn.k.weight"


def test_attention_param_bias():
    from shared.constants import Attention, attention_param

    result = attention_param(0, Attention.O_BIAS)
    assert result == "blocks.0.attn.o.bias"


# --------------------------------------------------------------------------- #
#  Step 3: layer_norm_param() helper                                           #
# --------------------------------------------------------------------------- #


def test_layer_norm_param_ln1():
    from shared.constants import layer_norm_param

    result = layer_norm_param(0, "ln1")
    assert isinstance(result, str)
    assert result == "blocks.0.ln1.gamma"


def test_layer_norm_param_ln2_is_bias():
    from shared.constants import layer_norm_param

    result = layer_norm_param(1, "ln2")
    assert result == "blocks.1.ln2.bias"


def test_layer_norm_param_final():
    from shared.constants import layer_norm_param

    result = layer_norm_param(2, "final")
    assert result == "blocks.2.final.gamma"


# --------------------------------------------------------------------------- #
#  Step 4: moe_param() helper                                                  #
# --------------------------------------------------------------------------- #


def test_moe_param_gate():
    from shared.constants import MoE, moe_param

    result = moe_param(0, 0, MoE.GATE_WEIGHT)
    assert result == "blocks.0.moe.gate.weight"


def test_moe_param_expert_w1():
    from shared.constants import MoE, moe_param

    result = moe_param(0, 0, MoE.EXPERT_W1)
    assert result == "blocks.0.moe.expert_0.w1"


def test_moe_param_expert_for_layer_2():
    from shared.constants import MoE, moe_param

    result = moe_param(2, 1, MoE.EXPERT_W3)
    assert result == "blocks.2.moe.expert_1.w3"


# --------------------------------------------------------------------------- #
#  Step 5: transformer_param() helper                                          #
# --------------------------------------------------------------------------- #


def test_transformer_param_embedding():
    from shared.constants import Transformer, transformer_param

    result = transformer_param(Transformer.EMBEDDING)
    assert result == "embed.weight"


def test_transformer_param_lm_head_weight():
    from shared.constants import Transformer, transformer_param

    result = transformer_param(Transformer.LM_HEAD_WEIGHT)
    assert result == "lm_head.weight"


def test_transformer_param_lm_head_bias():
    from shared.constants import Transformer, transformer_param

    result = transformer_param(Transformer.LM_HEAD_BIAS)
    assert result == "lm_head.bias"


# --------------------------------------------------------------------------- #
#  Step 6: get_all_params() helper                                             #
# --------------------------------------------------------------------------- #


def test_get_all_params_0_layers():
    from shared.constants import get_all_params

    result = get_all_params(0)
    assert isinstance(result, dict)

    # Transformer-level keys must be present
    assert "embed.weight" in result
    assert "lm_head.weight" in result
    assert "lm_head.bias" in result


def test_get_all_params_with_1_layer():
    from shared.constants import get_all_params

    result = get_all_params(1)

    # Must contain layer 0 attention params
    assert "blocks.0.attn.q.weight" in result
    assert "blocks.0.attn.k.weight" in result
    assert "blocks.0.attn.v.weight" in result
    assert "blocks.0.attn.o.weight" in result

    # Must contain layer 0 bias params
    assert "blocks.0.attn.q.bias" in result
    assert "blocks.0.attn.k.bias" in result
    assert "blocks.0.attn.v.bias" in result
    assert "blocks.0.attn.o.bias" in result

    # Must contain layer 0 layer_norm params
    assert "blocks.0.ln1.gamma" in result
    assert "blocks.0.ln2.bias" in result

    # Must contain layer 0 MoE params
    assert "blocks.0.moe.gate.weight" in result
    assert "blocks.0.moe.expert_0.w1" in result
    assert "blocks.0.moe.expert_0.w2" in result


def test_get_all_params_with_2_layers():
    from shared.constants import get_all_params

    result = get_all_params(2)

    # Layer 1 params must also be present
    assert "blocks.1.attn.q.weight" in result
    assert "blocks.1.ln1.gamma" in result
    assert "blocks.1.moe.gate.weight" in result


def test_get_all_params_no_duplicates():
    from shared.constants import get_all_params

    result = get_all_params(1)
    assert len(result) == len(set(result.keys()))


def test_get_all_params_all_values_are_strings():
    from shared.constants import get_all_params

    result = get_all_params(0)
    for v in result.values():
        assert isinstance(v, str)
