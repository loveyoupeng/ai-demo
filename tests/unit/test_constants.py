"""Tests for shared.constants — Parameter name constants for all transformer backends.

Covers: Attention, MoE, LayerNorm, Transformer classes + helper functions.
Tests: imports, values, helpers, counts, edge cases, round-trips, completeness.
"""

from __future__ import annotations

import pytest


class TestImportExisting:
    """Existing constants are importable (Attention, MoE, param)."""

    def test_attention_import(self):
        from shared.constants import Attention
        Attention.QKV_PROJ_WEIGHT
        Attention.QKV_PROJ_BIAS
        Attention.O_PROJ_WEIGHT
        Attention.O_PROJ_BIAS

    def test_moe_import(self):
        from shared.constants import MoE
        MoE.EXPERT_W1
        MoE.EXPERT_W2

    def test_param_import(self):
        from shared.constants import param
        assert callable(param)


class TestNewClassesExist:
    """All new classes and functions must exist."""

    def test_layernorm_exists(self):
        from shared.constants import LayerNorm
        assert LayerNorm is not None

    def test_transformer_exists(self):
        from shared.constants import Transformer
        assert Transformer is not None

    def test_attention_new_attributes(self):
        from shared.constants import Attention
        Attention.QKV
        Attention.Q
        Attention.K
        Attention.V
        Attention.O
        Attention.Q_WEIGHT
        Attention.K_WEIGHT
        Attention.V_WEIGHT
        Attention.O_WEIGHT
        Attention.Q_BIAS
        Attention.K_BIAS
        Attention.V_BIAS
        Attention.O_BIAS

    def test_moe_new_attributes(self):
        from shared.constants import MoE
        MoE.W1
        MoE.W2
        MoE.W3
        MoE.GATE_WEIGHT
        MoE.EXPERT_W1
        MoE.EXPERT_W2
        MoE.EXPERT_W3

    def test_layernorm_all_attributes(self):
        from shared.constants import LayerNorm
        LayerNorm.GN_BIAS
        LayerNorm.GN_GAMMA
        LayerNorm.LN_BIAS
        LayerNorm.LN_GAMMA

    def test_transformer_all_attributes(self):
        from shared.constants import Transformer
        Transformer.EMBEDDING
        Transformer.LM_HEAD_WEIGHT
        Transformer.LM_HEAD_BIAS
        Transformer.TRANSFORMER_LN_GAMMA
        Transformer.TRANSFORMER_LN_BIAS

    def test_helper_functions_exist(self):
        from shared.constants import (
            attention_param,
            block_param,
            get_all_params,
            layer_norm_param,
            moe_param,
            transformer_param,
        )
        assert all(
            fn is not None
            for fn in [
                attention_param, block_param, get_all_params,
                layer_norm_param, moe_param, transformer_param,
            ]
        )


class TestValuesAllStrings:
    """All constant values are non-empty strings."""

    def test_attention_values(self):
        from shared.constants import Attention
        for key in [
            "QKV", "Q", "K", "V", "O",
            "Q_WEIGHT", "K_WEIGHT", "V_WEIGHT", "O_WEIGHT",
            "Q_BIAS", "K_BIAS", "V_BIAS", "O_BIAS",
        ]:
            value = getattr(Attention, key)
            assert isinstance(value, str)
            assert len(value) > 0

    def test_moe_values(self):
        from shared.constants import MoE
        for key in ["W1", "W2", "W3", "GATE_WEIGHT", "EXPERT_W1", "EXPERT_W2", "EXPERT_W3"]:
            value = getattr(MoE, key)
            assert isinstance(value, str)
            assert len(value) > 0

    def test_layernorm_values(self):
        from shared.constants import LayerNorm
        for key in ["GN_BIAS", "GN_GAMMA", "LN_BIAS", "LN_GAMMA"]:
            value = getattr(LayerNorm, key)
            assert isinstance(value, str)
            assert len(value) > 0

    def test_transformer_values(self):
        from shared.constants import Transformer
        for key in [
            "EMBEDDING", "LM_HEAD_WEIGHT", "LM_HEAD_BIAS",
            "TRANSFORMER_LN_GAMMA", "TRANSFORMER_LN_BIAS",
        ]:
            value = getattr(Transformer, key)
            assert isinstance(value, str)
            assert len(value) > 0


class TestParamHelper:
    """param(prefix, name) constructs correct dotted names."""

    def test_param_simple(self):
        from shared.constants import param
        assert param("layers.0", "alpha") == "layers.0.alpha"

    def test_param_nested(self):
        from shared.constants import param
        assert param("a.b.c", "key") == "a.b.c.key"

    def test_param_edge_empty_prefix(self):
        from shared.constants import param
        assert param("", "key") == ".key"

    def test_param_edge_empty_name(self):
        from shared.constants import param
        assert param("weight", "") == "weight."

    def test_param_special_chars(self):
        from shared.constants import param
        assert param("blocks.0", "self_attn") == "blocks.0.self_attn"


class TestBlockParam:
    """block_param(layer_idx, component) generates correct keys."""

    def test_block_attn_layer0(self):
        from shared.constants import block_param
        assert block_param(0, "attn") == "blocks.0.attn"

    def test_block_attn_layer5(self):
        from shared.constants import block_param
        assert block_param(5, "attn") == "blocks.5.attn"

    def test_block_mlp_layer0(self):
        from shared.constants import block_param
        assert block_param(0, "mlp") == "blocks.0.mlp"

    def test_block_ln1(self):
        from shared.constants import block_param
        assert block_param(0, "ln1") == "blocks.0.ln1"

    def test_block_ln2(self):
        from shared.constants import block_param
        assert block_param(0, "ln2") == "blocks.0.ln2"

    def test_block_transformer(self):
        from shared.constants import block_param
        assert block_param(0, "transformer") == "blocks.0.transformer"

    def test_block_edge_empty_component(self):
        from shared.constants import block_param
        assert block_param(0, "") == "blocks.0."

    def test_block_large_index(self):
        from shared.constants import block_param
        assert block_param(100, "attn") == "blocks.100.attn"


class TestAttentionParam:
    """attention_param(layer_idx, key) generates complete parameter names."""

    def test_attention_qkv_weight_layer0(self):
        from shared.constants import Attention, attention_param
        assert attention_param(0, Attention.QKV) == "blocks.0.attn.qkv.weight"

    def test_attention_q_bias_layer1(self):
        from shared.constants import Attention, attention_param
        assert attention_param(1, Attention.Q_BIAS) == "blocks.1.attn.q.bias"

    def test_attention_k_weight(self):
        from shared.constants import Attention, attention_param
        assert attention_param(0, Attention.K_WEIGHT) == "blocks.0.attn.k.weight"

    def test_attention_all_keys_layer0(self):
        import inspect
        from shared.constants import Attention, attention_param

        # Get all str-valued attributes of Attention class
        keys = [
            v for k, v in inspect.getmembers(Attention)
            if isinstance(v, str) and not k.startswith("_")
        ]
        for key in keys:
            result = attention_param(0, key)
            assert isinstance(result, str)
            assert result.startswith("blocks.0.attn.")

    def test_attention_roundtrip_with_base(self):
        from shared.constants import Attention, attention_param, block_param

        block_key = block_param(0, "attn")
        q_key = attention_param(0, Attention.Q)
        assert f"{block_key}.{Attention.Q}" == q_key


class TestLayerNormParam:
    """layer_norm_param(layer_idx, part) generates correct LayerNorm names."""

    def test_ln1_gamma(self):
        from shared.constants import LayerNorm, block_param, layer_norm_param

        block_key = block_param(0, "ln1")
        assert layer_norm_param(0, "ln1") == f"{block_key}.gamma"
        assert f"{block_key}.gamma" == layer_norm_param(0, "ln1")
        assert f"{block_key}.bias" == layer_norm_param(0, "ln2")

    def test_ln2_layer0(self):
        from shared.constants import layer_norm_param
        result = layer_norm_param(0, "ln2")
        assert result == "blocks.0.ln2"

    def test_layer5(self):
        from shared.constants import layer_norm_param
        assert layer_norm_param(5, "ln1") == "blocks.5.ln1"

    def test_ln_roundtrip(self):
        from shared.constants import block_param, layer_norm_param
        from shared.constants import LayerNorm

        block_key = block_param(0, "ln1")
        gamma = layer_norm_param(0, "ln1")
        assert f"{block_key}.gamma" == gamma
        assert f"{block_key}.bias" == LayerNorm.LN_BIAS


class TestMoeParam:
    """moe_param(layer_idx, expert_idx, key) generates MoE names."""

    def test_expert0_w1(self):
        from shared.constants import MoE, moe_param
        assert moe_param(0, 0, MoE.EXPERT_W1) == "blocks.0.moe.expert_0.w1"

    def test_expert0_w2(self):
        from shared.constants import MoE, moe_param
        assert moe_param(0, 0, MoE.EXPERT_W2) == "blocks.0.moe.expert_0.w2"

    def test_expert0_w3(self):
        from shared.constants import MoE, moe_param
        assert moe_param(0, 0, MoE.EXPERT_W3) == "blocks.0.moe.expert_0.w3"

    def test_gate_weight(self):
        from shared.constants import MoE, moe_param
        assert moe_param(0, 0, MoE.GATE_WEIGHT) == "blocks.0.moe.gate.weight"

    def test_expert2_w1(self):
        from shared.constants import MoE, moe_param
        assert moe_param(0, 2, MoE.EXPERT_W1) == "blocks.0.moe.expert_2.w1"

    def test_layer3_expert1(self):
        from shared.constants import MoE, moe_param
        assert moe_param(3, 1, MoE.EXPERT_W1) == "blocks.3.moe.expert_1.w1"

    def test_moe_w1_non_expert(self):
        from shared.constants import MoE, moe_param
        assert moe_param(0, 0, MoE.W1) == "blocks.0.moe.w1"


class TestTransformerParam:
    """transformer_param(key) generates Transformer-level names."""

    def test_embedding(self):
        from shared.constants import Transformer, transformer_param
        assert transformer_param(Transformer.EMBEDDING) == "embed.weight"

    def test_lm_head_weight(self):
        from shared.constants import Transformer, transformer_param
        assert transformer_param(Transformer.LM_HEAD_WEIGHT) == "lm_head.weight"

    def test_lm_head_bias(self):
        from shared.constants import Transformer, transformer_param
        assert transformer_param(Transformer.LM_HEAD_BIAS) == "lm_head.bias"

    def test_transformer_ln_gamma(self):
        from shared.constants import Transformer, transformer_param
        assert transformer_param(Transformer.TRANSFORMER_LN_GAMMA) == "transformer_layernorm.gamma"

    def test_transformer_ln_bias(self):
        from shared.constants import Transformer, transformer_param
        assert transformer_param(Transformer.TRANSFORMER_LN_BIAS) == "transformer_layernorm.bias"


class TestGetAllParams:
    """get_all_params(n) returns dict with correct keys for n layers."""

    def test_key_count_1_layer(self):
        from shared.constants import get_all_params
        result = get_all_params(1)
        assert len(result) == 28

    def test_key_count_2_layers(self):
        from shared.constants import get_all_params
        result = get_all_params(2)
        assert len(result) == 43

    def test_key_count_3_layers(self):
        from shared.constants import get_all_params
        result = get_all_params(3)
        assert len(result) == 58

    def test_formula(self):
        from shared.constants import get_all_params
        for n in [1, 3, 5, 10]:
            result = get_all_params(n)
            expected = 4 * (n + 1) + 4 * n + 19 + 13
            assert len(result) == expected

    def test_1layer_has_attn_keys(self):
        from shared.constants import get_all_params
        result = get_all_params(1)
        assert "blocks.0.attn.qkv.weight" in result
        assert "blocks.0.attn.o.weight" in result

    def test_1layer_has_layernorm_keys(self):
        from shared.constants import get_all_params
        result = get_all_params(1)
        assert "blocks.0.ln1.gamma" in result
        assert "blocks.0.ln2.gamma" in result

    def test_1layer_has_moe_keys(self):
        from shared.constants import get_all_params
        result = get_all_params(1)
        assert "blocks.0.moe.expert_0.w1" in result
        assert "blocks.0.moe.gate.weight" in result

    def test_2layers_has_both(self):
        from shared.constants import get_all_params
        result = get_all_params(2)
        assert "blocks.0.attn.qkv.weight" in result
        assert "blocks.1.attn.qkv.weight" in result

    def test_all_keys_are_strings(self):
        from shared.constants import get_all_params
        result = get_all_params(2)
        assert all(isinstance(k, str) for k in result.keys())

    def test_edge_0_layers(self):
        from shared.constants import get_all_params
        result = get_all_params(0)
        assert "embed.weight" in result
        assert len(result) > 0

    def test_keys_unique(self):
        from shared.constants import get_all_params
        result = get_all_params(2)
        keys = list(result.keys())
        assert len(keys) == len(set(keys))

    def test_block_param_roundtrip_attn(self):
        from shared.constants import (
            Attention,
            attention_param,
            block_param,
        )
        # block_param(0, "attn") + key from attention_param composition
        base = block_param(0, "attn")
        key = attention_param(0, Attention.Q)
        full_key = f"{base}.{Attention.Q}"
        assert full_key == key

    def test_block_param_roundtrip_ln(self):
        from shared.constants import (
            LayerNorm,
            block_param,
            layer_norm_param,
        )
        base = block_param(0, "ln1")
        key = layer_norm_param(0, "ln1")
        assert f"{base}.{LayerNorm.LN_GAMMA}" == key

    def test_block_param_roundtrip_moe(self):
        from shared.constants import MoE, block_param, moe_param

        base = block_param(0, "moe")
        expected = f"{base}.expert_2.w1"
        assert expected == moe_param(0, 2, MoE.EXPERT_W1)

    def test_all_params_no_duplicates_across_layers(self):
        from shared.constants import get_all_params
        result = get_all_params(3)
        keys = list(result.keys())
        assert len(keys) == len(set(keys)), "Duplicate keys found"


class TestCompleteness:
    """Verify every expected attribute exists on each class."""

    def test_attention_full(self):
        from shared.constants import Attention

        required = {
            "QKV", "Q", "K", "V", "O",
            "Q_WEIGHT", "K_WEIGHT", "V_WEIGHT", "O_WEIGHT",
            "Q_BIAS", "K_BIAS", "V_BIAS", "O_BIAS",
        }
        missing = required - {attr for attr in dir(Attention) if not attr.startswith("_")}
        assert not missing, f"Missing attributes: {missing}"

    def test_moe_full(self):
        from shared.constants import MoE

        required = {"W1", "W2", "W3", "GATE_WEIGHT", "EXPERT_W1", "EXPERT_W2", "EXPERT_W3"}
        missing = required - {attr for attr in dir(MoE) if not attr.startswith("_")}
        assert not missing, f"Missing attributes: {missing}"

    def test_layernorm_full(self):
        from shared.constants import LayerNorm

        required = {"GN_BIAS", "GN_GAMMA", "LN_BIAS", "LN_GAMMA"}
        missing = required - {attr for attr in dir(LayerNorm) if not attr.startswith("_")}
        assert not missing, f"Missing attributes: {missing}"

    def test_transformer_full(self):
        from shared.constants import Transformer

        required = {
            "EMBEDDING", "LM_HEAD_WEIGHT", "LM_HEAD_BIAS",
            "TRANSFORMER_LN_GAMMA", "TRANSFORMER_LN_BIAS",
        }
        missing = required - {attr for attr in dir(Transformer) if not attr.startswith("_")}
        assert not missing, f"Missing attributes: {missing}"
