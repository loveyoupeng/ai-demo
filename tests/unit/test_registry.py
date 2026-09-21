"""Tests for shared.registry — the parameter registry (checkpoint format owner).

Covers:
- entry enumeration (dense and MoE) matches the Keys scheme,
- expected shapes as a function of the config,
- the PyTorch transpose rule (only nn.Linear-backed keys),
- validate(): clean dicts pass; stale checkpoints (missing/extra keys,
  wrong shapes) fail fast with a clear message.
"""

import numpy as np
import pytest

from shared.config import TransformerConfig
from shared.constants import Attn, Keys, LayerNorm, Mlp
from shared.registry import ParameterRegistry


def _dense_cfg(n_layers: int = 2) -> TransformerConfig:
    return TransformerConfig.from_dict(
        {
            "vocab_size": 32,
            "embed_dim": 8,
            "n_layers": n_layers,
            "n_heads": 2,
            "n_experts": 1,
            "top_k": 1,
            "expert_dim": 16,
        }
    )


def _moe_cfg(n_experts: int = 3) -> TransformerConfig:
    return TransformerConfig.from_dict(
        {
            "vocab_size": 32,
            "embed_dim": 8,
            "n_layers": 1,
            "n_heads": 2,
            "n_experts": n_experts,
            "top_k": 2,
            "expert_dim": 16,
        }
    )


class TestEntryEnumeration:
    """Registry entries match the Keys scheme and cover every parameter."""

    def test_dense_entry_count(self) -> None:
        reg = ParameterRegistry(_dense_cfg(n_layers=2))
        # 3 model-level + per layer: 2 norms + 4 attn + 3 ffn
        assert len(reg.entries) == 3 + 2 * (2 + 4 + 3)

    def test_moe_entry_count(self) -> None:
        reg = ParameterRegistry(_moe_cfg(n_experts=3))
        # 3 model-level + per layer: 2 norms + 4 attn + 1 router + 3*E experts
        assert len(reg.entries) == 3 + (2 + 4 + 1 + 3 * 3)

    def test_keys_match_all_param_keys(self) -> None:
        from shared.constants import all_param_keys

        cfg = _moe_cfg(n_experts=2)
        reg = ParameterRegistry(cfg)
        assert reg.keys() == all_param_keys(cfg.n_layers, has_moe=True, n_experts=2)

    def test_no_duplicate_keys(self) -> None:
        reg = ParameterRegistry(_moe_cfg(n_experts=4))
        keys = reg.keys()
        assert len(keys) == len(set(keys))

    def test_moe_layer_has_no_dense_ffn_keys(self) -> None:
        reg = ParameterRegistry(_moe_cfg())
        keys = set(reg.keys())
        assert Keys.ffn(0, Mlp.GATE_PROJ) not in keys
        assert Keys.moe_gate(0) in keys


class TestSharedExpertEntries:
    """Shared-expert keys (ADR 0002): present only when n_shared_experts > 0."""

    @staticmethod
    def _shared_cfg(n_shared_experts: int = 2) -> TransformerConfig:
        return TransformerConfig.from_dict(
            {
                "vocab_size": 32,
                "embed_dim": 8,
                "n_layers": 1,
                "n_heads": 2,
                "n_experts": 3,
                "top_k": 1,
                "expert_dim": 16,
                "n_shared_experts": n_shared_experts,
            }
        )

    def test_shared_expert_keys_and_shapes(self) -> None:
        reg = ParameterRegistry(self._shared_cfg(2))
        shapes = reg.expected_shapes()
        for s in (0, 1):
            assert shapes[Keys.moe_shared_expert(0, s, Mlp.GATE_PROJ)] == (8, 16)
            assert shapes[Keys.moe_shared_expert(0, s, Mlp.UP_PROJ)] == (8, 16)
            assert shapes[Keys.moe_shared_expert(0, s, Mlp.DOWN_PROJ)] == (16, 8)

    def test_default_has_no_shared_expert_keys(self) -> None:
        keys = ParameterRegistry(_moe_cfg()).keys()
        assert all("shared_experts" not in key for key in keys)

    def test_keys_match_all_param_keys_with_shared(self) -> None:
        from shared.constants import all_param_keys

        cfg = self._shared_cfg(1)
        reg = ParameterRegistry(cfg)
        assert reg.keys() == all_param_keys(cfg.n_layers, has_moe=True, n_experts=3, n_shared_experts=1)

    def test_dense_config_rejects_shared_experts(self) -> None:
        with pytest.raises(AssertionError, match="requires MoE"):
            TransformerConfig.from_dict(
                {
                    "vocab_size": 32,
                    "embed_dim": 8,
                    "n_layers": 1,
                    "n_heads": 2,
                    "n_experts": 1,
                    "n_shared_experts": 1,
                }
            )


class TestExpectedShapes:
    """Shapes are derived from the config (the format contract)."""

    def test_embedding_and_lm_head(self) -> None:
        shapes = ParameterRegistry(_dense_cfg()).expected_shapes()
        assert shapes[Keys.embed()] == (32, 8)  # (V, D)
        assert shapes[Keys.lm_head()] == (8, 32)  # (D, V)
        assert shapes[Keys.final_norm()] == (8,)

    def test_attention_shapes_gqa(self) -> None:
        cfg = TransformerConfig.from_dict(
            {
                "vocab_size": 32,
                "embed_dim": 8,
                "n_layers": 1,
                "n_heads": 4,
                "n_groups": 2,
            }
        )
        shapes = ParameterRegistry(cfg).expected_shapes()
        hd = 8 // 4
        assert shapes[Keys.attn(0, Attn.Q_PROJ)] == (8, 4 * hd)
        assert shapes[Keys.attn(0, Attn.K_PROJ)] == (8, 2 * hd)
        assert shapes[Keys.attn(0, Attn.O_PROJ)] == (4 * hd, 8)

    def test_ffn_shapes(self) -> None:
        shapes = ParameterRegistry(_dense_cfg()).expected_shapes()
        assert shapes[Keys.ffn(0, Mlp.GATE_PROJ)] == (8, 16)  # (D, FF)
        assert shapes[Keys.ffn(0, Mlp.DOWN_PROJ)] == (16, 8)  # (FF, D)

    def test_norm_shapes(self) -> None:
        shapes = ParameterRegistry(_dense_cfg()).expected_shapes()
        assert shapes[Keys.ln(0, LayerNorm.INPUT)] == (8,)
        assert shapes[Keys.ln(0, LayerNorm.POST_ATTENTION)] == (8,)


class TestTorchTransposeRule:
    """Only nn.Linear-backed weights are transposed in the PyTorch track."""

    def test_transposed_keys(self) -> None:
        cfg = _moe_cfg()
        reg = ParameterRegistry(cfg)
        transposed = {e.key for e in reg.entries if e.torch_transpose}
        expected = {
            Keys.attn(0, Attn.Q_PROJ),
            Keys.attn(0, Attn.K_PROJ),
            Keys.attn(0, Attn.V_PROJ),
            Keys.attn(0, Attn.O_PROJ),
            Keys.moe_gate(0),
            Keys.lm_head(),
        }
        assert transposed == expected

    def test_no_transpose_for_raw_weights(self) -> None:
        reg = ParameterRegistry(_dense_cfg())
        for entry in reg.entries:
            if entry.key.startswith("model.layers.0.mlp."):
                assert not entry.torch_transpose
            if entry.key in (Keys.embed(), Keys.final_norm()):
                assert not entry.torch_transpose


class TestValidate:
    """validate() rejects stale checkpoints with a clear message."""

    def test_clean_dict_passes(self) -> None:
        reg = ParameterRegistry(_moe_cfg())
        params = {k: np.zeros(s) for k, s in reg.expected_shapes().items()}
        reg.validate(params)  # no exception

    def test_missing_key_rejected(self) -> None:
        reg = ParameterRegistry(_dense_cfg())
        params = {k: np.zeros(s) for k, s in reg.expected_shapes().items()}
        del params[Keys.embed()]
        with pytest.raises(ValueError, match="missing keys"):
            reg.validate(params)

    def test_stale_key_rejected(self) -> None:
        reg = ParameterRegistry(_dense_cfg())
        params = {k: np.zeros(s) for k, s in reg.expected_shapes().items()}
        params["blocks.0.attn.q.weight"] = np.zeros((8, 8))  # pre-migration key
        with pytest.raises(ValueError, match="unexpected keys"):
            reg.validate(params)

    def test_shape_mismatch_rejected(self) -> None:
        reg = ParameterRegistry(_dense_cfg())
        params = {k: np.zeros(s) for k, s in reg.expected_shapes().items()}
        params[Keys.embed()] = np.zeros((16, 8))  # wrong vocab
        with pytest.raises(ValueError, match="Shape mismatch"):
            reg.validate(params)
