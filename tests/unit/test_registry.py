# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
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
from shared.constants import ATTN_PROJS, Attn, DeltaNet, Keys, LayerNorm, Mlp
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
        # 3 model-level + per layer: 1 post-norm + 4 mixer norms + 3 dn mixers x 8 + 1 attn mixer x 4 + 3 ffn projs
        assert len(reg.entries) == 3 + 2 * (1 + 4 + 3 * 8 + 4 + 3)

    def test_moe_entry_count(self) -> None:
        reg = ParameterRegistry(_moe_cfg(n_experts=3))
        # 3 model-level + per layer: 1 post norm + 4 mixer norms + 3 dn x 8 + 1 attn x 4 + 1 router + 3*E expert entries
        assert len(reg.entries) == 3 + (
            1 + 4 + 3 * 8 + 4 + 1 + 3 * 3
        )  # 1 post-norm + 4 mixer norms + 3 dn x 8 + 1 attn x 4 + router + 3 experts x 3 projs

    def test_keys_match_expected_scheme(self) -> None:
        cfg = _moe_cfg(n_experts=2)
        reg = ParameterRegistry(cfg)
        keys = reg.keys()
        # spot-check the scheme's shape: model-level keys + per-layer keys
        assert keys[0] == "model.embed_tokens"
        assert "model.layers.0.mlp.gate.weight" in keys  # router
        assert "model.layers.0.mlp.experts.1.down_proj.weight" in keys
        assert "model.layers.0.mlp.gate_proj.weight" not in keys  # MoE has no dense FFN

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

    def test_keys_include_shared_expert_keys(self) -> None:
        cfg = self._shared_cfg(1)
        reg = ParameterRegistry(cfg)
        keys = reg.keys()
        assert "model.layers.0.mlp.shared_experts.0.gate_proj.weight" in keys
        assert "model.layers.0.mlp.experts.2.up_proj.weight" in keys

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
        assert shapes[Keys.attn(0, 3, Attn.Q_PROJ)] == (8, 4 * hd)
        assert shapes[Keys.attn(0, 3, Attn.K_PROJ)] == (8, 2 * hd)
        assert shapes[Keys.attn(0, 3, Attn.O_PROJ)] == (4 * hd, 8)

    def test_ffn_shapes(self) -> None:
        shapes = ParameterRegistry(_dense_cfg()).expected_shapes()
        assert shapes[Keys.ffn(0, Mlp.GATE_PROJ)] == (8, 16)  # (D, FF)
        assert shapes[Keys.ffn(0, Mlp.DOWN_PROJ)] == (16, 8)  # (FF, D)

    def test_norm_shapes(self) -> None:
        shapes = ParameterRegistry(_dense_cfg()).expected_shapes()
        assert shapes[Keys.mixer_ln(0, 0)] == (8,)
        assert shapes[Keys.ln(0, LayerNorm.POST_ATTENTION)] == (8,)
        assert shapes[Keys.mixer_ln(0, 3)] == (8,)


class TestTorchTransposeRule:
    """Only nn.Linear-backed weights are transposed in the PyTorch track."""

    def test_transposed_keys(self) -> None:
        cfg = _moe_cfg()
        reg = ParameterRegistry(cfg)
        transposed = {e.key for e in reg.entries if e.torch_transpose}
        expected = {Keys.moe_gate(0), Keys.lm_head()}
        for m in range(cfg.n_mixers()):
            if cfg.mixer_kind(m) == "d":
                for param in (
                    DeltaNet.Q_PROJ,
                    DeltaNet.K_PROJ,
                    DeltaNet.V_PROJ,
                    DeltaNet.O_PROJ,
                    DeltaNet.A_PROJ,
                    DeltaNet.GATE_PROJ,
                ):
                    expected.add(Keys.deltanet(0, m, param))
            else:
                for proj in ATTN_PROJS:
                    expected.add(Keys.attn(0, m, proj))
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


def _hybrid_cfg(n_layers: int = 4, n_deltanet: int = 3) -> TransformerConfig:
    return TransformerConfig.from_dict(
        {
            "vocab_size": 32,
            "embed_dim": 8,
            "n_layers": n_layers,
            "n_heads": 2,
            "n_experts": 1,
            "top_k": 1,
            "expert_dim": 16,
            "n_deltanet_sublayers": n_deltanet,
        }
    )


class TestHybridDeltaNetEntries:
    def test_deltanet_keys_on_d_mixers_attn_on_f_mixer(self) -> None:
        reg = ParameterRegistry(_hybrid_cfg(4))
        keys = reg.keys()
        # every layer: mixers 0..2 are DeltaNet, mixer 3 is full attention
        for i in range(4):
            for m in range(3):
                assert Keys.deltanet(i, m, DeltaNet.Q_PROJ) in keys
                assert Keys.attn(i, m, Attn.Q_PROJ) not in keys
                assert Keys.mixer_ln(i, m) in keys
            assert Keys.attn(i, 3, Attn.Q_PROJ) in keys
            assert Keys.deltanet(i, 3, DeltaNet.Q_PROJ) not in keys

    def test_deltanet_shapes(self) -> None:
        reg = ParameterRegistry(_hybrid_cfg(4))
        shapes = reg.expected_shapes()
        D, H = 8, 2
        hd = D // H
        assert shapes[Keys.deltanet(1, 1, DeltaNet.Q_PROJ)] == (D, H * hd)
        assert shapes[Keys.deltanet(1, 1, DeltaNet.O_PROJ)] == (H * hd, D)
        assert shapes[Keys.deltanet(1, 1, DeltaNet.A_PROJ)] == (D, H)
        assert shapes[Keys.deltanet(1, 1, DeltaNet.DT_BIAS)] == (H,)
        assert shapes[Keys.deltanet(1, 1, DeltaNet.GATE_PROJ)] == (D, D)
        assert shapes[Keys.deltanet(1, 1, DeltaNet.NORM)] == (D,)
        assert shapes[Keys.mixer_ln(2, 0)] == (D,)

    def test_all_full_config_has_no_deltanet_keys(self) -> None:
        reg = ParameterRegistry(_hybrid_cfg(2, n_deltanet=0))
        assert not any(".deltanet." in k for k in reg.keys())  # noqa: SIM118
        assert Keys.attn(0, 0, Attn.Q_PROJ) in reg
