# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Tests for shared.config.TransformerConfig."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shared.config import TransformerConfig


class TestConstructionDefaults:
    """Verify default values when constructing with no arguments."""

    def test_vocab_size(self):
        cfg = TransformerConfig()
        assert cfg.vocab_size == 4096

    def test_context_length(self):
        cfg = TransformerConfig()
        assert cfg.context_length == 256

    def test_embed_dim(self):
        cfg = TransformerConfig()
        assert cfg.embed_dim == 512

    def test_n_layers(self):
        cfg = TransformerConfig()
        assert cfg.n_layers == 8

    def test_n_heads(self):
        cfg = TransformerConfig()
        assert cfg.n_heads == 8

    def test_n_groups(self):
        cfg = TransformerConfig()
        assert cfg.n_groups == 8

    def test_seed(self):
        cfg = TransformerConfig()
        assert cfg.seed == 42


class TestDerivedFields:
    """Verify computed head_dim, k_dim, v_dim, and expert_dim."""

    def test_head_dim_default(self):
        cfg = TransformerConfig()
        assert cfg.head_dim == cfg.embed_dim // cfg.n_heads  # 512 // 8 = 64

    def test_k_dim_default(self):
        cfg = TransformerConfig()
        assert cfg.k_dim == cfg.kv_heads * cfg.head_dim

    def test_v_dim_equals_k_dim(self):
        cfg = TransformerConfig()
        assert cfg.v_dim == cfg.k_dim

    def test_expert_dim_auto(self):
        cfg = TransformerConfig()
        assert cfg.expert_dim == cfg.embed_dim * 4  # 512 * 4 = 2048

    def test_expert_dim_explicit_override(self):
        cfg = TransformerConfig(expert_dim=1024)
        assert cfg.expert_dim == 1024

    def test_head_dim_custom(self):
        cfg = TransformerConfig(embed_dim=256, n_heads=4, n_groups=4)
        assert cfg.head_dim == 256 // 4  # 64


class TestValidation:
    """AssertionError on invalid values."""

    def test_vocab_size_zero(self):
        with pytest.raises(AssertionError):
            TransformerConfig(vocab_size=0)

    def test_context_length_zero(self):
        with pytest.raises(AssertionError):
            TransformerConfig(context_length=0)

    def test_embed_dim_zero(self):
        with pytest.raises(AssertionError):
            TransformerConfig(embed_dim=0)

    def test_n_layers_zero(self):
        with pytest.raises(AssertionError):
            TransformerConfig(n_layers=0)

    def test_n_heads_zero(self):
        with pytest.raises(AssertionError):
            TransformerConfig(n_heads=0)

    def test_n_groups_below_range(self):
        with pytest.raises(AssertionError):
            TransformerConfig(n_groups=0)

    def test_n_groups_above_range(self):
        with pytest.raises(AssertionError):
            TransformerConfig(n_groups=16)

    def test_n_experts_zero(self):
        with pytest.raises(AssertionError):
            TransformerConfig(n_experts=0)

    def test_top_k_below_range(self):
        with pytest.raises(AssertionError):
            TransformerConfig(top_k=0)

    def test_top_k_above_range(self):
        with pytest.raises(AssertionError):
            TransformerConfig(top_k=8)

    def test_rope_dim_zero_valid(self):
        cfg = TransformerConfig(rope_dim=0)
        assert cfg.rope_dim == 0

    def test_rope_dim_greater_than_head_dim(self):
        # With defaults head_dim=64, rope_dim=128 exceeds it
        with pytest.raises(AssertionError):
            TransformerConfig(rope_dim=128)


class TestQueryMethods:
    """is_gqa, has_moe."""

    def test_is_gqa_true(self):
        cfg = TransformerConfig(n_groups=2, n_heads=8)
        assert cfg.is_gqa() is True

    def test_is_gqa_false_equal(self):
        cfg = TransformerConfig(n_groups=8, n_heads=8)
        assert cfg.is_gqa() is False

    def test_has_moe_true(self):
        cfg = TransformerConfig(n_experts=4)
        assert cfg.has_moe() is True

    def test_has_moe_false(self):
        cfg = TransformerConfig(n_experts=1, top_k=1)
        assert cfg.has_moe() is False

    def test_mixer_kinds_default_3_deltanet_plus_full(self):
        cfg = TransformerConfig(n_layers=3)
        assert cfg.mixer_kinds() == ["d", "d", "d", "F"]
        assert cfg.n_mixers() == 4
        assert cfg.has_deltanet() is True

    def test_mixer_kinds_zero_deltanet_is_classic(self):
        cfg = TransformerConfig(n_layers=3, n_deltanet_sublayers=0)
        assert cfg.mixer_kinds() == ["F"]
        assert cfg.has_deltanet() is False

    def test_mixer_kind_index_bounds(self):
        cfg = TransformerConfig()
        assert cfg.mixer_kind(0) == "d"
        assert cfg.mixer_kind(cfg.n_mixers() - 1) == "F"
        with pytest.raises(AssertionError):
            cfg.mixer_kind(cfg.n_mixers())

    def test_negative_deltanet_count_rejected(self):
        with pytest.raises(AssertionError, match="n_deltanet_sublayers"):
            TransformerConfig(n_layers=1, n_deltanet_sublayers=-1)

    def test_serialization_roundtrip(self):
        cfg = TransformerConfig(n_layers=4, n_deltanet_sublayers=3)
        cfg2 = TransformerConfig.from_dict(cfg.to_dict())
        assert cfg2.n_deltanet_sublayers == 3
        assert cfg2.mixer_kinds() == ["d", "d", "d", "F"]

    def test_legacy_layer_pattern_key_ignored_on_load(self):
        cfg = TransformerConfig.from_dict(
            {"vocab_size": 64, "embed_dim": 32, "n_layers": 1, "n_heads": 4, "layer_pattern": "dddF"}
        )
        assert cfg.n_deltanet_sublayers == 3  # the default, not the legacy pattern
        assert cfg.mixer_kinds() == ["d", "d", "d", "F"]


class TestSerialization:
    """to_dict and from_dict roundtrip."""

    def test_to_dict_excludes_derived(self):
        cfg = TransformerConfig()
        d = cfg.to_dict()
        assert "head_dim" not in d
        assert "k_dim" not in d
        assert "v_dim" not in d

    def test_to_dict_includes_all_settable(self):
        cfg = TransformerConfig()
        d = cfg.to_dict()
        expected_keys = {
            "vocab_size",
            "context_length",
            "embed_dim",
            "n_layers",
            "n_heads",
            "n_groups",
            "swa_window",
            "mla_latent_dim",
            "mla_group_size",
            "rope_dim",
            "n_experts",
            "top_k",
            "expert_dim",
            "n_shared_experts",
            "n_deltanet_sublayers",
            "dtype",
            "seed",
            "norm_eps",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_roundtrip(self):
        cfg = TransformerConfig(embed_dim=256, n_layers=4, seed=99)
        d = cfg.to_dict()
        cfg2 = TransformerConfig.from_dict(d)
        assert cfg2.embed_dim == 256
        assert cfg2.n_layers == 4
        assert cfg2.seed == 99
        # Derived fields re-computed
        assert cfg2.head_dim == 256 // 8

    def test_from_dict_ignores_derived(self):
        """Derived fields in input dict should be silently ignored."""
        data = {
            "vocab_size": 1024,
            "head_dim": 64,
            "k_dim": 128,
            "v_dim": 128,
        }
        cfg = TransformerConfig.from_dict(data)
        assert cfg.vocab_size == 1024
        # The input dict should remain unchanged (from_dict builds a new dict)
        assert data["head_dim"] == 64
        assert data["k_dim"] == 128
        assert data["v_dim"] == 128

    def test_from_dict_ignores_legacy_knobs(self):
        """Old config.json files carry quant_type/kvcache_type/max_length/
        load_balance_loss keys; they must load cleanly after the knobs were
        removed from TransformerConfig."""
        data = {
            "vocab_size": 512,
            "max_length": 2048,
            "quant_type": "none",
            "kvcache_type": "naive",
            "load_balance_loss": 0.0,
        }
        cfg = TransformerConfig.from_dict(data)
        assert cfg.vocab_size == 512


class TestFrozen:
    """Dataclass is immutable — cannot modify fields after construction."""

    def test_cannot_modify_field(self):
        cfg = TransformerConfig()
        assert cfg.vocab_size == 4096
        with pytest.raises(FrozenInstanceError):
            cfg.vocab_size = 1000  # pyright: ignore[reportAttributeAccessIssue]

    def test_cannot_modify_derived(self):
        cfg = TransformerConfig()
        assert cfg.head_dim == 64
        with pytest.raises(FrozenInstanceError):
            cfg.head_dim = 100  # pyright: ignore[reportAttributeAccessIssue]
