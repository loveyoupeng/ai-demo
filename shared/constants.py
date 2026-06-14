"""Minimal parameter name constants for transformer modules."""

from __future__ import annotations


class Attention:
    QKV_PROJ_WEIGHT: str = "qkv_proj.weight"
    QKV_PROJ_BIAS: str = "qkv_proj.bias"
    O_PROJ_WEIGHT: str = "o_proj.weight"
    O_PROJ_BIAS: str = "o_proj.bias"


class MoE:
    EXPERT_W1: str = "w1"
    EXPERT_W2: str = "w2"


def param(prefix: str, name: str) -> str:
    return f"{prefix}.{name}"
