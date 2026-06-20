"""Minimal stubs to pass C0.2 import test.

Full implementations in C1–C7. ModelConfig defined here since it's needed
by both TorchModel (C7) and the cross-backend parity tests (C14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the PyTorch decoder-only transformer.

    Mirrors shared.config.TransformerConfig so both backends use
    identical hyperparameters for parity testing.

    Attributes:
        vocab_size: Size of the token vocabulary.
        embed_dim: Hidden dimension of the transformer.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        n_groups: Number of groups for grouped-query attention (GQA).
                  Must be in [1, n_heads]. When equal to n_heads, uses
                  standard multi-head attention.
        n_experts: Number of experts in the MoE feedforward layer.
        top_k: Number of experts activated per token.
        ff_dim: Dimension of the feedforward inner dimension.
        rope_dim: Dimension for rotary position encoding.
                  0 means full-dimension RoPE.
        max_length: Maximum sequence length for KV cache.
        seed: Random seed for weight initialization.

    """

    vocab_size: int = 4096
    embed_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_groups: int = 8
    n_experts: int = 4
    top_k: int = 2
    ff_dim: int = 0  # 0 = auto (4x embed_dim)
    rope_dim: int = 0
    max_length: int = 2048
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_groups < 1 or self.n_groups > self.n_heads:
            raise ValueError(f"n_groups must be in [1, n_heads], got {self.n_groups}")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError(f"top_k must be in [1, n_experts], got {self.top_k}")
        if self.rope_dim < 0 or self.rope_dim > self.embed_dim:
            raise ValueError("rope_dim must be in [0, embed_dim]")
        # Auto ff_dim: standard transformer is 4x embed_dim
        if self.ff_dim <= 0:
            object.__setattr__(self, "ff_dim", 4 * self.embed_dim)

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.n_heads


# Minimal TorchModel stub — full implementation in C7
class TorchModel:
    """Decoder-only transformer in PyTorch.

    See C7 for full implementation with embedding, attention, MoE, etc.
    This stub exists only so C0.2's import test passes.
    """

    def __init__(self, config: ModelConfig) -> None: ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def get_parameter_dict(self) -> dict[str, torch.Tensor]: ...
