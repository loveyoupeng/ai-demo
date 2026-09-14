"""Hyperparameter configuration for the decoder-only transformer.

This is the SINGLE SOURCE OF TRUTH — every backend (numpy, torch, triton, cuda)
reads from this module. All config values are validated on construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransformerConfig:
    """All hyperparameters for the decoder-only transformer.

    Example (defaults — a medium-sized model):
        vocab_size=4096, embed_dim=512, n_layers=8, n_heads=8
        → head_dim = 512 // 8 = 64

    Derived fields:
        head_dim = embed_dim // n_heads
        k_dim = n_groups * head_dim
        v_dim = k_dim
        expert_dim = 4 * embed_dim (when set to 0)
    """

    # Architecture
    vocab_size: int = 4096  # Token vocabulary size
    context_length: int = 256  # Max sequence length for training
    embed_dim: int = 512  # Hidden dimension (model width D)
    n_layers: int = 8  # Number of transformer blocks
    n_heads: int = 8  # Number of query heads
    n_groups: int | None = None  # None → n_heads (standard MHA); else K/V heads (must divide n_heads)

    # Positional encoding
    # 0 → rotate all head dimensions (standard RoPE).
    # 0 < rope_dim < head_dim → rotate the first rope_dim dimensions only.
    rope_dim: int = 0

    # Mixture of Experts (opt-in)
    # n_experts == 1 → standard dense SwiGLU feed-forward (no MoE).
    # n_experts > 1 → MoE: router + top-k over SwiGLU experts.
    n_experts: int = 1
    top_k: int = 1  # Top-k experts per token (1 <= top_k <= n_experts)
    expert_dim: int = 0  # FFN hidden dimension; 0 = 4 * embed_dim (standard)

    # Inference
    max_length: int = 2048  # Max generation length

    # KV Cache
    quant_type: str = "none"  # "none"/"1-bit"/"2-bit"/"4-bit"
    kvcache_type: str = "naive"  # "naive" or "turboquant"
    load_balance_loss: float = 0.0  # MoE load balancing weight

    # Training
    seed: int = 42
    norm_eps: float = 1e-6  # Epsilon for RMSNorm (one value for all tracks)

    # Derived (computed)
    head_dim: int = field(init=False, default=0)  # = embed_dim // n_heads
    k_dim: int = field(init=False, default=0)  # = n_groups * head_dim
    v_dim: int = field(init=False, default=0)  # = k_dim

    def __post_init__(self) -> None:
        """Validate and compute derived fields."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.context_length > 0, "context_length must be positive"
        assert self.embed_dim > 0, "embed_dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        # Resolve n_groups (None → n_heads) once; all derived math uses the int.
        n_groups = self.n_heads if self.n_groups is None else self.n_groups
        assert 1 <= n_groups <= self.n_heads, f"n_groups must be in [1, n_heads], got {n_groups}"
        object.__setattr__(self, "n_groups", n_groups)
        assert self.n_heads % n_groups == 0, f"n_heads must be divisible by n_groups ({self.n_heads} % {n_groups})"
        assert self.n_experts > 0, "n_experts must be positive"
        assert 1 <= self.top_k <= self.n_experts, f"top_k must be in [1, n_experts], got {self.top_k}"
        assert self.quant_type in ("none", "1-bit", "2-bit", "4-bit"), (
            f"quant_type must be 'none'/'1-bit'/'2-bit'/'4-bit', got {self.quant_type}"
        )
        assert self.kvcache_type in ("naive", "turboquant"), (
            f"kvcache_type must be 'naive' or 'turboquant', got {self.kvcache_type}"
        )

        head_dim = self.embed_dim // self.n_heads
        assert self.rope_dim == 0 or (self.rope_dim <= head_dim and self.rope_dim % 2 == 0), (
            f"rope_dim must be 0 or an even number <= head_dim ({head_dim}), got {self.rope_dim}"
        )
        object.__setattr__(self, "head_dim", head_dim)
        object.__setattr__(self, "k_dim", n_groups * head_dim)
        object.__setattr__(self, "v_dim", n_groups * head_dim)
        if self.expert_dim == 0:
            object.__setattr__(self, "expert_dim", self.embed_dim * 4)

    def is_gqa(self) -> bool:
        """GQA active when n_groups < n_heads (multiple query heads share one K/V head)."""
        return self.kv_heads < self.n_heads

    @property
    def kv_heads(self) -> int:
        """Resolved K/V head count (``n_groups``; ``None`` already resolved to ``n_heads``)."""
        assert self.n_groups is not None
        return self.n_groups

    def has_moe(self) -> bool:
        """MoE active when n_experts > 1."""
        return self.n_experts > 1

    def has_quantized_cache(self) -> bool:
        """True when quant_type != 'none'."""
        return self.quant_type != "none"

    def to_dict(self) -> dict[str, Any]:
        """Serialize, excluding derived fields head_dim/k_dim/v_dim."""
        skip = {"head_dim", "k_dim", "v_dim"}
        return {k: v for k, v in asdict(self).items() if k not in skip}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformerConfig:
        """Deserialize, ignoring derived fields and metadata (model_type, etc.)."""
        skip = {"head_dim", "k_dim", "v_dim", "model_type"}
        filtered = {k: v for k, v in data.items() if k not in skip}
        return cls(**filtered)
