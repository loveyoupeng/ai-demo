"""Parameter name constants for transformer modules.

All parameter names are defined as class-level string constants.
Helper functions that assemble compound paths use ONLY these constants —
no magic string literals are permitted.

Naming convention across backends
=================================
The keys below define the EXACT save/load format shared by NumPy, PyTorch,
and Triton backends.  Each backend's save_as_numpy() produces these keys,
and load_from_numpy_dict() consumes them.

NumPy keys:     impl/_np/model.py → get_all_parameters()
PyTorch keys:   impl/_torch/layers.py → save_as_numpy() / load_from_numpy()
Triton keys:    impl/_triton/model.py → save_as_numpy() / load_from_numpy_dict()
"""

from __future__ import annotations


class Mha:
    """Constants for Multi-Head Attention parameter names."""

    WQ: str = "Wq"
    BQ: str = "bq"
    WK: str = "Wk"
    BK: str = "bk"
    WV: str = "Wv"
    BV: str = "bv"
    WO: str = "Wo"
    BO: str = "bo"


class Attention:
    """Constants for attention layer parameter names."""

    Q_WEIGHT: str = "q.weight"
    K_WEIGHT: str = "k.weight"
    V_WEIGHT: str = "v.weight"
    O_WEIGHT: str = "o.weight"
    Q_BIAS: str = "q.bias"
    K_BIAS: str = "k.bias"
    V_BIAS: str = "v.bias"
    O_BIAS: str = "o.bias"

    # Save/load keys (cross-backend convention)
    WEIGHTS: str = "mha"


class LayerNorm:
    """Constants for LayerNorm parameter names."""

    LN_GAMMA: str = "ln_gamma"
    LN_BIAS: str = "ln_bias"

    # Save/load keys (cross-backend convention)
    LN1: str = "ln1_gamma"
    LN2: str = "ln2_gamma"


class MoE:
    """Constants for MoE (Mixture of Experts) parameter names."""

    W1: str = "w1"
    W2: str = "w2"
    W3: str = "w3"
    GATE_WEIGHT: str = "gate.weight"
    EXPERT_W1: str = "expert.w1"
    EXPERT_W2: str = "expert.w2"
    EXPERT_W3: str = "expert.w3"

    # Save/load keys (cross-backend convention)
    ROUTER: str = "router"
    BIAS: str = "bias"
    EXPERTS: str = "moe"
    GATE1: str = "gate1"
    GATE2: str = "gate2"


class Transformer:
    """Constants for Transformer-level parameter names."""

    EMBEDDING: str = "embed"
    LM_HEAD_WEIGHT: str = "lm_head"
    LM_HEAD_BIAS: str = "lm_head_bias"

    # Save/load keys (cross-backend convention)
    EMBEDDING_WEIGHTS: str = "embedding_weights"
    FINAL_GAMMA: str = "final_ln_gamma"  # also accept "final_gamma" for compat
    OUTPUT_W1: str = "output.W1"
    OUTPUT_W2: str = "output.W2"
    OUTPUT_W3: str = "output.W3"
    OUTPUT_PROJ_W: str = "output_proj_w"
    OUTPUT_PROJ_B: str = "output_proj_b"


class Block:
    """Constants for per-block (layer) save/load keys.

    Example usage:
        >>> Block.prefix(0)
        'blocks.0'
        >>> Block.ln1_gamma(0)
        'blocks.0.ln1_gamma'
    """

    PREFIX: str = "blocks"

    @staticmethod
    def prefix(layer_idx: int) -> str:
        """Generate the base path for a transformer block.

        Args:
            layer_idx: 0-based transformer block index.

        Returns:
            Base path like 'blocks.0'.
        """
        return f"{Block.PREFIX}.{layer_idx}"

    @staticmethod
    def ln1_gamma(layer_idx: int) -> str:
        """Layer norm 1 gamma save/load key.

        Args:
            layer_idx: 0-based transformer block index.

        Returns:
            Key like 'blocks.0.ln1_gamma'.
        """
        return f"{Block.prefix(layer_idx)}.ln1_gamma"

    @staticmethod
    def ln2_gamma(layer_idx: int) -> str:
        """Layer norm 2 gamma save/load key.

        Args:
            layer_idx: 0-based transformer block index.

        Returns:
            Key like 'blocks.0.ln2_gamma'.
        """
        return f"{Block.prefix(layer_idx)}.ln2_gamma"

    @staticmethod
    def mha(layer_idx: int, param_key: str) -> str:
        """MHA save/load key.

        Args:
            layer_idx: 0-based transformer block index.
            param_key: MHA parameter key (e.g., Mha.WQ).

        Returns:
            Key like 'blocks.0.mha.Wq'.
        """
        return f"{Block.prefix(layer_idx)}.mha.{param_key}"

    @staticmethod
    def moe_router(layer_idx: int) -> str:
        """MoE router save/load key.

        Args:
            layer_idx: 0-based transformer block index.

        Returns:
            Key like 'blocks.0.moe.router'.
        """
        return f"{Block.prefix(layer_idx)}.moe.{MoE.ROUTER}"

    @staticmethod
    def moe_bias(layer_idx: int) -> str:
        """MoE router bias save/load key.

        Args:
            layer_idx: 0-based transformer block index.

        Returns:
            Key like 'blocks.0.moe.bias'.
        """
        return f"{Block.prefix(layer_idx)}.moe.{MoE.BIAS}"

    @staticmethod
    def moe_expert(layer_idx: int, expert_idx: int, param: str) -> str:
        """MoE expert save/load key.

        Args:
            layer_idx: 0-based transformer block index.
            expert_idx: 0-based expert index.
            param: Parameter name (e.g., 'W1', 'W2', 'W3').

        Returns:
            Key like 'blocks.0.moe.experts.0.W1'.
        """
        return f"{Block.prefix(layer_idx)}.moe.experts.{expert_idx}.{param}"

    @staticmethod
    def gate1(layer_idx: int) -> str:
        """Gate 1 save/load key (PyTorch-specific).

        Args:
            layer_idx: 0-based transformer block index.

        Returns:
            Key like 'blocks.0.gate1'.
        """
        return f"{Block.prefix(layer_idx)}.gate1"

    @staticmethod
    def gate2(layer_idx: int) -> str:
        """Gate 2 save/load key (PyTorch-specific).

        Args:
            layer_idx: 0-based transformer block index.

        Returns:
            Key like 'blocks.0.gate2'.
        """
        return f"{Block.prefix(layer_idx)}.gate2"


def block_param(layer_idx: int, component: str) -> str:
    """Generate the base path for a transformer block component.

    Args:
        layer_idx: 0-based transformer block index.
        component: Component name (e.g., 'attn', 'mlp', 'ln1').

    Returns:
        Base path like 'blocks.0.attn'.
    """
    return f"blocks.{layer_idx}.{component}"


def attention_param(layer_idx: int, key: str) -> str:
    """Generate the full parameter name for an attention weight component.

    The key must be one of the Attention class attributes (e.g. Attention.Q_WEIGHT).

    Args:
        layer_idx: 0-based block index.
        key: Attention class attribute string (e.g. Attention.Q_WEIGHT).

    Returns:
        Full path like 'blocks.0.attn.q.weight'.
    """
    return f"{block_param(layer_idx, 'attn')}.{key}"


def layer_norm_param(layer_idx: int, part: str) -> str:
    """Generate the full parameter name for a LayerNorm component within a block.

    Args:
        layer_idx: 0-based transformer block index.
        part: Which layer norm component ("ln1", "ln2", "final").
              "ln2" uniquely returns .bias, all others return .gamma.

    Returns:
        Full path like 'blocks.0.ln1.gamma' or 'blocks.1.ln2.bias'.
    """
    suffix = ".gamma" if part != "ln2" else ".bias"
    return f"blocks.{layer_idx}.{part}{suffix}"


def moe_param(layer_idx: int, expert_idx: int, key: str) -> str:
    """Generate MoE parameter name.

    For expert weights, returns: blocks.{layer}.moe.expert_{expert}.{key_name}
    For gate/non-expert, returns: blocks.{layer}.moe.{key}

    Args:
        layer_idx: 0-based block index.
        expert_idx: 0-based expert index.
        key: MoE class attr (e.g. "expert.w1", "gate.weight", "w1").

    Returns:
        Full parameter path.
    """
    if key.startswith("expert."):
        return f"blocks.{layer_idx}.moe.expert_{expert_idx}.{key.split('.', 1)[1]}"
    return f"blocks.{layer_idx}.moe.{key}"


def transformer_param(key: str) -> str:
    """Generate a Transformer-level parameter name.

    Args:
        key: A Transformer class attribute that is a key in the mapping below.

    Returns:
        Full parameter name with appropriate suffixes.
    """
    mapping = {
        Transformer.EMBEDDING: "embed.weight",
        Transformer.LM_HEAD_WEIGHT: "lm_head.weight",
        Transformer.LM_HEAD_BIAS: "lm_head.bias",
    }
    return mapping[key]


def get_all_params(num_layers: int) -> dict[str, str]:
    """Generate all parameter names for a transformer with num_layers layers.

    Every key in the returned dictionary is built from the existing
    constants (Attention, LayerNorm, Transformer, MoE) — no magic strings.

    Args:
        num_layers: Number of transformer blocks (can be 0).

    Returns:
        Dict where all keys are parameter name strings and values are identical
        strings (used for parameter name lookup, not actual parameter values).

    Example:
        >>> params = get_all_params(0)
        >>> "embed.weight" in params
        True
        >>> len(params) == 3
        True
    """
    params: dict[str, str] = {}

    # Transformer-level parameters (always present)
    params[transformer_param(Transformer.EMBEDDING)] = transformer_param(
        Transformer.EMBEDDING
    )
    params[transformer_param(Transformer.LM_HEAD_WEIGHT)] = transformer_param(
        Transformer.LM_HEAD_WEIGHT
    )
    params[transformer_param(Transformer.LM_HEAD_BIAS)] = transformer_param(
        Transformer.LM_HEAD_BIAS
    )

    for layer_idx in range(num_layers):
        # Attention weights (use constants — no magic strings)
        params[attention_param(layer_idx, Attention.Q_WEIGHT)] = (
            attention_param(layer_idx, Attention.Q_WEIGHT)
        )
        params[attention_param(layer_idx, Attention.K_WEIGHT)] = (
            attention_param(layer_idx, Attention.K_WEIGHT)
        )
        params[attention_param(layer_idx, Attention.V_WEIGHT)] = (
            attention_param(layer_idx, Attention.V_WEIGHT)
        )
        params[attention_param(layer_idx, Attention.O_WEIGHT)] = (
            attention_param(layer_idx, Attention.O_WEIGHT)
        )
        # Attention biases
        params[attention_param(layer_idx, Attention.Q_BIAS)] = (
            attention_param(layer_idx, Attention.Q_BIAS)
        )
        params[attention_param(layer_idx, Attention.K_BIAS)] = (
            attention_param(layer_idx, Attention.K_BIAS)
        )
        params[attention_param(layer_idx, Attention.V_BIAS)] = (
            attention_param(layer_idx, Attention.V_BIAS)
        )
        params[attention_param(layer_idx, Attention.O_BIAS)] = (
            attention_param(layer_idx, Attention.O_BIAS)
        )
        # LayerNorm
        params[layer_norm_param(layer_idx, "ln1")] = layer_norm_param(
            layer_idx, "ln1"
        )
        params[layer_norm_param(layer_idx, "ln2")] = layer_norm_param(
            layer_idx, "ln2"
        )
        # MoE
        params[moe_param(layer_idx, 0, MoE.GATE_WEIGHT)] = moe_param(
            layer_idx, 0, MoE.GATE_WEIGHT
        )
        params[moe_param(layer_idx, 0, MoE.EXPERT_W1)] = moe_param(
            layer_idx, 0, MoE.EXPERT_W1
        )
        params[moe_param(layer_idx, 0, MoE.EXPERT_W2)] = moe_param(
            layer_idx, 0, MoE.EXPERT_W2
        )

    return params


# ──────────────────────────────────────────────────────────────────────
# Save/load key lookup functions (used by all three backends)
# ──────────────────────────────────────────────────────────────────────


def block_ln1_gamma(layer_idx: int) -> str:
    """Save/load key for block 1 layer norm gamma.

    Cross-backend convention — all three backends use this key.

    Args:
        layer_idx: 0-based transformer block index.

    Returns:
        Key like 'blocks.0.ln1_gamma'.
    """
    return Block.ln1_gamma(layer_idx)


def block_ln2_gamma(layer_idx: int) -> str:
    """Save/load key for block 2 layer norm gamma.

    Cross-backend convention — all three backends use this key.

    Args:
        layer_idx: 0-based transformer block index.

    Returns:
        Key like 'blocks.0.ln2_gamma'.
    """
    return Block.ln2_gamma(layer_idx)


def save_load_prefix() -> str:
    """Return the prefix used in all cross-backend save/load keys.

    Returns:
        Always 'blocks'.
    """
    return Block.PREFIX
