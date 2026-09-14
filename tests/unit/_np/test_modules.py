"""Test that the impl._np per-operator module package exists.

The former god module ``impl._np.modules`` was split into one module per
operator; each must be importable and expose its class.
"""

from __future__ import annotations


def test_import_operator_modules_succeeds() -> None:
    """Verify each per-operator module is importable and exposes its class."""
    from impl._np.attention import MultiHeadAttention
    from impl._np.block import TransformerBlock
    from impl._np.cross_entropy import CrossEntropyLoss
    from impl._np.embedding import Embedding
    from impl._np.ffn import SwiGLUFFN
    from impl._np.init import xavier_uniform
    from impl._np.layernorm import RMSNorm
    from impl._np.moe import MixtureOfExperts
    from impl._np.rope import RoPE
    from impl._np.stack import DecoderStack

    for cls in (
        Embedding,
        RMSNorm,
        RoPE,
        SwiGLUFFN,
        MultiHeadAttention,
        MixtureOfExperts,
        TransformerBlock,
        DecoderStack,
    ):
        assert callable(cls)
    assert callable(xavier_uniform)
    assert callable(CrossEntropyLoss)


def test_import_utils_succeeds() -> None:
    """Verify the impl._np.utils module is importable."""
    import impl._np.utils

    assert hasattr(impl._np.utils, "initialize_linear")
