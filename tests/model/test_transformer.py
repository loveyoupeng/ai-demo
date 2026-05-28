import pytest
import numpy as np
from model.transformer import TransformerBlock, Transformer


@pytest.mark.timeout(10)
def test_transformer_block_shape():
    """
    Test that a single Transformer Block preserves input shape.
    """
    batch_size = 2
    seq_len = 12
    embed_dim = 32
    num_heads = 4
    num_experts = 8

    from model.attention import MultiHeadAttention
    from model.moe import MoELayer

    mha = MultiHeadAttention(embed_dim, num_heads)
    moe = MoELayer(embed_dim, num_experts)

    block = TransformerBlock(embed_dim, mha, moe)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Causal mask
    mask = np.tril(np.ones((seq_len, seq_len)))

    output = block.forward(x, mask=mask)

    assert output.shape == (batch_size, seq_len, embed_dim)


@pytest.mark.timeout(20)
def test_transformer_model_shape():
    """
    Test that the full Transformer model produces correct logits shape.
    """
    batch_size = 2
    seq_len = 10
    vocab_size = 50
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    num_experts = 8

    # We need to mock/create components for the transformer
    # Since the transformer constructor builds them, we'll pass the config/params
    # or just let it build them if the constructor allows.

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=100,
    )

    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    logits = model.forward(input_ids)

    # Logits shape: [Batch, Seq_Len, Vocab_Size]
    assert logits.shape == (batch_size, seq_len, vocab_size)


@pytest.mark.timeout(20)
def test_transformer_causality():
    """
    Test that the transformer model is causal.
    The prediction at time t should not depend on input at time t+1.
    """
    batch_size = 1
    seq_len = 5
    vocab_size = 20
    embed_dim = 16
    num_layers = 1
    num_heads = 2
    num_experts = 4

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=10,
    )

    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Get logits for the first sequence
    logits1 = model.forward(input_ids)

    # Create a second sequence that is identical up to the last token, but differs at the last token
    input_ids_modified = input_ids.copy()
    input_ids_modified[0, -1] = (input_ids_modified[0, -1] + 1) % vocab_size

    logits2 = model.forward(input_ids_modified)

    # The logits for all tokens except the last one should be identical
    np.testing.assert_allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-7)
    # The last token logit should be different
    assert not np.allclose(logits1[:, -1, :], logits2[:, -1, :])
