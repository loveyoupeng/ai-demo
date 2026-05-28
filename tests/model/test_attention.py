import pytest
import numpy as np
from src.model.attention import MultiHeadAttention


@pytest.mark.timeout(10)
def test_mha_output_shape():
    """
    Test that Multi-Head Attention produces the correct output shape.
    """
    batch_size = 2
    num_heads = 4
    seq_len = 8
    embed_dim = 32

    mha = MultiHeadAttention(embed_dim, num_heads)

    # Mock input: [Batch, Seq_Len, Embed_Dim]
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Create a causal mask: [Seq_Len, Seq_Len]
    mask = np.tril(np.ones((seq_len, seq_len)))

    output = mha.forward(x, mask=mask)

    # Expected: [Batch, Seq_Len, Embed_Dim]
    assert output.shape == (batch_size, seq_len, embed_dim)


@pytest.mark.timeout(10)
def test_mha_causal_mask():
    """
    Test that the causal mask actually prevents attending to future tokens.
    If we change a future token, the current token's output should not change.
    """
    batch_size = 1
    num_heads = 1
    seq_len = 4
    embed_dim = 8

    mha = MultiHeadAttention(embed_dim, num_heads)

    x1 = np.random.randn(batch_size, seq_len, embed_dim)
    x2 = x1.copy()
    # Change the last token in x2
    x2[0, -1, :] += 10.0

    mask = np.tril(np.ones((seq_len, seq_len)))

    out1 = mha.forward(x1, mask=mask)
    out2 = mha.forward(x2, mask=mask)

    # The output for the first token (and all except the last) should be identical
    # because they shouldn't be able to "see" the changed last token.
    np.testing.assert_allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-7)
    # The last token output SHOULD be different
    assert not np.allclose(out1[:, -1, :], out2[:, -1, :])
