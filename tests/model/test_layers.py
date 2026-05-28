import pytest
import numpy as np
from src.model.layers import TokenEmbedding, PositionalEmbedding, FeedForward, LayerNorm

@pytest.mark.timeout(5)
def test_token_embedding_shape():
    batch_size = 2
    seq_len = 10
    vocab_size = 50
    embed_dim = 16
    
    embeddings = TokenEmbedding(vocab_size, embed_dim)
    indices = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    
    output = embeddings.forward(indices)
    
    # [Batch, Seq_Len, Embed_Dim]
    assert output.shape == (batch_size, seq_len, embed_dim)

@pytest.mark.timeout(5)
def test_positional_embedding_shape():
    seq_len = 10
    embed_dim = 16
    
    pos_embeddings = PositionalEmbedding(seq_len, embed_dim)
    # Positional embedding usually doesn't take batch/indices in same way, 
    # it returns the matrix for the sequence
    output = pos_embeddings.forward()
    
    # [Seq_Len, Embed_Dim]
    assert output.shape == (seq_len, embed_dim)

@pytest.mark.timeout(5)
def test_feed_forward_shape():
    batch_size = 2
    seq_len = 10
    embed_dim = 16
    dim_feedforward = 64
    
    ffn = FeedForward(embed_dim, dim_feedforward)
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    output = ffn.forward(x)
    
    # [Batch, Seq_Len, Embed_Dim]
    assert output.shape == (batch_size, seq_len, embed_dim)

@pytest.mark.timeout(5)
def test_layer_norm_shape_and_stats():
    batch_size = 2
    seq_len = 10
    embed_dim = 16
    
    ln = LayerNorm(embed_dim)
    x = np.random.randn(batch_size, seq_len, embed_dim) * 10 + 5 # non-zero mean/std
    
    output = ln.forward(x)
    
    # [Batch, Seq_Len, Embed_Dim]
    assert output.shape == (batch_size, seq_len, embed_dim)
    
    # Check if mean is close to 0 and std is close to 1 across the embed_dim
    # We normalize over the last dimension
    mean = np.mean(output, axis=-1)
    std = np.std(output, axis=-1)
    
    assert np.allclose(mean, 0, atol=1e-5)
    assert np.allclose(std, 1, atol=1e-5)
