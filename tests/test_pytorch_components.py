import torch
import pytest
from model.pytorch.attention import MultiHeadAttention
from model.pytorch.layers import FeedForward, LayerNorm, TokenEmbedding, PositionalEmbedding

def test_token_embedding():
    vocab_size = 100
    embed_dim = 32
    batch_size = 4
    seq_len = 10
    model = TokenEmbedding(vocab_size, embed_dim)
    indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(indices)
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_positional_embedding():
    max_seq_len = 50
    embed_dim = 32
    model = PositionalEmbedding(max_seq_len, embed_dim)
    pe = model()
    assert pe.shape == (max_seq_len, embed_dim)

def test_layer_norm():
    embed_dim = 32
    batch_size = 4
    seq_len = 10
    model = LayerNorm(embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_len, embed_dim)
    # LayerNorm output has mean ~0 and std ~1 (population std)
    assert torch.allclose(output.mean(dim=-1), torch.zeros(batch_size, seq_len), atol=1e-5)
    assert torch.allclose(output.std(dim=-1, correction=0), torch.ones(batch_size, seq_len), atol=1e-2)

def test_feed_forward():
    embed_dim = 32
    intermediate_dim = 128
    batch_size = 4
    seq_len = 10
    model = FeedForward(embed_dim, intermediate_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_multi_head_attention():
    embed_dim = 32
    num_heads = 4
    batch_size = 4
    seq_len = 10
    model = MultiHeadAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test basic forward
    output, cache = model(x)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert "attn_weights" in cache
    
    # Test causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len))
    output_masked, _ = model(x, mask=mask)
    assert output_masked.shape == (batch_size, seq_len, embed_dim)

def test_mha_kv_cache():
    embed_dim = 32
    num_heads = 4
    batch_size = 2
    seq_len = 5
    model = MultiHeadAttention(embed_dim, num_heads)
    
    # First step
    x1 = torch.randn(batch_size, 1, embed_dim)
    output1, _ = model(x1, use_cache=True, cache_idx=0)
    assert output1.shape == (batch_size, 1, embed_dim)
    assert 0 in model.kv_cache
    
    # Second step
    x2 = torch.randn(batch_size, 1, embed_dim)
    output2, _ = model(x2, use_cache=True, cache_idx=1)
    assert output2.shape == (batch_size, 1, embed_dim)
    
    # Check cache shape: if we fixed MultiHeadAttention, K and V should have seq_len 2 now at index 1
    k, v = model.kv_cache[1]
    assert k.shape[2] == 2
    assert v.shape[2] == 2
