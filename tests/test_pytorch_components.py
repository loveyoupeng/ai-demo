from __future__ import annotations

import torch
from model.pytorch.attention import PyTorchMultiHeadAttention
from model.pytorch.layers import (
    PyTorchTokenEmbedding,
    PyTorchLayerNorm,
    PyTorchFeedForward,
    PyTorchPositionalEmbedding,
)


def test_token_embedding():
    vocab_size = 100
    embed_dim = 32
    batch_size = 4
    seq_len = 10
    model = PyTorchTokenEmbedding(vocab_size, embed_dim)
    indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(indices)
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_positional_embedding():
    max_seq_len = 50
    embed_dim = 32
    model = PyTorchPositionalEmbedding(max_seq_len, embed_dim)
    x = torch.zeros(1, 10, embed_dim)
    out = model(x)
    assert out.shape == (1, 10, embed_dim)


def test_layer_norm():
    embed_dim = 32
    batch_size = 4
    seq_len = 10
    model = PyTorchLayerNorm(embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert torch.allclose(
        output.mean(dim=-1), torch.zeros(batch_size, seq_len), atol=1e-5
    )
    assert torch.allclose(
        output.std(dim=-1, correction=0), torch.ones(batch_size, seq_len), atol=1e-2
    )


def test_feed_forward():
    embed_dim = 32
    intermediate_dim = 128
    batch_size = 4
    seq_len = 10
    model = PyTorchFeedForward(embed_dim, intermediate_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_multi_head_attention():
    embed_dim = 32
    num_heads = 4
    batch_size = 4
    seq_len = 10
    model = PyTorchMultiHeadAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    output, cache = model(x)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert "attn_weights" in cache
    mask = torch.tril(torch.ones(seq_len, seq_len))
    output_masked, _ = model(x, mask=mask)
    assert output_masked.shape == (batch_size, seq_len, embed_dim)


def test_mha_kv_cache():
    embed_dim = 32
    num_heads = 4
    batch_size = 2
    model = PyTorchMultiHeadAttention(embed_dim, num_heads)
    x1 = torch.randn(batch_size, 1, embed_dim)
    output1, _ = model(x1)
    assert output1.shape == (batch_size, 1, embed_dim)
    x2 = torch.randn(batch_size, 1, embed_dim)
    output2, _ = model(x2)
    assert output2.shape == (batch_size, 1, embed_dim)
