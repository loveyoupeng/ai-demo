import torch
import pytest
from model.pytorch.transformer import TransformerBlock, Transformer

def test_transformer_block_shape():
    embed_dim = 64
    num_heads = 8
    ffn_intermediate_dim = 256
    batch_size = 2
    seq_len = 16
    block = TransformerBlock(embed_dim, num_heads, ffn_intermediate_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = block(x)
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_transformer_shape():
    vocab_size = 100
    embed_dim = 64
    num_heads = 8
    num_layers = 2
    max_seq_len = 128
    ffn_intermediate_dim = 256
    batch_size = 2
    seq_len = 16
    model = Transformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, ffn_intermediate_dim)
    indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(indices)
    assert logits.shape == (batch_size, seq_len, vocab_size)

def test_transformer_causal_mask():
    vocab_size = 100
    embed_dim = 64
    num_heads = 8
    num_layers = 1
    max_seq_len = 128
    ffn_intermediate_dim = 256
    batch_size = 1
    seq_len = 8
    model = Transformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, ffn_intermediate_dim)
    indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Causal mask (triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    logits = model(indices, mask=mask)
    assert logits.shape == (batch_size, seq_len, vocab_size)

def test_transformer_gradients():
    vocab_size = 100
    embed_dim = 64
    num_heads = 8
    num_layers = 1
    max_seq_len = 128
    ffn_intermediate_dim = 256
    batch_size = 2
    seq_len = 8
    
    model = Transformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, ffn_intermediate_dim)
    indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(indices)
    loss = logits.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
