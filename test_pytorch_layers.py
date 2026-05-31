import torch
from src.model.pytorch.layers import TokenEmbedding, PositionalEmbedding, LayerNorm

def test_token_embedding():
    vocab_size = 100
    embed_dim = 32
    layer = TokenEmbedding(vocab_size, embed_dim)
    indices = torch.randint(0, vocab_size, (2, 10))
    out = layer.forward(indices)
    assert out.shape == (2, 10, 32)
    print("test_token_embedding passed")

def test_positional_embedding():
    max_len = 50
    embed_dim = 32
    layer = PositionalEmbedding(max_len, embed_dim)
    out = layer.forward()
    assert out.shape == (max_len, embed_dim)
    print("test_positional_embedding passed")

def test_layernorm():
    embed_dim = 32
    layer = LayerNorm(embed_dim)
    x = torch.randn(2, 10, embed_dim)
    out = layer.forward(x)
    assert out.shape == (2, 10, 32)
    print("test_layernorm passed")

if __name__ == "__main__":
    test_token_embedding()
    test_positional_embedding()
    test_layernorm()
