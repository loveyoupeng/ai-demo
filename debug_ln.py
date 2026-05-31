import torch
import torch.nn as nn

embed_dim = 32
batch_size = 4
seq_len = 10
eps = 1e-6
ln = nn.LayerNorm(embed_dim, eps=eps)
x = torch.randn(batch_size, seq_len, embed_dim)
output = ln(x)
print(f"Mean: {output.mean(dim=-1)}")
print(f"Std: {output.std(dim=-1)}")
print(f"Mean close to 0: {torch.allclose(output.mean(dim=-1), torch.zeros(batch_size, seq_len), atol=1e-5)}")
print(f"Std close to 1: {torch.allclose(output.std(dim=-1), torch.ones(batch_size, seq_len), atol=1e-2)}")
