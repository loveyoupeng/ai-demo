import numpy as np
from model.numpy.moe import Expert

np.random.seed(42)
batch_size, seq_len, embed_dim, dim_ff = 2, 3, 4, 8

exp = Expert(embed_dim, dim_ff)
x = np.random.randn(batch_size, seq_len, embed_dim)
exp.forward(x)
grad_output = np.random.randn(batch_size, seq_len, embed_dim)
dx_analytical, grads = exp.backward(x, grad_output)

for key in ["w1", "b1", "w2", "b2"]:
    g = grads[key]
    print(f"{key}: shape={g.shape}, all_zero={np.allclose(g, 0)}, max_val={np.max(np.abs(g))}")

print("\nGrad w1:")
print(grads["w1"])
