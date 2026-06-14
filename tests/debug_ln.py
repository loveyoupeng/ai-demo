"""Debug LayerNorm backward — trace intermediates to find gradient mismatch."""
import numpy as np
import torch
from src.model.layers import LayerNorm as NL
from src.model.pytorch.layers import PyTorchLayerNorm as PL

np.random.seed(42)
torch.manual_seed(42)

# Same config as transformer: embed_dim=64
B, L, D = 1, 128, 64

x_np = np.random.randn(B, L, D)
x_pt = torch.from_numpy(x_np)

# NumPy LayerNorm
ln_np = NL(D)
ln_np.gamma[:] = 1.0  # reset to ones

# PyTorch LayerNorm  
ln_pt = PL(D)
ln_pt.gamma.data[:] = 1.0  # reset to ones

# ---- Forward ----
out_np = ln_np.forward(x_np)
out_pt = ln_pt.forward(x_pt)

print("=== FORWARD ===")
print(f"Outputs match: {np.allclose(out_np, out_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"x_norm match: {np.allclose(ln_np.x_norm, ln_pt.x_norm.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"x_mean match: {np.allclose(ln_np.mean, ln_pt.x_mean.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"x_var match: {np.allclose(ln_np.var, ln_pt.x_var.detach().numpy(), rtol=1e-10, atol=1e-10)}")

# ---- Backward (single LayerNorm, no chaining) ----
grad_out = np.random.randn(B, L, D)
grad_out_pt = torch.from_numpy(grad_out)

dx_np, grads_np = ln_np.backward(grad_out)
dx_pt, grads_pt = ln_pt.backward(grad_out_pt)

print("\n=== BACKWARD PARAM GRADS ===")
gamma_np = grads_np["gamma"]
gamma_pt = grads_pt["weight"]
print(f"gamma match: {np.allclose(gamma_np, gamma_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
beta_np = grads_np["beta"]
beta_pt = grads_pt["bias"]
print(f"beta match: {np.allclose(beta_np, beta_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"gamma diff max: {np.max(np.abs(gamma_np - gamma_pt.numpy()))}")
print(f"gamma idx that differs: {np.where(np.abs(gamma_np - gamma_pt.numpy()) > 1e-6)[0][:5]}")

print("\n=== BACKWARD dx ===")
print(f"dx match: {np.allclose(dx_np, dx_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"dx diff max: {np.max(np.abs(dx_np - dx_pt.numpy()))}")

# ---- Now trace step by step to find where the split happens ----
print("\n=== STEP-BY-STEP TRACE ===")

# NumPy intermediates
grad_x_norm_np = grad_out * ln_np.gamma
mean_grad_x_norm_np = np.mean(grad_x_norm_np, axis=-1, keepdims=True)
mean_grad_x_norm_x_norm_np = np.mean(grad_x_norm_np * ln_np.x_norm, axis=-1, keepdims=True)
var_np = np.sqrt(ln_np.var + ln_np.eps)
dx_np_trace = (1.0 / var_np) * (grad_x_norm_np - mean_grad_x_norm_np - ln_np.x_norm * mean_grad_x_norm_x_norm_np)

# PyTorch intermediates
gamma_pt_data = ln_pt.gamma.data  # clone to be safe
grad_x_norm_pt = grad_out_pt * gamma_pt_data
mean_grad_x_norm_pt = torch.mean(grad_x_norm_pt, dim=-1, keepdim=True)
mean_grad_x_norm_x_norm_pt = torch.mean(grad_x_norm_pt * ln_pt.x_norm, dim=-1, keepdim=True)
var_pt = torch.sqrt(ln_pt.x_var + ln_pt.eps)
dx_pt_trace = (1.0 / var_pt) * (grad_x_norm_pt - mean_grad_x_norm_pt - ln_pt.x_norm * mean_grad_x_norm_x_norm_pt)

print(f"grad_x_norm match: {np.allclose(grad_x_norm_np, grad_x_norm_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"mean_grad_x_norm match: {np.allclose(mean_grad_x_norm_np, mean_grad_x_norm_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"mean_grad_x_norm_x_norm match: {np.allclose(mean_grad_x_norm_x_norm_np, mean_grad_x_norm_x_norm_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"sqrt(var+eps) match: {np.allclose(var_np, var_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")

# And the gamma gradient step by step
print("\n=== GAMMA GRAD TRACE ===")
# NumPy
gamma_grad_np_trace = np.sum(grad_out * ln_np.x_norm, axis=(0, 1))
# PyTorch
gamma_grad_pt_trace = torch.sum(grad_out_pt * ln_pt.x_norm, dim=tuple(range(x_pt.ndim - 1)))

print(f"grad_out * x_norm match: {np.allclose(grad_out * ln_np.x_norm, (grad_out_pt * ln_pt.x_norm).detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"gamma grad match: {np.allclose(gamma_grad_np_trace, gamma_grad_pt_trace.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"stored gamma grad match: {np.allclose(gamma_np, gamma_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
