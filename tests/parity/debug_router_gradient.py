"""Debug: Isolate router backward — compare NumPy vs PyTorch step-by-step."""

import sys
sys.path.insert(0, 'src')
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from model.moe import MoELayer
from model.pytorch.moe import PyTorchMoELayer, PyTorchRouter

np_moe = MoELayer(embed_dim=64, num_experts=4, dim_ff=128, num_experts_per_token=2)
pt_moe = PyTorchMoELayer(embed_dim=64, num_experts=4, dim_ff=128, num_experts_per_token=2)
pt_moe.double()

# Sync weights
with torch.no_grad():
    pt_moe.router.w.data.copy_(torch.from_numpy(np_moe.router.weights).double())
    for ei in range(len(np_moe.experts)):
        ne = np_moe.experts[ei]
        pe = pt_moe.experts[ei]
        for pname in ["W1", "W2", "b1", "b2"]:
            np_val = getattr(ne.ffn, pname)
            getattr(pe, pname.lower()).data.copy_(
                torch.from_numpy(np_val).double())

x_np = np.random.randn(2, 8, 64).astype(np.float64)
x_pt = torch.from_numpy(x_np).double()

# === FORWARDS ===
np_out, np_cache = np_moe.forward(x_np)
pt_out, pt_cache = pt_moe.forward(x_pt)

# Verify forward parity
fwd_diff = np.abs(np_out - pt_out.detach().numpy()).max()
print(f"=== Forward diff: {fwd_diff:.2e} ===")
assert fwd_diff < 1e-10, f"Forward mismatch: {fwd_diff:.2e}"

# === STEP 1: d_top_k_weights ===
print("\n=== Step 1: d_top_k_weights ===")
np_top_k_weights = np_cache["top_k_weights"]
pt_top_k_weights = pt_cache["top_k_weights"]
np_all_expert_outputs = np_cache["all_expert_outputs"]
pt_all_expert_outputs = pt_cache["all_expert_outputs"]
np_top_k_indices = np_cache["top_k_indices"]
pt_top_k_indices = pt_cache["top_k_indices"]

d_top_k_np = np.zeros_like(np_top_k_weights)
d_top_k_pt = torch.zeros_like(pt_top_k_weights)

batch_size, seq_len = 2, 8
for b in range(batch_size):
    for s in range(seq_len):
        for k_idx in range(2):
            expert_idx = np_top_k_indices[b, s, k_idx]
            weight = np_top_k_weights[b, s, k_idx]
            expert_out = np_all_expert_outputs[expert_idx, b, s, :]
            d_out = np_out[b, s, :]
            d_top_k_np[b, s, k_idx] = np.sum(d_out * expert_out)

            expert_idx_pt = int(pt_top_k_indices[b, s, k_idx].item())
            weight_pt = pt_top_k_weights[b, s, k_idx].item()
            expert_out_pt = pt_all_expert_outputs[expert_idx_pt, b, s, :]
            d_out_pt = pt_out[b, s, :]
            d_top_k_pt[b, s, k_idx] = torch.sum(d_out_pt * expert_out_pt)

np_diff_step1 = np.abs(d_top_k_np - d_top_k_pt.detach().numpy()).max()
print(f"  d_top_k_weights diff: {np_diff_step1:.2e}")
assert np_diff_step1 < 1e-10, f"Step 1 mismatch: {np_diff_step1:.2e}"

# === STEP 2: d_all_expert_outputs ===
d_all_expert_outputs_np = np.zeros_like(np_all_expert_outputs)
d_all_expert_outputs_pt = torch.zeros_like(pt_all_expert_outputs)

for b in range(batch_size):
    for s in range(seq_len):
        for k_idx in range(2):
            expert_idx = np_top_k_indices[b, s, k_idx]
            weight = np_top_k_weights[b, s, k_idx]
            d_all_expert_outputs_np[expert_idx, b, s, :] += np_out[b, s, :] * weight
            expert_idx_pt = int(pt_top_k_indices[b, s, k_idx].item())
            weight_pt = pt_top_k_weights[b, s, k_idx].item()
            d_all_expert_outputs_pt[expert_idx_pt, b, s, :] += pt_out[b, s, :] * weight_pt

np_diff_step2 = np.abs(d_all_expert_outputs_np - d_all_expert_outputs_pt.detach().numpy()).max()
print(f"  d_all_expert_outputs diff: {np_diff_step2:.2e}")
assert np_diff_step2 < 1e-10, f"Step 2 mismatch: {np_diff_step2:.2e}"

# === STEP 3: Normalization ===
print("\n=== Step 3: Normalization (d_w_normalized) ===")
S_np = np.sum(np_top_k_weights, axis=-1, keepdims=True) + 1e-8
term_to_subtract_np = np.sum(np_top_k_weights * d_top_k_np, axis=-1, keepdims=True)
d_w_normalized_np = (d_top_k_np - term_to_subtract_np) / S_np

top_k_sum_pt = pt_cache["top_k_sum"]
S_pt = top_k_sum_pt.detach().numpy()
term_to_subtract_pt = torch.sum(pt_top_k_weights * d_top_k_pt, dim=-1, keepdim=True)
d_w_normalized_pt = (d_top_k_pt - term_to_subtract_pt) / (top_k_sum_pt + 1e-8)

print(f"  NumPy S[0, 0, :3]: {S_np[0, 0, :3]}")
print(f"  PT   top_k_sum[0, 0, :3]: {S_pt[0, 0, :3]}")
np_s_diff = np.abs(S_np - S_pt).max()
print(f"  S diff: {np_s_diff:.2e}")
assert np_s_diff < 1e-10, f"S mismatch: {np_s_diff:.2e}"

np_diff_norm = np.abs(d_w_normalized_np - d_w_normalized_pt.detach().numpy()).max()
print(f"  d_w_normalized diff: {np_diff_norm:.2e}")

# === STEP 4: Placement of d_routing_weights ===
print("\n=== Step 4: d_routing_weights placement ===")
d_routing_weights_np = np.zeros((batch_size, seq_len, 4))
for b in range(batch_size):
    for s in range(seq_len):
        for k_idx in range(2):
            exp_idx = np_top_k_indices[b, s, k_idx]
            d_routing_weights_np[b, s, exp_idx] += d_w_normalized_np[b, s, k_idx]

d_routing_weights_pt = torch.zeros((batch_size, seq_len, 4), dtype=torch.float64)
for k_idx in range(2):
    d_routing_weights_pt[
        torch.arange(batch_size).unsqueeze(1).unsqueeze(2),
        torch.arange(seq_len).unsqueeze(0).unsqueeze(2),
        pt_top_k_indices[:, :, k_idx].unsqueeze(-1),
    ] = d_w_normalized_pt[:, :, k_idx].unsqueeze(-1)

np_placement_diff = np.abs(d_routing_weights_np - d_routing_weights_pt.detach().numpy()).max()
print(f"  d_routing_weights diff: {np_placement_diff:.2e}")
print(f"  NumPy [0, :, 0]:\n{d_routing_weights_np[0, :, 0]}")
print(f"  PT   [0, :, 0]:\n{d_routing_weights_pt[0, :, 0].detach().numpy()}")

# === STEP 5: Router backward ===
print("\n=== Step 5: Router backward ===")
dx_router_np, grads_router_np = np_moe.router.backward(x_np, d_routing_weights_np)
dx_router_pt, grads_router_pt = pt_moe.router.backward(x_pt, d_routing_weights_pt)

router_grad_diff = np.abs(grads_router_np["weights"] - grads_router_pt["w"].detach().numpy()).max()
print(f"  Router weights diff: {router_grad_diff:.2e}")
print(f"  NumPy router.grads['weights'][0, :]: {grads_router_np['weights'][0, :]}")
print(f"  PT   router.grads['w'][0, :]: {grads_router_pt['w'][0, :].detach().numpy()}")

# === STEP 6: Full backward comparison ===
print("\n=== Step 6: Full backward ===")
np_dx, np_grads = np_moe.backward(x_np, np_out, np_cache)
pt_dx, pt_grads = pt_moe.backward(x_pt, pt_out, pt_cache)

print(f"  NumPy grad keys: {list(np_grads.keys())}")
print(f"  PT   grad keys: {list(pt_grads.keys())}")

for np_key in np_grads:
    name_part = np_key.rsplit(".", 1)[-1]
    pt_key = "router." + name_part if "router" in np_key else np_key
    if pt_key in pt_grads:
        np_val = np_grads[np_key]
        pt_val = pt_grads[pt_key].detach().numpy() if hasattr(pt_grads[pt_key], 'detach') else pt_grads[pt_key]
        diff = np.abs(np_val - pt_val).max()
        status = "✓" if diff < 1e-10 else "✗"
        size = np_val.size if hasattr(np_val, 'size') else np.array(np_val).size
        print(f"  {status} {np_key:40s} <-> {pt_key:30s} diff={diff:.2e} size={size}")
