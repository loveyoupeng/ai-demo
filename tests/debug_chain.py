"""Debug full forward/backward chain divergence."""
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from model.transformer import Transformer as NPT
from model.pytorch.transformer import PyTorchTransformer as PPT

B, S, V, D = 2, 8, 64, 64

model_np = NPT(V, D, 2, 4, 4)
model_pt = PPT(V, D, 2, 4, 4)
model_pt.double()

# Sync ALL params
np_params = model_np.get_params()
for name, p in np_params.items():
    if name.startswith("blocks."):
        parts = name.split(".", 3)
        i = int(parts[1])
        sublayer = parts[2]
        param_name = parts[3]
        block = model_pt.blocks[i]
        if sublayer == "ln1" and param_name == "gamma":
            block.ln1.gamma.data = torch.from_numpy(p)
        elif sublayer == "ln1" and param_name == "beta":
            block.ln1.beta.data = torch.from_numpy(p)
        elif sublayer == "ln2" and param_name == "gamma":
            block.ln2.gamma.data = torch.from_numpy(p)
        elif sublayer == "ln2" and param_name == "beta":
            block.ln2.beta.data = torch.from_numpy(p)
        elif sublayer == "mha":
            canonical = (
                f"qkv.{param_name}" if param_name in ("W_q", "W_k", "W_v") else f"o.{param_name}"
            )
            block.mha.set_params({canonical: p})
        elif sublayer == "moe":
            if param_name.startswith("router.weights"):
                block.moe.router.set_params({"w": p})
            elif param_name.startswith("router"):
                block.moe.router.set_params({param_name.replace("weights", "w"): p})
            elif param_name.startswith("expert."):
                block.moe.set_params({param_name.replace(".W", ".w"): p})
    elif name == "lm_head":
        model_pt.lm_head.weight.data = torch.from_numpy(p.T)
model_pt.token_embedding.weight.data = torch.from_numpy(np_params["token_embedding.weights"])

input_ids = np.random.randint(0, V, (B, S))
mask = np.tril(np.ones((S, S))).astype(np.float64)
grad_logits = np.random.randn(B, S, V).astype(np.float64)

_, ncache = model_np.forward(input_ids, mask)
_, pcache = model_pt.forward(torch.from_numpy(input_ids).long(), torch.from_numpy(mask))

# Trace forward step by step  
x_np = model_np.token_embedding.forward(input_ids)
x_pt = model_pt.token_embedding(torch.from_numpy(input_ids).long())

print("=== FORWARD ===")
print(f"token_embedding match: {np.allclose(x_np, x_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")

for i in range(2):
    res_id = i + 1
    # NumPy forward
    res1 = x_np if i == 0 else res_prev_np
    ln1x_np = model_np.blocks[i].ln1.forward(x_np)
    mha_out_np, mha_cache_np = model_np.blocks[i].mha.forward(ln1x_np, mask=mask)
    x_mha_np = res1 + mha_out_np
    
    # PyTorch forward
    res1_pt = x_pt if i == 0 else res_prev_pt
    ln1x_pt = model_pt.blocks[i].ln1.forward(x_pt)
    mha_out_pt, mha_cache_pt = model_pt.blocks[i].mha.forward(ln1x_pt, mask=torch.from_numpy(mask))
    x_mha_pt = res1_pt + mha_out_pt
    
    print(f"\nBlock {i}:")
    print(f"  x_in match: {np.allclose(x_np, x_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
    print(f"  ln1_x match: {np.allclose(ln1x_np, ln1x_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
    print(f"  mha_out match: {np.allclose(mha_out_np, mha_out_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
    print(f"  x_mha match: {np.allclose(x_mha_np, x_mha_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
    
    max_diff = max(
        np.max(np.abs(x_np - x_pt.detach().numpy())),
        np.max(np.abs(ln1x_np - ln1x_pt.detach().numpy())),
        np.max(np.abs(mha_out_np - mha_out_pt.detach().numpy())),
        np.max(np.abs(x_mha_np - x_mha_pt.detach().numpy())),
    )
    print(f"  max diff: {max_diff}")
    
    # MoE
    moe_out_np, _ = model_np.blocks[i].moe.forward(x_mha_np)
    moe_out_pt, _ = model_pt.blocks[i].moe.forward(x_mha_pt)
    
    print(f"  moe_out match: {np.allclose(moe_out_np, moe_out_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
    if not np.allclose(moe_out_np, moe_out_pt.detach().numpy(), rtol=1e-10, atol=1e-10):
        print(f"  moe_out max diff: {np.max(np.abs(moe_out_np - moe_out_pt.detach().numpy()))}")
    
    x_np = x_mha_np + moe_out_np
    x_pt = x_mha_pt + moe_out_pt
    res_prev_np = x_mha_np
    res_prev_pt = x_mha_pt

print(f"\nFinal logits match: {np.allclose(x_np @ model_np.lm_head, model_pt(x_pt).detach().numpy(), rtol=1e-10, atol=1e-10)}")

# ============================================================
# Now trace backward step by step
# ============================================================
print("\n\n=== BACKWARD ===")

# Step 1: Compute d_lm_head_input (same for both backends)
d_lm_head_input_np = np.dot(grad_logits, model_np.lm_head.T)
d_lm_head_input_pt = torch.from_numpy(d_lm_head_input_np).double()

print(f"d_lm_head_input match: {np.allclose(d_lm_head_input_np, d_lm_head_input_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")

# Block 1 backward
print("\n--- Block 1 backward ---")
dx1_np, b1g_np = model_np.blocks[1].backward(d_lm_head_input_np, ncache["blocks_cache"][1])
dx1_pt, b1g_pt = model_pt.blocks[1].backward(d_lm_head_input_pt, pcache["blocks_cache"][1])

print(f"dx1 match: {np.allclose(dx1_np, dx1_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"dx1 max diff: {np.max(np.abs(dx1_np - dx1_pt.detach().numpy()))}")

# Check individual grad differences in block 1
for key, pt_key in [("ln1.gamma", "ln1.weight"), ("ln2.gamma", "ln2.weight")]:
    g1 = b1g_np.get(key)
    g2 = b1g_pt.get(pt_key).detach().numpy() if g1 is not None else None
    if g1 is not None and g2 is not None:
        m = np.allclose(g1, g2, rtol=1e-2, atol=1e-2)
        diff = np.max(np.abs(g1 - g2)) if not m else 0
        print(f"  {key} match: {m}, max diff: {diff}")

# Block 0 backward  
print("\n--- Block 0 backward (using actual dx1 from block.1 backward) ---")
dx0_np, b0g_np = model_np.blocks[0].backward(dx1_np, ncache["blocks_cache"][0])
dx0_pt, b0g_pt = model_pt.blocks[0].backward(dx1_pt, pcache["blocks_cache"][0])

print(f"dx0 match: {np.allclose(dx0_np, dx0_pt.detach().numpy(), rtol=1e-10, atol=1e-10)}")
print(f"dx0 max diff: {np.max(np.abs(dx0_np - dx0_pt.detach().numpy()))}")

for key, pt_key in [("ln1.gamma", "ln1.weight"), ("ln1.beta", "ln1.bias")]:
    g1 = b0g_np.get(key)
    g2 = b0g_pt.get(pt_key).detach().numpy() if g1 is not None else None
    if g1 is not None and g2 is not None:
        m = np.allclose(g1, g2, rtol=1e-2, atol=1e-2)
        if not m:
            diff = np.max(np.abs(g1 - g2))
            idx = np.argmax(np.abs(g1 - g2))
            print(f"  {key} MISMATCH: max diff {diff:.6f} at idx {idx}")
            print(f"    np[{idx}]={g1[idx]}, pt[{idx}]={g2[idx]}")
        else:
            print(f"  {key} OK")

# ============================================================
# Now test: what if we use the same dx for both backends?
# ============================================================
print("\n\n=== CROSS-BACKWARD TEST ===")
# Use dx1_np to call PyTorch backward
dx0_cross_pt, b0g_cross_pt = model_pt.blocks[0].backward(torch.from_numpy(dx1_np).double(), pcache["blocks_cache"][0])
# Use dx1_pt to call NumPy backward
dx0_cross_np, b0g_cross_np = model_np.blocks[0].backward(dx1_pt.detach().numpy(), ncache["blocks_cache"][0])

print("PyTorch block.0 backward with NumPy dx1:")
for key, pt_key in [("gamma", "weight"), ("beta", "bias")]:
    g1 = b0g_cross_np.get(f"ln1.{key}") or b0g_cross_np.get(f"ln2.{key}")
    g2 = b0g_cross_pt.get(f"ln1.{pt_key}") or b0g_cross_pt.get(f"ln2.{pt_key}")
    if g1 is not None:
        m = np.allclose(g1, g2.detach().numpy(), rtol=1e-2, atol=1e-2)
        if not m:
            diff = np.max(np.abs(g1 - g2.detach().numpy()))
            print(f"  ln1.{key} MISMATCH: max diff {diff:.6f}")
        else:
            print(f"  ln1.{key} OK")
