"""Debug KV cache cross-backend difference."""
import numpy as np
import torch
from tokenizer.char_tokenizer import CharTokenizer
from model.transformer import Transformer
from model.pytorch.transformer import PyTorchTransformer


def _copy_params(np_model: Transformer, pt_model: PyTorchTransformer) -> None:
    np_params = np_model.get_params()
    pt_params = pt_model.state_dict()
    key_map = {
        "token_embedding.weights": "token_embedding.weight",
        "lm_head": "lm_head.weight",
        "blocks.0.ln1.gamma": "blocks.0.ln1.gamma",
        "blocks.0.ln1.beta": "blocks.0.ln1.beta",
        "blocks.0.ln2.gamma": "blocks.0.ln2.gamma",
        "blocks.0.ln2.beta": "blocks.0.ln2.beta",
        "blocks.0.mha.W_q": "blocks.0.mha.W_q",
        "blocks.0.mha.W_k": "blocks.0.mha.W_k",
        "blocks.0.mha.W_v": "blocks.0.mha.W_v",
        "blocks.0.mha.W_o": "blocks.0.mha.W_o",
        "blocks.0.moe.router.weights": "blocks.0.moe.router.w",
        "blocks.0.moe.expert.0.W1": "blocks.0.moe.experts.0.w1",
        "blocks.0.moe.expert.0.b1": "blocks.0.moe.experts.0.b1",
        "blocks.0.moe.expert.0.W2": "blocks.0.moe.experts.0.w2",
        "blocks.0.moe.expert.0.b2": "blocks.0.moe.experts.0.b2",
        "blocks.0.moe.expert.1.W1": "blocks.0.moe.experts.1.w1",
        "blocks.0.moe.expert.1.b1": "blocks.0.moe.experts.1.b1",
        "blocks.0.moe.expert.1.W2": "blocks.0.moe.experts.1.w2",
        "blocks.0.moe.expert.1.b2": "blocks.0.moe.experts.1.b2",
    }
    for np_key, pt_key in key_map.items():
        if np_key == "lm_head":
            pt_params[pt_key] = torch.from_numpy(np_params[np_key].T)
        elif np_key in np_params and pt_key in pt_params:
            pt_params[pt_key] = torch.from_numpy(np_params[np_key])
    pt_model.load_state_dict(pt_params)


torch.manual_seed(42)
np.random.seed(42)

np_model = Transformer(vocab_size=50, embed_dim=32, num_layers=1,
                       num_heads=2, num_experts=2, max_seq_len=32)
pt_model = PyTorchTransformer(vocab_size=50, embed_dim=32, num_layers=1,
                              num_heads=2, num_experts=2, max_seq_len=32)
_copy_params(np_model, pt_model)

tok = CharTokenizer("the quick brown fox jumps over the lazy dog. " * 20)
prompt = "abc"
current_ids = tok.encode(prompt).reshape(1, -1).astype(np.int32)
prompt_len = current_ids.shape[1]

# Step 1: Process token 'd' (1st new token) with cache of 3 tokens
# Step 2: Process token 'e' (2nd new token) with cache of 4 tokens

# NumPy step 1
logits_np_step1, _ = np_model.forward(current_ids, use_cache=True, cache_idx=prompt_len)
print(f"NumPy step1: logits shape={logits_np_step1.shape}, logits[:, -1, :]=shape={logits_np_step1[:, -1, :].shape}")
# Just get the position 1 token (the new token) from the sequence
np_pos_narrow = logits_np_step1[:, 0:1, :]
print(f"NumPy step1 pos_narrow: {np_pos_narrow}")

# Let me get the correct position - the last token is at position 0 when processing one new token
# but we need position 0 because cache_idx=3 means position 3 in full sequence, which is the last position
# In a [1, 4] sequence, the 3rd new token is at index 3
# When processing with use_cache=True, current_ids still is [1, 4] but we only compute position 0 of output
print(f"NumPy step1 full: {logits_np_step1[0, -1, :]}")

# PyTorch step 1
current_ids_pt = torch.tensor(current_ids, dtype=torch.int64)
mask1 = torch.tril(torch.ones((1, prompt_len), device=current_ids_pt.device))
from model.pytorch.attention_kvcache import PyTorchTurboQuantCache
kv_caches = [
    PyTorchTurboQuantCache(
        embed_dim=32, num_heads=2, max_seq_len=64,
        head_dim=16, batch_size=1,
    )
    for _ in range(1)
]
logits_pt_step1, _ = pt_model(current_ids_pt, mask=mask1, kv_caches=kv_caches)
print(f"PyTorch step1: logits shape={logits_pt_step1.shape}")
print(f"PyTorch step1 last token: {logits_pt_step1[0, -1, :]}")

# Now step 2 - append token and process next
# Get next token from step 1
np_next = logits_np_step1[:, -1, :].argmax(axis=-1).reshape(1, 1).astype(np.int32)
current_ids = np.concatenate([current_ids, np_next], axis=1)
pt_next = logits_pt_step1[:, -1, :].argmax(dim=-1).unsqueeze(0)
current_ids_pt = torch.cat([current_ids_pt, pt_next], dim=1)

print(f"\n--- Step 2 ---")
print(f"Current IDs (np): {current_ids}")
print(f"Current IDs (pt): {current_ids_pt}")

# NumPy step 2
logits_np_step2, _ = np_model.forward(current_ids, use_cache=True, cache_idx=prompt_len + 1)
print(f"NumPy step2 full: {logits_np_step2[0, -1, :]}")

# PyTorch step 2
mask2 = torch.tril(torch.ones((1, prompt_len + 1), device=current_ids_pt.device))
kv_caches2 = [
    PyTorchTurboQuantCache(
        embed_dim=32, num_heads=2, max_seq_len=64,
        head_dim=16, batch_size=1,
    )
    for _ in range(1)
]
logits_pt_step2, _ = pt_model(current_ids_pt[:, -1:], mask=mask2, kv_caches=kv_caches2)
print(f"PyTorch step2: logits shape={logits_pt_step2.shape}")
print(f"PyTorch step2 last token: {logits_pt_step2[0, -1, :]}")

# Also test: NumPy processing the same single token with same cache_idx
print(f"\n--- Cross-check: NumPy single token at position 4 ---")
np_single = np.concatenate([current_ids, np_next], axis=1)
logits_np_single, _ = np_model.forward(current_ids[:, -1:], use_cache=True, cache_idx=prompt_len + 1)
print(f"NumPy single: {logits_np_single[0, -1, :]}")

# Also check: what does PT produce if we feed full sequence?
print(f"\n--- Cross-check: PT full sequence ---")
current_ids_pt_full = torch.tensor(current_ids, dtype=torch.int64)
mask_full = torch.tril(torch.ones((1, current_ids_pt_full.shape[1]), device=current_ids_pt_full.device))
logits_pt_full, _ = pt_model(current_ids_pt_full, mask=mask_full)
print(f"PT full pos 4: {logits_pt_full[0, 4, :]}")
