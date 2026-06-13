"""KV cache invariant tests.

Correct KV cache behavior:
- Step N processes only the NEW token (last position)
- Previous tokens' K,V are stored in cache and reused
- Output at position N must match full sequence output at position N
"""
import numpy as np
import torch

from tokenizer.char_tokenizer import CharTokenizer
from model.transformer import Transformer
from model.pytorch.transformer import PyTorchTransformer
from model.pytorch.attention_kvcache import PyTorchTurboQuantCache


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


def _ar_generate_numpy(model, tokenizer, prompt, num_new_tokens, temperature=0.0):
    """Simulate KV-cache-aware AR generation with NumPy model.

    Matches PyTorch: pre-fill prompt through model to build KV cache,
    then generate tokens one at a time.
    """
    current_ids = tokenizer.encode(prompt).reshape(1, -1).astype(np.int32)
    prompt_len = current_ids.shape[1]
    generated = []

    # Pre-fill KV cache with prompt tokens first
    _, _ = model.forward(current_ids, use_cache=True, cache_idx=0)

    for step in range(num_new_tokens):
        new_token_ids = current_ids[:, -1:]  # [1, 1] — only new token
        logits, _ = model.forward(new_token_ids, use_cache=True, cache_idx=step + prompt_len)
        next_token = logits[:, -1, :].argmax(axis=-1)
        next_token = next_token.reshape(1, 1).astype(np.int32)
        current_ids = np.concatenate([current_ids, next_token], axis=1)
        generated.append(int(next_token[0, 0]))
    return generated


def _ar_generate_pytorch(model, tokenizer, prompt, num_new_tokens, temperature=0.0, use_kv_cache=True):
    """Simulate KV-cache-aware AR generation with PyTorch model.

    Pre-fills prompt through model to build KV cache, then generates
    tokens one at a time. Matches NumPy semantics.
    """
    current_ids = torch.tensor(tokenizer.encode(prompt).reshape(1, -1), dtype=torch.int64)
    prompt_len = current_ids.shape[1]
    generated = []

    embed_dim = model.embed_dim
    num_layers = model.num_layers
    num_heads = model.num_heads
    head_dim = embed_dim // num_heads

    kv_caches = None
    if use_kv_cache:
        kv_caches = [
            PyTorchTurboQuantCache(
                embed_dim=embed_dim, num_heads=num_heads, max_seq_len=64,
                head_dim=head_dim, batch_size=1,
            )
            for _ in range(num_layers)
        ]

    # Pre-fill KV cache with prompt tokens first
    if use_kv_cache:
        model(torch.tensor(current_ids, dtype=torch.int64), kv_caches=kv_caches)

    for step in range(num_new_tokens):
        x = current_ids[:, -1:]  # [1, 1]
        cur_mask = torch.tril(torch.ones((1, prompt_len + step), device=current_ids.device))
        if not use_kv_cache:
            x = current_ids
        logits, _ = model(x, mask=cur_mask, kv_caches=kv_caches if use_kv_cache else None)
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)  # [1, 1]
        current_ids = torch.cat([current_ids, next_token], dim=1)
        generated.append(int(next_token[0, 0].item()))

    return generated


def test_01_full_sequence_base_match():
    """Full sequence forward must produce same logits on both backends."""
    torch.manual_seed(42)
    np.random.seed(42)

    np_model = Transformer(vocab_size=50, embed_dim=32, num_layers=1,
                           num_heads=2, num_experts=2, max_seq_len=32)
    pt_model = PyTorchTransformer(vocab_size=50, embed_dim=32, num_layers=1,
                                  num_heads=2, num_experts=2, max_seq_len=32)
    _copy_params(np_model, pt_model)

    tok = CharTokenizer("the quick brown fox jumps over the lazy dog. " * 20)
    ids = tok.encode("abc").reshape(1, -1).astype(np.int32)

    np_out = np_model.forward(ids)[0]
    pt_out = pt_model(torch.tensor(ids, dtype=torch.int64))[0].detach().numpy()

    max_diff = np.max(np.abs(np_out - pt_out))
    print(f"[01] Full seq max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Full seq mismatch: {max_diff}"


def test_02_cache_step_matches_full_position():
    """After step N cache operations, single-token forward must match full seq at that position.

    Correct behavior:
    - Step 0: Input = [a,b,c] (3 tokens), cache_idx=3, output should match full seq position 2
    - Step 1: Input extended to [a,b,c,d] (4 tokens), cache_idx=4, NEW token processing
              output at position 3 should match full [a,b,c,d] position 3
    """
    torch.manual_seed(42)
    np.random.seed(42)

    np_model = Transformer(vocab_size=50, embed_dim=32, num_layers=1,
                           num_heads=2, num_experts=2, max_seq_len=32)
    pt_model = PyTorchTransformer(vocab_size=50, embed_dim=32, num_layers=1,
                                  num_heads=2, num_experts=2, max_seq_len=32)
    _copy_params(np_model, pt_model)

    tok = CharTokenizer("the quick brown fox jumps over the lazy dog. " * 20)
    base_ids = np.array([[0, 1, 2]])  # Fake IDs: a=0, b=1, c=2

    # Step 0: forward on 3 tokens with cache_idx=3
    np_cache0, _ = np_model.forward(base_ids, use_cache=True, cache_idx=3)
    np_full0 = np_model.forward(base_ids)[0]
    print(f"[02-0] NP cache0 shape: {np_cache0.shape}, pos2: {np_cache0[0, 2, :5]}")
    print(f"[02-0] NP full0 pos2: {np_full0[0, 2, :5]}")
    diff0 = np.max(np.abs(np_cache0[0, 2] - np_full0[0, 2]))
    print(f"[02-0] Diff: {diff0:.2e}")
    assert diff0 < 1e-5, f"Cache at step 0 doesn't match full: {diff0}"

    # Step 1: extend to 4 tokens, get next token via argmax
    np_next0 = int(np_cache0[0, 2].argmax())
    base_ids = np.concatenate([base_ids, np.array([[np_next0]])], axis=1)
    print(f"[02-1] base_ids now: {base_ids.flatten().tolist()}")

    # Full seq baseline for 4 tokens
    np_full1 = np_model.forward(base_ids)[0]
    print(f"[02-1] NP full1 pos3: {np_full1[0, 3, :5]}")

    # Cache step 1: should use cache from step 0 + new token
    np_cache1, _ = np_model.forward(base_ids, use_cache=True, cache_idx=4)
    print(f"[02-1] NP cache1 shape: {np_cache1.shape}")
    print(f"[02-1] NP cache1 pos3: {np_cache1[0, 3, :5]}")
    diff1 = np.max(np.abs(np_cache1[0, 3] - np_full1[0, 3]))
    print(f"[02-1] Diff at pos3: {diff1:.2e}")
    assert diff1 < 1e-5, f"Cache at step 1 doesn't match full: {diff1}"

    # Step 2: extend to 5 tokens
    np_next1 = int(np_cache1[0, 3].argmax())
    base_ids2 = np.concatenate([base_ids, np.array([[np_next1]])], axis=1)
    np_full2 = np_model.forward(base_ids2)[0]
    np_cache2, _ = np_model.forward(base_ids2, use_cache=True, cache_idx=5)
    print(f"[02-2] NP cache2 pos4: {np_cache2[0, 4, :5]}")
    print(f"[02-2] NP full2 pos4: {np_full2[0, 4, :5]}")
    diff2 = np.max(np.abs(np_cache2[0, 4] - np_full2[0, 4]))
    print(f"[02-2] Diff at pos4: {diff2:.2e}")
    assert diff2 < 1e-5, f"Cache at step 2 doesn't match full: {diff2}"


def test_03_cross_backend_ar_generate():
    """AR generation: NumPy cache mode must produce same tokens as PT cache mode."""
    tok = CharTokenizer("the quick brown fox jumps over the lazy dog. " * 20)
    prompt = "abc"
    num_new_tokens = 5

    torch.manual_seed(42)
    np.random.seed(42)

    np_model = Transformer(vocab_size=50, embed_dim=32, num_layers=1,
                           num_heads=2, num_experts=2, max_seq_len=32)
    pt_model = PyTorchTransformer(vocab_size=50, embed_dim=32, num_layers=1,
                                  num_heads=2, num_experts=2, max_seq_len=32)
    _copy_params(np_model, pt_model)

    np_tokens = _ar_generate_numpy(np_model, tok, prompt, num_new_tokens, temperature=0.0)
    pt_tokens = _ar_generate_pytorch(pt_model, tok, prompt, num_new_tokens, temperature=0.0, use_kv_cache=True)

    print(f"[03] NumPy tokens: {np_tokens}")
    print(f"[03] PT tokens:    {pt_tokens}")

    assert np_tokens == pt_tokens, f"Mismatch: NP={np_tokens} vs PT={pt_tokens}"


if __name__ == "__main__":
    test_01_full_sequence_base_match()
    test_02_cache_step_matches_full_position()
    test_03_cross_backend_ar_generate()
    print("ALL PASSED")
