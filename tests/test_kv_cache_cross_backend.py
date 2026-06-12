"""KV cache end-to-end: NumPy KV cache ↔ PyTorch KV cache cross-backend parity.

Both backends generate the same tokens in autoregressive mode when KV cache is enabled.
"""
from __future__ import annotations

import numpy as np
import torch

from tokenizer.char_tokenizer import CharTokenizer
from model.transformer import Transformer
from model.pytorch.transformer import PyTorchTransformer


def _copy_params(np_model: Transformer, pt_model: PyTorchTransformer) -> None:
    """Copy weights from NumPy model to PyTorch model using get_params/set_params."""
    key_map = {
        "token_embedding.embedding.weight": "token_embedding.weight",
        "lm_head": "lm_head.weight",
        "blocks.0.ln1.weight": "blocks.0.ln1.weight",
        "blocks.0.ln1.bias": "blocks.0.ln1.bias",
        "blocks.0.ln2.weight": "blocks.0.ln2.weight",
        "blocks.0.ln2.bias": "blocks.0.ln2.bias",
        "blocks.0.mha.W_q.weight": "blocks.0.mha.W_q.weight",
        "blocks.0.mha.W_k.weight": "blocks.0.mha.W_k.weight",
        "blocks.0.mha.W_v.weight": "blocks.0.mha.W_v.weight",
        "blocks.0.mha.W_o.weight": "blocks.0.mha.W_o.weight",
        "blocks.1.ln1.weight": "blocks.1.ln1.weight",
        "blocks.1.ln2.weight": "blocks.1.ln2.weight",
        "blocks.1.mha.W_q.weight": "blocks.0.mha.W_q.weight",
        "blocks.1.mha.W_k.weight": "blocks.0.mha.W_k.weight",
        "blocks.1.mha.W_v.weight": "blocks.0.mha.W_v.weight",
        "blocks.1.mha.W_o.weight": "blocks.0.mha.W_o.weight",
    }

    np_params = np_model.get_params()
    pt_params = pt_model.state_dict()
    for np_key, pt_key in key_map.items():
        if np_key == "lm_head":
            pt_params[pt_key] = torch.from_numpy(np_params[np_key].T)
        elif np_key in np_params and pt_key in pt_params:
            pt_params[pt_key] = torch.from_numpy(np_params[np_key])
    pt_model.load_state_dict(pt_params)


def _make_identical_model_params(vocab_size: int, embed_dim: int, num_layers: int,
                                  num_heads: int, num_experts: int, max_seq_len: int):
    """Create NumPy and PyTorch models with identical initial weights."""
    torch.manual_seed(42)
    np.random.seed(42)

    np_model = Transformer(vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers,
                           num_heads=num_heads, num_experts=num_experts, max_seq_len=max_seq_len)
    pt_model = PyTorchTransformer(vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers,
                                  num_heads=num_heads, num_experts=num_experts, max_seq_len=max_seq_len)
    _copy_params(np_model, pt_model)

    return np_model, pt_model


def _ar_generate_numpy(model, tokenizer, prompt, num_new_tokens, temperature=0.0):
    """Simulate KV-cache-aware AR generation with NumPy model."""
    current_ids = tokenizer.encode(prompt).reshape(1, -1).astype(np.int32)
    prompt_len = current_ids.shape[1]
    generated = []

    for step in range(num_new_tokens):
        logits, _ = model.forward(current_ids, use_cache=True, cache_idx=step + prompt_len)
        # For deterministic generation (temperature=0), argmax
        next_token = logits[:, -1, :].argmax(axis=-1)
        next_token = next_token.reshape(1, 1).astype(np.int32)
        current_ids = np.concatenate([current_ids, next_token], axis=1)
        generated.append(int(next_token[0, 0]))

    return generated


def _ar_generate_pytorch(model, tokenizer, prompt, num_new_tokens, temperature=0.0, use_kv_cache=True):
    """Simulate KV-cache-aware AR generation with PyTorch model."""
    current_ids = torch.tensor(tokenizer.encode(prompt).reshape(1, -1), dtype=torch.int64)
    prompt_len = current_ids.shape[1]
    generated = []

    # Build per-layer KV caches for PyTorch model
    embed_dim = model.embed_dim
    num_layers = model.num_layers
    num_heads = model.num_heads
    head_dim = embed_dim // num_heads

    kv_caches = None
    if use_kv_cache:
        from model.pytorch.attention_kvcache import PyTorchTurboQuantCache
        kv_caches = [
            PyTorchTurboQuantCache(
                embed_dim=embed_dim, num_heads=num_heads, max_seq_len=64,
                head_dim=head_dim, batch_size=1,
            )
            for _ in range(num_layers)
        ]

    for step in range(num_new_tokens):
        x = current_ids[:, -1:]  # [1, 1]
        cur_mask = torch.tril(torch.ones((1, prompt_len + step), device=current_ids.device))
        if not use_kv_cache:
            # Cacheless: full sequence
            x = current_ids

        logits, _ = model(x, mask=cur_mask, kv_caches=kv_caches if use_kv_cache else None)

        # For deterministic generation (temperature=0), argmax
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)  # [1, 1]
        current_ids = torch.cat([current_ids, next_token], dim=1)
        generated.append(int(next_token[0, 0].item()))

    return generated


def test_kv_cache_cross_backend_parity():
    """NumPy KV cache AR should produce IDENTICAL tokens as PyTorch KV cache AR."""
    vocab_size = 50
    embed_dim = 32
    num_layers = 1
    num_heads = 2
    num_experts = 2
    max_seq_len = 32
    prompt = "abc"
    num_new_tokens = 5

    np_model, pt_model = _make_identical_model_params(
        vocab_size, embed_dim, num_layers, num_heads, num_experts, max_seq_len
    )
    tok = CharTokenizer("the quick brown fox jumps over the lazy dog. " * 20)

    torch.manual_seed(42)
    np_model.train()

    # Generate with both backends
    np_tokens = _ar_generate_numpy(np_model, tok, prompt, num_new_tokens, temperature=0.0)
    pt_tokens = _ar_generate_pytorch(pt_model, tok, prompt, num_new_tokens, temperature=0.0, use_kv_cache=True)

    assert np_tokens == pt_tokens, (
        f"Cross-backend KV cache mismatch:\n"
        f"  NumPy tokens: {np_tokens}\n"
        f"  PyTorch tokens: {pt_tokens}\n"
        f"  Diff at position: {[i for i, (a, b) in enumerate(zip(np_tokens, pt_tokens)) if a != b]}"
    )


def test_kv_cache_cross_backend_parity_2_layers():
    """Extend to 2 layers."""
    vocab_size = 50
    embed_dim = 32
    num_layers = 2
    num_heads = 2
    num_experts = 2
    max_seq_len = 32
    prompt = "abc"
    num_new_tokens = 5

    np_model, pt_model = _make_identical_model_params(
        vocab_size, embed_dim, num_layers, num_heads, num_experts, max_seq_len
    )
    tok = CharTokenizer("the quick brown fox jumps over the lazy dog. " * 20)

    torch.manual_seed(42)

    np_tokens = _ar_generate_numpy(np_model, tok, prompt, num_new_tokens, temperature=0.0)
    pt_tokens = _ar_generate_pytorch(pt_model, tok, prompt, num_new_tokens, temperature=0.0, use_kv_cache=True)

    assert np_tokens == pt_tokens, (
        f"Cross-backend 2-layer KV cache mismatch:\n"
        f"  NumPy tokens: {np_tokens}\n  PyTorch tokens: {pt_tokens}"
    )
