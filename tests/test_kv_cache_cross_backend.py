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
    np_params = np_model.get_params()
    pt_params = pt_model.state_dict()

    # NumPy key (from np_model.get_params()) -> PT key (from pt_model.state_dict())
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
    if np_model.num_layers >= 2:
        for layer_idx in range(2, np_model.num_layers):
            layer_key = f"blocks.{layer_idx}"
            key_map[f"{layer_key}.ln1.gamma"] = f"{layer_key}.ln1.gamma"
            key_map[f"{layer_key}.ln1.beta"] = f"{layer_key}.ln1.beta"
            key_map[f"{layer_key}.ln2.gamma"] = f"{layer_key}.ln2.gamma"
            key_map[f"{layer_key}.ln2.beta"] = f"{layer_key}.ln2.beta"
            key_map[f"{layer_key}.mha.W_q"] = f"{layer_key}.mha.W_q"
            key_map[f"{layer_key}.mha.W_k"] = f"{layer_key}.mha.W_k"
            key_map[f"{layer_key}.mha.W_v"] = f"{layer_key}.mha.W_v"
            key_map[f"{layer_key}.mha.W_o"] = f"{layer_key}.mha.W_o"
            for expert_idx in range(np_model.blocks[0].moe.num_experts):
                key_map[f"{layer_key}.moe.router.weights"] = f"{layer_key}.moe.router.w"
                key_map[f"{layer_key}.moe.expert.{expert_idx}.W1"] = f"{layer_key}.moe.experts.{expert_idx}.w1"
                key_map[f"{layer_key}.moe.expert.{expert_idx}.b1"] = f"{layer_key}.moe.experts.{expert_idx}.b1"
                key_map[f"{layer_key}.moe.expert.{expert_idx}.W2"] = f"{layer_key}.moe.experts.{expert_idx}.w2"
                key_map[f"{layer_key}.moe.expert.{expert_idx}.b2"] = f"{layer_key}.moe.experts.{expert_idx}.b2"

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
    """Simulate KV-cache-aware AR generation with NumPy model.

    Pure step-by-step generation: each step embeds + processes only the new token.
    The KV cache is empty at start, matching the PyTorch implementation.
    """
    current_ids = tokenizer.encode(prompt).reshape(1, -1).astype(np.int32)
    prompt_len = current_ids.shape[1]
    generated = []

    # Pre-fill KV cache with prompt tokens first
    _, _ = model.forward(current_ids, use_cache=True, cache_idx=0)

    for step in range(num_new_tokens):
        new_token_ids = current_ids[:, -1:]  # [1, 1] — only new token
        logits, _ = model.forward(new_token_ids, use_cache=True, cache_idx=prompt_len + step)
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

    # Pre-fill KV cache with prompt tokens first
    if use_kv_cache:
        model(torch.tensor(current_ids, dtype=torch.int64), kv_caches=kv_caches)

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
