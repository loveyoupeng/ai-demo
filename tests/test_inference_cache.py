"""Ensure NumPy AutoregressiveGenerator uses KV cache with correct behaviour."""
from __future__ import annotations

import pytest

from tokenizer.char_tokenizer import CharTokenizer
from model.transformer import Transformer
from inference import AutoregressiveGenerator


@pytest.fixture()
def model():
    vocab_size = 50
    model = Transformer(vocab_size=vocab_size, embed_dim=32, num_layers=1,
                        num_heads=2, num_experts=2, max_seq_len=32)
    return model


@pytest.fixture()
def tokenizer():
    text = "the quick brown fox jumps over the lazy dog. " * 20
    return CharTokenizer(text)


def test_ar_generator_wires_kv_cache(model):
    """AutoregressiveGenerator should call model.forward with use_cache=True."""
    tokenizer = CharTokenizer("the quick brown fox " * 20)
    gen = AutoregressiveGenerator(model, tokenizer, temperature=0.0)

    # Patch forward to capture kwargs
    orig_forward = model.forward
    used_cache = []

    def patched_forward(*args, **kwargs):
        used_cache.append(kwargs.get("use_cache", False))
        return orig_forward(*args, **kwargs)
    model.forward = patched_forward

    gen.generate("the", num_new_tokens=3)

    # We expect 3 calls (one per generated token), all with use_cache=True
    assert len(used_cache) == 3
    assert all(used_cache)


def test_ar_generator_use_cache_false(model):
    """AutoregressiveGenerator.generate() should accept a use_cache=False option."""
    tokenizer = CharTokenizer("the quick brown fox " * 20)
    gen = AutoregressiveGenerator(model, tokenizer, temperature=0.0)

    orig_forward = model.forward
    used_cache = []

    def patched_forward(*args, **kwargs):
        used_cache.append(kwargs.get("use_cache", False))
        return orig_forward(*args, **kwargs)
    model.forward = patched_forward

    gen.generate("the", num_new_tokens=3, use_cache=False)

    # When use_cache=False is passed, all calls should have use_cache=False
    assert all(not c for c in used_cache)
