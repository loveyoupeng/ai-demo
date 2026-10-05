"""Tokenization entry point for the decoder-only transformer.

Provides the GPT-2 BPE tokenizer used by the real-data training pipeline
(``scripts/train_real_tinystories.py``). Tracks receive an
``AutoTokenizer`` and call its ``encode``/``decode`` directly — there are
no wrapper helpers here on purpose: the transformers API is already the
stable interface, and SFT/tool pipelines train their own BPE vocabularies
(see ``scripts/train_tokenizer.py``).
"""

from __future__ import annotations

from transformers import AutoTokenizer  # type: ignore[import-untyped]


def create_tokenizer(model_path: str = "gpt2") -> AutoTokenizer:
    """Create and configure a tokenizer.

    Uses the GPT-2 tokenizer (~50k vocab). The first call downloads the
    tokenizer files (~500KB) — this is expected and cached by
    transformers for subsequent runs.

    Args:
        model_path: HuggingFace model ID (default: "gpt2").

    Returns:
        An AutoTokenizer with the pad token set to the EOS token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
