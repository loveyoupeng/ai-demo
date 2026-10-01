#!/usr/bin/env python3
"""Train a small BPE tokenizer on the project's datasets (one operation).

Pipeline::

    1. Read the datasets the project teaches from:
       - ``resource/tinystories/*.json``    (pre-training corpus)
       - ``resource/code_instructions.json`` (SFT coding)
       - ``resource/tool_calls.json``        (SFT tool calling)
    2. Concatenate every text field into one character stream.
    3. Train a byte-pair-encoding tokenizer on that stream
       (Hugging Face ``tokenizers`` lib — merges the most frequent byte
       pairs until ``vocab_size`` tokens).

The result is stored as ``resource/bpe_tokenizer.json`` (one file,
``Tokenizer.save`` format — the same format ``Tokenizer.from_file`` loads).

Usage::

    uv run python -m scripts.train_tokenizer            # 512-token vocab
    uv run python -m scripts.train_tokenizer --vocab 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resource"
DEFAULT_VOCAB = 512

# Special tokens the SFT trainer relies on; they have fixed IDs at the
# start of the vocab so the masked prefixes always begin with them.
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|prompt|>",  # marks where the *loss* starts in SFT
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|tool|>",
    "<|tool_call|>",
    "<|tool_response|>",
]


def _unicode_fold(text: str) -> str:
    """Kept for old-checkpoint compat; NOT used for training — the coding-agent
    model must see real casing, code indentation, and whitespace verbatim."""
    import re as _re

    return _re.sub(r"\s+", " ", text.lower().strip())


def _python_texts() -> list[str]:
    """Real Python functions + SFT instructions + tool calls — verbatim."""
    parts: list[str] = []
    corpus = RESOURCE_DIR / "corpus_python.json"
    if corpus.exists():
        parts.extend(json.loads(corpus.read_text()))
    code_file = RESOURCE_DIR / "code_instructions.json"
    if code_file.exists():
        for row in json.loads(code_file.read_text()):
            parts.append(f"INSTRUCTION: {row['instruction']}\nANSWER: {row['output']}")
    tool_file = RESOURCE_DIR / "tool_calls.json"
    if tool_file.exists():
        for row in json.loads(tool_file.read_text()):
            for msg in row.get("messages", []):
                if msg.get("content"):
                    parts.append(msg["content"])
            for tool in row.get("tools", []):
                parts.append(json.dumps(tool, ensure_ascii=False))
    return parts


def _tool_row_texts(row: dict) -> list[str]:
    """Flatten one tool_calls row to its text (content + tool JSON blobs)."""
    texts = []
    for msg in row["messages"]:
        if msg.get("content"):
            texts.append(msg["content"])
        for tc in msg.get("tool_calls") or []:
            texts.append(json.dumps(tc["function"], ensure_ascii=False))
    texts.extend(json.dumps(t["function"], ensure_ascii=False) for t in row["tools"])
    return texts


def collect_text() -> str:
    """All training text: tinystories corpus + SFT instructions + snippets."""
    parts = []

    ts_all = RESOURCE_DIR / "tinystories_train.json"
    if ts_all.exists():
        parts.extend(json.loads(ts_all.read_text()))

    code_file = RESOURCE_DIR / "code_instructions.json"
    if code_file.exists():
        for row in json.loads(code_file.read_text()):
            parts.append(" ".join([row["instruction"], row["output"]]))

    tool_file = RESOURCE_DIR / "tool_calls.json"
    if tool_file.exists():
        for row in json.loads(tool_file.read_text()):
            parts.extend(_tool_row_texts(row))

    return "\n".join(parts)


def train(tokenizer_path: Path | None = None, vocab_size: int = DEFAULT_VOCAB) -> None:
    """Train BPE on the collected text and save."""
    # The coding-agent tokenizer trains on REAL code: casing + indentation
    # + whitespace verbatim. Python corpus dominates; SFT texts join it.
    lines: list[str] = []
    for doc in _python_texts():
        lines.extend(doc.split("\n"))
    text = "\n".join(lines)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # ByteLevel decoder reverses the Ġ-style byte-level encoding on decode.
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tok.train_from_iterator(text.split("\n"), trainer=trainer)

    dest = tokenizer_path or (RESOURCE_DIR / "bpe_tokenizer.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(dest))
    print(f"  vocab size: {tok.get_vocab_size()}")
    print(f"  saved: {dest} ({dest.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="Where to save the tokenizer JSON")
    ap.add_argument("--vocab", type=int, default=DEFAULT_VOCAB, help="Target vocabulary size")
    args = ap.parse_args()
    train(args.out, args.vocab)


if __name__ == "__main__":
    main()
