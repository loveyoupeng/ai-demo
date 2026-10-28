#!/usr/bin/env python3
# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Download a small GSM8K slice for the thinking-SFT and RL stages.

Mirrors ``scripts/download_sft_data.py`` — same streaming-download-to-JSON
pattern. GSM8K (Cobbe et al., "Training Verifiers to Solve Math Word
Problems") is the classic reasoning dataset: every row carries a
**step-by-step rationale** plus a final numeric answer, which is exactly the
shape the thinking stage needs (the "teacher traces" the model imitates;
the R1-Distill recipe — SFT on a stronger trace-maker's chain of thought).

    uv run python -m scripts.download_gsm8k

Preprocessing (the standard cleanup, done once here so the training
pipeline stays dead simple):

1. Strip the ``<<2+3=5>>`` calculator annotations from the rationale —
   they are supervision artifacts for GSM8K's verifiers, not text a
   language model should learn to emit.
2. Split out the gold answer from the ``####`` marker.
3. Keep the rows that FIT THE TOY CONTEXT: every scanned row is
   tokenized with the project's BPE tokenizer and the SHORTEST-N win
   (a raw GSM8K row is ~300 tokens median, ctx is 128) — the on-disk
   slice is the short tail of the real distribution, the honest lesson
   that reasoning length is a budget.
4. If ``resource/bpe_tokenizer.json`` is missing (fresh clone), write the
   raw cleaned rows instead and let the pipeline re-filter after it trains
   the tokenizer.

Output: ``resource/gsm8k.json`` —
``[{"question", "reasoning", "answer"}, ...]``.

Also emits ``resource/gsm8k_test.json`` (100 held-out problems, same
filter) — the RL stage's held-out correctness probe.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    print("Error: 'datasets' package required. Install with:\n  pip install datasets", file=sys.stderr)
    sys.exit(1)

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resource"

# <<3+4=7>> calculator annotations (GSM8K verifier supervision, not text).
_CALC_RE = re.compile(r"<<[^>]*>>")


def _clean_row(row: Mapping[str, object]) -> dict[str, str] | None:
    question = (str(row.get("question", "") or "")).strip()
    full = (str(row.get("answer", "") or "")).strip()
    if not question or "####" not in full:
        return None
    rationale, _, answer = full.rpartition("####")
    reasoning = _CALC_RE.sub("", rationale).strip()
    answer = answer.strip().replace(",", "")
    if not reasoning or not answer:
        return None
    return {"question": question, "reasoning": reasoning, "answer": answer}


def _encode_row(row: dict[str, str], tokenizer) -> list[int]:
    """The thinking row's token IDs (measured for the keep-shortest-N curation)."""
    from shared.thinking import format_thinking_row

    return tokenizer.encode(format_thinking_row(row["question"], row["reasoning"], row["answer"])).ids


def _scan_split(split: str, cap: int, keep: int, tokenizer) -> list[tuple[int, dict[str, str]]]:
    """Stream one split; return the ``keep`` shortest rows as
    (token_length, row) pairs (longest first). ``tokenizer=None`` keeps
    the first ``keep`` rows unmeasured (fresh clone — no BPE file yet).
    The heap entry carries the arrival index as a tie-break (dicts do
    not order, and equal lengths are common)."""
    import heapq

    best: list[tuple[int, int, dict[str, str]]] = []
    for seen, row in enumerate(load_dataset("openai/gsm8k", "main", split=split, streaming=True)):
        cleaned = _clean_row(row)
        if cleaned is not None:
            n_tokens = 0 if tokenizer is None else len(_encode_row(cleaned, tokenizer))
            entry = (-n_tokens, seen, cleaned)
            if len(best) < keep:
                heapq.heappush(best, entry)
            elif n_tokens < -best[0][0]:
                heapq.heapreplace(best, entry)
        if seen >= cap - 1:
            break
    return [(-neg, cleaned) for neg, _i, cleaned in best]


def download_gsm8k(out_dir: Path, limit: int = 7_479, test_limit: int = 100) -> None:
    """openai/gsm8k → cleaned {question, reasoning, answer} rows.

    Curation is keep-the-SHORTEST-N, not a hard cutoff: a raw GSM8K row
    tokenizes to ~300 BPE tokens (median), which does not fit the toy
    ctx=128 — a hard "≤100 tokens" cutoff over the first rows would keep
    nothing. Instead every scanned row is measured and the N shortest
    KEPT rows win the budget, so the on-disk slice is the fittest the
    dataset offers (the honest lesson: at toy scale you train on the
    short tail of the real distribution).
    """
    KEEP = 500  # rows on disk (train)
    tokenizer = None
    tok_path = out_dir / "bpe_tokenizer.json"
    if tok_path.is_file():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tok_path))
    else:
        print("  bpe_tokenizer.json not found — writing unmeasured rows (re-run after the pipeline trains the BPE)")

    train_keep = _scan_split("train", limit, KEEP, tokenizer)
    test_keep = _scan_split("test", test_limit, test_limit, tokenizer)

    train_rows = [cleaned for _neg, cleaned in sorted(train_keep, key=lambda p: -p[0])]
    dest = out_dir / "gsm8k.json"
    dest.write_text(json.dumps(train_rows, indent=1))
    if train_rows and tokenizer is not None:
        lens = [n for n, _r in train_keep]
        print(
            f"  gsm8k train: {len(train_rows)} shortest rows kept "
            f"({min(lens)}–{max(lens)} tokens) → {dest} ({dest.stat().st_size / 1024:.0f} KB)"
        )
    else:
        print(f"  gsm8k train: {len(train_rows)} rows → {dest}")
    test_rows = [cleaned for _neg, cleaned in test_keep]
    dest_test = out_dir / "gsm8k_test.json"
    dest_test.write_text(json.dumps(test_rows, indent=1))
    print(f"  gsm8k test: {len(test_rows)} held-out rows → {dest_test}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=RESOURCE_DIR)
    ap.add_argument("--limit", type=int, default=7_479, help="Raw train rows to scan (default: the full train split)")
    ap.add_argument("--test-limit", type=int, default=100, help="Held-out test rows (default 100)")
    args = ap.parse_args()

    print("downloading openai/gsm8k (main config, streaming)...")
    download_gsm8k(args.out_dir, args.limit, args.test_limit)
    print("Done.")


if __name__ == "__main__":
    main()
