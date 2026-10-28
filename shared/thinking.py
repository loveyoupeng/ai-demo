# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""The thinking-section contract — shared by data prep, SFT, RL, and serving.

The thinking stage (ADR 0004) teaches the model to separate *reasoning*
(the visible step-by-step "how I get there") from *the committed answer*
(the parseable "what I conclude"), using two special tokens:

    <|think|>   opens the thinking section
    <|answer|>  opens the answer section

A full thinking row has the shape::

    question <|think|> step-by-step rationale <|answer|> final number

One module owns the format so every consumer agrees byte-for-byte:

- ``scripts/download_gsm8k.py`` — filters rows to fit the toy context.
- ``scripts/train_demo_model_v3.py`` — builds thinking-SFT batches.
- ``scripts/rl.py`` + ``impl/*/rl.py`` — the GRPO reward (the tags are
  what make the reward computable).
- the learning page — prompt templates for the think-tagged model.

The reward parser implements the ADR 0004 decision: the answer is the text
after the **last** ``<|answer|>`` tag (the model may draft-and-revise inside
``<|think|>``; the last tag is the commitment), ``$``/``,`/trailing ``.``
are stripped, and the value parses as a number.
"""

from __future__ import annotations

import re

# The special tokens (also passed to the BPE trainer as specials —
# scripts/train_demo_model_v3.py). Plain-Python constants so the data
# pipeline can use them before a tokenizer exists.
THINK_TAG = "<|think|>"
ANSWER_TAG = "<|answer|>"

# Reward weights (ADR 0004): correctness dominates, format is the training
# wheel that makes the section shape learnable before correctness moves.
REWARD_CORRECTNESS = 1.0
REWARD_FORMAT = 0.2


def format_thinking_row(question: str, reasoning: str, answer: str) -> str:
    """Assemble the canonical thinking row (question → think → answer)."""
    return f"{question} {THINK_TAG} {reasoning} {ANSWER_TAG} {answer}"


def _clean_number(raw: str) -> float | None:
    """Parse a numeric answer, tolerating $ 1,234. and stray whitespace."""
    text = raw.strip().strip("$").replace(",", "").rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_completion(text: str) -> dict[str, object]:
    """Parse one model completion into the GRPO reward components.

    The contract (ADR 0004):

    - the answer is the text after the **last** ``<|answer|>`` tag;
    - format holds when BOTH tags appear in order and the answer side
      contains a parseable number;
    - a missing ``<|answer|>`` (truncated mid-think) is format 0,
      correctness 0 — the honest truncation lesson.

    Returns a dict with keys ``format_ok`` (bool), ``answer`` (float or
    None), and ``answer_text`` (str, the raw committed segment).
    """
    think_pos = text.find(THINK_TAG)
    answer_matches = list(re.finditer(re.escape(ANSWER_TAG), text))
    if think_pos < 0 or not answer_matches:
        return {"format_ok": False, "answer": None, "answer_text": ""}
    # both tags present — enforce think-before-answer (find() gives the
    # FIRST think tag; the model may open the section only once)
    if answer_matches[0].start() < think_pos:
        return {"format_ok": False, "answer": None, "answer_text": ""}
    answer_text = text[answer_matches[-1].end() :]
    value = _clean_number(answer_text)
    return {
        "format_ok": value is not None,
        "answer": value,
        "answer_text": answer_text.strip(),
    }


def reward(text: str, gold: str) -> tuple[float, dict[str, object]]:
    """Two-component GRPO reward for one completion (ADR 0004).

    r = 1.0 · correctness + 0.2 · format, where correctness is an exact
    numeric match against ``gold`` (``|got − gold| < 1e-6`` — the numbers
    are integers in practice) and format means both tags present, in
    order, with a parseable final answer.

    Returns ``(r, details)`` so the trainer can log format rate and
    correctness rate separately — the two curves the teaching story
    lives on (format rises first, correctness follows).
    """
    parsed = parse_completion(text)
    gold_value = _clean_number(gold)
    correct = (
        parsed["answer"] is not None and gold_value is not None and abs(float(parsed["answer"]) - gold_value) < 1e-6  # type: ignore[arg-type]
    )
    r = (REWARD_CORRECTNESS if correct else 0.0) + (REWARD_FORMAT if bool(parsed["format_ok"]) else 0.0)
    details: dict[str, object] = dict(parsed)
    details["correct"] = correct
    return float(r), details
