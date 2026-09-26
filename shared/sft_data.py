"""SFT data loader: prompt/response pairs with response-only loss masking.

The single teaching difference between pre-training and SFT is **which
tokens the loss sees**: pre-training computes next-token loss over every
position; SFT masks the prompt so only the *response* tokens earn
gradient. This module is where that happens.

A batch row is a dict:

    {"input_ids":  [...],  # full sequence: prompt + response (padded to
                          #  max_len)
     "target_ids": [...],  # same length; prompt positions = ignore_index
                          #  (=-100), response positions = the real next-
                          #  token IDs — the loss of a masked position is
                          #  zero AND its gradient is zero.
     "response_mask": [...]}  # bool: True where the response runs

Usage
-----
    loader = SFTLoader(tokenizer)          # tokenizer: shared.tokenizer.Tokenizer
    batch = loader.load_sft_data(
        Path("resource/code_instructions.json"),
        max_len=256,
    )
    inputs, targets = loader.as_tensors(batch)    # (B, S) int arrays
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


@dataclass
class SFTBatch:
    """One batch of SFT training examples (prompt-masked)."""

    input_ids: list[list[int]]  # (B, S) — whole sequence, prompt + response
    target_ids: list[list[int]]  # (B, S) — response only, prompt = ignore_index
    response_mask: list[list[bool]]  # (B, S) — True at response positions


class SFTLoader:
    """Build (input_ids, target_ids) batches from SFT JSON files.

    The tokenizer is from ``tokenizers.Tokenizer`` (config.vocab_size must
    match ``tokenizer.get_vocab_size()`` after training).

    Parameters
    ----------
    tokenizer : tokenizers.Tokenizer
        A trained BPE tokenizer (``resource/bpe_tokenizer.json``).
    ignore_index : int
        The loss-skip marker used at prompt positions (=-100).
    template : str
        How to linearize (prompt, response) in the token stream:
        ``"<|prompt|>{prompt}<|assistant|>{response}"``. The loader masks
        everything that comes from ``{prompt}``, never the response.
    """

    TG_PROMPT = "<|prompt|>"  # id=1 — marks "everything before here is context"
    TG_ASSIST = "<|assistant|>"  # id=2 — the response starts here
    TG_USER = "<|user|>"  # id=3
    TG_SYSTEM = "<|system|>"  # id=4
    TG_TOOL = "<|tool|>"  # id=5
    TG_TOOL_CALL = "<|tool_call|>"  # id=6
    TG_TOOL_RESP = "<|tool_response|>"  # id=7

    def __init__(self, tokenizer: Tokenizer, max_len: int = 256, ignore_index: int = -100) -> None:
        self.tok = tokenizer
        self.max_len = max_len
        self.ignore_index = ignore_index
        # Special-token IDs — fixed by the trainer (SPECIAL_TOKENS order)
        self.prompt_id = self.tok.token_to_id(self.TG_PROMPT)
        self.assist_id = self.tok.token_to_id(self.TG_ASSIST)
        if self.prompt_id is None or self.assist_id is None:
            raise ValueError("tokenizer is missing <|prompt|> / <|assistant|> tokens")

    def _mask_pair(self, prompt: str, response: str) -> tuple[list[int], list[int], list[bool]]:
        """Encode one (prompt, response) pair into (input_ids, target_ids, response_mask).

        The *prompt* token run ends just before ``<|assistant|>``, and the
        *response* token run runs from ``<|assistant|>`` to the end of the
        response. Response tokens in ``target_ids`` are shifted by one
        (next-token supervision), so the model learns to produce the
        response given the prompt.
        """
        # The whole sequence is prompt + response.
        full_text = f"{self.TG_PROMPT}{prompt}{self.TG_ASSIST}{response}"

        enc = self.tok.encode(full_text)
        ids = enc.ids[: self.max_len]

        # Find where the response starts: the token AFTER <|assistant|>.
        n_prompt = len(ids)
        up_to = self.tok.encode(f"{self.TG_PROMPT}{prompt}").ids  # prompt only, no assist tag
        n_prompt = len(up_to)  # prompt token count (no assist tag)
        # add the <|assistant|> tag: it belongs to the prompt too
        if n_prompt < len(ids) and ids[n_prompt] == self.assist_id:
            n_prompt += 1
        resp_len = len(ids) - n_prompt

        # input_ids is the sequence; target_ids shifts left by one so
        # each position predicts its natural next-token; mask prompt.
        tgt = [self.ignore_index] * len(ids)
        for pos in range(n_prompt, len(ids) - 1):
            tgt[pos] = ids[pos + 1]  # each position learns the next response token
        # the last response position has nothing to learn from (pads ignored)
        pads = 0 if len(ids) >= self.max_len else (self.max_len - len(ids))
        if len(ids) < self.max_len:
            tgt.extend([self.ignore_index] * pads)
            ids = ids + [0] * pads

        # response mask: True where the response runs (ignores pads)
        resp_mask = [False] * n_prompt + [True] * resp_len + [True] * pads
        return ids[: self.max_len], tgt[: self.max_len], resp_mask[: self.max_len]

    def load_sft_data(self, path: Path) -> list[dict]:
        """Parse a JSON SFT dataset file into a ready-to-batch list of
        (instruction, output) / OpenAI-messages rows.

        - ``code_instructions.json``: each row has ``instruction`` + ``coutput``.
        - ``tool_calls.json``: each row is OpenAI-messages — assemble with
          the full role template, mask the user/system turns.
        """
        rows = json.loads(Path(path).read_text())
        out = []
        for row in rows:
            if "instruction" in row:  # code_instructions format
                out.append({"instruction": row["instruction"], "output": row["output"]})
            elif "messages" in row:  # tool_calls format
                out.append(row)
        return out

    def example_at(self, path: Path, index: int = 0) -> dict:
        """Convenience: load one parsed row for prints/tests."""
        rows = self.load_sft_data(path)
        return rows[index % len(rows)]

    def batch(self, rows: list[dict]) -> SFTBatch:
        """Assemble one batch of (input_ids, target_ids, mask) rows.

        Two formats:
          - Code instructions: prompt = instruction, response = output.
          - Tool calling: prompt = the user turn, response = the rest of
            the conversation (all downstream tokens are predicted).
        """
        inputs, targets, masks = [], [], []
        for row in rows:
            if "instruction" in row:
                inp, tgt, mask = self._mask_pair(row["instruction"], row["output"])
            else:
                # OpenAI messages: fold the whole conversation. Everything
                # before the first assistant turn is "prompt"; the rest is
                # the response the model learns to produce.
                prompt = self._messages_to_text(row["messages"], assistant_start=True)
                response = self._messages_to_text(row["messages"], assistant_start=False)
                inp, tgt, mask = self._mask_pair(prompt, response)
            inputs.append(inp)
            targets.append(tgt)
            masks.append(mask)
        return SFTBatch(inputs, targets, masks)

    def _messages_to_text(self, messages: list[dict], assistant_start: bool) -> str:
        """Linearize OpenAI messages into a single token sequence.

        The bot sub-object is a conversation; we fold it to text once, then
        mask: everything BEFORE the first assistant turn is ~prompt~,
        everything FROM that turn to the end is the ~response~ (the model
        learns to produce the response given the prompt).

        - ``assistant_start=True``  → prompt prefix (last byte is
          ``<|assistant|>``).
        - ``assistant_start=False`` → response suffix (starts at
          the assistant's content/tool calls; the assistant *tag itself*
          lives in the prompt prefix).
        """
        first_assistant_idx = next(
            (i for i, m in enumerate(messages) if m["role"] == "assistant"),
            len(messages),
        )
        if first_assistant_idx >= len(messages):
            return ""  # no assistant turn → nothing to train on

        # prefix = everything up to (and including) the <|assistant|> tag of
        # the first assistant message; suffix = everything after the tag.
        prefix_parts = []
        for i, msg in enumerate(messages[: first_assistant_idx + 1]):
            if i == first_assistant_idx:
                prefix_parts.append(self.TG_ASSIST)  # tag only, not the body
            else:
                prefix_parts.append(self._format_message(msg))
        prefix = "".join(prefix_parts)

        first_body = self._format_message(messages[first_assistant_idx])[len(self.TG_ASSIST) :]
        tail = "".join(self._format_message(m) for m in messages[first_assistant_idx + 1 :])
        return prefix if assistant_start else (first_body + tail)

    def _format_message(self, msg: dict) -> str:
        """Human-readable linearization of one OpenAI message."""
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            calls = json.dumps([c["function"] for c in msg["tool_calls"]], ensure_ascii=False)
            return f"{self.TG_ASSIST}{self.TG_TOOL_CALL}{calls}"
        if msg["role"] == "tool":
            return f"{self.TG_TOOL_RESP}{msg.get('content', '')}"
        tag = (
            self.TG_ASSIST
            if msg["role"] == "assistant"
            else (self.TG_USER if msg["role"] == "user" else self.TG_SYSTEM)
        )
        return f"{tag}{msg.get('content') or ''}"
