#!/usr/bin/env python3
"""Download small SFT datasets (code instruction + tool calling).

Mirrors ``scripts/download_tinystories.py`` — same streaming-download-to-JSON
pattern, two datasets for the two SFT behaviors the project teaches:

    uv run python -m scripts.download_sft_data

Datasets
--------
1. ``HuggingFaceH4/CodeAlpaca_20K`` — instruction/response pairs where the
   response is a code snippet. ~20k rows; we take the first 2,000 (≈3 MB).
   Teaches: *instruction following in the code domain* — the model learns
   "the response comes after the prompt, and it should be Python."

2. Tool calling — ``glaiveai/glaive-function-calling-v2`` (200 conversations,
   capped). Glaive stores rows as *text*: ``SYSTEM:`` (JSON schemas), then
   ``USER:``/``ASSISTANT:``/``FUNCTION RESPONSE:`` turns — we parse that shape
   into the canonical OpenAI-messages JSON, which is the real producer/consumer
   format at inference time (the same JSON vLLM streams back). The student sees
   exactly what a real model produces.

Usage::

    uv run python -m scripts.download_sft_data            # both
    uv run python -m scripts.download_sft_data --code     # code instruct only
    uv run python -m scripts.download_sft_data --tools    # tool calling only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    print("Error: 'datasets' package required. Install with:\n  pip install datasets", file=sys.stderr)
    sys.exit(1)

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resource"


# ── Code instructions (HuggingFaceH4/CodeAlpaca_20K) ──────────────────────────


def download_code_instructions(out_dir: Path, limit: int = 2_000) -> None:
    """HuggingFaceH4/CodeAlpaca_20K → list of (instruction, output).

    Alpaca formatter columns in this release are ``prompt``/``completion``;
    we normalize to the more familiar ``instruction``/``output`` pair so both
    SFT datasets speak the same vocabulary downstream.
    """
    rows = []
    for idx, row in enumerate(load_dataset("HuggingFaceH4/CodeAlpaca_20k", split="train", streaming=True)):
        text = row.get("prompt", "")
        out_ = row.get("completion", "")
        if text.strip() and out_.strip():
            rows.append({"instruction": text, "output": out_})
        if idx >= limit - 1:
            break

    dest = out_dir / "code_instructions.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=1))
    print(f"  code_instructions: {len(rows)} rows → {dest} ({dest.stat().st_size / 1024:.0f} KB)")


# ── Tool calling (glaiveai/glaive-function-calling-v2) ────────────────────────
#
# Glaive's rows are plain text: a SYSTEM: header carrying the tool JSON
# schemas, then USER:/ASSISTANT:/FUNCTION RESPONSE: turns separated by
# blank lines, with function calls wrapped as ``<functioncall> {…}`` and a
# trailing ``<|endoftext|>``. We parse that shape into the canonical
# OpenAI messages format, which is *the* production format for tool
# calling — so the JSON the student sees in the training file is the same
# JSON vLLM returns at inference time.


def _parse_function_call(payload: str) -> dict | None:
    """Parse glaive's ``<functioncall> {json}`` into a tool_call object.

    Glaive's arguments field is a JSON-string with *unescaped* inner quotes,
    so strict ``json.loads`` on the whole payload fails. We extract the
    name and the raw arguments body with a regex, then parse the arguments
    as its own JSON document.
    """
    m = re.match(r'^\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.+)\'\s*\}\s*$', payload.strip(), re.S)
    if not m:
        # zero-arg functions: {"name": "foo"} (no arguments key)
        m0 = re.match(r'^\{\s*"name"\s*:\s*"([^"]+)"\s*\}\s*$', payload.strip())
        if m0:
            return {"name": m0.group(1), "arguments": {}}
        return None
    name, args_raw = m.group(1), m.group(2)
    try:
        arguments = json.loads(args_raw)
    except json.JSONDecodeError:
        arguments = {"raw": args_raw}
    return {"name": name, "arguments": arguments}


def _convert_glaive_row(row: dict, max_turns: int = 8) -> dict | None:
    """One glaive row → OpenAI-messages dict, or None to skip.

    Returns
    -------
    {"messages": [{"role": ..., "content"/"tool_calls"}, ...], "tools": [...]}
    """
    system = row["system"].strip()
    # Tools live in the system prompt after "functions. Use them if required -"
    # as MULTIPLE CONCATENATED JSON objects (one per tool), not an array —
    # so we parse successive top-level values with raw_decode.
    tools_blob = system.split("following functions. Use them if required -", 1)[-1].strip()
    tools = []
    try:
        decoder = json.JSONDecoder()
        txt = tools_blob
        while txt:
            txt = txt.lstrip()
            if not txt:
                break
            obj, end = decoder.raw_decode(txt)
            tools.append(obj)
            txt = txt[end:]
    except json.JSONDecodeError:
        return None

    # Split the chat into labeled turns by role markers at line starts
    # (USER:/ASSISTANT:/FUNCTION RESPONSE:). NOTE: `endoftext` only closes
    # ASSISTANT *text* turns — function calls and user turns have no marker,
    # so splitting on `<|endoftext|>` alone merges turns into one chunk.
    turns = _parse_chat_turns(row["chat"])
    if turns is None or turns[0][0] != "user":
        return None

    # One tool_call → one assistant message, then the tool result.
    messages = []
    call_idx = 0
    for kind, text in turns:
        msg = _turn_to_openai_message(kind, text, call_idx)
        if msg is None:
            continue
        if kind == "tool_call":
            call_idx += len(msg["tool_calls"])
        messages.append(msg)

    # Keep conversations bounded (first max_turns turns).
    messages = messages[:max_turns]
    # OpenAI wraps each function schema once more: {"type": "function",
    # "function": {...}} — keep that surface so the file is drop-in usable.
    wrapped = [{"type": "function", "function": t} for t in tools]
    return {"messages": messages, "tools": wrapped}


def _parse_chat_turns(chat: str) -> list[tuple[str, str | dict]] | None:
    """Split GLaive's chat into (kind, text) turns, or None if malformed.

    Kind: "user" | "assistant" | "tool_call" | "tool_result". ``chat`` is
    parsed once to a flat list of top-level operations.
    """
    parts = re.split(r"(?m)^(USER:|ASSISTANT:|FUNCTION RESPONSE:) ", chat)
    out: list[tuple[str, str | dict]] = []
    for i in range(1, len(parts) - 1, 2):
        mark, body = parts[i], parts[i + 1].strip()
        if not body:
            continue
        body = body.split("<|endoftext|>")[0].strip()
        if not body:
            continue

        if mark == "USER:":
            out.append(("user", body))
        elif mark == "ASSISTANT:":
            if body.startswith("<functioncall>"):
                payload = body[len("<functioncall>") :].strip()
                call = _parse_function_call(payload)
                if call is None:
                    out.append(("assistant", payload))  # unparsed call: keep as text
                else:
                    out.append(("tool_call", call))
            else:
                out.append(("assistant", body))
        elif mark == "FUNCTION RESPONSE:":
            out.append(("tool_result", body))

    return out if out else None


def _turn_to_openai_message(kind: str, text: str | dict, call_idx: int) -> dict | None:
    """Map a (kind, text) turn to one OpenAI message (or None to skip).

    `text` is a plain string for user/assistant/tool_result, and a dict for
    tool_call (the parsed functioncall payload).
    """
    if kind == "user":
        return {"role": "user", "content": text}
    if kind == "tool_call":
        assert isinstance(text, dict)
        calls = [
            {
                "id": f"call_{call_idx}",
                "type": "function",
                "function": {"name": text["name"], "arguments": json.dumps(text["arguments"], ensure_ascii=False)},
            }
        ]
        return {"role": "assistant", "content": None, "tool_calls": calls}
    if kind == "tool_result":
        assert isinstance(text, str)
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = {"raw": text}
        return {"role": "tool", "content": json.dumps(body, ensure_ascii=False)}
    return {"role": "assistant", "content": text}


def download_tool_calling(out_dir: Path, limit: int = 200) -> None:
    """glaiveai/glaive-function-calling-v2 → OpenAI-messages JSON shape."""
    out_rows = []
    seen = 0
    items = iter(load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True))
    while len(out_rows) < limit and seen < limit * 10:
        seen += 1
        try:
            row = next(items)
        except StopIteration:
            break
        converted = _convert_glaive_row(row)
        if converted is not None:
            out_rows.append(converted)

    dest = out_dir / "tool_calls.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out_rows, indent=1))
    print(f"  tool_calls: {len(out_rows)}/{seen} rows parsed → {dest} ({dest.stat().st_size / 1024:.0f} KB)")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=RESOURCE_DIR, help="Output directory")
    ap.add_argument("--code", action="store_true", help="Download only code instructions")
    ap.add_argument("--tools", action="store_true", help="Download only tool calling")
    args = ap.parse_args()

    do_all = not (args.code or args.tools)
    if do_all or args.code:
        download_code_instructions(args.out_dir)
    if do_all or args.tools:
        download_tool_calling(args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
