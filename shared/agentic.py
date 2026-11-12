# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Multi-turn agentic RL environment (ADR 0010) — the tool sandbox + turn assembly.

The RL loop (``scripts/rl.py --multi-turn``) runs a policy that may emit a
``<|tool_call|>`` JSON block; this module executes the tool call in a tiny
deterministic sandbox and returns the ``<|tool_response|>`` text to append.

Safety: the calculator evaluates an arithmetic expression parsed with
``ast`` and a strict node whitelist — no names, no calls, no attribute
access, no eval of arbitrary code. Anything outside the whitelist returns
an error string (which is itself a valid tool response the policy can
learn from).

Turn assembly: one trajectory is a list of turns; each turn is
``(prompt_or_tool_text, completion_text)``. The reward (ADR 0004's
``shared.thinking.reward``) is computed on the FINAL completion's
``<|answer|>``; a small tool-use bonus (ADR 0010) rewards well-formed
tool calls that were actually consumed.
"""

from __future__ import annotations

import ast
import json
import operator
import re

# Special tokens (the tool-call format from the v3 pipeline's tool SFT stage).
TOOL_CALL_TAG = "<|tool_call|>"
TOOL_RESPONSE_TAG = "<|tool_response|>"

# ADR 0010 reward shaping: correctness dominates (shared.thinking), format is
# the wheel, and the tool bonus keeps the agentic behavior alive at demo scale.
REWARD_TOOL_USE = 0.1

# The calculator's whitelist: arithmetic over numeric literals only.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorTool:
    """The single tool of the demo environment: safe arithmetic evaluation."""

    name = "calculator"

    @staticmethod
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return CalculatorTool._eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](CalculatorTool._eval(node.left), CalculatorTool._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](CalculatorTool._eval(node.operand))
        raise ValueError(f"disallowed expression node: {type(node).__name__}")

    def run(self, args: dict) -> str:
        """Evaluate ``args['expr']``; return the result string (or an error)."""
        expr = str(args.get("expr", "")).strip()
        if not expr:
            return "error: empty expr"
        try:
            tree = ast.parse(expr, mode="eval")
            result = self._eval(tree)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
            return f"error: {e}"
        # Render ints without a trailing .0 for readability.
        if result.is_integer():
            return str(int(result))
        return repr(result)


_TOOL = CalculatorTool()
_TOOLS = {"calculator": _TOOL}

_TOOL_CALL_RE = re.compile(re.escape(TOOL_CALL_TAG) + r"\s*(\[.*?\]|\{.*?\})\s*$", re.DOTALL)


def parse_tool_call(text: str) -> list[dict] | None:
    """Extract the tool-call JSON array from a completion that ends with
    ``<|tool_call|>{...}``. Returns the parsed list of {name, arguments}
    dicts, or None when the completion does not end in a tool call.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    calls = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        args = item.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        calls.append({"name": name, "arguments": args})
    return calls or None


def execute_tool_calls(calls: list[dict]) -> str:
    """Run the parsed tool calls; return the ``<|tool_response|>`` body text.

    Each result is a JSON object {name, result}; unknown tools return an
    error string. The response is wrapped in the TOOL_RESPONSE_TAG by the
    caller (``format_tool_response``).
    """
    results = []
    for c in calls:
        tool = _TOOLS.get(c["name"])
        if tool is None:
            results.append({"name": c["name"], "error": f"unknown tool: {c['name']}"})
        else:
            results.append({"name": c["name"], "result": tool.run(c["arguments"])})
    return json.dumps(results, ensure_ascii=False)


def format_tool_response(calls: list[dict]) -> str:
    """The full ``<|tool_response|>...`` string to append after a tool call."""
    return TOOL_RESPONSE_TAG + execute_tool_calls(calls)


def split_turns(trajectory_text: str) -> list[str]:
    """Split a full trajectory into turns at each tool-call boundary.

    Turn 0 is the user prompt + the model's first completion up to (and
    including) the first ``<|tool_call|>``; each later turn starts with the
    ``<|tool_response|>`` and ends at the next ``<|tool_call|>`` or the end.
    """
    parts = re.split(f"({re.escape(TOOL_CALL_TAG)}|{re.escape(TOOL_RESPONSE_TAG)})", trajectory_text)
    turns: list[str] = []
    cur = ""
    for p in parts:
        cur += p
        if p in (TOOL_CALL_TAG, TOOL_RESPONSE_TAG):
            turns.append(cur)
            cur = ""
    if cur:
        turns.append(cur)
    return turns
