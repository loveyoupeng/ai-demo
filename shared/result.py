"""Result[RetType, CtxType] — the generic result/capture wrapper.

**The problem it solves.** The NumPy track's forward methods return
``(real_result, captured_state)`` tuples — but a flat tuple can't tell a
reader (or a caller) which part is THE answer and which part is captured
side-channel data for the learning-mode record and the analytic backward.
The capture dict mixes two roles with different consumers:

- **backward context**: the intermediates the closed-form backward
  recomputes from (``x``'s projections, post-RoPE q/k, attention weights,
  the merged context, the scores scale, the RoPE angles) — the gradient
  chain's inputs;
- **display context**: the intermediates ONLY the learning-mode record
  shows (the raw score matrix history, the causal mask, the cache-ready
  K/V views) — never touched by gradients.

``Result`` separates the two: ``.value`` is THE result (the attention
output, the logits, the generated sequence); ``.ctx`` is the captured
context, sub-grouped into ``.ctx.backward`` and ``.ctx.display``; and a
``Result.nocapture(value)`` constructor marks a capture-free call (the
plain forward path) so callers can tell "no capture happened" from "the
capture is empty".

Contract (every Result-returning method holds):
- ``.value`` is bit-identical whether the context was captured or not —
  the capture is an overlay, never a second code path;
- ``.ctx`` is a plain immutable view (a dict of dicts); consumers read
  it, never mutate it;
- tuple unpacking still works (``out, state = result`` returns
  ``(value, ctx.backward | {} merged with display)`` — the pre-wrapper
  flat-dict shape) so existing callers/tests keep working during the
  migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

R = TypeVar("R")
C = TypeVar("C")


@dataclass(frozen=True)
class Context(Generic[C]):
    """The captured side-channel data, sub-grouped by consumer.

    backward: the intermediates the analytic backward recomputes from
        (projections, post-RoPE q/k, attention weights, merged context,
        scale, RoPE angles) — the gradient chain's inputs.
    display: the intermediates ONLY the learning-mode record shows (raw
        score history, causal mask, cache-ready K/V views) — never
        touched by gradients.
    record: per-operator extra keys that don't fit either group (e.g.
        the MoE per-expert outs; the drafter's per-round inputs/outputs)
        — a free-form dict, consumer-specific.
    """

    backward: dict
    display: dict
    record: dict = field(default_factory=dict)

    def merged(self) -> dict:
        """The pre-wrapper flat-dict view (backward ∪ display ∪ record) —
        the shape every pre-Result caller/test read. On key collision the
        backward group wins (the gradient chain's inputs are canonical)."""
        out = dict(self.display)
        out.update(self.record)
        out.update(self.backward)
        return out

    def __getitem__(self, key: str):
        """Flat-dict-style access (merged view) — the migration path."""
        return self.merged()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.backward or key in self.display or key in self.record

    def get(self, key: str, default=None):
        try:
            return self.merged()[key]
        except KeyError:
            return default


@dataclass(frozen=True)
class Result(Generic[R, C]):
    """THE result + the captured context, separated.

    value: THE result — what the method is FOR (the attention output, the
        logits, the generated sequence).
    ctx: the captured context (Context(backward, display, record)) — the
        side-channel data the backward and the learning-mode record
        consume. ``ctx`` of a ``Result.nocapture(value)`` is empty.
    """

    value: R
    ctx: Context[C]

    @classmethod
    def nocapture(cls, value: R) -> Result[R, C]:
        """A capture-free result (the plain forward path): value only;
        ctx is empty — callers can tell "no capture happened" from "the
        capture is empty"."""
        return cls(value, Context({}, {}, {}))

    @classmethod
    def capture(cls, value: R, backward: dict, display: dict, record: dict | None = None) -> Result[R, C]:
        """A captured result: value + the context sub-grouped by consumer."""
        return cls(value, Context(backward, display, record or {}))

    # ── the tuple-unpacking migration path ────────────────────────────────
    def __iter__(self):
        """``out, state = result`` returns (value, flat merged ctx) — the
        pre-wrapper shape, so existing callers/tests keep working."""
        return iter((self.value, self.ctx.merged()))

    def __getitem__(self, key: int | str):
        """Two access modes, both contract: an int is the tuple position
        (0 = value, 1 = the flat merged ctx — the migration path); a str
        is a capture-key lookup into the flat merged ctx (so
        ``result["logits"]`` works like the pre-wrapper flat dict did)."""
        if isinstance(key, str):
            return self.ctx.merged()[key]
        return (self.value, self.ctx.merged())[key]

    def __len__(self) -> int:
        return 2
