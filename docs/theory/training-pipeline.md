# Training Pipeline: Pre-train → SFT → Tool-Calling

The full pipeline *in reverse order of difficulty*: tool-calling is the furthest
thing from "just predict text," so we motivate it last. At every step we ask
"what is the cold hard math-side difference between this inner-loop model and
the last iteration?"—and the answer is always the same thing.

---

## Stage 1 — Pre-Training (what it is)

**Goal:** teach the model the *shape* of the data (syntax, repeated
argument/variable pairing, the "code-ness" of Python), using tokens only as
indices. The data is **raw text**: (prompt is a token sequence, response is
the same sequence shifted left by one, loss calculated on *every* position).

```python
for x in batch:
    logits = model(x)            # (B, S, V)
    loss   = CE(logits, shift(x)) # next-token loss, every position
```

**One-data-wrangling detail (real-world):** TinyStories, GSM8K-CoT, labels
"real pieces of text" (not math problems), so the model pre-trains on the
torch-text (`the cat', a.`), which teaches grammar but nothing someone would
*do* with it.

**Where it lives:** `scripts/train.py` (all backends; `--synth` adds a
synthetic generator so (a) and (b) sums can be measured identically).

## Stage 2 — SFT (Supervised Fine-Tuning)

**What it adds:** a *loss mask* that says "this side (prompt) is context;
that side (response) is what we're actually teaching." `CrossEntropyLoss`
still handles the ignore-index (=-100), so masked positions are on the loss
surface AND produce zero gradient. The math is identical: everything is next-
token CE — we just grade the response.

```
targets = [ignore, ignore, ..., resp_0, resp_1, ..., resp_T]
```

**Why this is a semantic shift (not just a parameter):** Pre-training
presents "continue the text." SFT presents "given this input, produce this
*specific* answer"—the model stops learning raw next-token, and starts
learning the *task shift*. The model learns to *follow* the instruction, not
just continue the prompt.

**Where it lives:** `scripts/sft.py` (per-track trainers:
`impl/_np/sft.py` = teaching baseline, torch/triton/cuda = production shape).

## Stage 3 — Tool Calling (SFT)

**What makes it different:** SFT trains a *response* that's chunked into JSON:
`{"thinking":"...", "tool_calls":[...]}` — the model learns that *at the
assistant's turn*, the right response to a specific input shape isn't text.
It emits a function call you can parse.

**The training trick:** The same prompt-masked CE trick as Stage 2 —
but now the `<|tool_call|>` token sits between the user question and the
JSON. The model's output at the assistant position is the *serialized*
tool call. This is behaviour SFT teaches (not the abstracted tool-call
loop) — the data pipeline owns *routing the rest*.

**Where it lives:** same data (the loader builds masked prompts), same
trainer class, and instructions stay the same as the file's README
(dialogue inference shows the model emit the tool_calls JSON it got).

## The Contract

The user-visible entry: `scripts/sft.py`. Stages: ``pre`` (pre-train),
``post`` (SFT-only), ``prepost`` (the real pipeline: pre-train then SFT,
with the SFT § masking being the only thing that changes the underlying
math).

## Common Pitfalls This Repository Avoids

Like other small SFT setups, we want to *teach* the difference between:
- Masking the prompt logits vs. masking the loss (both work, one is DUMB).
- not giving a "decision" examples (a message must have `tool_calls` when
the user message is a USER turn) — else the model learns `contents`
instead of the tool call.

What they do *not* model is RoPE + GQA and why they're required to make
the response logits cross the sequence boundary at pocket scale. See
`docs/specs/architecture-fixes.md`.
