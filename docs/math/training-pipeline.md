# Training Pipeline: Pre-train → SFT → Thinking → RL → Drafters

The full pipeline in pipeline order: pre-train (predict text) → SFT
(follow the prompt) → tool-calling (emit JSON) → thinking SFT (reason in
the open, then commit to an answer) → RL (optimize a reward on the
model's own sampled behavior) → drafter distillation (speed up
inference, losslessly). At every step we ask "what is the cold hard
math-side difference between this inner-loop model and the last
iteration?"—and the answer is always the same thing.

---

Shape letters used below (full definitions: [CONTEXT.md](../../CONTEXT.md) →
"Shape notation"): **B** = batch size (sequences in parallel), **S** =
sequence length (token positions in one pass), **D** = `embed_dim` (model
width), **V** = `vocab_size`.

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

**Where it lives:** `scripts/train.py` (all backends; `--synthetic` adds a
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

## Stage 4 — Thinking SFT (reasoning distillation, ADR 0004)

**What it adds:** a response format with TWO sections, marked by the
special tokens ``<|think|>`` and ``<|answer|>`` —
``question <|think|> step-by-step rationale <|answer|> final number``.
The tags separate *how I get there* from *what I commit to*, and they
are what makes the next stage's reward computable.

**The data is the R1-Distill recipe at teaching scale:** GSM8K gold
rationales (cleaned — ``<<3+4=7>>`` calculator annotations stripped —
curated to the shortest rows that fit the toy context; see
`scripts/download_gsm8k.py`) plus ~2,000 synthetic two-step arithmetic
problems with template rationales. The "teacher traces" are gold
rationales and templates — no real teacher model exists at this scale,
and the docs say so plainly. The stage logs print sample trace pairs
(the ``[distill trace]`` lines).

**Where it lives:** `scripts/train_demo_model_v3.py` stage 4 →
`learning_think` / `learning_moe_think`.

## Stage 5 — RL (GRPO, ADR 0004)

**What it adds:** optimization of a REWARD on the model's own sampled
behavior. SFT minimizes distance to fixed targets; RL lets the model
exceed every trace in its data (or fail in instructive ways). The loop:

```python
for question, gold in prompts:                  # one GROUP per prompt
    completions = [sample(policy, question) for _ in range(8)]   # rollouts
    rewards = [format(c)*0.2 + correct(c, gold)*1.0 for c in completions]
    A = (rewards - mean(rewards)) / (std(rewards) + 1e-4)        # no critic
    loss = -A * log π(y_t|·) + β·KL(π ‖ π_ref)                   # reweighted CE
    loss.backward(); optimizer.step()            # the SAME backward chain
```

Three formulas and the existing machinery: the group-relative advantage
is a critic-free baseline (the 8 completions of one prompt carry the
counterfactual), the k3 KL estimator keeps the policy near the frozen
thinking-SFT reference (β=0.04), and — the key implementation insight —
the policy gradient IS the cross-entropy gradient scaled by the
advantage, so every track reuses its SFT backward.

**Where it lives:** `impl/_np/rl.py` (the derivations — the group-baseline
theorem, k3, the reweighted-CE backward), `impl/_torch/rl.py`
(+ triton/cuda mirrors), `scripts/rl.py` (the unified 4-backend entry),
and the demo pipeline's stage 5 → `learning_rl` / `learning_moe_rl` —
THE checkpoints the learning page serves (its Model dropdown's
``*-rl`` entries; ``*`` vs ``*-rl`` isolates RL's delta because the RL
run started from exactly the shown SFT checkpoint).


## Stage 6 — Drafter Distillation (speculative decoding, ADR 0003)

(Pipeline order: the drafters are distilled LAST, from the RL target —
ADR 0003's per-target contract. Pedagogically the drafter math has
nothing to do with RL, so it is presented after both.)

**What makes it different:** every stage above trains the *target* on
tokens from a dataset. This stage trains a *drafter* on the **target's own
behavior** — a tiny model whose only job is to guess what the target will
pick next, cheaply.

**The key insight (why not train on raw corpus?):** a drafter trained on
the corpus learns the corpus's distribution; a drafter accepted by the
target needs the *target's* distribution. So the distillation runs the
FROZEN target greedily over corpus prompts (rollouts), records at every
position the target's final-norm hidden state and its greedy next token,
and trains the drafter so its proposal at offset j matches the target's
continuation at offset j+1:

```python
for anchor_token, anchor_hidden, targets in distill_batch(rollouts):
    _toks, probs = drafter.draft(anchor_token, anchor_hidden, k)
    loss = CE(probs, targets, ignore_index=-100)   # per draft position
```

**What the drafter looks like:** either one tiny transformer block run k
times (MTP — sequential, order from recurrence) or a non-causal parallel
backbone plus a causal refinement module (DSpark — the whole block in one
pass). Both share the target's embedding + lm_head (the biggest matrices,
and sharing them puts the drafter's proposals in the target's exact
vocabulary space) and save as sidecar checkpoints beside the target.

**The contract that makes it safe:** greedy verification only commits a
drafted token when it is exactly what plain greedy would have picked — so
a badly-distilled drafter only wastes the drafting compute, it can never
change the output (the lossless contract; acceptance below 50% prints a
warning at build time, never a failure).

**Where it lives:** `scripts/train_drafters.py` (rollouts on the NumPy
track — the oracle; drafter training on the torch track — autograd;
acceptance validated by reloading the saved sidecars into the NumPy
engine, which also proves the cross-track round trip every build).

## The Contract

The user-visible entries: `scripts/sft.py` (pre/SFT stages),
`scripts/rl.py` (the RL stage, all four backends). The full demo
pipeline — the six stages in the order above plus the drafter
distillation of ADR 0003 — lives in
`scripts/train_demo_model_v3.py` (supersedes v2, deleted):

- `scripts/train_demo_model_v3.py` — the dense family: BPE-512 (with
  the thinking specials) → pre-train → code SFT → tool SFT → thinking
  SFT → GRPO RL → drafter sidecars, producing `learning_base` /
  `learning_sft` / `learning_tool` / `learning_think` / `learning_rl`.
- `scripts/train_demo_model_v3.py --variant moe` — the MoE twin: the
  same tokenizer, corpus, stages, and seed with the FFN swapped for the
  PARAM-MATCHED MoE of ADR 0005 (E=8 routed + 1 always-on shared
  expert at dim 32, top-1: ~1.08× the dense model's total parameters,
  ~54% active per token — the total-vs-active trade the page renders),
  producing `learning_moe_*` … `learning_moe_rl`.

Every checkpoint's `config.json` carries `"pipeline": 3` — the
staleness marker `scripts/learning.py` checks (a missing/older marker
retrains).

## Common Pitfalls This Repository Avoids

Like other small SFT setups, we want to *teach* the difference between:
- Masking the prompt logits vs. masking the loss (both work, one is DUMB).
- not giving a "decision" examples (a message must have `tool_calls` when
the user message is a USER turn) — else the model learns `contents`
instead of the tool call.

What they do *not* model is RoPE + GQA and why they're required to make
the response logits cross the sequence boundary at pocket scale. See
`docs/architecture/design.md`.
