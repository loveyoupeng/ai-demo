# ADR 0004: Thinking-SFT + GRPO RL stage — reasoning chain as a trained stage

- Status: accepted
- Date: 2026-10-25

## Context

The demo pipeline stops at SFT (pre-train → code → tool-call), and the
training docs note nothing RL-shaped exists in the repo. Production
post-training is a three-act play — pre-train → SFT → RL — where RL is
what turns "imitate the response" into "get the answer right". The missing
stage is the one that teaches *why* RL exists: SFT minimizes distance to a
fixed target; RL optimizes a reward on the model's OWN sampled behavior,
so the model can exceed every trace in its data (or fail in instructive
ways).

Two families dominate small-scale practice:

- **PPO** (Schulman et al.): policy gradient with a clipped importance
  ratio — but needs a trained **critic** network (value function). A second
  trained model is the largest single chunk of new machinery a teaching
  repo could add, and it buries the reward math under value-learning math.
- **GRPO** (Shao et al., DeepSeekMath; the DeepSeek-R1 recipe): sample a
  **group** of G completions per prompt, score each with the reward, and
  use the group's own mean/std as the baseline — no critic. The advantage
  is one readable formula, and the "watch 8 completions fight" loop is the
  best teaching artifact in the feature. R1 additionally popularized the
  two-component reward (format compliance + answer correctness) with
  explicit thinking sections.

A second gap: the pipeline's SFT stages teach instruction→code and
tool-call JSON, but never a **thinking section** — model-visible markup
that separates reasoning ("how I get there") from the final answer ("what
I commit to"). R1-style training needs that format as an explicit,
reinforceable behavior, and it has to be learned in SFT first (RL only
improves what the model already samples sometimes).

Constraints: the checkpoint seam (`ParameterRegistry`) must survive
untouched — RL/SFT change only *values*, never keys or shapes; every stage
must exist in all four tracks (the repo's invariant); the NumPy track
owns the math reference; vocab is BPE-512 and every existing checkpoint
family is retrained anyway (v3 pipeline), so new special tokens are safe
this time (and only this time — the decision below pins them).

## Decision

1. **Thinking tags as special tokens**: `<|think|>` and `<|answer|>` join
   the BPE trainer's special list. A thinking row is
   `question <|think|> step-by-step rationale <|answer|> final number`.
   The tool-call stage is untouched (`<|tool_call|>` stays its own
   behavior). Tokenizer version is pinned by the `"pipeline": 3` marker in
   `config.json` — a checkpoint without the marker (or with an older one)
   fails the staleness check and retrains.

2. **Thinking SFT (reasoning distillation framing)**: a fourth SFT stage
   trains on GSM8K gold rationales (cleaned: `<<…>>` annotations stripped,
   token-length-filtered to the toy context) plus synthetic two-step
   arithmetic with template rationales — data whose "teacher" is the gold
   rationale, the R1-Distill recipe at teaching scale. Training logs
   print sample trace pairs; the docs state plainly that no real teacher
   model exists at this scale.

3. **GRPO RL stage, all four tracks** (`impl/_np/rl.py` …
   `impl/_cuda/rl.py`, unified entry `scripts/rl.py`): sample G=8
   completions per prompt (temp 1.0, top-k 40, ≤80 new tokens),
   reward = 1.0·correctness + 0.2·format, advantage
   `A = (r − μ_group)/(σ_group + 1e-4)`, importance-ratio clip ε=0.2
   against the SFT policy that sampled the rollout, KL-to-reference
   β=0.04 against the frozen thinking-SFT model, AdamW lr 1e-4. The RL
   backward is the existing masked-CE analytic backward **reweighted by
   per-token advantage** — the policy-gradient theorem
   `∇θ J = E[A_t · ∇θ log π(y_t|·)]` is literally "CE gradient times
   scalar weight", so every track reuses its SFT backward path.

4. **NumPy owns the derivations** (`impl/_np/rl.py`): the group-baseline
   theorem (why subtracting μ makes the gradient unbiased without a
   critic), the k3 KL estimator with its low-variance derivation, and the
   reweighted-CE backward — each derived in the docstring, checked by
   tests against finite differences. The torch-family tracks run the
   same math through autograd; triton/cuda run their kernels.

5. **Checkpoint family**: RL loads the thinking-SFT checkpoint,
   reinforces it, and saves `learning_rl` (dense) / `learning_moe_rl`
   (moe) — registry-valid, loadable in all four tracks (round-trip
   guarantee unchanged). Drafter sidecars are re-distilled from the RL
   target (ADR 0003's per-target contract).

## Alternatives considered

- **PPO**: canonical, but the critic network doubles the machinery and
  hides the reward math — rejected for this scale.
- **DPO**: no rollouts, no reward — it trains on *pairs of pre-scored
  completions*, which removes exactly the loop this stage exists to
  teach — rejected.
- **REINFORCE with a learned baseline**: still a second model; the group
  mean IS a baseline, and it is free — rejected.
- **RL without thinking SFT** (R1-Zero, emergent format from format
  reward alone): at V=512/D=64 the model cannot discover `<|think|>`
  usage from a 0/1 format reward in any affordable number of steps —
  rejected; SFT seeds the format, RL sharpens the content.
- **Rationale-in-prompt variants, no tags**: unparseable rewards — the
  tags are what make the reward computable — rejected.

## Consequences

- The demo pipeline becomes pre-train → code SFT → tool SFT → thinking
  SFT → GRPO RL → drafter distillation (`train_demo_model_v3.py`); v2 is
  deleted and every reference migrated.
- The learning page's Model dropdown gains the RL checkpoints
  (`dense-rl`, `moe-rl`, default `moe-rl`), so RL-vs-SFT is a controlled
  comparison (RL starts from exactly the shown SFT model).
- New parity surface: cross-track RL gradient parity runs on **fixed**
  rollouts (no sampling in the parity path — same trick as the drafter
  acceptance tests), so RL math is drift-checked like every other stage.
- `CONTEXT.md` gains the RL vocabulary: policy, rollout, reward,
  group-relative advantage, reference model, KL penalty, thinking chain,
  reasoning distillation.
- Honest expectations: at toy scale, format reward rises sharply and
  correctness reward rises modestly; both are visible on the page, and
  the gap between them is itself the lesson (RL amplifies what the base
  model can almost do).
