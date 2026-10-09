# ADR 0003: Speculative decoding — MTP + DSpark drafters as sidecars

- Status: accepted
- Date: 2026-10-08

## Context

The repo teaches production inference idioms (KV-step path, TurboQuant,
sampling) with one real algorithm per feature. Speculative decoding is absent:
generation is strictly one token per target forward in all four tracks.

Two 2026 methods define the design space:

- **DSpark** (arXiv 2607.05147, DeepSeek): a *semi-autoregressive* drafter —
  a parallel backbone predicts the whole block, a lightweight sequential
  module models intra-block dependency; plus *confidence-scheduled
  verification* (the verified length adapts from estimated prefix-survival
  probability).
- **DFlash** (arXiv 2602.06036): a masked block-diffusion drafter
  conditioned on target context features.
- **MTP** (DeepSeek-V3 style) is the classic sequential baseline: a small
  head on the target's last hidden state drafting one token per step.

The core invariant — any track's checkpoint loads and runs on any other —
must survive; the parameter registry validates exact key sets, so draft
weights cannot casually join `model.npz`.

## Decision

1. **Drafter protocol, shared by all drafter families.** One interface
   (`draft(context_ids, hidden, k) -> (tokens, logprobs)` + per-drafter
   state) that MTP and DSpark implement today and DFlash can implement
   later. DFlash is deliberately **not built now**; it is a plug-in point.

2. **Two drafters now**: MTP (k=4, sequential, shares the target's
   embedding + lm_head — no duplicated V×D weights) and DSpark (k=8,
   semi-AR: parallel backbone + sequential module; confidence-scheduled
   verification with a 0.8 prefix-survival target, statistics tracked
   online).

3. **Sidecar checkpoints, not registry keys.** The target's `model.npz` and
   `ParameterRegistry` are untouched. Each drafter lives in its own
   directory (`<model>/draft_mtp/`, `<model>/draft_dspark/`) with its own
   mini key-scheme and loader. Sidecars load in every track, so the
   cross-track round-trip guarantee extends to drafters: a drafter distilled
   on one track accepts on all four.

4. **Chunked verification everywhere.** Every track gains
   `forward_chunk(ids, position, cache)` — k tokens' K/V appended in one
   pass, every drafted position scored in parallel. NumPy's is the readable
   reference; torch uses masked batched SDPA; triton/cuda run their kernels.
   No track verifies by re-running the full sequence (that is a naive-demo
   shortcut, not the real setup).

5. **Lossless contract.** Greedy verification (accept while target argmax
   agrees; first mismatch truncates, target token wins) in all four tracks —
   output must be token-identical to plain greedy decoding. The NumPy track
   additionally implements the rejection-sampling theorem
   (`min(1, p_t/p_d)` accept, resample `norm(max(0, p_t−p_d))`) for
   temperature-mode speculation: the theory lives in the reference track,
   the production tracks run what production runs.

6. **Distillation training** (`scripts/train_drafters.py`): freeze the
   target, roll it out over the mixed corpus, train drafters to predict the
   target's own greedy continuations (+ hidden-state conditioning for
   MTP/DSpark). `scripts/learning.py` bootstraps it on first run and loads
   the sidecars when present (falls back to plain with a warning).

7. **Defaults.** The trained demo ships with **MTP active** in the GUI and
   CLI (`--spec plain|mtp|dspark`, default `mtp`; missing sidecar → plain +
   warning). Plain is the opt-out comparison mode, not the baseline.

8. **Learning mode teaches the mechanism.** Records gain a per-round spec
   block (draft tokens, per-position accept/reject, accepted length,
   schedule state); the page shows accepted rate, accepted length, and a
   **paired same-prompt plain-decoding run** so tokens/sec is an honest
   same-machine comparison (plain must be slower). Block size k is a slider
   capped at the trained max — dragging it down shows suffix decay, DSpark's
   core motivation. DFlash is absent from the GUI until built.

## Alternatives considered

- **Draft weights inside the main registry**: one file, but every existing
  checkpoint's key set changes and the compare-tab models all implicitly
  carry drafters — rejected.
- **Verify by full re-forward per round**: zero new model code, but not the
  production shape and it hides the real parallel-verification lesson —
  rejected.
- **Greedy-only in NumPy too**: simpler, but drops the rejection-sampling
  theorem, the best teaching math in the feature — rejected for the
  reference track.
- **Build all three drafters now**: DFlash's diffusion training loop is the
  largest single chunk of new machinery for a third demonstration of the
  same verification pipeline; the protocol makes it additive later —
  deferred.

## Consequences

- `forward_chunk` is a new method on all four tracks — a new parity surface
  (chunked forward == step-by-step forward) covered by cross-backend tests
  and two new `verify_equivalence` scenarios (`spec_mtp`, `spec_dspark`):
  speculative greedy output must be token-identical to plain greedy.
- Drafter acceptance quality on the toy model is a teaching artifact; the
  build logs an acceptance-rate warning below 50% but never fails on it.
- The glossary (`CONTEXT.md`) gains the speculative-decoding vocabulary
  (drafter, target model, draft block, verification, accepted length, MTP,
  DSpark, DFlash).
