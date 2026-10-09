# Spec: Speculative Decoding — MTP + DSpark Drafters (DFlash-Ready Protocol)

**Status:** implemented (2026-10-09 — all slices: protocol+sidecars, all-track forward_chunk, MTP/DSpark drafters on both stacks, engines, DSpark schedule, distillation, CLI/server wiring, page UI, tests + verify_equivalence scenarios; see Further Notes → Implementation record)
**Date:** 2026-10-08 (design) / 2026-10-09 (implementation)
**Tracks affected:** NumPy (math reference), PyTorch (production reference), Triton (kernel reference), CUDA (bare-metal reference) + learning mode
**ADR:** `docs/adr/0003-speculative-decoding-drafters.md` (accepted)

## Problem Statement


The project teaches production inference idioms — KV-step decoding, sampling,
a quantized cache — but generation is strictly one target forward per token.
Nothing in the repo, the learning page, or the CLI demonstrates *speculative
decoding*: the technique modern serving systems use to confirm several
tokens per target forward pass (DeepSeek's DSpark and MTP, block-diffusion
drafters like DFlash). A learner visiting the page cannot see a drafter
propose a block, watch the target verify it in one parallel pass, or
measure the benefit (accepted length, tokens per target pass, tokens per
second). The trained demo pipeline produces no drafter, so the fastest
inference story the project can currently tell is "one token per step".

## Solution

Add real speculative decoding to all four tracks, at teaching scale but with
the real process flow:

- A shared **drafter protocol** (the one new seam) with two implementations:
  **MTP** (DeepSeek-V3 style sequential head, shares the target's embedding
  and lm_head) and **DSpark** (semi-autoregressive parallel drafter +
  confidence-scheduled verification). A third family, **DFlash**
  (masked block diffusion), is a documented plug-in point — interface
  reserved, deliberately not built.
- **Chunked verification** in all four tracks: `forward_chunk` on the
  KV-step interface appends k tokens' K/V in one pass and scores every
  drafted position in parallel.
- A **lossless contract**: speculative greedy decoding is token-identical to
  plain greedy decoding on every backend; the NumPy track additionally
  implements the rejection-sampling theorem for temperature mode.
- Drafters trained by **distillation** from the frozen target
  (`scripts/train_drafters.py`), stored as **sidecar checkpoints** (the
  target's registry/format untouched), round-tripping across all four
  backends.
- **Learning mode ships speculative by default** (MTP active; plain and
  DSpark switchable in the GUI/CLI), with extended inference records
  (draft tokens, per-position accept/reject, accepted length, DSpark
  schedule state, phase timings) and an honest speed comparison: measured
  tokens/sec of the speculative run vs a paired same-prompt plain run.

## User Stories

1. As a learner, I want the architecture diagram to show the drafter
   module next to the target model, so that I understand the two roles in
   speculative decoding (cheap proposer, quality owner).
2. As a learner, I want a decoding-mode selector (plain | MTP | DSpark) in
   the page, so that I can run the same prompt under different mechanisms
   and compare.
3. As a learner, I want the page to default to MTP when drafters are
   trained, so that the shipped demo shows the acceleration story out of
   the box.
4. As a learner, I want each round's draft tokens displayed above the
   output with per-position accept/reject marks, so that verification is
   visible token by token.
5. As a learner, I want accepted-length and acceptance-rate statistics per
   request, so that I can quantify how much the drafter helps.
6. As a learner, I want measured tokens/sec for the speculative run next
   to a paired plain-decoding run of the same prompt and seed, so that the
   speedup claim is an honest same-machine comparison.
7. As a learner, I want phase timings (draft ms, verify ms, total) per
   round, so that I see where generation time actually goes.
8. As a learner, I want a block-size slider capped at the trained maximum,
   so that dragging it down shows suffix decay — why longer blocks accept
   progressively worse.
9. As a learner, I want DSpark's confidence schedule state in the record
   (per-prefix survival estimates and the chosen verify length per round),
   so that I understand adaptive, load-aware verification.
10. As a learner, I want click-to-inspect math for the drafter forward and
    the verification pass, so that the formulas are traceable the same way
    as the rest of the page.
11. As a learner, I want sampled (temperature) speculative decoding in the
    NumPy track to implement the rejection-sampling rule, so that I learn
    the lossless theorem (accept with min(1, p_t/p_d); resample from
    norm(max(0, p_t−p_d)) at the first rejection).
12. As a CLI user, I want `--spec plain|mtp|dspark` on the inference and
    learning entry points (default `mtp`), so that all modes run headlessly.
13. As a CLI user, I want a clear warning and automatic fallback to plain
    when a requested drafter's sidecar is missing, so that commands never
    break on an untrained drafter.
14. As a maintainer, I want one dedicated drafter-training script
    (distillation from the frozen target), so that drafters are
    reproducibly rebuilt after any target change.
15. As a maintainer, I want the learning entry point to bootstrap drafter
    training on first run, so that the demo remains a one-command
    experience.
16. As a track developer, I want `forward_chunk(ids, position, cache)` on
    all four KV-step interfaces, so that verification is one parallel pass
    everywhere — no full re-forward shortcut.
17. As a track developer, I want chunked == stepwise parity pinned by
    tests on every track, so that the fast verification path is exact.
18. As a track developer, I want drafters stored as sidecar checkpoints
    with their own small key scheme, so that the target checkpoint and
    parameter registry stay untouched and all existing checkpoints remain
    valid.
19. As a track developer, I want the drafter protocol as the only new
    seam, so that MTP and DSpark (and a future DFlash) are interchangeable
    behind one interface.
20. As a track developer, I want MTP to share the target's embedding and
    lm_head, so that the drafter stays tiny and matches the DeepSeek-V3
    design (no duplicated V×D weights).
21. As a test engineer, I want `spec_mtp` and `spec_dspark`
    verify-equivalence scenarios, so that speculative greedy output is
    proven token-identical to plain greedy on every backend.
22. As a test engineer, I want cross-backend drafter round-trip tests, so
    that a drafter trained on one track accepts on all four.
23. As a test engineer, I want an acceptance-rate sanity check at build
    time that warns below 50% but never fails, so that drafter quality is
    visible without a flaky gate.
24. As a test engineer, I want the extended record schema pinned by
    schema-identity tests across all four learning adapters, so that the
    page consumes any backend's records interchangeably.
25. As a maintainer, I want the record JSON extension to be backward
    compatible (optional spec block, absent for plain), so that old
    records and old page code keep working.
26. As a teacher, I want the compare tab to stay speculative-free, so
    that the SFT first-divergence story stays clean.
27. As a maintainer, I want the domain glossary and ADR to define the
    speculative vocabulary, so that docs and code agree (done:
    CONTEXT.md inference section, ADR 0003).
28. As a maintainer, I want lint, typecheck, and the full test bar green
    after the change, so that the repo's quality bar holds.

## Implementation Decisions


- **Drafter protocol (new seam).** One interface implemented by every
  drafter family: draft a block of at most k candidate tokens from the
  verify-context (token ids plus the target's final hidden state at the
  anchor position) returning tokens and their log-probabilities, with
  per-drafter mutable state (DSpark's online survival statistics). MTP and
  DSpark implement it now; DFlash's slot is reserved and documented, absent
  from the GUI until built.
- **MTP drafter.** DeepSeek-V3 style: a single transformer block conditioned
  on the target's final hidden state (normalized) concatenated with the
  next token's embedding; drafts sequentially with its own KV cache;
  **shares the target's embedding matrix and lm_head**; trained block size
  k=4.
- **DSpark drafter.** Semi-autoregressive per the paper, at teaching scale:
  a parallel backbone predicts all block positions in one pass from the
  anchor hidden state plus learned block-position embeddings; a lightweight
  sequential (causal) module then refines the block for intra-block
  dependency; trained block size k=8. **Confidence-scheduled verification**:
  per-prefix-length acceptance statistics tracked online; each round's
  verify length is the longest prefix whose estimated survival probability
  meets a 0.8 target. Schedule state is per-request and part of the record.
- **Chunked verification.** Every track's KV-step interface gains
  `forward_chunk`: append the chunk's K/V in one pass, causal within the
  chunk, attend against the cache, return logits for all chunk positions.
  NumPy is the readable two-pass reference with shape comments; the PyTorch
  track uses one masked batched SDPA call; Triton and CUDA run their
  kernels with the same contract.
- **Lossless contract.** Greedy verification on all four tracks: accept
  drafted tokens while the target's argmax agrees; at the first mismatch
  the target's own token wins and the block tail is discarded; the bonus
  token after a fully accepted block comes from the target's last logits.
  Output must be token-identical to plain greedy. The NumPy track
  additionally implements the rejection-sampling theorem for sampled mode;
  the torch-family tracks run greedy verification only (what production
  runs at temperature 0).
- **Sidecar checkpoints.** Each drafter lives in its own directory beside
  the target checkpoint with its own config and a small dedicated key
  scheme (MTP block weights + projection; DSpark backbone + sequential
  module + position embeddings; shared embedding/lm_head are *referenced*
  from the target at load time, not copied). A drafter saved from any
  track loads on all four; the target's `model.npz` key set is untouched.
- **Distillation training.** A dedicated script freezes the trained target
  (the learning-mode pipeline's final checkpoint), rolls it out greedily
  over the mixed corpus, and trains both drafters to predict the target's
  own continuations, with the target's final hidden states as conditioning.
  Logs acceptance-rate estimates; warns below 50%, never fails. Standalone
  runnable; also invoked by the learning entry point's first-run
  bootstrap.
- **Surface wiring.** `--spec plain|mtp|dspark` (default `mtp`) on the
  learning and inference entry points; the inference API gains a `spec`
  field; the page gains a decoding-mode dropdown and a block-size slider
  capped at the trained k. Missing sidecar → warn + plain. DSpark's
  schedule state rides in the record.
- **Records.** The generation record gains an optional per-round spec
  block: drafter id, requested k, draft tokens with per-position
  accept/reject, accepted length, schedule state (DSpark), phase timings,
  measured tokens/sec — plus the paired plain-run timing for the same
  prompt/seed. All four adapters serialize the identical schema. Spec-off
  records are unchanged.
- **Where things live.** The NumPy spec engine (greedy + rejection
  sampling) is the math reference in the NumPy track; the torch family
  (torch/triton/cuda) shares one production engine over its generator seam;
  drafters for the torch family are torch modules (the CUDA track keeps its
  NVRTC kernels for the *target*; the tiny drafter is a torch module — the
  production-tech-stack choice at this scale).

## Testing Decisions

- Good tests assert external behavior at existing seams — outputs, parity,
  schema — never internals.
- **Seams (highest first; one new seam total):**
  1. *Generation seam (existing)* — the lossless oracle: speculative greedy
     output must equal plain greedy output, per drafter, per backend.
  2. *KV-step interface (existing)* — `forward_chunk` must equal repeated
     `forward_step`, and prefill + steps/chunks must equal one full forward
     (dense/GQA/MoE), on every track.
  3. *verify_equivalence (existing)* — two new scenarios, `spec_mtp` and
     `spec_dspark`, asserting greedy equivalence across backends using a
     distilled-drafter fixture.
  4. *Cross-backend suite (existing)* — drafter sidecar round-trip (train
     on one track, accept on all four) and chunk-parity additions.
  5. *Learning-record seam (existing)* — schema-identity: all four adapters
     emit the same spec-block fields; spec-off records unchanged.
  6. *Drafter protocol (new)* — exercised through seams 1–4 rather than
     mocked; plus a distillation smoke test asserting trained drafters
     accept above zero (sanity) with the <50% warning path covered.
- Prior art: the 49 cross-backend parity tests (tiered float64 tolerances),
  the TurboQuant parity-budget test, the 7 verify_equivalence scenarios, the
  learning adapters' existing record tests.
- GPU tests keep the repo's discipline: separate invocations for triton and
  cuda suites; CUDA tests one file per invocation (NVRTC isolation).

## Out of Scope

- **DFlash implementation** — protocol slot and docs only; no training, no
  GUI entry.
- Wall-clock/serving throughput optimization beyond honest measurement on
  the toy model (no batching, no multi-request scheduling engine).
- Response-quality goals beyond the existing demo bar ("English words");
  drafter acceptance on a D=64 target is a teaching artifact, not a
  production claim.
- Speculation in the compare tab (three models × modes would be noise).
- Sampling-based verification outside the NumPy track.

## Further Notes

Implementation record (2026-10-09 — all slices landed):
- One new seam as planned: `shared/draft.py` (Drafter protocol + sidecar
  scheme). `forward_chunk` joined the existing KV-step interface on all
  four tracks; the engines live at `impl/_np/spec.py` (reference: greedy +
  rejection sampling) and `shared/spec_engine.py` (torch-family production
  greedy).
- Two cross-track bugs found by the new tests and fixed: the DSpark
  `pos_emb` broadcast (both NumPy and torch drafters drafted correctly
  only at batch 1), and the NumPy MTP drafter's fixed batch-1 cache (now
  rebuilt on batch change, matching the torch drafter).
- Acceptance on the distilled drafters is drafter-quality-dependent
  (pos-0 top-1 agreement ≈75% at full training budget; measured block
  acceptance ≈25–30%); the lossless contract is
  drafter-quality-INdependent and is what the tests pin.
- The CLI default resolved to `--spec auto` (mtp when sidecars exist) —
  one default everywhere, per the design session.
- `verify_equivalence` runs 9 scenarios (the two spec scenarios assert
  spec-greedy ≡ plain greedy on BOTH engines with an untrained drafter).

- ADR 0003 records the irreversible calls (sidecar format, chunked
  verification everywhere, NumPy-carries-the-theorem, MTP default on).
- The glossary (CONTEXT.md, Inference section) already defines: speculative
  decoding, target model, drafter, draft block, verification, accepted
  length, acceptance, MTP, DSpark, DFlash.
- History: the design started as "disabled by default"; the confirmed final
  state is **MTP active by default** with plain as the opt-out comparison
  mode.
- Order of implementation: shared protocol + sidecar I/O + NumPy
  `forward_chunk` + NumPy MTP (the seam everything hangs off) → torch-family
  `forward_chunk` + shared engine → DSpark → distillation script → server,
  page, records → tests → docs bar.
