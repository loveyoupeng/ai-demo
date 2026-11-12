# ADR 0008: Sliding-window attention (SWA) on the full-attention mixer

- Status: accepted
- Date: 2026-11-09
- Supersedes: nothing (additive, opt-in via config)

## Context

The full (RoPE) attention mixer (ADR 0007's 'F' mixer) is O(S²) in the
score matrix and attends to **every** past token. Production long-context
models (Mistral, Gemma, LongCat) interleave *local* windowed attention with
occasional *global* attention, and 2026 analyses ("Sliding-window beats
linear attention", arXiv 2608.28444; "Rethinking the Role of Efficient
Attention in Hybrid Architectures", EMNLP 2026) show SWA can beat linear
attention on recall while keeping decode memory O(window). Our hybrid
(ADR 0007) already dedicates 3 of 4 mixers to linear attention; making the
one full-attention mixer **windowed** is the complementary lever and the
cheapest possible architecture change: no new parameters, only a different
mask.

## Decision

New config field `TransformerConfig.swa_window: int = 0`:

- `0` (default) — full causal attention (unchanged; all existing
  checkpoints/configs behave exactly as before).
- `w > 0` — the full-attention mixer masks key `j` for query `i` when
  `i - j > w` IN ADDITION to the causal rule (`j > i`). With absolute
  positions, query at absolute `p` attends to absolute keys in
  `[p - w, p]`.

The mask is applied uniformly in all three forward paths:

- `forward_state` (training/prefill): banded causal mask
  `mask[i,j] = (j > i) or (i - j > w)` over the (S, S) block.
- `forward_step` (decode): the query is at absolute `position`; only the
  last `w + 1` cache rows are visible — implemented by **slicing** the
  cache (`k[:, :, max(0, t+1-w):]`), so decode cost is O(w) instead of
  O(t).
- `forward_chunk` (speculative verification): the (c, t+c) block mask
  becomes `mask[i, col] = (col_abs > row_abs) or (row_abs - col_abs > w)`
  with `row_abs = position + i`, `col_abs = col`.

## Why not alternatives

- **SWA on the DeltaNet mixers too**: DeltaNet is already O(1) per token
  in state size; a window there would need state eviction (rolling S),
  which breaks the exact BPTT rollback the spec-decoding contract relies
  on. Not worth it at teaching scale — noted as a `# PROD:` follow-up.
- **Learned sparse patterns (DSA/lightning indexer)**: that is the
  *selection* counterpart to SWA's fixed band; a bigger change (extra
  indexer head + new cache). Kept as a possible future ADR; SWA first
  because it is parameter-free.

## Consequences

- No new checkpoint keys (mask only) — `swa_window` is a config knob, and
  the registry is unchanged by it. Old checkpoints load with
  `swa_window=0` semantics.
- All four tracks implement the same mask; the triton/cuda SDPA kernels
  take a `window` parameter (0 = pure causal) in addition to
  `is_causal`.
- The learning page renders the banded mask in the existing
  `attn.mask` heatmap (the band is visually obvious) and shows an
  "SWA w=…" badge; the decode path shows the sliced cache.
- The demo pipeline (v4) trains with `swa_window=32` at `context_length=128`.
