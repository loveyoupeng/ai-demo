# ADR 0005: Parameter-matched dense-vs-MoE — depth vs expert bank

- Status: accepted (revised 2026-10-28 — the first version matched only
  the FFN scale, leaving both models at 3 layers; the revision makes the
  TOTAL budgets match and splits them the way production pairs do)
- Date: 2026-10-25
- Supersedes: the **scale choice** in ADR 0002 (E=3 routed + 1 shared at
  the dense model's FFN width). ADR 0002's *decision* — an always-on
  ungated shared expert — is untouched; only the geometry is superseded.

## Context

The learning page sells the MoE variant with the sentence "same
tokenizer, same data, same seed — only the FFN differs," which makes
dense-vs-moe a controlled experiment. But the twin MoE (E=3 + 1 shared,
each expert at the dense FFN's full width 256) is NOT param-matched: it
holds ~4× the dense model's FFN parameters. The comparison taught
"MoE = more capacity," which is not the property MoE architectures are
actually chosen for. Production MoE models (Qwen3-235B-A22B,
Llama-4-Maverick, DeepSeek-V3) are chosen for the opposite trade:

- **total parameters** are allowed to grow (capacity lives in the
  expert bank),
- **active parameters per token** stay small (a top-1 or top-2 routed
  slice + a small shared expert), so per-token FLOPs stay near a much
  smaller dense model.

The repo cannot show that trade if the two models differ in total size
in the least interesting direction. The learning page also renders
whatever config `/api/model` returns (the diagram interpolates
`n_experts`), so a geometry change is a config change plus a params
line, not a page rewrite.

## Decision

The two families get **the same total budget, spent differently** — the
production pairing's shape (Qwen3-32B dense is DEEPER — 64 layers —
than the 30B-A3B MoE's 48; the dense model buys depth, the MoE buys the
expert bank):

1. **Dense: 6 layers, thin FFN** (`expert_dim=96` = 1.5·D):
   275,264 total = 275,264 active. Depth is the dense model's edge.
2. **MoE: 3 layers, wide expert bank** (E=8 routed + 1 always-on shared
   at expert_dim=32, top_k=1): 282,560 total (1.03× dense) but only
   153,536 ACTIVE per token (~54% of its own parameters) — the
   routed slice is 1-of-8 experts (32-wide) + the shared expert. The
   total-vs-active contrast IS the lesson, at a scale the page renders.
   - 16-dim experts (a 15-expert alternative) would have been more
     A3B-like in ratio but too degenerate to learn at D=64; 8×32 is the
     middle that keeps both models trainable.
3. **The page shows the trade explicitly**: `/api/model` gains
   `n_params_active` next to `n_params`, and the badge line renders
   "total X · active Y/token" — the two numbers the A3B story lives on.
4. **ADR 0002's shared-expert decision is unchanged**: the shared expert
   remains always-on and ungated; only its width follows expert_dim (32),
   so the routed+shared contrast stays visible in the diagram.
5. **First revision's flaw (why this is revised)**: the initial cut
   matched only the FFN scale and left both models at 3 layers — the MoE
   held 1.08× the dense TOTAL, so the comparison was not budget-matched
   and both diagrams showed the same depth. Splitting depth (6 dense vs
   3 MoE) fixes both: budgets match to 3%, and the page visibly shows
   the two ways to spend the same budget.

## Alternatives considered

- **Keep E=3 + 1 @ 256**: zero migration, but teaches the wrong property
  (capacity growth, not sparse activation) — rejected.
- **Param-match by shrinking experts to 64 with E=6 + shared**: splits
  the difference but the active fraction (≈25% with shared) is the same
  while the router scores get less interesting — rejected.
- **Raise dense to match MoE totals**: makes the dense model the
  expensive one and confuses the "which is the sparse one?" story —
  rejected.

## Consequences

- BOTH families re-train (the dense geometry changed too — its
  checkpoints carry `"pipeline": 3` and go stale; the marker is now 4);
  the registry/schema are untouched (`expert_dim`/`n_layers` are already
  config knobs honored by all four tracks).
- Router scores stay (S, 8) — small enough to render per cell; the
  dense diagram now shows 6 blocks vs the MoE's 3, which is itself half
  the lesson (the same budget, two shapes).
- The diagram auto-adjusts (layer and expert counts are read from
  config), and `/api/model`'s `n_params_active` makes the trade visible
  in the badge line.
