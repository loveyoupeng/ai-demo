# ADR 0005: Parameter-matched MoE — 8 routed experts + 1 shared at dim 32

- Status: accepted
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

Dense reference at the toy scale (D=64, L=3, H=4, FFN=256): 262,592
parameters, all active.

## Decision

1. **MoE geometry becomes E=8 routed + 1 shared expert, expert_dim=32,
   top_k=1** (one config change: `n_experts=8, n_shared_experts=1,
   expert_dim=32, top_k=1`).
   - Total: 282,560 (1.08× dense) — capacity-matched.
   - Active per token: 153,536 (0.59× dense) — the routed slice is
     1-of-8 experts (32-wide) + the shared expert (32-wide), so ~54% of
     the MoE's own parameters and well under the dense model's FFN
     budget stand behind any single token. The total-vs-active contrast
     IS the lesson, at a scale the page can render.
   - 16-dim experts (a 15-expert alternative) would have been more
     A3B-like in ratio but too degenerate to learn at D=64; 3 wide
     experts (the status quo) shows no contrast. 8×32 is the middle
     that keeps both models trainable.
2. **The page shows the trade explicitly**: `/api/model` gains
   `n_params_active` next to `n_params`, and the badge line renders
   "total X · active Y/token" — the two numbers the A3B story lives on.
3. **ADR 0002's shared-expert decision is unchanged**: the shared expert
   remains always-on and ungated; only its width follows the new
   expert_dim (32), so the routed+shared contrast stays visible in the
   diagram.

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

- The MoE twin re-trains (v3 pipeline trains both families anyway); the
  registry/schema are untouched (`expert_dim` is already a config knob
  honored by all four tracks).
- Router scores become (S, 8) — still small enough to render per cell.
- The diagram auto-adjusts (expert count is read from config), but the
  page's model-info line must add the active-param count or the
  redesign is invisible.
- `learning_moe*` checkpoints are stale on disk until retrained — the
  `"pipeline": 3` marker in `config.json` (ADR 0004) drives the
  staleness check.
