# ADR 0011: Group-of-2 blocks with alternating global mixer (MHA / MLA)

- Status: accepted
- Date: 2026-11-10
- Extends: ADR 0007 (per-block mixers), ADR 0009 (MLA)

## Context

ADR 0007/0009 gave every block the same mixer list: 3 Gated DeltaNet + 1
global attention, where the global slot is either classic MHA (KV cache
`{"k","v"}`) or MLA (latent cache `{"c"}`, ADR 0009) — chosen globally by
`mla_latent_dim`. The question: put BOTH global mixers in the model, one per
block, alternating.

Production precedent is layer-interleaved hybrids with a 3:1 linear:global
ratio: Kimi-Linear-48B-A3B config.json sets
`linear_attn_config.full_attn_layers = [4, 8, 12, 16, 20, 24, 27]` — one
global (MLA) layer every 4th layer, last layer always global; Kimi K3 is
69 KDA + 24 gated-MLA layers. Qwen3-Next/Qwen3.5 do the 3:1 split per block
with plain MHA. No open model puts MHA and MLA in the same stack — MLA
dominates MHA on KV-cache memory at matched quality, so the MHA arm here is
a **comparison arm**, not a memory win.

## Decision

Treat **groups of `mla_group_size` (default 2) blocks as one logical
layer**. Within a group, the global mixer alternates by block parity:

- even block index in the group (`layer_idx % mla_group_size == 0`) →
  **MLA** global mixer (latent cache `{"c"}`, ADR 0009 keys
  `self_attn.{kv_down,k_up,v_up}_proj`);
- odd block index → **classic MHA** global mixer (cache `{"k","v"}`,
  keys `self_attn.{k,v}_proj`).

Config: `mla_group_size: int = 1` (1 = every block MLA when
`mla_latent_dim > 0` — the ADR 0009 behavior, backward compatible) and the
alternation is on whenever `mla_group_size > 1`. Helper
`TransformerConfig.has_mla_at(layer_idx) -> bool` = `mla_latent_dim > 0 and
(layer_idx % mla_group_size == 0)`. `has_mla()` stays true when any block
is MLA.

Example (the v7 demo): `mla_group_size=2`, `mla_latent_dim=16` — MoE =
4 layers, dense = 8 layers, so BOTH families carry the identical
MLA/MHA interleave (blocks 0,2,4,… MLA; 1,3,5,… MHA). Each block still
runs `n_deltanet_sublayers` DeltaNet mixers + its one global mixer.
Param-matched at the new depth: MoE E=10+1 shared at `expert_dim=20`
(~543K, the 30B-A3B shape); dense 8L thin FFN `expert_dim=48` (~752K —
deeper stack, more total params; the active-per-token trade of ADR 0005
is what the page compares).

## Why this shape

- 3:1 linear:global per block matches Qwen3-Next/Kimi-Linear; grouping into
  2-block logical layers doubles the global density (1 global per 2 blocks
  instead of per 4 layers) and alternates the global KIND, giving a
  controlled MHA-vs-MLA comparison inside one checkpoint (the learning
  page renders both cache kinds side by side).
- Alternating keeps the two cache schemas interleaved through depth, so
  neither mixer's representation is confined to early/late layers.
- Kimi-Linear puts the LAST layer global; we keep the default parity rule
  (even index → MLA) and note the alternative as a config follow-up.

## Consequences

- Registry: per-layer branch — `Keys.attn(i, m, …)` emits MLA keys for
  `has_mla_at(i)` blocks, classic keys otherwise. Old checkpoints
  (`mla_group_size=1`) validate unchanged.
- Every track's block constructor, `make_cache`, prefill backfill,
  `_param_slots`/`_param_tensors`, and spec rollback switch on
  `config.has_mla_at(i)`.
- The learning page labels each block's global mixer "attention (MLA)" /
  "attention" and its cache node "latent c cache" / "K/V cache" per block.
- Pipeline marker: 6 (v6, `mla_group_size=2`, 3L/6L) → 7 (v7: MoE 4L / dense 8L, same interleave).
