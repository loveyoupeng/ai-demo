# ADR 0007: Per-block mixing sublayers — 3 Gated DeltaNet + 1 full attention per layer

- Status: accepted
- Date: 2026-11-03
- Supersedes: ADR 0006's per-layer `layer_pattern` selection (ADR 0006's
  Gated DeltaNet math is unchanged and still lives in `impl/_np/deltanet.py`)

## Context

ADR 0006 made each *layer* either full attention or DeltaNet via
`layer_pattern="dddF"`. The user-visible inspector showed one mixer per block.
The requested shape (Qwen3-Next style) is different: **every block contains
FOUR sequential mixing sublayers — 3 Gated DeltaNet + 1 full RoPE attention** —
each with its own pre-norm, i.e. inside each layer:

```
h_0 = x
h_{m+1} = h_m + mixer_m(RMSNorm_m(h_m))     for m = 0..2  (Gated DeltaNet)
h_4     = h_3 + attn(RMSNorm_3(h_3))         (full RoPE attention)
out     = h_4 + FFN(RMSNorm(h_4))
```

## Decision

Replace `layer_pattern: str | None` with `n_deltanet_sublayers: int = 3`.
Every block owns `n_mixers() = n_deltanet_sublayers + 1` mixers:
`mixer_kinds() == ["d", "d", "d", "F"]` — mixers 0..n-1 are Gated DeltaNet,
the last is full RoPE attention. `n_deltanet_sublayers = 0` restores the
classic single-attention block.

Concretely:

- **Keys gain a mixer index**: `Keys.attn(i, m, proj)` →
  `model.layers.{i}.mixers.{m}.self_attn.{proj}.weight`;
  `Keys.deltanet(i, m, param)` → `model.layers.{i}.mixers.{m}.deltanet.{param}.weight`;
  new `Keys.mixer_ln(i, m)` → `model.layers.{i}.mixers.{m}.input_layernorm.weight`
  (one pre-norm gamma per mixer; `post_attention_layernorm` stays per block).
  `layer_pattern` is a legacy key ignored on load.
- **Cache is a list per layer**: `cache[i]` is a list of `n_mixers()` dicts —
  `{"s", "s_hist"}` for DeltaNet mixers, `{"k", "v"}` for the attention mixer.
  Spec rollback (`_trim_cache` in `impl/_np/spec.py` and
  `shared/spec_engine.py`) iterates `cache[i][m]`.
- **Block record**: learning records carry `mixers: [...]` (one MixerRecord
  per mixer, each with `ln`/`h`/`out` plus `attn` or `deltanet`) instead of a
  single `attn`/`deltanet` key.
- The page renders one LN + mixer + cache node per mixer inside each block.

## Consequences

- All four tracks implement the same per-block mixer chain; NumPy carries the
  analytic BPTT backward through every mixer (gradcheck ~1e-9).
- Checkpoints written before this ADR use the pre-mixer key scheme and do NOT
  load (the registry validation fails fast); retrain or migrate.
- Demo geometry: `scripts/train_demo_model_v3.py` sets
  `n_deltanet_sublayers=3` on both variants.
