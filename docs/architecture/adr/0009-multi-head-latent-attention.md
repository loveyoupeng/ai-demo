# ADR 0009: Multi-head Latent Attention (MLA) — latent KV cache

- Status: accepted
- Date: 2026-11-09
- Supersedes: nothing (opt-in via config; extends GQA's cache compression)

## Context

At inference the KV cache is the memory bottleneck: it grows as
`(B, G, t, hd)` per layer (GQA already shrank it H/G vs MHA). DeepSeek-V2/V3
introduced **Multi-head Latent Attention (MLA)**: instead of caching K and V
per group, cache ONE low-rank **latent vector** `c_t` per token and
*reconstruct* K/V from it on the fly:

    c_t   = x_t @ W_kv_down            (B, S, L)      L = latent dim
    k_t   = RoPE(c_t @ W_k_up)         (B, G, S, hd)  up-projected per group
    v_t   = c_t @ W_v_up               (B, G, S, hd)
    q_t   = x_t @ W_q                  (B, H, S, hd)  (query stays full-rank)

The cache stores only `c` — `(B, L, t)` per layer instead of
`(B, G, 2, t, hd)` — an order-of-magnitude memory cut when
`L << 2·G·hd`, at the price of two small matmuls per reconstruction.
2026 frontier models (DeepSeek-V3/V4, Kimi, GLM-5) all ship MLA or its
sparse-attention extension (DSA).

## Decision

New config field `TransformerConfig.mla_latent_dim: int = 0`:

- `0` (default) — current behavior: `k/v_proj` (D, G·hd) each, cache
  `{"k", "v"}` per group. Registry unchanged.
- `L > 0` — MLA mode. The full-attention mixer's K/V projections are
  REPLACED by three projections (new registry keys):

      model.layers.{i}.mixers.{m}.self_attn.kv_down_proj.weight  (D, L)
      model.layers.{i}.mixers.{m}.self_attn.k_up_proj.weight     (L, G·hd)
      model.layers.{i}.mixers.{m}.self_attn.v_up_proj.weight     (L, G·hd)

  `q_proj`/`o_proj` are unchanged. The cache becomes `{"c": (B, L, t)}`;
  K/V are reconstructed from the whole cached latent at every step/chunk:
  `k = RoPE(c_cache @ W_k_up)`, `v = c_cache @ W_v_up` (then GQA repeat as
  usual). RoPE after up-projection is mathematically identical to DeepSeek's
  decoupled-rope variant (RoPE is linear in k); the decoupled form saves
  compute and is marked `# PROD:` in the NumPy reference.

- TurboQuant (1-bit cache) in MLA mode quantizes the reconstructed
  per-head K/V exactly as in non-MLA mode.

## Why not alternatives

- **Keep k/v_proj and add a latent on top**: two sources of K/V would
  double cache traffic — MLA's whole point is the cache holds ONLY c.
- **MQA-style single shared latent without up-projections**: that is just
  GQA with G=1 on the latent; MLA keeps per-group up-projections so quality
  stays at GQA level.

## Consequences

- Registry: when `mla_latent_dim > 0` the attention mixer's `k_proj`/`v_proj`
  entries are replaced by `kv_down_proj`/`k_up_proj`/`v_up_proj`. Old
  checkpoints (mla off) still validate against old configs; the demo
  pipeline bumps its `pipeline` marker so stale checkpoints bootstrap.
- All four tracks implement the same math: NumPy = analytic backward
  through the low-rank factorization (chain rule over the two matmuls),
  gradchecked; torch/triton/cuda = the same projections as ordinary
  `nn.Linear`s plus a latent cache dict `{"c": …}`.
- `make_cache`/prefill backfill store `c` (the latent), and
  `forward_step`/`forward_chunk` reconstruct K/V from the cache slice.
  Spec-decoding rollback trims `c` rows like K/V rows.
- The learning page labels the cache node "latent c cache (MLA)" and shows
  the reconstruction `k = c·W_k_up` in the attn inspector; an "MLA L=…"
  badge shows the cache-size saving vs GQA.
- Demo pipeline (v4) trains with `mla_latent_dim=16` at D=64, G=4, hd=16
  (GQA K/V would be 2·4·16=128 floats/token; latent = 16 — an 8× cache cut).
