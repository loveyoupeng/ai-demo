# ADR 0002: Ungated shared expert in the MoE block

- Status: accepted
- Date: 2026-09-20

## Context

The MoE FFN (`softmax router → top-k mask → renormalize → Σ wⱼ·Eⱼ(x)`) routes
every token exclusively through its top-k experts. DeepSeek-V2/V3 showed that
adding a *shared expert* — one (or a few) experts every token always passes
through — improves both quality and router load: common features land in the
shared expert, freeing routed experts to specialize.

Because this repo's core invariant is **cross-track checkpoint equivalence**
(the same `model.npz` loads and runs identically on all four tracks), the
change must land in the checkpoint format (registry keys), `TransformerConfig`,
all four tracks, gradcheck, and every parity test — or not at all.

## Decision

`TransformerConfig.n_shared_experts: int = 0`. When > 0, each MoE block adds
`n_shared_experts` always-active SwiGLU experts and computes

```
out = (Σ_s E_shared_s(x)) / n_shared_experts + Σ_j∈topk wⱼ·E_j(x)
```

- The shared expert is **ungated**: it does not participate in the router
  softmax and does not consume router probability mass. The routed branch is
  untouched (top-k softmax renormalization stays exactly as ADR-free baseline).
- Shared-expert output is **averaged** over `n_shared_experts` so turning the
  knob from 1 → 2 does not rescale the block output.
- `n_shared_experts = 0` reproduces the pre-ADR behavior bit-for-bit: no new
  parameters, no output change, all existing checkpoints remain valid.
- Registry gets one entry group per block per shared expert:
  `model.layers.{i}.mlp.shared_experts.{s}.{gate,up,down}_proj.weight`.
- The learning-mode diagram gains a `shared` cell in the MoE module when the
  loaded model enables it.

## Alternatives considered

- **Router-integrated shared expert** (Mixtral-style: shared expert is just
  expert E+1 that always wins the top-k): simpler keys, but it consumes
  probability mass from the softmax and entangles the shared expert's gradient
  with the routing distribution — pedagogically muddier, and numerically no
  better in the literature.
- **Per-layer choice of shared/routed (some layers dense)**: rejected — one
  knob for all layers keeps the config legible; this is a teaching repo.
- **NumPy-track-only implementation**: rejected; it would break the
  cross-backend equivalence invariant that justifies the repo's existence.

## Consequences

- Checkpoint format grows new keys only when `n_shared_experts > 0`; the
  registry validates them like any other entry, so old checkpoints load
  unchanged and new checkpoints fail fast on old code (unknown key).
- `tests/cross_backend/` gains a shared-expert MoE parity scenario; the NumPy
  gradcheck covers the additive branch (no top-k kink on that path).
- The learning-mode MoE inspector shows the shared expert's output as an
  always-on additive term next to the routed sum.
