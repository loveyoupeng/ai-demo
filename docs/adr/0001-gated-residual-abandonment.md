# ADR 0001: Abandon "gated residual" for the standard additive residual

**Status**: Accepted (2026-09-14)
**Related**: [architecture-fixes.md](../specs/architecture-fixes.md)

## Context

The original block design used a non-standard "gated residual": the residual
stream was multiplied by a learned scalar (or per-channel vector) before
being added to the sublayer output, i.e. `out = g * x + f(ln(x))` where `g`
was a learned gate. This was a design invented in this repo; no published
model (LLaMA, GPT-3, GPT-4, Mixtral, etc.) uses a scalar-gated residual.

The spec's goal is a **standard, web-searchable architecture**: pre-norm
LLaMA block with the plain additive residual `out = x + f(ln(x))`. The
standard form is what every paper, course, and reference implementation
discusses, so a reader looking up "how does a LLaMA block work?" finds
exactly what is in this repo.

## Decision

Drop the gated residual. The block uses the standard additive residual:

```
h = x + attn(ln1(x))
out = h + mlp(ln2(h))
```

where `ln1`/`ln2` are RMSNorms applied *before* the sublayer (pre-norm) and
the residual stream `x` is added unchanged to each sublayer's output.

## Consequences

- **Positive**: the block matches LLaMA and every modern LLM reference
  exactly; the math is web-searchable; the cross-backend parity tests compare
  against a standard form (not a repo-specific variant); the backward pass is
  the standard `dx = dout + dsublayer` (no extra gate-gradient term).
- **Positive**: the parameter count and the key scheme are identical to
  LLaMA's (no extra `gate` parameter per block), so checkpoints are directly
  comparable to HF-Llama-style checkpoints.
- **Negative**: none identified. The gated residual had no published
  equivalent to defend, and no test or script depended on it.
