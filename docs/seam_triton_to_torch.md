# Triton → PyTorch Shared Seam

The Triton track is a *thin layer of Triton kernels over a PyTorch model*. The
model skeleton (the `nn.Module` classes, the parameter registry, the
`load_from_numpy_dict` / `save_as_numpy` interchange, the training loop) is
PyTorch; only the attention core and the FFN activation run on Triton kernels.
This is a deliberate, documented one-way dependency: **the Triton track
depends on the PyTorch track, never the other way around.** The two "torch-
family" tracks share the same model skeleton so they cannot drift.

## What the Triton track imports from the PyTorch track

| Import | Where | Why |
|---|---|---|
| `impl._torch.layers.RoPE` | `impl/_triton/transformer.py:20` | RoPE is a simple element-wise rotation; kernelizing it would not buy enough to justify a second implementation. The Triton track reuses the PyTorch reference so the math is bit-identical across the two torch-family tracks. |

That is the *only* cross-track import. Everything else in the Triton track is
either Triton kernels (`attn.py`, `rope.py`, `layernorm.py`, `activation.py`)
or plain PyTorch wiring (`transformer.py`, `model.py`, `inference.py`,
`training.py`).

## Why a seam and not a shared module

A shared `impl/shared_torch/` module would force the Triton track to import
through a third location, hiding the fact that the Triton track is a PyTorch
model with a Triton attention core. The explicit import from `impl._torch`
makes the dependency visible at the import site (a one-line comment:
`# shared-seam: torch building block reused by triton`), and it makes the
one-way rule easy to enforce: the PyTorch track has *no* imports from the
Triton track (verified by a grep), so a cycle is impossible.

## Rules

1. **One-way.** `impl/_torch/**` must not import from `impl/_triton/**`.
   If a change would require that, the shared code belongs in a new
   `impl/shared_torch/` module that *both* tracks import — and this document
   must be updated to describe the new seam.
2. **Documented at the import site.** Every cross-track import carries a
   `# shared-seam:` comment naming the block and the reason.
3. **Same math.** The shared block is the *reference* implementation; the
   Triton track does not re-derive it. If the PyTorch track changes the block,
   the Triton track inherits the change (and the cross-backend parity tests
   verify it).

## Current shared blocks

- **RoPE** (`impl._torch.layers.RoPE`) — used by
  `TritonMultiHeadAttention` to rotate q and k. RoPE is a simple element-
  wise rotation; the Triton track reuses the PyTorch reference so the math
  is bit-identical across the two torch-family tracks. (The Triton track
  also has a standalone Triton RoPE kernel in `impl/_triton/rope.py` for
  experiments, but the block itself uses the shared PyTorch `RoPE`.)

## Verifying the seam

```bash
# The PyTorch track must not import from the Triton track (one-way rule).
grep -r "from impl._triton" impl/_torch/   # expect: no matches

# The Triton track's cross-track imports (the seam, documented).
grep -rn "from impl._torch" impl/_triton/  # expect: the RoPE import above
```

