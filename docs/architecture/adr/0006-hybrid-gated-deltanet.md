# ADR 0006: Gated DeltaNet as a mixing sublayer

- Status: **superseded in placement by ADR 0007** (the DeltaNet math below is
  unchanged and still lives in `impl/_np/deltanet.py`; only the *where* moved:
  ADR 0007 puts 3 DeltaNet mixers + 1 full-attention mixer inside EVERY block
  instead of selecting one mixer per layer via `layer_pattern`)
- Date: 2026-11-01
- Supersedes: nothing

## Context

Full (RoPE) attention is O(S²) in the score matrix and O(S) per-token decode
against the KV cache. Production hybrids (Qwen3-Next, Kimi Linear) put a
**3:1 ratio of linear-attention (Gated DeltaNet) layers to full-attention
layers** in the stack: the DeltaNet layers give O(hd²) per-token state
updates (no per-token history read), the full-attention layer keeps global
retrieval. The repo had no linear-attention layer at all (grep: no
`deltanet`/`linear attention`/`recurrent` code before this ADR).

## Decision

Add a **Gated DeltaNet** mixing sublayer (Yang et al., "Gated DeltaNet",
2024/2025) selected per layer by a new config field `layer_pattern`
(e.g. `"dddF"`, repeating over `n_layers`):

- **'F' layer** — the existing RoPE full attention, unchanged.
- **'d' layer** — Gated DeltaNet:
  - `q̂, k̂ = L2norm(xWq), L2norm(xWk)` per head (no RoPE — linear layers
    carry no absolute position);
  - per-head forget gate `β_t = sigmoid(a_proj(x_t) + dt_bias)`;
  - recurrent state `S_t = (1-β_t)·S_{t-1}·(I − k̂_t k̂_tᵀ) + β_t·k̂_t v_tᵀ`
    (the delta rule: subtract the key's old value before writing the new one);
  - read-out `o_t = S_t q̂_t`, output gate `α_t = sigmoid(gate_proj(x_t))`,
    `out = α_t ⊙ RMSNorm(o_t @ W_o)`.

The state is a per-head `(hd, hd)` matrix carried in the layer's cache
(`{"s", "s_hist"}`) instead of K/V rows; `forward_step` updates it in O(hd²);
`forward_chunk` (speculative verification) scans the chunk **sequentially**
through DeltaNet layers and snapshots S per position so spec rollback
(`_trim_cache`) restores the state at the accepted prefix — greedy spec stays
**token-identical to plain** (the lossless contract).

## Why not alternatives

- **All full attention**: O(S) decode per token forever; the hybrid is the
  production shape for long context.
- **All DeltaNet (pure linear)**: loses global retrieval; the 1-in-4 full
  attention layer is what keeps retrieval.
- **Chunked-WY parallel scan in the NumPy reference**: unnecessary at
  teaching scale (ctx=128, D=64); the sequential scan IS the readable math.
  The GPU tracks note the fused-kernel path as a `# PROD:` comment.

## Consequences

- Checkpoint keys: `model.layers.{i}.deltanet.{q,k,v,o,a,gate}_proj.weight`,
  `dt_bias.weight`, `norm.weight` on 'd' layers (registry-owned); old
  all-attention checkpoints load unchanged (`layer_pattern` defaults to "").
- All four tracks implement the same math: NumPy = analytic backward (BPTT
  through the recurrence, gradchecked vs finite differences), torch = autograd
  scan, triton = torch-op scan over the kernel-backed projections, cuda = same
  via cuBLAS + the NVRTC-backed FFN/norm kernels.
- The learning page renders a `deltanet` module per 'd' layer (projections,
  β, the per-position S matrices, the read-out) and labels the cache node
  "S state"; records carry a `deltanet` block key beside `attn`.
- BF16 is the running serving dtype on the GPU tracks (torch/triton/cuda);
  NumPy stays float64/float32; checkpoints stay FP32 on disk.
- Demo geometry stays param-matched at ~279K total: dense 6L FFN=80 vs MoE
  3L E=10+1 shared @ dim 24 (`dddF` on both).
- Drafters (ADR 0003) are untouched: they consume `forward_chunk`'s
  `(logits, x_final)`, and the sidecar format is unchanged.
