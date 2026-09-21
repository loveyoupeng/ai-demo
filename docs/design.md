# Design Document: Decoder-Only Transformer Learning Project

**Date:** 2026-06-26 (last synced: 2026-09-19)
**Goal:** Build a fully functional decoder-only transformer LLM in 4 equivalent implementations (NumPy, PyTorch, Triton, CUDA) for educational purposes.

## Track intent (doctrine)

Each track is written to teach a different skill — this guides every docstring
and comment choice:

- **NumPy = how the math works.** Every operator is hand-derived with formula
  citations and shape comments; the analytic backward is the lesson.
- **PyTorch = how to do it properly with a framework.** Production idiom:
  `nn.Module` composition, autograd, SDPA, `torch.optim`. Read it to learn
  how this model should be written in real code.
- **Triton = how to write the kernels.** Memory-traffic-aware kernels
  (online softmax) over the PyTorch skeleton.
- **CUDA = the bare metal.** NVRTC kernels, explicit launches, no framework.

When a track deviates from its simplest correct form, the deviation must
answer that track's question (e.g. Triton fuses because kernel fusion *is*
the lesson); pedagogical shortcuts that would mislead about production
practice get a `# PROD:` note instead.

## Cross-track naming (audited 2026-09)

Deliberate: every track owns `forward` + `load_from_numpy_dict` +
`get_all_parameters`/`save_as_numpy`, and classes carry the track prefix
(`NumPyModel`, `TorchModel`, `TritonModel`, `CUDAModel`; same for the
`TextGenerator`s). Known deltas, kept consciously:

- The NumPy `TextGenerator` and `NaiveKVCache` skip the `NumPy…` prefix —
  that track is the reference, so its names are the unadorned canonical ones.
- Only the NumPy track publicizes `forward_prefill`/`forward_step`; the
  PyTorch/Triton/CUDA generators own their cache loop internally (the
  pedagogical value of the exact step path belongs to the reference track).
- `CUDAModel` is not an `nn.Module` — bare-metal track, plain tensor
  attributes; training collects grads by walking the block's tensor
  attributes (fixed 2026-09 — an earlier collector walked stale names).

---

## Architecture

```text
                    +----------+
                    | PyTorch  |
                    | Triton   |
                    | CUDA     |
                      |  |  |
                      v  v  v
              +-----------------+
              |   Shared Config  |  (json + env vars)
              +-----------------+
                      |
                 +----v----+
                 |  NumPy   |  (reference/benchmark)
                 +---------+
```

### Weight Flow

- All 4 backends accept the same JSON config + env vars → same model topology
- **Save**: NumPy `get_all_parameters()` / PyTorch `save_as_numpy()` / Triton `save_as_numpy()` / CUDA flat → `model.npz` (flat dict, via the shared `ParameterRegistry`)
- **Load**: NumPy `load_from_*` / PyTorch `load_from_*` / Triton `load_from_*` / CUDA flat assign
- **PyTorch↔Triton**: direct load via compatible `save_as_numpy()` / `load_from_numpy_dict()`
- **NumPy↔NumPy**: direct load (identical API)
- **NumPy↔CUDA**: direct load via shared flat format (no conversion needed)

---

## Project Structure

```text
project/
├── shared/           # Shared across backends: config, constants, checkpoint, data
│   ├── config.py     # TransformerConfig (frozen dataclass, one per model)
│   ├── config_utils.py # Unified config reader (CLI > env > file > defaults)
│   ├── constants.py  # Keys — parameter-name constants (HF-Llama scheme)
│   ├── registry.py   # ParameterRegistry: owner of the flat-dict checkpoint format
│   ├── tokenizer.py  # GPT-2 BPE (primary) + char-level tokenizer for demos
│   ├── dataset.py    # TinyStories loading → (input, target) batches
│   └── checkpoint.py # save/restore helpers (config.json + model.npz + vocab.json)
├── impl/
│   ├── _np/          # NumPy track (reference: hand-rolled math + backward)
│   │   ├── embedding.py / layernorm.py / rope.py     # single-purpose components
│   │   ├── attention.py  # MultiHeadAttention (GQA, KV cache, TurboQuant)
│   │   ├── ffn.py        # SwiGLU FFN
│   │   ├── moe.py        # MixtureOfExperts (router + top-k experts)
│   │   ├── block.py      # TransformerBlock (RMSNorm → attn → residual → ...)
│   │   ├── stack.py      # DecoderStack (N blocks)
│   │   ├── model.py      # NumPyModel (embedding → stack → final norm → lm_head)
│   │   ├── inference.py  # TextGenerator (greedy / sampled decoding)
│   │   ├── training.py / optimizer.py / cross_entropy.py / gradcheck.py
│   │   ├── learning.py   # instrumented_forward + generate_with_records (records)
│   │   ├── learning_server.py # learning-mode HTTP server (numpy/torch backends)
│   │   └── web/          # the learning page (vanilla JS + KaTeX, no framework)
│   ├── _torch/       # PyTorch track (production ops: F.scaled_dot_product_attention)
│   │   ├── layers.py     # TorchModel + all nn.Module components
│   │   ├── learning.py   # torch record adapter (same JSON shapes as the np one)
│   │   └── inference.py / training.py / cross_entropy.py / kv_cache.py
│   ├── _triton/      # Triton GPU kernels (flash attention, etc.)
│   ├── _cuda/        # CUDA bare-metal (kernels/, NVRTC compiler, per-track CLI)
│   └── (per-track) cli.py entry points: uv run python -m impl._<track>.cli
├── tests/
│   ├── unit/         # per-backend unit tests (_np/, _torch/, _triton/, _cuda/) + shared
│   └── cross_backend/ # parity tests between tracks (3-way, GPU parity)
└── docs/             # design.md, docstring_style.md, adr/, specs/, task_plan.md
```

---

## Implementation Status

| Backend | Components | Tests | Status |
| --------- | ----------- | ------- | -------- |
| NumPy | Complete (Embedding, RMSNorm, RoPE, MHA, SwiGLU FFN, GQA, MoE, TransformerBlock, DecoderStack) | All pass | ✅ Complete |
| PyTorch | Complete (all layers match NumPy) | All pass | ✅ Complete |
| Triton | Complete (all kernels match NumPy) | All pass | ✅ Complete |
| CUDA | Complete (all kernels compiled + execute) | All pass | ✅ Complete |

### NumPy Layer Summary

- **Embedding**: `vocab_size × embed_dim` trainable lookup table, no bias
- **RMSNorm**: per-dim scale — `x / rms(x) * gamma`, no bias (Zhang & Sennrich 2019)
- **RoPE**: 2D orthogonal rotation of (q, k) pairs by position (Su et al. 2021)
- **MHA**: Multi-Head Attention — supports KV cache and non-cached inference
- **SwiGLU FFN**: gated linear unit variant with SiLU activation
- **MoE**: Mixture of Experts — multiple feed-forward expert sub-layers, gated routing
- **GQA**: Grouped-Query Attention — groups query heads into key/value groups for KV cache efficiency
- **Residual**: standard additive skip connection `out = x + f(ln(x))` (see [ADR-0001](adr/0001-gated-residual-abandonment.md))
- **TransformerBlock**: one layer — RMSNorm → MHA → residual → RMSNorm → FFN → residual
- **DecoderStack**: N-layer stack; the model applies embedding → stack → final RMSNorm → lm_head

---

## Cross-Backend Parity

### Training Parity (Weight Diff After 1 Iteration)

| Comparison | Weight Diff | Inference Diff |
| ----------- | ------------ | ---------------- |
| NumPy vs PyTorch | `~1e-4` | `~1e-5` |
| NumPy vs Triton | `~1e-2` | `~1e-1` |
| NumPy vs CUDA | `~1e-2` | `~1e-1` |

### Inference Parity (Same Weights, Same Input)

| Comparison | Token Diff | Max Prob Diff |
| ----------- | ----------- | --------------- |
| NumPy vs PyTorch | `~1e-4` | `~1e-5` |
| NumPy vs Triton | `~1e-2` | `~1e-2` |
| NumPy vs CUDA | `~1e-2` | `~1e-2` |

---

## Training & Inference

### Training

- **Loss function**: Cross-entropy (label smoothing = 0.0)
- **Optimizer**: AdamW (β1=0.9, β2=0.999, eps=1e-8)
- **Scheduler**: Cosine annealing with warmup
- **Gradient clipping**: Norm clipping (max_norm=1.0)
- **Batch size**: Context-length chunks for next-token prediction (e.g., 32×256)
- **Data pipeline**: Tokenizer → TokenizedDataset → DataLoader → forward → loss → step

### Inference

- **Greedy decoding**: `argmax(logits)` → deterministic, best for testing
- **Weighted sampling**: Sample from softmax(logits / temperature) → stochastic
- **KV Cache**: Full caching of past key/value tokens for efficiency
- **Token buffer**: Circular buffer for fixed window
- **Multi-level cache**: Supports LRU/LFU for long context caching

### Multi-Level KV Cache

- **Full cache** (L=seq_len): Stores all past K/V — used for training
- **Partial cache** (L≤L_max): Stores recent K/V tokens — used for long context
- **Circular buffer** (L≤L_max): Fixed-size circular buffer — used for short context

---

## Platform Target

| Component | Value |
| ----------- | ------- |
| **Device** | NVIDIA Jetson AGX Orin 64GB |
| **OS** | Ubuntu 22.04 with JetPack 6.2.2 |
| **CUDA** | CUDA 12.6 (nvcc 12.6) |
| **PyTorch** | PyTorch 2.2.0 with CUDA 12.6 |
| **GPU** | 2048 CUDA cores, 64-bit memory, ~20 TFLOPS |

---

## Key Design Decisions

1. **4 backends, same topology** — All accept JSON config, produce same model structure
2. **NumPy as truth benchmark** — Pure Python/numpy implementation used as the reference for all other backends
3. **Independent training** — Each backend trains independently with its own random seed, but should produce equivalent weights
4. **Round-trip tests** — Save to NumPy format, load into any backend, verify inference matches
5. **Flat checkpoint format** — All backends save/load `model.npz` as a flat dict (keys from the shared `Keys` scheme), enabling cross-backend transfer
6. **Standard additive residual** — the block uses the plain pre-norm skip `out = x + f(ln(x))`, matching LLaMA and every modern reference (the repo-specific "gated residual" was abandoned — see [ADR-0001](adr/0001-gated-residual-abandonment.md))
7. **Multi-level KV caching** — Configurable cache length for efficient training vs inference
8. **PyTorch nn.Module wrapper** — PyTorch/Triton models are `nn.Module` instances (training via `.parameters()`); `CUDAModel` is a plain class whose tensor attributes carry `requires_grad`
