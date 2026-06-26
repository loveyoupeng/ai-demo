# Design Document: Decoder-Only Transformer Learning Project

**Date:** 2026-06-26 (last synced: 2026-06-26)
**Goal:** Build a fully functional decoder-only transformer LLM in 4 equivalent implementations (NumPy, PyTorch, Triton, CUDA) for educational purposes.

---

## Architecture

```
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
- **Save**: NumPy `get_all_parameters()` / PyTorch `save_as_numpy()` / Triton `save_as_numpy()` / CUDA flat → `ckpt.npz` (flat dict)
- **Load**: NumPy `load_from_*` / PyTorch `load_from_*` / Triton `load_from_*` / CUDA flat assign
- **PyTorch↔Triton**: direct load via compatible `save_as_numpy()` / `load_from_numpy_dict()`
- **NumPy↔NumPy**: direct load (identical API)
- **NumPy↔CUDA**: direct load via shared flat format (no conversion needed)

---

## Project Structure

```
project/
├── shared/           # Shared: config, constants, tokenizer, dataset, checkpoint
│   ├── config.py     # ModelConfig (torch+numpy), NpModelConfig (np)
│   ├── constants.py  # Parameter name constants for all backends
│   ├── tokenizer.py  # Byte-level BPE tokenizer
│   ├── dataset.py    # TokenizedDataset (torch.Dataset + np.ndarray)
│   └── checkpoint.py # save/restore helpers (ckpt.npz)
├── impl/
│   ├── _np/          # NumberPy implementation (reference)
│   │   ├── model.py  # NumPyModel: Embedding → RMSNorm → RoPE → MultiQueryAttention → SwiGLUFeedForward → TransformerBlock → GatedResidualTransformer
│   │   ├── layers.py # All layer components
│   │   ├── loss.py   # CrossEntropyLoss → NLLLoss
│   │   └── optimizer.py # AdamW
│   ├── _torch/       # PyTorch implementation
│   │   ├── model.py  # TorchModel wrapper
│   │   └── training.py # PyTorch-specific training utilities
│   ├── _triton/      # Triton GPU kernels
│   ├── _cuda/        # CUDA bare-metal
│   └── cli.py        # CLI entry point
└── tests/
    ├── unit/         # Unit tests per backend
    ├── cross_backend/# Cross-backend parity tests
    └── integration/  # E2E tests
```

---

## Implementation Status

| Backend | Components | Tests | Status |
|---------|-----------|-------|--------|
| NumPy | Complete (Embedding, RMSNorm, RoPE, MHA, SwiGLU FFN, GQA, MoE, GatedResidual, TransformerBlock, GatedResidualTransformer) | All pass | ✅ Complete |
| PyTorch | Complete (all layers match NumPy) | All pass | ✅ Complete |
| Triton | Complete (all kernels match NumPy) | All pass | ✅ Complete |
| CUDA | Complete (all kernels compiled + execute) | All pass | ✅ Complete |

### NumPy Layer Summary
- **Embedding**: `vocab_size × embed_dim` trainable lookup table, no bias
- **RMSNorm**: per-dim gamma `× input × scale`, no bias, fused bias add supported
- **RoPE**: 2D orthogonal rotation for key/query per head, fused with QK mult
- **MHA**: Multi-Head Attention — supports KV cache (multi-level) and non-cached inference
- **SwiGLU FFN**: Gated linear unit variant with SiLU activation
- **MoE**: Mixture of Experts — multiple feed-forward expert sub-layers, gated routing
- **GQA**: Grouped-Query Attention — groups query heads into query groups for KV cache efficiency
- **Gated Residual**: Residual connection with adaptive scalar gating (learnable, not sigmoid)
- **TransformerBlock**: Single layer — RMSNorm → MHA → Gated Residual → RMSNorm → SwiGLU FFN → Gated Residual
- **GatedResidualTransformer**: N-layer stack with precomputed RoPE, token embedding, layer normalization (final_ln), and output projection (lm_head)

---

## Cross-Backend Parity

### Training Parity (Weight Diff After 1 Iteration)

| Comparison | Weight Diff | Inference Diff |
|-----------|------------|----------------|
| NumPy vs PyTorch | `~1e-4` | `~1e-5` |
| NumPy vs Triton | `~1e-2` | `~1e-1` |
| NumPy vs CUDA | `~1e-2` | `~1e-1` |

### Inference Parity (Same Weights, Same Input)

| Comparison | Token Diff | Max Prob Diff |
|-----------|-----------|---------------|
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
|-----------|-------|
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
5. **Flat checkpoint format** — All backends save/load `ckpt.npz` as a flat dict, enabling cross-backend transfer
6. **Gated residual connections** — Not sigmoid-based; learnable scalar gates provide smoother training dynamics
7. **Multi-level KV caching** — Configurable cache length for efficient training vs inference
8. **PyTorch nn.Module wrapper** — PyTorch/Triton/CUDA models are `nn.Module` instances, enabling gradient-based training via `.parameters()`
