# Task Plan: Decoder-Only Transformer Learning Project

## Goal
Build a fully functional decoder-only transformer LLM in 4 equivalent implementations (NumPy, PyTorch, Triton, CUDA) with identical behavior, trained on TinyStories, for educational purposes.

## Architecture
```
+----------+
| Shared   | - config.json + env vars → model topology
| Backbone | - tokenizer, dataset, checkpoint
+-----+----+
      |
  +---+---+---+
  |   |   |   |
NumPy PyT Trit CUDA (4 backends, same structure)
+---+---+---+
      |
   ckpt.npz (flat dict, cross-backend compatible)
```

## Completed Phases
| Phase | Title | Status | Tests | Commits |
|-------|-------|--------|-------|---------|
| A | Shared Foundation | ✅ Done | 131 | (multiple) |
| B | NumPy Backend | ✅ Done | ~100 | (multiple) |
| C | PyTorch Backend | ✅ Done | 310 | 36 |
| C+ | E2E Scripts (train/infer) | ✅ Done | ~450 | (multiple) |
| 3++ | Post-Norm + Dropout | ✅ Done | ~421 | (multiple) |
| D | Platform Setup | ✅ Done | 440 | (multiple) |
| E | Triton Kernels | ✅ Done | 551 | (multiple) |
| F | CUDA Bare-Metal | ✅ Done | 121 | (multiple) |
| G | Weight Diff Tests | ✅ Done | 10 | (multiple) |

## CLI Commands
```bash
# Training
uv run python -m scripts.train --backend numpy/torch/triton/cuda

# Inference
uv run python -m scripts.infer --model resource/models/ckpt.npz --prompt "hello"

# Equivalence verification
uv run python -m scripts.verify_equivalence --fast

# Dataset download
uv run python -m scripts.download_tinystories
```

## Testing Commands
```bash
# All tests
uv run pytest tests/ -v

# Cross-backend parity
uv run pytest tests/cross_backend/ -v
```

## Remaining Work
| Item | Priority | Description |
|------|----------|-------------|
| Training on TinyStories | Medium | Currently synthetic data only |
| 4-way equivalence | Medium | Verify all backends produce same model |
| Clean up old phase plans | Low | Consolidate into design.md |

## Key Decisions
1. **4 backends, same topology** — All accept JSON config + env vars
2. **NumPy as truth benchmark** — Pure Python/numpy reference implementation
3. **Independent training** — Each backend trains independently with its own RNG
4. **Round-trip tests** — Save NumPy format → load into any backend → compare
5. **Flat checkpoint** — All save/load `ckpt.npz` as flat dict
6. **Gated residuals** — Learnable scalar gates, not sigmoid
7. **Multi-level KV** — Configurable cache length (training vs inference)
8. **PyTorch nn.Module** — PyTorch/Triton/CUDA models are nn.Module instances

## Platform Notes
- NVIDIA Jetson AGX Orin 64GB, JetPack 6.2.2, CUDA 12.6, ~20 TFLOPS
- Must use `torch.zeros()`/`torch.ones()` for tensor init (never `torch.empty()`)
- nvgpu driver requires contiguous tensors for `copy_()`
