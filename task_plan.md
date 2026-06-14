# Task Plan: Decoder-Only Transformer Learning Project

## Goal
Build a fully functional decoder-only transformer LLM from scratch in 4 implementations (NumPy, PyTorch, Triton, CUDA) with identical behavior, trained on TinyStories, featuring RoPE, MHA, MoE, GQA, and multi-level KV caching for educational purposes.

## Current Phase
Phase 1 - Planning Review

## Phases

### Phase 0: Project Initialization
- [ ] Verify empty project structure after cleanup
- [ ] Set up pyproject.toml with dependencies
- [ ] Create directory structure
- [ ] Set up CI/lint/config files
- [ ] Confirm plan with user

### Phase 1: Shared Foundations & Testing Infrastructure
- [ ] Create config module (all hyperparameters)
- [ ] Create tokenizer (BytePair or CharLevel)
- [ ] Create dataset loader (TinyStories)
- [ ] Create shared base model interface
- [ ] Set up cross-backend test infrastructure
- [ ] Create parameter naming conventions & constants

### Phase 2: NumPy Implementation (Learning-Focused)
- [ ] Core layers: Embedding, LayerNorm, RMSNorm, SiLU, GeLU
- [ ] RoPE position encoding (configurable)
- [ ] MHA (configurable: heads, dim, GQA toggle)
- [ ] FeedForward + MoE (configurable: num_experts, topk)
- [ ] TransformerBlock + DecoderStack
- [ ] Full model class with forward/backward
- [ ] Loss functions: CrossEntropy
- [ ] Optimizers: SGD, Adam
- [ ] KV Cache: Naive implementation
- [ ] KV Cache: TurboQuant implementation
- [ ] Training loop (full)
- [ ] Inference engine (autoregressive)
- [ ] Checkpoint save/load
- [ ] CLI for training and inference
- [ ] Comprehensive unit tests
- [ ] Cross-backend reference tests

### Phase 3: PyTorch Implementation (Production-Ready)
- [ ] All layers using PyTorch nn.Module with same interfaces
- [ ] Same forward/backward behavior as NumPy
- [ ] Same KV cache, training loop, inference
- [ ] Proper docstrings for production use
- [ ] Cross-backend parity tests
- [ ] Performance benchmarks comparison

### Phase 4: Triton Implementation (GPU Kernel Optimization)
- [ ] Compute kernels: LayerNorm, attention, MoE routing, activation
- [ ] Same model architecture using Triton kernels
- [ ] Inference with triton-accelerated kernels
- [ ] Cross-backend parity tests
- [ ] Profiling vs NumPy/PyTorch comparison

### Phase 5: CUDA Implementation (Lowest Level)
- [ ] Compute kernels via nvidia/cuda-python bindings
- [ ] Same model architecture using CUDA kernels
- [ ] Inference with CUDA-accelerated kernels
- [ ] Cross-backend parity tests
- [ ] Performance benchmarks

### Phase 6: Integration & Verification
- [ ] Train model on TinyStories using each backend
- [ ] Save/load checkpoint cross-validation
- [ ] Generate identical outputs for same query across backends
- [ ] Final e2e verification script

## Key Questions
1. ~~Tokenizer choice for TinyStories~~ (confirmed: BytePair + Char fallback)
2. ~~Dataset source~~ (confirmed: TinyStories, ~8MB from HuggingFace)
3. ~~KV cache approach~~ (confirmed: naive full-precision + TurboQuant 1-bit)
4. ~~GQA~~ (confirmed: opt-in, toggle via config)
5. MoE: top-k (default 2) or all experts?
6. Training: which loss + optimizer? default?
7. Project structure: shared code vs per-backend standalone?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| NumPy first, then torch/triton/cuda | Learning path; NumPy is reference |
| TinyStories dataset | Small, clean, ideal for demo |
| Shared config + tokenizer | Single source of truth across backends |
| TurboQuant for KV | Google research, 1-bit compression |
| All backends produce identical results | Deterministic with same seed |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| N/A | - | Clean slate, no errors yet |

## Notes
- Cleaned existing codebase to start fresh
- Planning files created for tracking progress
- Awaiting user approval before implementation begins
