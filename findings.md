# Findings & Decisions

## Requirements

### Core Architecture
- Decoder-only text-to-text transformer (MHA)
- Configurable: layers, heads, dimensions, context_length
- RoPE position encoding (configurable)
- GQA (Grouped-Query Attention) – opt-in config toggle
- MoE (Mixture of Experts) – configurable num_experts
- KV Cache: Naive (full precision) + TurboQuant (1-bit compressed)

### Implementations (4 backends, equivalent behavior)
1. **NumPy** – Learning-focused, heavy comments, mathematical explanations
2. **PyTorch** – Production-ready, proper OOP, clean interfaces
3. **Triton** – GPU kernel optimization, learn custom kernel patterns
4. **CUDA** – Lowest-level GPU programming via nvidia/cuda-python

### Pipeline
- Train from tiny dataset (TinyStories)
- Save/load checkpoints in shared format
- Inference engine with autoregressive generation
- CLI tool for interactive text input/ouput

### Quality Standards
- TDD approach (tests guide development)
- pyright + ruff free (code and warnings)
- OOP designed, Python best practices
- Cross-backend equivalent behavior verified by tests

## Research Findings

### Dataset
- **TinyStories**: ~8MB, simple English stories, AI-generated but clean
- Available free on HuggingFace (`allenai/tinystories`)
- Vocabulary size manageable for demo (smaller than Wikipedia datasets)

### Tokenizer
- BytePair Encoding (BPE) as default – standard for LLMs
- Character-level tokenizer as fallback for simplicity
- Configurable vocab_size: 512, 1024, 4096

### RoPE (Rotary Positional Embeddings)
- Introduced in Yang et al. (2021)
- Injects position info into Q and K via rotation matrices
- Configurable: rope_dim (can be full d_head or partial)
- Works with GQA naturally

### MoE (Mixture of Experts)
- Top-k routing (default k=2) – select top-k experts per token
- All-gate: use all experts
- Load balancing loss: Optional, encourages even expert usage
- Each expert = 2-layer FeedForward with GELU/SiGLU

### GQA (Grouped-Query Attention)
- Multiple query heads share the same KV head
- Toggle: n_groups = 1 for GQA, n_heads for self-attention
- Intermediate: e.g., 8 heads, 2 groups = 2 KV heads shared by 4 query heads each

### KV Cache
- **Naive**: Full fp32/fp16 KV tensors, simple indexing by position
- **TurboQuant** (Google): 1-bit compact KV cache
  - KV values quantized to single bit per value
  - Calibration step to find scaling factors
  - Reduces memory by ~32x for KV storage
  - Configurable: block_size, quant_type (1-bit, 2-bit, 4-bit)

### Checkpoint Format
- Binary JSON format (.npz) for tensor data
- Separate JSON for model config, hyperparameters, vocab
- Compatible: NumPy can load torch checkpoints and vice versa
- Seed stored with checkpoint for reproducibility

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| NumPy first, then torch/triton/cuda | NumPy is the "source of truth" – everyone learns from it first |
| Shared config module | Single place to change architecture → changes all backends |
| Shared tokenizer + dataset | Same training data is crucial for cross-backend equivalence |
| BPE tokenizer + char fallback | Industry standard, but char for very small demos |
| Default: CrossEntropy + Adam | Standard for LLM training, easy to understand |
| Top-2 MoE routing | Default 2 experts per token – enough capacity, not too sparse |
| TurboQuant: 1-bit KV | Google's approach, dramatic memory savings for long sequences |
| Checkpoint shared format | Any backend trains → any backend infers |

## Validation Strategy

| Scenario | Test | Method |
|----------|------|--------|
| Standalone layer parity | NumPy vs PyTorch forward | rtol=1e-4, atol=1e-4 |
| Single-layer backward parity | Full grad chain per layer | rtol=1e-3, atol=1e-3 |
| Full model checkpoint equivalence | Same input → same output | max diff < 1e-5 |
| Training convergence parity | Same loss curve shape | qualitative comparison |
| Inference output equivalence | Same prompt → same tokens | exact string match |
| Cross-format checkpoint | Torch saves → NumPy loads | roundtrip test |

## Resources

- TinyStories: `huggingface.co/allenai/tinystories`
- RoPE: "Attention Is All You Need" + RoPE original paper (Su et al. 2021)
- GQA: "GQA: Generalized Query Attention" (Du et al. 2022)
- MoE: "Mixtral of Experts" (Jiang et al. 2024), Switch Transformer (Fedus et al. 2021)
- TurboQuant: Google research on KV cache quantization (1-bit compression)
