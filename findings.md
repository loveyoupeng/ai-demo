# Findings & Decisions

## Key Findings

### GPU Initialization
- `torch.empty()` → uninitialized memory → garbage values → NaN during training
- Always use `torch.zeros()`, `torch.ones()`, or `torch.nn.init.*()` for weight initialization
- CUDA model uses `nn.init.normal_()` (fixed from `torch.empty()`)

### Resource Conflicts
- nvgpu requires exclusive GPU access; parallel training causes `cudaErrorIllegalAddress`
- Sequential training resolves the issue

### Weight Drift
- PyTorch vs NumPy weight drift is **expected** — different RNG implementations diverge
- The drift is not a bug; it's a consequence of independent training

### MoE Layout
- NumPy: MoE params stored as `(n_experts, ...)` flat tensor
- CUDA: MoE params stored as separate `expert_W1[i]`, `expert_W3[i]`, `expert_B[i]`
- Both layouts produce identical outputs; only the flat vs nested structure differs

### MHA→RoPE Shape Mismatch
- MHA returns `(*batch, heads, ctx, dim)` where `ctx` is context window size
- RoPE needs `(*batch, ctx, embed_dim)`
- Fix: reshape `MHA_out` by flattening head+context, then reshape `ctx` to last dim
- Applies to both NumPy and PyTorch implementations

### Contiguous Tensors
- nvgpu requires `torch.contiguous()` before `copy_()` for non-contiguous views
- Added `.contiguous()` in `_load_from_numpy_dict()` in NumPyModel

## Training Notes
- Cross-entropy loss on token predictions: predict next token from context
- Gradient clipping at norm=1.0 prevents exploding gradients
- Cosine scheduler with warmup improves stability
- Batch size = context_length chunks for next-token prediction

## Platform Constraints (Jetson)
- 64GB unified memory: all backends share CPU+GPU memory
- Limited VRAM: small batch sizes required
- nvgcu driver: strict tensor contiguity requirements
