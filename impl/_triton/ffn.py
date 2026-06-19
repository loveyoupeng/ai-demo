"""SwiGLU Feed-Forward Network — PyTorch matmul-based gating.

SwiGLU ( gated SiLU activation) is the FFN activation used in modern
large language models (LLaMA, PaLM, Gemma). It replaces the standard
ReLU/gated activations with a more expressive gating mechanism.

Architecture
------------
Unlike attention or Layernorm, SwiGLU is NOT implemented as a Triton
kernel because it consists of highly-optimized PyTorch matmuls which
already use cuBLAS. Writing a Triton kernel for three matmuls + activations
would add overhead for what is essentially a memory-bound operation
(weights are large and don't fit in shared memory efficiently).

Instead of fusing, we keep it as a PyTorch native operation where:
- Matmuls use cuBLAS (GPU-optimized, shared library)
- Silu activation is element-wise (negligible cost)
- The bottleneck is memory bandwidth, not compute

SwiGLU formula
--------------
For input x ∈ R^D, hidden dimension ff_dim, and learnable weights:

  gate = SiLU(x · W1)        → vector of size ff_dim
  proj = x · W3              → vector of size ff_dim
  gated = gate ⊙ proj        → element-wise multiply, size ff_dim
  output = gated · W2        → vector of size D

where:
  W1 ∈ R^{D×ff_dim}, W3 ∈ R^{D×ff_dim}, W2 ∈ R^{ff_dim×D}
  SiLU(x) = x / (1 + exp(-x))  — smooth gating between 0 and x


Why SiLU gating?
----------------
Compared to ReLU gating (used in some architectures) or ELU gating:

  1. Smooth: SiLU is C^1 continuous everywhere (no hard zero)
     → Better gradient flow, no dead neurons
  
  2. Learned gating: The gate signal comes from x itself (xW1),
     not from a learnable threshold (as in ReLU)
     → Adaptive: different dimensions gate differently based on input
  
  3. Bounded below: SiLU(x) ≥ 0 for x ≥ 0, and for x < 0,
     SiLU(x) → 0 with a soft decay (not hard zero cut-off)
  
  4. Near-identity for large x: SiLU(x) ≈ x when x is large
     → Gradients don't vanish for high-magnitude activations

Why three weight matrices (W1, W3, W2)?
-----------------------------------------
Standard FFN:   output = ReLU(x·W1) · W2
SwiGLU FFN:     output = SiLU(x·W1) ⊙ (x·W3) · W2

The key difference: instead of one projection → activation → output,
SwiGLU has TWO projections (W1 → SiLU, W3 → identity) multiplied
element-wise, then projected to output. This gives:

  - More parameters for same hidden dim (3M vs 2M, where M=ff_dim*D)
  - Better expressivity (the gate is input-dependent, not fixed)
  - Improved gradient flow through multiple paths


Memory access pattern (PyTorch + cuBLAS)
----------------------------------------
Each matmul is a GEMM (General Matrix Multiply):
  x [B*S, D] @ W1 [D, ff_dim] → gate [B*S, ff_dim]
  x [B*S, D] @ W3 [D, ff_dim] → proj [B*S, ff_dim]
  gated [B*S, ff_dim] @ W2 [ff_dim, D] → out [B*S, D]

cuBLAS loads weight matrices into shared memory (L2 cache), then
computes in a tiled fashion for maximum throughput. The activation
(silu) is element-wise and runs on the host (CPU) or as a simple
GPU kernel with coalesced memory access.


Why NOT a Triton kernel?
------------------------
Three arguments against custom Triton kernel:

  1. Memory-bound: Matmuls are already bandwidth-limited.
     Fusing them into one kernel doesn't save memory access
     (each matmul reads W, writes output — that's the bottleneck).

  2. cuBLAS quality: NVIDIA's cuBLAS matmul implementation is
     hand-optimized at the assembly level for every GPU architecture.
     A Python-written Triton kernel can't match this performance.

  3. Kernel launch overhead: Three separate matmuls → three cuBLAS calls.
     Fusing them into one kernel would remove kernel launch overhead
     (microseconds) but add compute overhead (no shared memory reuse).
     The tradeoff is negative for large matrices.

When WOULD we use a Triton kernel for FFN?
  - When ff_dim is small (< 128) and matmuls are tiny GEMMs
    (cuBLAS overhead dominates)
  - When using INT8/FP8 quantization (requires custom kernels)
  - When implementing novel activation functions not in PyTorch

For our purposes (float32/float64, large ff_dim), PyTorch matmuls
are optimal.

Reference
---------
Shazeer, "GLU Variants Improve Transformer" (2020)
https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import torch


def swiglu_ffn(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU Feed-Forward Network with smooth gating.

    Computes: output = (SiLU(xW1) ⊙ xW3) @ W2

    This is a standard PyTorch implementation using cuBLAS matmuls
    (not a custom Triton kernel). See the module docstring for why.

    Forward pass
    ------------
    Step 1: Gate path — element-wise activation
      gate = x @ W1  → (..., ff_dim)
      gate = SiLU(gate) → (..., ff_dim)
      # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
      # Produces smooth gating: [0, ∞) for positive, (-∞, 0) for negative

    Step 2: Parallel projection — identity path
      proj = x @ W3  → (..., ff_dim)
      # W3 is independent of W1, providing a parallel signal

    Step 3: Element-wise gating
      gated = gate * proj  → (..., ff_dim)
      # Multiplication: only elements where both gate AND proj are
      # positive/strongly negative propagate. Other elements are suppressed.

    Step 4: Output projection
      out = gated @ W2  → (..., D)
      # Projects back to original dimension

    Dimensions
    ----------
    Input:       x   ~  (..., D)
    W1:         w1   ~  (D, ff_dim)       — gate projection
    W3:         w3   ~  (D, ff_dim)       — proj path
    W2:         w2   ~  (ff_dim, D)       — output projection

    Output:      out ~   (..., D)
                 gate ~   (..., ff_dim)
                 proj ~   (..., ff_dim)
                 gated ~  (..., ff_dim)

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
        Input activations. Can be any shape where the last
        dimension matches D (e.g., (B,S,D) or (B*S,D)).
    w1 : torch.Tensor, shape (D, ff_dim)
        Gate projection weights. The activation applied to
        xW1 (SiLU) forms the gating signal.
    w3 : torch.Tensor, shape (D, ff_dim)
        Parallel projection weights. xW3 is multiplied with
        SiLU(xW1) — W3 is independent from W1.
    w2 : torch.Tensor, shape (ff_dim, D)
        Output projection weights. Projects the gated
        (..., ff_dim) back to (..., D).

    Returns
    -------
    torch.Tensor, shape (..., D)
        Gated feedforward output. Same shape as input x but
        last dimension unchanged (D in, D out).

    Notes
    -----
    Memory layout: All tensors are row-major (C-order).
    Batch dimensions (...) can vary — the function supports
    arbitrary batch shapes via PyTorch's broadcasting.

    Activation: SiLU is the native torch.nn.functional.silu() which
    uses the formula x / (1 + exp(-x)). For large negative x, this
    approaches 0 (suppressing the gate). For large positive x, this
    approaches x (near-identity, passing the activation through).

    Example
    -------
    >>> import torch
    >>> x = torch.randn(2, 8, 64, device='cuda')        # (B, S, D)
    >>> w1 = torch.randn(64, 256, device='cuda')         # (D, ff_dim)
    >>> w3 = torch.randn(64, 256, device='cuda')         # (D, ff_dim)
    >>> w2 = torch.randn(256, 64, device='cuda')         # (ff_dim, D)
    >>> out = swiglu_ffn(x, w1, w3, w2)
    >>> out.shape
    torch.Size([2, 8, 64])

    Gradient flow
    -------------
    Gradients flow through FOUR paths:
      1. dL/dx ← dL/dout ← ... ← dL/dgated ← dL/dproj (via W2, W3)
      2. dL/dx ← dL/dout ← ... ← dL/dgated ← dL/dgate (via W2, W1)
      3. dL/dW2 ← gated^T @ dL/dout (outer product, sparse)
      4. dL/dW1, dL/dW3 ← gradients through SiLU gates

    The dual-path architecture (W1→gate AND W3→proj) provides
    richer gradient flow than single-path FFNs.
    """
    # ── Step 1: Gate path ─────────────────────────────────────
    # x:       (..., D)
    # x @ W1:  (..., ff_dim) — project to inner dim
    # SiLU:    (..., ff_dim) — element-wise activation (smooth gate)
    #
    # SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    # For x ≥ 0: SiLU(x) ∈ [0, x]  (passes positive values, scales them)
    # For x <  0: SiLU(x) ∈ (-sigmoid(0^+), 0) → 0 (suppresses negatives)
    gate = torch.nn.functional.silu(x @ w1)  # (..., ff_dim)

    # ── Step 2: Parallel projection ───────────────────────────
    # x:       (..., D)
    # x @ W3:  (..., ff_dim) — independent path
    #
    # W3 is NOT the same as W1. This means the gate signal (W1)
    # and the projection signal (W3) learn different features.
    # The element-wise multiply in step 3 determines which
    # features are "activated" by both paths simultaneously.
    proj = x @ w3  # (..., ff_dim)

    # ── Step 3: Element-wise gating ──────────────────────────
    # gate:   (..., ff_dim)  — from SiLU(xW1)
    # proj:   (..., ff_dim)  — from xW3
    # out:    (..., ff_dim)  — both must be non-zero
    #
    # Only dimensions where BOTH gate AND proj are strong propagate.
    # This is more selective than ReLU (which only checks one path).
    gated = gate * proj  # (..., ff_dim)

    # ── Step 4: Output projection ────────────────────────────
    # gated:  (..., ff_dim) — gated signals
    # @ W2:   (..., D)      — project back to embedding dim
    return gated @ w2  # (..., D)