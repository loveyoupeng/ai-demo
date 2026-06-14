from __future__ import annotations

import numpy as np
from model.attention import MultiHeadAttention


def test_mha_rope_backward_numerical():
    """Compare numerical vs analytical gradient for MHA with RoPE."""
    np.random.seed(0)
    B, L, D, H = 2, 4, 32, 4
    dk = D // H  # 8

    mha = MultiHeadAttention(D, H)
    x = np.random.randn(B, L, D).astype(np.float64)
    mask = np.tril(np.ones((L, L)))

    out, cache = mha.forward(x, mask=mask, use_rope=True)
    d_out = np.ones_like(out)

    # Numerical gradient
    eps = 1e-5
    dx_numerical = np.zeros_like(x)
    for bi in range(B):
        for li in range(L):
            for di in range(D):
                orig = x[bi, li, di]
                
                x[bi, li, di] = orig + eps
                out_up, _ = mha.forward(x, mask=mask.copy(), use_rope=True)
                
                x[bi, li, di] = orig - eps
                out_down, _ = mha.forward(x, mask=mask.copy(), use_rope=True)
                
                x[bi, li, di] = orig
                dx_numerical[bi, li, di] = (out_up - out_down).sum() / (2 * eps)

    # Analytical gradient - backward uses positional (mask, Q, K, V, attn_weights, context)
    dx, _ = mha.backward(x, d_out, mask, cache["Q"], cache["K"], cache["V"], cache["attn_weights"], cache["context"])

    print("Numerical gradient (first 3 flat):", dx_numerical.flatten()[:3])
    print("Analytical gradient (first 3 flat):", dx.flatten()[:3])
    print(f"Max absolute error: {np.max(np.abs(dx - dx_numerical))}")
    print(f"Max relative error: {np.max(np.abs(dx - dx_numerical) / (np.abs(dx_numerical) + 1e-8))}")

    assert np.allclose(dx, dx_numerical, rtol=1e-1, atol=1e-1), (
        "Analytical gradient doesn't match numerical gradient"
    )
    print("TEST PASSED - gradients match")

if __name__ == "__main__":
    test_mha_rope_backward_numerical()
