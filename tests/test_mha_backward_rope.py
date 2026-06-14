from __future__ import annotations

import numpy as np
from model.attention import MultiHeadAttention


def test_mha_backward_gradient_numerical():
    """Numerical gradient of MHA output w.r.t. x should match analytical with use_rope=True."""
    np.random.seed(0)
    B, L, D, H = 2, 4, 32, 4

    mha = MultiHeadAttention(D, H)
    x = np.random.randn(B, L, D).astype(np.float64)
    mask = np.tril(np.ones((L, L)))

    out, cache = mha.forward(x, mask=mask, use_rope=True)
    d_out = np.ones_like(out)

    # Numerical gradient w.r.t. x
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

    # Analytical gradient — backward uses positional arguments (mask, Q, K, V, attn_weights, context)
    dx, _ = mha.backward(
        x,
        d_out,
        mask,
        np.asarray(cache["Q"]),
        np.asarray(cache["K"]),
        np.asarray(cache["V"]),
        np.asarray(cache["attn_weights"]),
        np.asarray(cache["context"]),
    )

    print("Numerical gradient (first 3 flat):", dx_numerical.flatten()[:3])
    print("Analytical gradient (first 3 flat):", dx.flatten()[:3])
    max_abs_err = np.max(np.abs(dx - dx_numerical))
    max_rel_err = np.max(np.abs(dx - dx_numerical) / (np.abs(dx_numerical) + 1e-8))
    print(f"Max abs error: {max_abs_err:.6e}")
    print(f"Max rel error: {max_rel_err:.6e}")

    assert np.allclose(dx, dx_numerical, rtol=5e-2, atol=5e-2), (
        f"dx gradient mismatch: max_abs_err={max_abs_err:.2e}"
    )


def test_mha_backward_gradient_numerical_false():
    """Same test with use_rope=False — should match (baseline)."""
    np.random.seed(0)
    B, L, D, H = 2, 4, 32, 4

    mha = MultiHeadAttention(D, H)
    x = np.random.randn(B, L, D).astype(np.float64)
    mask = np.tril(np.ones((L, L)))

    out, cache = mha.forward(x, mask=mask, use_rope=False)
    d_out = np.ones_like(out)

    eps = 1e-5
    dx_numerical = np.zeros_like(x)
    for bi in range(B):
        for li in range(L):
            for di in range(D):
                orig = x[bi, li, di]
                x[bi, li, di] = orig + eps
                out_up, _ = mha.forward(x, mask=mask.copy(), use_rope=False)
                x[bi, li, di] = orig - eps
                out_down, _ = mha.forward(x, mask=mask.copy(), use_rope=False)
                x[bi, li, di] = orig
                dx_numerical[bi, li, di] = (out_up - out_down).sum() / (2 * eps)

    dx, _ = mha.backward(
        x,
        d_out,
        mask,
        np.asarray(cache["Q"]),
        np.asarray(cache["K"]),
        np.asarray(cache["V"]),
        np.asarray(cache["attn_weights"]),
        np.asarray(cache["context"]),
    )

    print("Baseline (no RoPE) max_abs_err:", np.max(np.abs(dx - dx_numerical)))

    assert np.allclose(dx, dx_numerical, rtol=5e-2, atol=5e-2)
