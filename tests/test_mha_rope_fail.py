from __future__ import annotations

import numpy as np
from model.attention import MultiHeadAttention


class TestMHAWithRoPE:
    def test_mha_forward_accepts_use_rope_param(self):
        mha = MultiHeadAttention(embed_dim=32, num_heads=4)
        x = np.random.randn(2, 5, 32).astype(np.float64)
        out, cache = mha.forward(x, use_rope=True)
        assert out.shape == (2, 5, 32)
        assert np.isfinite(out).all()

    def test_mha_with_rope_differs_from_without(self):
        np.random.seed(99)
        mha0 = MultiHeadAttention(64, 8)
        mha1 = MultiHeadAttention(64, 8)

        for key in ["W_q", "W_k", "W_v", "W_o"]:
            getattr(mha1, key)[...] = getattr(mha0, key)[:]

        x = np.random.randn(2, 8, 64).astype(np.float64)
        out0, _ = mha0.forward(x.copy(), use_rope=False)
        out1, _ = mha1.forward(x.copy(), use_rope=True)

        assert not np.allclose(out0, out1, atol=1e-6), "RoPE should change output"
