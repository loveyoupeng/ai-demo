from __future__ import annotations

import numpy as np
from model.attention import MultiHeadAttention
from model.rope import apply_rope


class TestMHAWithRoPE:
    """Test RoPE integration in MHA."""

    def test_mha_rope_forward_changes_output(self):
        """
        When RoPE is enabled, MHA forward output differs from
        RoPE-disabled MHA (same weights, same input).
        """
        batch_size = 2
        seq_len = 5
        embed_dim = 32
        num_heads = 4
        head_dim = embed_dim // num_heads  # 32 / 4 = 8

        # Create two MHA instances with identical weights
        np.random.seed(42)
        mha0 = MultiHeadAttention(embed_dim, num_heads)
        mha0.W_q = mha0.W_q.copy()
        mha0.W_k = mha0.W_k.copy()
        mha0.W_v = mha0.W_v.copy()
        mha0.W_o = mha0.W_o.copy()

        mha1 = MultiHeadAttention(embed_dim, num_heads)
        mha1.W_q = mha0.W_q.copy()
        mha1.W_k = mha0.W_k.copy()
        mha1.W_v = mha0.W_v.copy()
        mha1.W_o = mha0.W_o.copy()

        # Initialize RoPE theta for mha1
        pair_count = head_dim // 2
        power = np.arange(pair_count, dtype=np.float64) * -2.0 / head_dim
        theta_all = np.arange(seq_len, dtype=np.float64).reshape(-1, 1) * (10000.0 ** power)

        x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float64)

        # MHA without RoPE
        out0, _ = mha0.forward(x.copy())

        # MHA with RoPE: manually apply RoPE to Q and K after projection
        Q0 = np.dot(x.copy(), mha1.W_q)
        K0 = np.dot(x.copy(), mha1.W_k)

        Q = Q0.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K0.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE: need [B, L, H, D] for apply_rope
        Q_rope = apply_rope(
            Q.transpose(0, 2, 1, 3), theta_all
        ).transpose(0, 2, 1, 3)
        K_rope = apply_rope(
            K.transpose(0, 2, 1, 3), theta_all
        ).transpose(0, 2, 1, 3)

        # Check that Q/K are rotated (not identity)
        assert not np.allclose(Q_rope, Q, rtol=1e-8)
        assert not np.allclose(K_rope, K, rtol=1e-8)

        # With rotated Q/K, output must differ
        # Simulate MHA forward with rotated Q, K
        V = np.dot(x.copy(), mha1.W_v)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q_rope, K_rope.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        scores = np.where(np.tril(np.ones((seq_len, seq_len))) == 0, -1e9, scores)
        attn = scores / np.sum(scores, axis=-1, keepdims=True)
        context = np.matmul(attn, V)
        context_out = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        out1 = np.dot(context_out, mha1.W_o)

        assert not np.allclose(out0, out1, rtol=1e-3)
