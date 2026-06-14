from __future__ import annotations

import numpy as np

from model.rope import apply_rope, compute_theta, reverse_rope


class MultiHeadAttention:
    r"""
    Multi-Head Attention (MHA) mechanism.

    Intuition:
    MHA allows the model to jointly attend to information from different
    representation subspaces at different positions. Instead of one single
    attention pass, we split the embedding dimension into multiple "heads,"
    where each head can learn to focus on different aspects of the sequence
    (e.g., one head focuses on grammar, another on semantic meaning).

    Mathematical context:
    Single-head scaled dot-product attention:
    $$ \text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k})V $$

    Multi-head: project $X \in \mathbb{R}^{B \times L \times D}$ into $h$ heads
    using $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$.
    Output projected back via $W_O \in \mathbb{R}^{D \times D}$.

    Dimension tracking:
    - Input $x$: $[B, L, D]$
    - $Q, K, V$ (after projection): $[B, L, D]$
    - $Q, K, V$ (after head split): $[B, h, L, d_k]$
    - Scores: $[B, h, L, L]$
    - Attn weights: $[B, h, L, L]$
    - Context: $[B, h, L, d_k]$
    - Context output (after merge): $[B, L, D]$
    """

    def __init__(self, embed_dim: int, num_heads: int):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, and V
        # Shape: [Embed_Dim, Embed_Dim]
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01

        # Output projection [Embed_Dim, Embed_Dim]
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.01

        # Cache for KV values during inference (KV Cache)
        # Keyed by layer number (0, 1, 2, ...) for accumulation across steps
        self.kv_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_step: int = 0  # Track current step for cache accumulation

    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
        use_cache: bool = False,
        cache_idx: int | None = None,
        use_rope: bool = True,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len] (1 for keep, 0 for mask)
            use_cache: Whether to use/update KV cache
            cache_idx: Index of the current token for KV cache update
        Returns:
            output: [Batch, Seq_Len, Embed_Dim]
            cache: Dictionary containing intermediate values for backward pass
        """
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections
        # Q, K, V shape: [Batch, Seq_Len, Embed_Dim]
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 2. Split into multiple heads
        # Q, K, V shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Store RoPE theta for backward pass (reverse rotation of d_Q/d_K)
        rope_theta = None
        # Apply RoPE to Q and K (in [B, h, L, d] format via apply_rope which handles 4D)
        if use_rope:
            rope_theta = compute_theta(
                np.arange(seq_len, dtype=np.float64), self.head_dim
            )
            # apply_rope expects [B, L, H, D] format, Q/K are [B, H, L, D]
            Q = apply_rope(Q.transpose(0, 2, 1, 3), rope_theta).transpose(0, 2, 1, 3)
            K = apply_rope(K.transpose(0, 2, 1, 3), rope_theta).transpose(0, 2, 1, 3)
            self._forward_rope_theta = rope_theta

        # --- KV CACHE LOGIC (for autoregressive generation) ---
        if use_cache and cache_idx is not None:
            # Accumulate K/V across single-token inference steps.
            # When processing a single token at a time (the correct AR pattern),
            # each step contributes exactly 1 token to the K,V cache, so we can
            # safely concatenate the previous step's cache with the current one.
            # For multi-token inputs with cache enabled, the cache tracks all
            # input tokens as a single block without accumulating across calls.
            step = self._cache_step
            self._cache_step += 1
            if seq_len == 1 and step > 0 and (step - 1) in self.kv_cache:
                prev_K, prev_V = self.kv_cache[step - 1]
                # Concatenate along the sequence dimension (axis 2)
                # [Batch, Num_Heads, Prev_Seq_Len + 1, Head_Dim]
                K = np.concatenate([prev_K, K], axis=2)
                V = np.concatenate([prev_V, V], axis=2)
            self.kv_cache[step] = (K, V)
        # ----------------------

        # 3. Scaled Dot-Product Attention
        # Scores shape: [Batch, Num_Heads, Q_Seq_Len, K_Seq_Len]
        d_k = self.head_dim
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        # 4. Apply causal mask if provided.
        # In KV cache mode, mask is [Q_seq, K_seq] where K_seq includes cached.
        # If mask was built for the full sequence it may be oversized; we clip
        # it to [Q_Len, K_Len] to match scores' final two dims.
        if mask is not None:
            q_len, k_len = scores.shape[-2], scores.shape[-1]
            if mask.shape[0] != q_len or mask.shape[1] != k_len:
                # Build a fresh causal mask matching the actual Q/K lengths
                causal = np.tril(np.ones((q_len, k_len)))
                mask = causal
            scores = np.where(mask == 0, -1e9, scores)

        # 5. Softmax to get attention weights: [Batch, Num_Heads, Q_Seq_Len, K_Seq_Len]
        attn_weights = self._softmax(scores, axis=-1)

        # 6. Weighted sum of values: [Batch, Num_Heads, Q_Seq_Len, Head_Dim]
        context = np.matmul(attn_weights, V)

        # 7. Concatenate heads: [Batch, Seq_Len, Embed_Dim]
        context_out = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )

        # 8. Final output projection
        output = np.dot(context_out, self.W_o)

        # Prepare cache for backward pass
        cache = {
            "Q": Q,
            "K": K,
            "V": V,
            "attn_weights": attn_weights,
            "context": context_out,
            "mask": mask,
            "rope_theta": rope_theta,
        }

        return output, cache

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Numerical stable softmax."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def backward(
        self,
        x: np.ndarray,
        d_out: np.ndarray,
        mask: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        K: np.ndarray | None = None,
        V: np.ndarray | None = None,
        attn_weights: np.ndarray | None = None,
        context: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Backward pass for Multi-Head Attention.

        Args:
            x: Input tensor $[B, L, D]$
            d_out: Gradient w.r.t. output $[B, L, D]$
            mask: Causal mask $[L, L]$
            Q: Cached query tensor $[B, h, L, d_k]$
            K: Cached key tensor $[B, h, L, d_k]$
            V: Cached value tensor $[B, h, L, d_k]$
            attn_weights: Cached attention weights $[B, h, L, L]$
            context: Cached context output $[B, L, D]$

        Returns:
            dx: Gradient w.r.t. input $x$ $[B, L, D]$
            grads: Dictionary of gradients for parameters:
                   {'W_q': $[D, D]$, 'W_k': $[D, D]$, 'W_v': $[D, D]$, 'W_o': $[D, D]$}
        """
        batch_size, seq_len, _ = x.shape

        if context is None:
            raise ValueError("context must be provided for backward pass")

        # 1. Gradient w.r.t. W_o and context
        # [Embed_Dim, Embed_Dim]
        d_W_o = np.dot(
            context.reshape(-1, self.embed_dim).T, d_out.reshape(-1, self.embed_dim)
        )
        d_context = np.dot(d_out, self.W_o.T)

        # 1. Gradient w.r.t. attn_weights and V
        # d_context_heads shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
        d_context_heads = d_context.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if V is None or attn_weights is None:
            raise ValueError("V and attn_weights must be provided for backward pass")

        # d_V shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
        d_V = np.matmul(attn_weights.transpose(0, 1, 3, 2), d_context_heads)
        # d_attn_weights shape: [Batch, Num_Heads, Seq_Len, Seq_Len]
        d_attn_weights = np.matmul(d_context_heads, V.transpose(0, 1, 3, 2))

        # 3. Gradient w.r.t. scores (after softmax)
        d_scores = attn_weights * (
            d_attn_weights
            - np.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True)
        )

        # 4. Apply mask gradient
        if mask is not None:
            d_scores = d_scores * mask

        # 5. Gradient w.r.t. Q and K
        d_scores = d_scores * np.sqrt(self.head_dim)
        if Q is None or K is None:
            raise ValueError("Q and K must be provided for backward pass")
        d_Q = np.matmul(d_scores, K)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        # 5b. Reverse-rotate d_Q/d_K: these are ∂L/∂Q_rotated, need ∂L/∂Q_raw
        if getattr(self, "_forward_rope_theta", None) is not None:
            d_Q = reverse_rope(d_Q.transpose(0, 2, 1, 3), self._forward_rope_theta).reshape(
                batch_size, seq_len, self.embed_dim
            )
            d_K = reverse_rope(d_K.transpose(0, 2, 1, 3), self._forward_rope_theta).reshape(
                batch_size, seq_len, self.embed_dim
            )
        else:
            d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
            d_K = d_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # 7. Gradients for W_q, W_k, W_v
        # d_W_q, d_W_k, d_W_v shape: [Embed_Dim, Embed_Dim]
        d_W_q = np.dot(x.reshape(-1, self.embed_dim).T, d_Q.reshape(-1, self.embed_dim))
        d_W_k = np.dot(x.reshape(-1, self.embed_dim).T, d_K.reshape(-1, self.embed_dim))
        d_W_v = np.dot(x.reshape(-1, self.embed_dim).T, d_V.reshape(-1, self.embed_dim))

        # 8. Gradient w.r.t. input x
        # [Batch, Seq_Len, Embed_Dim]
        dx = np.dot(d_Q, self.W_q.T) + np.dot(d_K, self.W_k.T) + np.dot(d_V, self.W_v.T)

        # Store gradients
        self.grad_W_q = d_W_q
        self.grad_W_k = d_W_k
        self.grad_W_v = d_W_v
        self.grad_W_o = d_W_o

        grads = {"W_q": d_W_q, "W_k": d_W_k, "W_v": d_W_v, "W_o": d_W_o}
        return dx, grads

    def get_params(self) -> dict[str, np.ndarray]:
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            if k == "W_q":
                self.W_q = v
            elif k == "W_k":
                self.W_k = v
            elif k == "W_v":
                self.W_v = v
            elif k == "W_o":
                self.W_o = v

    def get_grads(self) -> dict[str, np.ndarray]:
        return {
            "W_q": self.grad_W_q,
            "W_k": self.grad_W_k,
            "W_v": self.grad_W_v,
            "W_o": self.grad_W_o,
        }
