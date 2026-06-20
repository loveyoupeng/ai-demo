"""Neural network module implementations in pure NumPy.

All forward passes accept numpy arrays and return numpy arrays.
Matrix dimensions are annotated in docstrings for clarity.
"""

import numpy as np


class Embedding:
    """Lookup table that maps token IDs to dense vectors.

    Parameters
    ----------
    None — forward pass takes weight explicitly for standalone testing.

    Forward
    -------
    input_ids : np.ndarray, shape (batch_size, seq_len)
        Token IDs, integers in [0, vocab_size).
    weight : np.ndarray, shape (vocab_size, embed_dim)
        Embedding lookup table.

    Returns
    -------
    out : np.ndarray, shape (batch_size, seq_len, embed_dim)
        Dense embedding vectors for each token.

    Notes
    -----
    This is a simple table lookup: out[b, s, :] = weight[tokens[b, s]].

    """

    def forward(
        self,
        input_ids: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        """Look up embeddings for each token ID.

        Parameters
        ----------
        input_ids : np.ndarray, shape (batch_size, seq_len)
            Token IDs to look up.
        weight : np.ndarray, shape (vocab_size, embed_dim)
            Embedding weight matrix.

        Returns
        -------
        np.ndarray, shape (batch_size, seq_len, embed_dim)
            Embedded vectors.

        Examples
        --------
        >>> import numpy as np
        >>> emb = Embedding()
        >>> tokens = np.array([[0, 1], [2, 3]], dtype=np.int32)
        >>> W = np.arange(12).reshape(4, 3).astype(np.float32)
        >>> out = emb.forward(tokens, W)
        >>> out.shape
        (2, 2, 3)
        >>> np.allclose(out[0, 0, :], W[0])
        True

        """
        # input_ids:  (batch_size, seq_len)
        # weight:     (vocab_size, embed_dim)
        # weight[input_ids]: broadcasts indexing to (batch_size, seq_len, embed_dim)
        return weight[input_ids]


class RMSNorm:
    """Root Mean Square Layer Normalization, a simplified LayerNorm variant.

    Parameters
    ----------
    None — forward pass takes input and gamma explicitly for standalone testing.

    Forward
    -------
    x : np.ndarray, shape (batch_size, seq_len, embed_dim)
        Input activations.
    gamma : np.ndarray, shape (embed_dim,)
        Learnable scale parameter.

    Returns
    -------
    out : np.ndarray, shape (batch_size, seq_len, embed_dim)
        Normalized output scaled by gamma.

    Notes
    -----
    RMSNorm formula:  out = x / sqrt(mean(x^2) + eps) * gamma
    where mean is taken over the last dimension (embed_dim).

    """

    def forward(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        """Apply RMS normalization.

        Parameters
        ----------
        x : np.ndarray, shape (..., embed_dim)
            Input activations (any leading batch dimensions).
        gamma : np.ndarray, shape (embed_dim,)
            Learnable scale.

        Returns
        -------
        np.ndarray, shape (..., embed_dim)
            RMS-normalized, scaled output.

        """
        # x:       (..., embed_dim)
        # mean(x^2): (..., 1) — mean over last dim
        # rms:     (..., 1)   — sqrt(mean(x^2) + eps)
        # output:  (..., embed_dim) — broadcast gamma over batch dims
        eps = 1e-6
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + eps  # (..., 1)
        return (x / rms) * gamma  # (..., embed_dim)


class SiLULayer:
    """Sigmoid Linear Unit (SiLU / Swish) activation: f(x) = x * sigmoid(x).

    Parameters
    ----------
    None — activation is stateless.

    Forward
    -------
    x : np.ndarray, shape (..., embed_dim)
        Input activations.

    Returns
    -------
    out : np.ndarray, shape (..., embed_dim)
        SiLU activation applied element-wise.

    Notes
    -----
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    Properties:
      - For large positive x: f(x) ≈ x (near-identity)
      - For large negative x: f(x) ≈ 0 (suppressed)
      - For x = 0: f(0) = 0
      - Smooth, non-monotonic gating that enables feature selection

    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply SiLU activation element-wise.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of any shape.

        Returns
        -------
        np.ndarray, same shape as x
            SiLU(x) = x * sigmoid(x).

        """
        # x:   (..., embed_dim)
        # sigmoid(x): (..., embed_dim) = 1 / (1 + exp(-x))
        # out: (..., embed_dim) = x * sigmoid(x)
        sigmoid_x = 1.0 / (1.0 + np.exp(-x))  # (..., embed_dim)
        return x * sigmoid_x  # (..., embed_dim)


class SwiGLUFFN:
    """SwiGLU Feedforward Network: SiLU(w1 @ x) * (w3 @ x) @ w2.

    A modern feedforward with gating that uses SiLU to provide
    smooth feature selection between w1 and w3 projections.

    Parameters
    ----------
    embed_dim : int
        Input/output dimension.
    ff_dim : int
        Intermediate (hidden) dimension.
    seed : int, optional
        Random seed for weight initialization (default 42).

    Forward
    -------
    x : np.ndarray, shape (batch_size, seq_len, embed_dim)
        Input activations.

    Returns
    -------
    out : np.ndarray, shape (batch_size, seq_len, embed_dim)
        Feedforward output.

    Notes
    -----
    SwiGLU formula:
      gate = SiLU(w1 @ x)          → (..., ff_dim)
      proj = w3 @ x                → (..., ff_dim)
      gated = gate * proj          → (..., ff_dim)  — element-wise
      out = gated @ w2             → (..., embed_dim)

    where:
      w1: (embed_dim, ff_dim)
      w3: (embed_dim, ff_dim)
      w2: (ff_dim, embed_dim)

    """

    def __init__(self, embed_dim: int, ff_dim: int, seed: int = 42) -> None:
        """Initialize SwiGLU weights.

        Parameters
        ----------
        embed_dim : int
            Input/output dimension.
        ff_dim : int
            Hidden dimension.
        seed : int
            Random seed for reproducibility.

        """
        rng = np.random.default_rng(seed)
        # Xavier initialization to keep activations at reasonable scale
        self.W1: np.ndarray = rng.uniform(
            -np.sqrt(6.0 / (embed_dim + ff_dim)),
            np.sqrt(6.0 / (embed_dim + ff_dim)),
            size=(embed_dim, ff_dim),
        ).astype(np.float32)
        self.W2: np.ndarray = rng.uniform(
            -np.sqrt(6.0 / (ff_dim + embed_dim)),
            np.sqrt(6.0 / (ff_dim + embed_dim)),
            size=(ff_dim, embed_dim),
        ).astype(np.float32)
        self.W3: np.ndarray = rng.uniform(
            -np.sqrt(6.0 / (embed_dim + ff_dim)),
            np.sqrt(6.0 / (embed_dim + ff_dim)),
            size=(embed_dim, ff_dim),
        ).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute SwiGLU feedforward with gating.

        Parameters
        ----------
        x : np.ndarray, shape (..., embed_dim)
            Input activations.

        Returns
        -------
        np.ndarray, shape (..., embed_dim)
            Gated feedforward output.

        """
        # x:              (..., embed_dim)
        # w1 @ x:         (..., ff_dim)     — first linear projection
        # SiLU(w1 @ x):   (..., ff_dim)     — smooth gating signal
        # w3 @ x:         (..., ff_dim)     — second linear projection (parallel)
        # gate * proj:    (..., ff_dim)     — element-wise gating
        # (gate * proj) @ w2: (..., embed_dim) — final linear projection
        gate = SiLULayer().forward(x @ self.W1)  # (..., ff_dim)
        proj = x @ self.W3  # (..., ff_dim)
        gated_output = gate * proj  # (..., ff_dim) — gating combines w1 and w3
        return gated_output @ self.W2  # (..., embed_dim)


class RoPE:
    """Rotary Positional Embedding — injects position via 2D rotation in key dimensions.

    Parameters
    ----------
    None — all parameters (freqs) are computed on-the-fly from position and head_dim.

    Forward
    -------
    x : np.ndarray, shape (batch_size, seq_len, n_heads, head_dim)
        Q or K tensor.
    position : np.ndarray, shape (seq_len,) or (batch_size, seq_len) or int
        Position indices for each sequence element.

    Returns
    -------
    out : np.ndarray, shape (batch_size, seq_len, n_heads, head_dim)
        Rotated Q or K with positional information embedded.

    Notes
    -----
    RoPE rotates each (odd, even) pair of dimensions by a position-dependent angle:
      x_m' = x_m * cos(mθ) - x_{m+1} * sin(mθ)
      x_{m+1}' = x_m * sin(mθ) + x_{m+1} * cos(mθ)

    where θ = 10000^(-2k/d) for the k-th dimension pair.
    rope_dim=0 means full head_dim rotation; rope_dim=n means only the first
    rope_dim dimensions are rotated (last head_dim-rope_dim dims pass through).

    """

    def forward(
        self,
        x: np.ndarray,
        position: int | np.ndarray,
        *,
        rope_dim: int = 0,
    ) -> np.ndarray:
        """Apply rotary positional embedding to Q or K tensor.

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, seq_len, n_heads, head_dim)
            Q or K inputs.
        position : int or np.ndarray, shape (seq_len,) or (batch_size, seq_len)
            Position indices — scalar int gives all positions same angle.
        rope_dim : int, optional
            Number of head_dims to rotate. 0 = full head_dim rotation.

        Returns
        -------
        np.ndarray, shape (batch_size, seq_len, n_heads, head_dim)
            Rotary-embedded tensor with position-dependent rotation.

        """
        # x:          (B, S, H, D)  — batch, seq, heads, head_dim
        # position:   scalar or (S,) or (B, S)  — position indices
        # freqs:      (S,) or (B, S, 1, D//2)   — precomputed frequencies
        # output:     (B, S, H, D)               — rotated Q/K

        # Separate unrotated dims (after rope_dim)
        if rope_dim > 0 and rope_dim < x.shape[-1]:
            x_rotated = x[..., :rope_dim]
            x_unchanged = x[..., rope_dim:]
        else:
            x_rotated = x
            x_unchanged = None

        # Handle both batched and single position
        batch_size, seq_len = x_rotated.shape[0], x_rotated.shape[1]

        if np.isscalar(position):
            pos = np.full((batch_size, seq_len), position, dtype=np.int32)
        else:
            pos = np.asarray(position, dtype=np.int32)

        # Handle both batched (B, S) and unbatched (S,) positions
        if pos.ndim == 1 and pos.shape[0] == seq_len:
            pos = np.broadcast_to(pos, (batch_size, seq_len))  # (B, S)

        # Compute rotation frequencies for each (odd, even) pair in the pair_dim
        pair_dim = x_rotated.shape[-1] // 2  # number of rotating pairs = head_dim // 2

        # freqs: (pair_dim,) = 1 / 10000^(2k / D) where k=0,1,...,pair_dim-1
        freqs = 1.0 / (10000.0 ** (np.arange(pair_dim, dtype=np.float32) * 2.0 / x_rotated.shape[-1]))
        # shape: (pair_dim,)  — one frequency per pair

        # For each position and each pair, compute rotation angle: pos * freq
        # freqs: (pair_dim,)
        # pos.reshape: (B, S, 1)  — broadcast to (B, S, pair_dim)
        # angles: (B, S, pair_dim) — pos * freq for each position-head_dim pair
        angles = pos[:, :, np.newaxis] * freqs[np.newaxis, np.newaxis, :]  # (B, S, pair_dim)

        cos = np.cos(angles)  # (B, S, pair_dim) — cosines
        sin = np.sin(angles)  # (B, S, pair_dim) — sines

        # Reshape for pairwise operations
        # x_flat: (B, S, H, pair_dim, 2) — last dim is (m, m+1) pair
        x_flat = x_rotated.reshape(x_rotated.shape[:-1] + (pair_dim, 2))  # (B, S, H, pair_dim, 2)

        # Extract odd and even components (paired by position in last dim)
        # Even: x[..., 0] → x_m, Odd: x[..., 1] → x_{m+1}
        # After reshape: (B, S, H, pair_dim, 2)
        x_even = x_flat[..., 0]  # (B, S, H, pair_dim)  — x_m
        x_odd = x_flat[..., 1]  # (B, S, H, pair_dim)  — x_{m+1}

        # Broadcast cos/sin for pairwise rotation
        # cos: (B, S, pair_dim,) → (B, S, 1, pair_dim,) — broadcast over heads
        cos_broad = cos[:, :, np.newaxis, :]  # (B, S, 1, pair_dim)
        sin_broad = sin[:, :, np.newaxis, :]  # (B, S, 1, pair_dim)

        # Apply 2D rotation: each (odd, even) pair rotates independently
        # y_m = x_m * cos - x_odd * sin
        # y_{m+1} = x_m * sin + x_odd * cos
        y_even = x_even * cos_broad - x_odd * sin_broad  # (B, S, H, pair_dim)
        y_odd = x_even * sin_broad + x_odd * cos_broad  # (B, S, H, pair_dim)

        # Reassemble: (B, S, H, pair_dim, 2) → (B, S, H, head_dim)
        rotated = np.stack([y_even, y_odd], axis=-1).reshape(x_rotated.shape)

        if x_unchanged is not None:
            return np.concatenate([rotated, x_unchanged], axis=-1)
        return rotated


class MoE:
    """Mixture of Experts — each token is routed to top-k experts for processing.

    Parameters
    ----------
    embed_dim : int
        Input/output dimension (hidden size).
    n_experts : int
        Total number of experts.
    ff_dim : int
        Hidden dimension for each expert (SwiGLU FFN).
    k : int
        Number of top experts to select per token.
    seed : int
        Random seed for weight initialization.

    Forward
    -------
    x : np.ndarray, shape (batch_size, seq_len, embed_dim)

    Returns
    -------
    out : np.ndarray, shape (batch_size, seq_len, embed_dim)

    Architecture
    ------------
    x [B,S,D] → router_scores = softmax(x @ W_router.T + b_router) [B,S,E]

    For each token t:
      top_k = argtopk(router_scores[t], k)    # k expert indices
      expert_out[t] = sum(router_scores[t, idx] * experts[idx](x[t]) for idx in top_k)

    Final: out = expert_out [B,S,D]

    Notes
    -----
    Each expert is a SwiGLU FFN (SiLU(w1 @ x) * (w3 @ x) @ w2).
    Softmax is applied across ALL experts (not just top-k), but only top-k
    contribute to the output (others get multiplied by 0 via gating).

    """

    NP_ROUTER: str = "moe.router"
    NP_BIAS: str = "moe.bias"

    def __init__(
        self,
        embed_dim: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        seed: int = 0,
    ) -> None:
        self.embed_dim = embed_dim
        self.n_experts = n_experts
        self.ff_dim = ff_dim
        self.k = k

        # Router: linear layer from embed_dim to n_experts
        # Shape: [embed_dim, n_experts]
        self.router = (
            np.random.default_rng(seed).normal(0, 1.0 / np.sqrt(embed_dim), (embed_dim, n_experts)).astype(np.float32)
        )
        self.bias = np.zeros(n_experts, dtype=np.float32)

        # Experts: each expert is a SwiGLUFFN
        # Each expert: input embed_dim → output embed_dim
        self.experts = [SwiGLUFFN(embed_dim, ff_dim, seed=seed + expert_idx) for expert_idx in range(n_experts)]

    def forward(
        self,
        x: np.ndarray,
        router: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute MoE forward pass.

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, seq_len, embed_dim)
        router : np.ndarray, shape (embed_dim, n_experts) — if None, use self.router
        bias : np.ndarray, shape (n_experts,) — if None, use self.bias

        Returns
        -------
        out : np.ndarray, shape (batch_size, seq_len, embed_dim)

        """
        batch_size, seq_len, embed_dim = x.shape

        rout_w = router or self.router
        rout_b = bias or self.bias

        # Router scores: (B, S, D) @ (D, E) + (E,) → (B, S, E)
        # Compute scores for all experts, all tokens
        router_scores = x @ rout_w + rout_b  # (B, S, E)

        # Softmax over expert dimension for routing weights
        # Normalize across experts: sum over E = 1 for each (batch, seq)
        router_scores_max = np.max(router_scores, axis=-1, keepdims=True)  # (B, S, 1)
        router_scores = router_scores - router_scores_max  # (B, S, E) — stable
        exp_scores = np.exp(router_scores)  # (B, S, E)
        score_sum = np.sum(exp_scores, axis=-1, keepdims=True)  # (B, S, 1)
        routing_weights = exp_scores / score_sum  # (B, S, E) — softmax over experts

        # For each token, select top-k experts
        # Compute expert outputs for all tokens and all experts
        # Then combine weighted by top-k routing weights

        # Mask to top-k experts: only the top-k selected experts contribute
        # For each token, find the k-th highest routing weight
        # and zero out all other experts below this threshold
        if self.k < self.n_experts:
            # Get the threshold value: k-th largest routing weight per token
            # We use a stable approach: sort weights and take the k-th largest
            sorted_indices = np.argsort(routing_weights, axis=-1)[:, :, ::-1]  # (B, S, E) descending
            kth_indices = sorted_indices[:, :, self.k - 1 : self.k]  # (B, S, 1)
            kth_values = np.take_along_axis(routing_weights, kth_indices, axis=-1)  # (B, S, 1)

            # Zero out experts with weights below the k-th threshold
            # For experts at exactly the threshold, keep them (they could be any of the top-k)
            routing_weights = np.where(routing_weights >= kth_values, routing_weights, 0.0)  # (B, S, E)

            # Renormalize to sum to 1 across experts
            renorm_sum = np.sum(routing_weights, axis=-1, keepdims=True)  # (B, S, 1)
            # Avoid division by zero
            renorm_sum = np.maximum(renorm_sum, 1e-8)
            routing_weights = routing_weights / renorm_sum  # (B, S, E)

        # Zero output buffer: (B, S, D)
        out = np.zeros((batch_size, seq_len, self.embed_dim), dtype=np.float32)

        # Process each token (vectorized over tokens for speed)
        # For each expert, compute expert_output for all (batch, seq) positions
        for expert_idx, expert in enumerate(self.experts):
            # expert: (B, S, D) → expert_output: (B, S, D)
            expert_output = expert.forward(x)  # (B, S, D)

            # Get routing weight for this expert: (B, S, 1)
            w = routing_weights[:, :, expert_idx : expert_idx + 1]  # (B, S, 1)

            # Weighted contribution: w * expert_output
            out = out + w * expert_output  # (B, S, D)

        return out


class MultiHeadAttention:
    """Scaled dot-product multi-head attention (with GQA support).

    Parameters
    ----------
    embed_dim : int
        Input/output embedding dimension.
    n_heads : int
        Number of query heads.
    n_groups : int, optional (default=embed_dim // (embed_dim // n_heads) or n_heads)
        Number of K/V groups for grouped-query attention.
        If n_groups == n_heads → standard MHA (no GQA).
        If n_groups < n_heads → GQA: K/V shared across groups.
    rope_dim : int
        Number of head dimensions to apply RoPE to (0 = no RoPE).
    seed : int
        Random seed for weight initialization.

    Forward
    -------
    x : np.ndarray, shape (batch_size, seq_len, embed_dim)

    Returns
    -------
    out : np.ndarray, shape (batch_size, seq_len, embed_dim)

    Architecture
    ------------
    X [B,S,D] → Q_proj [W_q: (D, H*head_dim), b_q: (H*head_dim)] → Q [B,S,H,D]
                  K_proj [W_k: (D, G*head_dim), b_k: (G*head_dim)] → K [B,S,G,D]
                  V_proj [W_v: (D, G*head_dim), b_v: (G*head_dim)] → V [B,S,G,D]

    QK^T / sqrt(d) [B,H,S,S] → softmax → multiply V

    For GQA (G=3, H=6): K and V are broadcast from G to H heads.
    For non-GQA (G=H): no broadcasting needed.

    Final: reshape → O_proj [W_o: (H*head_dim, D), b_o: (D)] → [B,S,D]

    Notes
    -----
    head_dim = embed_dim // n_heads.
    All weights are trained via backpropagation.

    """

    NP_QW: str = "mha.Wq"
    NP_QB: str = "mha.bq"
    NP_KW: str = "mha.Wk"
    NP_KB: str = "mha.bk"
    NP_VW: str = "mha.Wv"
    NP_VB: str = "mha.bv"
    NP_OW: str = "mha.Wo"
    NP_OB: str = "mha.bo"

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        rope_dim: int = 0,
        n_groups: int | None = None,
        seed: int = 0,
    ) -> None:
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.rope_dim = rope_dim
        if n_groups is None:
            n_groups = n_heads

        self.n_groups = n_groups
        rng = np.random.default_rng(seed)

        # Q projection: [D, H * head_dim]
        # fan_in for Xavier: input_dim = D
        scale_q = np.sqrt(2.0 / (embed_dim + self.n_heads * self.head_dim))  # noqa: F841
        self.Wq = rng.normal(0, 1.0 / np.sqrt(embed_dim), (embed_dim, self.n_heads * self.head_dim)).astype(np.float32)
        self.bq = np.zeros(self.n_heads * self.head_dim, dtype=np.float32)

        # K projection: [D, G * head_dim]
        self.Wk = rng.normal(0, 1.0 / np.sqrt(embed_dim), (embed_dim, self.n_groups * self.head_dim)).astype(np.float32)
        self.bk = np.zeros(self.n_groups * self.head_dim, dtype=np.float32)

        # V projection: [D, G * head_dim]
        self.Wv = rng.normal(0, 1.0 / np.sqrt(embed_dim), (embed_dim, self.n_groups * self.head_dim)).astype(np.float32)
        self.bv = np.zeros(self.n_groups * self.head_dim, dtype=np.float32)

        # Output projection: [H * head_dim, D]
        self.Wo = rng.normal(0, 1.0 / np.sqrt(embed_dim), (self.n_heads * self.head_dim, embed_dim)).astype(np.float32)
        self.bo = np.zeros(embed_dim, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        Wq: np.ndarray | None = None,
        bq: np.ndarray | None = None,
        Wk: np.ndarray | None = None,
        bk: np.ndarray | None = None,
        Wv: np.ndarray | None = None,
        bv: np.ndarray | None = None,
        Wo: np.ndarray | None = None,
        bo: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute multi-head attention forward pass.

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, seq_len, embed_dim)
        Wq : np.ndarray, shape (embed_dim, n_heads * head_dim) — if None, use self.Wq
        ... (similar for Wk, bk, Wv, bv, Wo, bo)

        Returns
        -------
        out : np.ndarray, shape (batch_size, seq_len, embed_dim)

        """
        batch_size, seq_len, embed_dim = x.shape

        # Use passed weights or instance weights (for testing)
        q_w = Wq or self.Wq
        q_b = bq or self.bq
        k_w = Wk or self.Wk
        k_b = bk or self.bk
        v_w = Wv or self.Wv
        v_b = bv or self.bv
        o_w = Wo or self.Wo
        o_b = bo or self.bo

        # Q, K, V projections
        # Q: (B, S, D) @ (D, H*d) → (B, S, H*d)  — add bias
        q = x @ q_w + q_b  # (B, S, H*d)  where H=d=16, d=4

        # K, V: (B, S, D) @ (D, G*d) → (B, S, G*d)
        k = x @ k_w + k_b  # (B, S, G*d)  where G=3, d=4

        # Reshape Q: (B, S, H*d) → (B, H, S, d)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, H, S, d)

        # Reshape K, V: (B, S, G*d) → (B, G, S, d)
        k = k.reshape(batch_size, seq_len, self.n_groups, self.head_dim).transpose(0, 2, 1, 3)  # (B, G, S, d)

        # Apply RoPE to Q and K if rope_dim > 0
        if self.rope_dim > 0:
            q = q.reshape(batch_size, self.n_heads, seq_len, self.head_dim)  # (B, H, S, d)
            k = k.reshape(batch_size, self.n_groups, seq_len, self.head_dim)  # (B, G, S, d)

            # RoPE only on first rope_dim dims of head_dim
            q = RoPE().forward(q, np.arange(seq_len), rope_dim=self.rope_dim)
            k = RoPE().forward(k, np.arange(seq_len), rope_dim=self.rope_dim)

            # Reshape back for the following transpose
            q = q.reshape(batch_size, self.n_heads, seq_len, self.head_dim)
            k = k.reshape(batch_size, self.n_groups, seq_len, self.head_dim)

        # Scaled dot-product attention: Q @ K^T / sqrt(d)
        # Q: (B, H, S, d), K^T: (B, G, d, S)
        # scores: (B, H, S, S)

        # For GQA: K has G heads, Q has H heads.
        # We need to broadcast K and V from G to H heads.
        # K for attention: (B, H, S, d) where K_h = K_{h % G}
        # V for attention: (B, H, S, d) where V_h = V_{h % G}

        # Expand K and V from (B, G, S, d) to (B, H, S, d) by repeating
        # This handles both standard MHA (G=H, no-op) and GQA (G<H)
        if self.n_groups != self.n_heads:
            # GQA: expand K and V from G to H heads
            expansion = np.zeros((self.n_heads, self.n_groups), dtype=np.float32)
            for h in range(self.n_heads):
                expansion[h, h % self.n_groups] = 1.0
            # k: (B, H, S, d) = (H, G) @ (B, G, S, d) — but this is matrix multiply over head dim
            # Better: just index-repeat K
            k_expanded = np.zeros((batch_size, self.n_heads, seq_len, self.head_dim), dtype=np.float32)
            for h in range(self.n_heads):
                k_expanded[:, h, :, :] = k[:, h % self.n_groups, :, :]
            k = k_expanded

            v = x @ v_w + v_b  # (B, S, G*d)
            v = v.reshape(batch_size, seq_len, self.n_groups, self.head_dim).transpose(0, 2, 1, 3)  # (B, G, S, d)

            v_expanded = np.zeros((batch_size, self.n_heads, seq_len, self.head_dim), dtype=np.float32)
            for h in range(self.n_heads):
                v_expanded[:, h, :, :] = v[:, h % self.n_groups, :, :]
            v = v_expanded
        else:
            v = x @ v_w + v_b  # (B, S, H*d)
            v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, H, S, d)

        # Scaled dot-product attention
        # Q: (B, H, S, d), K: (B, H, S, d) → K^T: (B, H, d, S)
        # scores: (B, H, S, S)
        # dim d = self.head_dim = 16, 4 = 4 → sqrt(4) = 2.0
        head_dim_sqrt = np.sqrt(float(self.head_dim))  # sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / head_dim_sqrt  # (B, H, S, S)  — QK^T / sqrt(d)

        # Softmax over the last dim (key positions) — attention weights for each query position
        # scores: (B, H, S, S)  — for each batch, head, query pos, scores over all key positions
        scores_max = np.max(scores, axis=-1, keepdims=True)  # (B, H, S, 1)
        scores = scores - scores_max  # (B, H, S, S) — numerical stability
        exp_scores = np.exp(scores)  # (B, H, S, S)
        scores_sum = np.sum(exp_scores, axis=-1, keepdims=True)  # (B, H, S, 1)
        attn_weights = exp_scores / scores_sum  # (B, H, S, S) — softmax over keys

        # Attention output: attn_weights @ V
        # (B, H, S, S) @ (B, H, S, d) → (B, H, S, d)
        attn_out = attn_weights @ v  # (B, H, S, d)

        # Reshape and output projection
        # (B, H, S, d) → (B, S, H*d) — put sequence dim back
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.n_heads * self.head_dim
        )  # (B, S, H*d)

        # Final output projection: (B, S, H*d) @ (H*d, D) + (D,)  — → (B, S, D)
        out = attn_out @ o_w + o_b  # (B, S, D)

        return out


class Linear:
    """Fully connected linear layer: y = x @ W + b.

    Parameters
    ----------
    None — forward pass takes weights explicitly for standalone testing.

    Forward
    -------
    x : np.ndarray, shape (..., input_dim)
        Input activations.
    weight : np.ndarray, shape (input_dim, output_dim)
        Weight matrix.
    bias : np.ndarray, shape (output_dim,)
        Bias vector (optional, defaults to zero).

    Returns
    -------
    out : np.ndarray, shape (..., output_dim)
        Linear transformation output.

    Notes
    -----
    Standard affine transformation with broadcasting over leading dimensions.

    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """Perform matrix multiplication with optional bias.

        Parameters
        ----------
        x : np.ndarray, shape (..., input_dim)
            Input tensor.
        weight : np.ndarray, shape (input_dim, output_dim)
            Weight matrix.
        bias : np.ndarray, shape (output_dim,), optional
            Bias vector added after multiplication.

        Returns
        -------
        np.ndarray, shape (..., output_dim)
            Transformed output.

        Examples
        --------
        >>> import numpy as np
        >>> lin = Linear()
        >>> x = np.ones((2, 3), dtype=np.float32)     # (batch=2, in=3)
        >>> w = np.eye(3, 4, dtype=np.float32)         # (in=3, out=4)
        >>> b = np.zeros(4, dtype=np.float32)
        >>> out = lin.forward(x, w, b)
        >>> out.shape
        (2, 4)

        """
        # x:     (..., input_dim)
        # W:     (input_dim, output_dim)
        # x @ W: (..., output_dim)
        # b:     (output_dim,)   — broadcast over batch dimensions
        out = x @ weight  # (..., output_dim)
        if bias is not None:
            out += bias  # broadcast bias over batch dims
        return out


class TransformerBlock:
    """Standard decoder-only transformer block.

        Parameters
        ----------
        embed_dim : int
            Input/output embedding dimension.
        n_heads : int
            Number of attention heads.
        n_experts : int
            Number of MoE experts.
        ff_dim : int
            Hidden dimension for MoE experts.
        k : int
            Number of top experts to activate per token.
        rope_dim : int
            Number of head dimensions for RoPE (0 = no RoPE).
        seed : int
            Random seed for weight initialization.

        Forward
        -------
        x : np.ndarray, shape (batch_size, seq_len, embed_dim)

        Returns
        -------
        out : np.ndar

    [...4 lines truncated...]

            h = x + MHA(ln1(x)) + MoE(ln2(x + MHA(ln1(x))))

            where:
              ln1 = RMSNorm, ln2 = RMSNorm
              MHA = MultiHeadAttention (with optional RoPE)
              MoE = Mixture of Experts (with top-k routing)

    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        rope_dim: int = 0,
        seed: int = 0,
    ) -> None:
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.k = k

        # RMSNorm weights — gamma for layer normalization
        self.ln1_gamma = np.ones(embed_dim, dtype=np.float32)
        self.ln2_gamma = np.ones(embed_dim, dtype=np.float32)

        # Gated residuals — learnable scalar gates, initialized to zero
        # Gate controls signal flow: h = h + sigmoid(gate) * gated_value
        # Initialized to zero → identity at start → gates learn to open
        self.gate1 = np.zeros(1, dtype=np.float32)
        self.gate2 = np.zeros(1, dtype=np.float32)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(embed_dim, n_heads=n_heads, rope_dim=rope_dim, seed=seed + 2)

        # Mixture of Experts
        self.moe = MoE(embed_dim, n_experts=n_experts, ff_dim=ff_dim, k=k, seed=seed + 3)

    def forward(
        self, x: np.ndarray, dropout: float = 0.0, training: bool = False, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Forward pass through the transformer block.

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, seq_len, embed_dim)
        dropout : float
            Dropout rate for dropout regularization (default: 0.0, no dropout)
        training : bool
            Whether in training mode (dropout active) or inference mode (default: False)
        rng : np.random.Generator or None
            Random number generator for dropout (uses default if None)

        Returns
        -------
        out : np.ndarray, shape (batch_size, seq_len, embed_dim)

        """
        # ── Stream 1: Attention ─────────────────────────────────────
        # MHA: (B, S, D) → (B, S, D) — self-attention output
        attn_out = self.mha.forward(x)  # (B, S, D)

        # First residual: x + attn_out  — (B, S, D)
        h = x + attn_out  # (B, S, D)

        # Post-norm: h = RMSNorm(h)  — (B, S, D) → (B, S, D)
        h = RMSNorm().forward(h, self.ln1_gamma)  # (B, S, D)

        # Gated residual: h = h + sigmoid(gate1) * h
        gate1_val: float = float(1.0 / (1.0 + np.exp(-self.gate1[0])))  # sigmoid
        h = h + gate1_val * h  # (B, S, D)

        # Optional dropout: h = h * Bernoulli(1 - dropout) with scaling
        if training and dropout > 0.0:
            mask = (
                (rng.random(h.shape) >= dropout).astype(np.float32)
                if rng is not None
                else (np.random.random(h.shape) >= dropout).astype(np.float32)
            )
            h = h * (mask / (1.0 - dropout))  # (B, S, D) — scaled dropout

        # ── Stream 2: MoE ──────────────────────────────────────────
        # MoE: (B, S, D) → (B, S, D) — mixture of experts output
        moe_out = self.moe.forward(h)  # (B, S, D)

        # Second residual: h + moe_out  — (B, S, D)
        out = h + moe_out  # (B, S, D)

        # Post-norm: out = RMSNorm(out)  — (B, S, D) → (B, S, D)
        out = RMSNorm().forward(out, self.ln2_gamma)  # (B, S, D)

        # Gated residual: out = out + sigmoid(gate2) * out
        gate2_val: float = float(1.0 / (1.0 + np.exp(-self.gate2[0])))  # sigmoid
        out = out + gate2_val * out  # (B, S, D)

        # Optional dropout: out = out * Bernoulli(1 - dropout) with scaling
        if training and dropout > 0.0:
            mask = (
                (rng.random(out.shape) >= dropout).astype(np.float32)
                if rng is not None
                else (np.random.random(out.shape) >= dropout).astype(np.float32)
            )
            out = out * (mask / (1.0 - dropout))  # (B, S, D) — scaled dropout

        return out


class DecoderStack:
    """Stack of TransformerBlocks — chains n_layers of decoder blocks.

    Parameters
    ----------
    n_layers : int
        Number of TransformerBlocks to stack.
    embed_dim : int
        Input/output embedding dimension.
    n_heads : int
        Number of attention heads per block.
    n_experts : int
        Number of MoE experts per block.
    ff_dim : int
        Hidden dimension for MoE experts per block.
    k : int
        Number of top experts to activate per token.
    rope_dim : int
        Number of head dimensions for RoPE (0 = no RoPE).
    seed : int
        Random seed for weight initialization.

    Forward
    -------
    x : np.ndarray, shape (batch_size, seq_len, embed_dim)

    Returns
    -------
    out : np.ndarray, shape (batch_size, seq_len, embed_dim)

    Architecture
    ------------
    X [B,S,D] → block_0 → block_1 → ... → block_{n_layers-1} → output [B,S,D]

    Each block:
      h = x + MHA(RMSNorm(x)) + MoE(RMSNorm(x + MHA(x)))

    """

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        rope_dim: int = 0,
        seed: int = 0,
    ) -> None:
        self.n_layers = n_layers

        # Create TransformerBlocks in sequence — each uses a different seed
        # to avoid weight collisions
        self.blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_experts=n_experts,
                ff_dim=ff_dim,
                k=k,
                rope_dim=rope_dim,
                seed=seed + layer_idx,
            )
            for layer_idx in range(n_layers)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all stacked blocks.

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, seq_len, embed_dim)

        Returns
        -------
        out : np.ndarray, shape (batch_size, seq_len, embed_dim)

        """
        out = x  # (B, S, D)

        for block in self.blocks:
            out = block.forward(out)  # (B, S, D) → (B, S, D)

        return out
