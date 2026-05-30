import numpy as np
from typing import Optional, Tuple, Dict, Any

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    Allows the model to attend to different parts of the sequence simultaneously.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, and V
        # We combine them into one large weight matrix for efficiency
        # [Embed_Dim, Embed_Dim]
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01

        # Output projection
        # [Embed_Dim, Embed_Dim]
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.01

        # Cache for KV values during inference
        self.kv_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def forward(
        self, 
        x: np.ndarray, 
        mask: Optional[np.ndarray] = None, 
        use_cache: bool = False,
        cache_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len] (1 for keep, 0 for mask)
            use_cache: Whether to use/update KV cache
            cache_idx: Index of the current token for KV cache update (used in inference)
        Returns:
            output: [Batch, Seq_Len, Embed_Dim]
            cache: Dictionary containing intermediate values for backward pass
        """
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections
        # [Batch, Seq_Len, Embed_Dim]
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 2. Split into multiple heads
        # Reshape to [Batch, Seq_Len, Num_Heads, Head_Dim]
        # Then transpose to [Batch, Num_Heads, Seq_Len, Head_Dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # --- KV CACHE LOGIC ---
        if use_cache and cache_idx is not None:
            if cache_idx in self.kv_cache:
                prev_K, prev_V = self.kv_cache[cache_idx]
                # Concat along the sequence dimension (axis 2)
                # [Batch, Num_Heads, Prev_Seq_Len + Current_Seq_Len, Head_Dim]
                K = np.concatenate([prev_K, K], axis=2)
                V = np.concatenate([prev_V, V], axis=2)
            
            self.kv_cache[cache_idx] = (K, V)
            current_kv_cache = self.kv_cache.copy()
        else:
            current_kv_cache = None
        # ----------------------

        # 3. Scaled Dot-Product Attention
        # Scores = (Q @ K^T) / sqrt(d_k)
        # [Batch, Num_Heads, Q_Seq_Len, Head_Dim] @ [Batch, Num_Heads, Head_Dim, K_Seq_Len] -> [Batch, Num_Heads, Q_Seq_Len, K_Seq_Len]
        d_k = self.head_dim
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        # 4. Apply causal mask if provided
        if mask is not None:
            # mask is [Seq_Len, Seq_Len], broadcast to [Batch, Num_Heads, Q_Seq_Len, K_Seq_Len]
            scores = np.where(mask == 0, -1e9, scores)

        # 5. Softmax to get attention weights
        attn_weights = self._softmax(scores, axis=-1)

        # 6. Weighted sum of values
        context = np.matmul(attn_weights, V)

        # 7. Concatenate heads
        context_out = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # 8. Final output projection
        output = np.dot(context_out, self.W_o)

        # Prepare cache for backward pass
        cache = {
            "Q": Q,
            "K": K,
            "V": V,
            "attn_weights": attn_weights,
            "context": context_out,
            "mask": mask
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
        mask: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
        attn_weights: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Backward pass for Multi-Head Attention.
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Gradient w.r.t. W_o and context
        # [Embed_Dim, Embed_Dim]
        if context is None:
             raise ValueError("context must be provided for backward pass")

        d_W_o = np.dot(context.reshape(-1, self.embed_dim).T, d_out.reshape(-1, self.embed_dim))
        d_context = np.dot(d_out, self.W_o.T)

        # 2. Gradient w.r.t. attn_weights and V
        d_context_heads = d_context.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        if V is None: raise ValueError("V must be provided for backward pass")
        d_V = np.matmul(attn_weights.transpose(0, 1, 3, 2), d_context_heads)
        d_attn_weights = np.matmul(d_context_heads, V.transpose(0, 1, 3, 2))

        # 3. Gradient w.r.t. scores (after softmax)
        if attn_weights is None: raise ValueError("attn_weights must be provided for backward pass")
        d_scores = attn_weights * (d_attn_weights - np.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True))

        # 4. Apply mask gradient
        if mask is not None:
            d_scores = d_scores * mask

        # 5. Gradient w.r.t. Q and K
        d_scores = d_scores * np.sqrt(self.head_dim)
        if Q is None or K is None: raise ValueError("Q and K must be provided for backward pass")

        d_Q = np.matmul(d_scores, K)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        # 6. Reshape gradients back to [Batch, Seq_Len, Embed_Dim]
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # 7. Gradients for W_q, W_k, W_v
        d_W_q = np.dot(x.reshape(-1, self.embed_dim).T, d_Q.reshape(-1, self.embed_dim))
        d_W_k = np.dot(x.reshape(-1, self.embed_dim).T, d_K.reshape(-1, self.embed_dim))
        d_W_v = np.dot(x.reshape(-1, self.embed_dim).T, d_V.reshape(-1, self.embed_dim))

        # 8. Gradient w.r.t. input x
        dx = np.dot(d_Q, self.W_q.T) + np.dot(d_K, self.W_k.T) + np.dot(d_V, self.W_v.T)

        # Store gradients in self
        self.grad_W_q = d_W_q
        self.grad_W_k = d_W_k
        self.grad_W_v = d_W_v
        self.grad_W_o = d_W_o

        grads = {
            "W_q": d_W_q,
            "W_k": d_W_k,
            "W_v": d_W_v,
            "W_o": d_W_o
        }

        return dx, grads

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W_q": self.W_q,
            "W_k": self.W_k,
            "W_v": self.W_v,
            "W_o": self.W_o
        }

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {
            "W_q": self.grad_W_q,
            "W_k": self.grad_W_k,
            "W_v": self.grad_W_v,
            "W_o": self.grad_W_o
        }


