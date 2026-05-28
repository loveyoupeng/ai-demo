import numpy as np
from typing import Optional, Tuple, Dict

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
    ) -> Tuple[np.ndarray, Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]]:
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len] (1 for keep, 0 for mask)
            use_cache: Whether to use/update KV cache
            cache_idx: Index of the current token for KV cache update (used in inference)
        Returns:
            output: [Batch, Seq_Len, Embed_Dim]
            cache: Updated KV cache
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
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # 8. Final output projection
        output = np.dot(context, self.W_o)

        return output, current_kv_cache

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
        
        Args:
            x: Input [Batch, Seq_Len, Embed_Dim]
            d_out: Gradient of loss w.r.t. output [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len]
            Q, K, V: Forward pass intermediate tensors (optional, for optimization)
            attn_weights: Forward pass attention weights (optional)
            context: Forward pass context tensor (optional)
            
        Returns:
            dx: Gradient of loss w.r.t. input x [Batch, Seq_Len, Embed_Dim]
            grads: Dictionary of gradients for parameters (W_q, W_k, W_v, W_o)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Gradient w.r.t. W_o and context
        # output = context @ W_o
        # d_W_o = context.T @ d_out
        # d_context = d_out @ W_o.T
        
        # context shape is [Batch, Seq_Len, Embed_Dim]
        # Flatten batch and seq_len for matrix multiplication
        context_flat = x.reshape(-1, self.embed_dim) # This is NOT correct, context is context.
        # Wait, we need context from forward pass. 
        # If context not provided, we'd need to recompute it, but usually we pass it.
        # For simplicity, let's assume context is passed or we'll use the logic from forward.
        # Since we are in backward, we MUST have the values from forward.
        
        # Re-calculating context if not provided is inefficient, but we'll assume 
        # user provides what is necessary or we derive it.
        # Let's look at the forward: context = np.matmul(attn_weights, V)
        # We need context to compute d_W_o.
        # But we can also compute d_W_o using the fact that output = context @ W_o
        
        # Let's assume we pass context.
        if context is None:
             # This is a bit of a hack, in a real engine context is cached.
             # For now, let's try to derive it or expect it.
             raise ValueError("context must be provided for backward pass")

        d_W_o = np.dot(context.transpose(0, 2, 1).reshape(-1, self.embed_dim).T, 
                       d_out.reshape(-1, self.embed_dim))
        # Error in above: context is [B, S, E], d_out is [B, S, E]. 
        # context_flat: [B*S, E], d_out_flat: [B*S, E].
        # d_W_o: [E, E]
        d_W_o = np.dot(context.reshape(-1, self.embed_dim).T, d_out.reshape(-1, self.embed_dim))
        
        d_context = np.dot(d_out, self.W_o.T) # [B, S, E]

        # 2. Gradient w.r.t. attn_weights and V
        # context = attn_weights @ V
        # [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
        # d_V = attn_weights.T @ d_context_heads
        # d_attn_weights = d_context_heads @ V.T
        
        # We need d_context in heads shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
        d_context_heads = d_context.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # If V was not provided, we can't do this.
        if V is None: raise ValueError("V must be provided for backward pass")
        
        # attn_weights: [B, H, S, S], V: [B, H, S, D]
        # d_V: [B, H, S, D]
        d_V = np.matmul(attn_weights.transpose(0, 1, 3, 2), d_context_heads)
        
        # d_attn_weights: [B, H, S, S]
        d_attn_weights = np.matmul(d_context_heads, V.transpose(0, 1, 3, 2))

        # 3. Gradient w.r.t. scores (after softmax)
        # attn_weights = softmax(scores)
        # d_scores = attn_weights * (d_attn_weights - sum(d_attn_weights * attn_weights, axis=-1, keepdims=True))
        if attn_weights is None: raise ValueError("attn_weights must be provided for backward pass")
        
        d_scores = attn_weights * (d_attn_weights - np.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True))

        # 4. Apply mask gradient
        if mask is not None:
            # Masked positions should have 0 gradient
            d_scores = d_scores * mask

        # 5. Gradient w.r.t. Q and K
        # scores = (Q @ K.T) / sqrt(d_k)
        # d_scores = d_scores * sqrt(d_k)
        # d_Q = d_scores @ K
        # d_K = d_scores.T @ Q
        
        d_scores = d_scores * np.sqrt(self.head_dim)
        
        if Q is None or K is None: raise ValueError("Q and K must be provided for backward pass")

        # d_Q: [B, H, S, D]
        d_Q = np.matmul(d_scores, K)
        # d_K: [B, H, S, D]
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        # 6. Reshape gradients back to [Batch, Seq_Len, Embed_Dim]
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # 7. Gradients for W_q, W_k, W_v
        # Q = x @ W_q => d_W_q = x.T @ d_Q
        # d_W_q: [E, E]
        d_W_q = np.dot(x.reshape(-1, self.embed_dim).T, d_Q.reshape(-1, self.embed_dim))
        d_W_k = np.dot(x.reshape(-1, self.embed_dim).T, d_K.reshape(-1, self.embed_dim))
        d_W_v = np.dot(x.reshape(-1, self.embed_dim).T, d_V.reshape(-1, self.embed_dim))

        # 8. Gradient w.r.t. input x
        # d_x = d_Q @ W_q.T + d_K @ W_k.T + d_V @ W_v.T
        dx = np.dot(d_Q, self.W_q.T) + np.dot(d_K, self.W_k.T) + np.dot(d_V, self.W_v.T)

        grads = {
            "W_q": d_W_q,
            "W_k": d_W_k,
            "W_v": d_W_v,
            "W_o": d_W_o
        }

        return dx, grads
