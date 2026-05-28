import numpy as np
from typing import Optional

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

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len] (1 for keep, 0 for mask)
        Returns:
            [Batch, Seq_Len, Embed_Dim]
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
        
        # 3. Scaled Dot-Product Attention
        # Scores = (Q @ K^T) / sqrt(d_k)
        # [Batch, Num_Heads, Seq_Len, Head_Dim] @ [Batch, Num_Heads, Head_Dim, Seq_Len] -> [Batch, Num_Heads, Seq_Len, Seq_Len]
        d_k = self.head_dim
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # 4. Apply causal mask if provided
        if mask is not None:
            # mask is [Seq_Len, Seq_Len], broadcast to [Batch, Num_Heads, Seq_Len, Seq_Len]
            # We fill masked positions with a very large negative number so softmax ignores them
            scores = np.where(mask == 0, -1e9, scores)
            
        # 5. Softmax to get attention weights
        # softmax is applied along the last dimension (the 'keys')
        # [Batch, Num_Heads, Seq_Len, Seq_Len]
        attn_weights = self._softmax(scores, axis=-1)
        
        # 6. Weighted sum of values
        # [Batch, Num_Heads, Seq_Len, Seq_Len] @ [Batch, Num_Heads, Seq_Len, Head_Dim] -> [Batch, Num_Heads, Seq_Len, Head_Dim]
        context = np.matmul(attn_weights, V)
        
        # 7. Concatenate heads
        # Transpose back to [Batch, Seq_Len, Num_Heads, Head_Dim]
        # Then reshape to [Batch, Seq_Len, Embed_Dim]
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # 8. Final output projection
        # [Batch, Seq_Len, Embed_Dim]
        output = np.dot(context, self.W_o)
        
        return output

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Numerical stable softmax."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
