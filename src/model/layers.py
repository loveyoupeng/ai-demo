import numpy as np
from typing import Tuple

class TokenEmbedding:
    """
    Learned token embeddings.
    Maps integer token IDs to continuous vectors.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Initialize weights randomly
        # [Vocab_Size, Embed_Dim]
        self.weights = np.random.randn(vocab_size, embed_dim) * 0.01

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Args:
            indices: [Batch, Seq_Len] integer token IDs
        Returns:
            [Batch, Seq_Len, Embed_Dim]
        """
        # Using numpy indexing to retrieve embeddings for each index
        return self.weights[indices]

class PositionalEmbedding:
    """
    Fixed sinusoidal positional embeddings.
    Helps the model understand the relative/absolute position of tokens.
    """
    def __init__(self, max_seq_len: int, embed_dim: int):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Precompute positional encoding matrix
        # [Max_Seq_Len, Embed_Dim]
        pe = np.zeros((max_seq_len, embed_dim))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        # div_term = 1 / (10000 ** (2i / d_model))
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe

    def forward(self) -> np.ndarray:
        """
        Returns the precomputed positional encoding matrix.
        Returns:
            [Max_Seq_Len, Embed_Dim]
        """
        return self.pe

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    Consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, embed_dim: int, dim_feedforward: int):
        self.embed_dim = embed_dim
        self.dim_feedforward = dim_feedforward
        
        # W1: [Embed_Dim, Dim_FF]
        self.W1 = np.random.randn(embed_dim, dim_feedforward) * 0.01
        # b1: [Dim_FF]
        self.b1 = np.zeros(dim_feedforward)
        
        # W2: [Dim_FF, Embed_Dim]
        self.W2 = np.random.randn(dim_feedforward, embed_dim) * 0.01
        # b2: [Embed_Dim]
        self.b2 = np.zeros(embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
        Returns:
            [Batch, Seq_Len, Embed_Dim]
        """
        # 1. First linear projection: [Batch, Seq_Len, Dim_FF]
        # (x @ W1) + b1
        h = np.dot(x, self.W1) + self.b1
        
        # 2. ReLU activation: max(0, h)
        h = np.maximum(0, h)
        
        # 3. Second linear projection: [Batch, Seq_Len, Embed_Dim]
        # (h @ W2) + b2
        output = np.dot(h, self.W2) + self.b2
        return output

class LayerNorm:
    """
    Layer Normalization.
    Normalizes the activations across the feature dimension.
    """
    def __init__(self, embed_dim: int, eps: float = 1e-6):
        self.embed_dim = embed_dim
        self.eps = eps
        # Learnable parameters: Gamma (scale) and Beta (shift)
        # [Embed_Dim]
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
        Returns:
            [Batch, Seq_Len, Embed_Dim]
        """
        # Calculate mean and variance along the last dimension (Embed_Dim)
        # axis=-1 means we normalize each vector individually
        mean = np.mean(x, axis=-1, keepdims=True) # [Batch, Seq_Len, 1]
        std = np.std(x, axis=-1, keepdims=True)   # [Batch, Seq_Len, 1]
        
        # Normalize: (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / np.sqrt(std**2 + self.eps)
        
        # Scale and shift using learned parameters
        # [Batch, Seq_Len, Embed_Dim]
        return self.gamma * x_norm + self.beta
