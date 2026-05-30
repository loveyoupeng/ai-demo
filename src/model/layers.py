import numpy as np
from typing import Tuple, Dict


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
        self.grad_weights = np.zeros_like(self.weights)

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Args:
            indices: [Batch, Seq_Len] integer token IDs
        Returns:
            [Batch, Seq_Len, Embed_Dim]
        """
        # Store indices for backward pass
        self.indices = indices
        # Using numpy indexing to retrieve embeddings for each index
        return self.weights[indices]

    def backward(
        self, grad_output: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Args:
            grad_output: [Batch, Seq_Len, Embed_Dim] gradient from next layer
        Returns:
            [Batch, Seq_Len, Embed_Dim] gradient w.r.t. input indices, and gradients for weights
        """
        # grad_output is [Batch, Seq_Len, Embed_Dim]
        # We need to accumulate gradients for weights
        # Flatten indices and grad_output to use np.add.at
        batch_size, seq_len, embed_dim = grad_output.shape

        # Create flattened indices for weight update
        # self.indices is [Batch, Seq_Len]
        # We need to broadcast it or use it to index into weights
        rows = self.indices.ravel()
        grad_output_flat = grad_output.reshape(-1, embed_dim)

        np.add.at(self.grad_weights, rows, grad_output_flat)

        # Gradient w.r.t indices is just grad_output
        dx = grad_output
        grads = self.get_grads()
        return dx, grads

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"weights": self.weights}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {"weights": self.grad_weights}


class PositionalEmbedding:
    """
    Fixed sinusoidal positional embeddings.
    Helps the model understand the relative/absolute position of tokens.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Precompute positional encoding matrix using sine and cosine functions.
        # This allows the model to learn relative positions because for any fixed offset 'k',
        # PE(pos+k) can be represented as a linear function of PE(pos).
        # [Max_Seq_Len, Embed_Dim]
        pe = np.zeros((max_seq_len, embed_dim))
        position = np.arange(0, max_seq_len)[:, np.newaxis]

        # The frequency term: 1 / (10000 ** (2i / d_model))
        # We use the log-space implementation for numerical stability.
        # emb_dim // 2 because we apply it to both sin and cos components.
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

        # Apply sine to even indices (0, 2, 4...) and cosine to odd indices (1, 3, 5...)
        # position: [Max_Seq_Len, 1], div_term: [Embed_Dim/2]
        # product: [Max_Seq_Len, Embed_Dim/2]
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

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Since this is fixed, gradient is zero.
        """
        return np.zeros_like(grad_output)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {}


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

        # Gradients
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
        Returns:
            [Batch, Seq_Len, Embed_Dim]
        """
        # Cache for backward pass
        self.x = x

        # 1. First linear projection: [Batch, Seq_Len, Dim_FF]
        # (x @ W1) + b1
        self.z1 = np.dot(x, self.W1) + self.b1

        # 2. ReLU activation: max(0, z1)
        self.h = np.maximum(0, self.z1)

        # 3. Second linear projection: [Batch, Seq_Len, Embed_Dim]
        # (h @ W2) + b2
        output = np.dot(self.h, self.W2) + self.b2
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Args:
            grad_output: [Batch, Seq_Len, Embed_Dim] gradient from next layer
        Returns:
            [Batch, Seq_Len, Embed_Dim] gradient w.r.t. input x
        """
        # grad_output shape: [Batch, Seq_Len, Embed_Dim]

        # 3. Backprop through second linear projection
        # grad_W2 = h.T @ grad_output (with batch/seq dims)
        # h is [Batch, Seq_Len, Dim_FF]
        # grad_output is [Batch, Seq_Len, Embed_Dim]
        # We need to sum over Batch and Seq_Len
        h_flat = self.h.reshape(-1, self.dim_feedforward)
        grad_output_flat = grad_output.reshape(-1, self.embed_dim)

        self.grad_W2 = np.dot(h_flat.T, grad_output_flat)
        self.grad_b2 = np.sum(grad_output, axis=(0, 1))

        # grad_h = grad_output @ W2.T
        grad_h = np.dot(grad_output, self.W2.T)  # [Batch, Seq_Len, Dim_FF]

        # 2. Backprop through ReLU
        # grad_z1 = grad_h * (z1 > 0)
        grad_z1 = grad_h * (self.z1 > 0)

        # 1. Backprop through first linear projection
        self.grad_b1 = np.sum(grad_z1, axis=(0, 1))
        self.grad_W1 = np.dot(
            self.x.reshape(-1, self.embed_dim).T,
            grad_z1.reshape(-1, self.dim_feedforward),
        )

        # grad_x = grad_z1 @ W1.T
        grad_x = np.dot(grad_z1, self.W1.T)

        return grad_x

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.grad_W1,
            "b1": self.grad_b1,
            "W2": self.grad_W2,
            "b2": self.grad_b2,
        }


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

        # Gradients
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
        Returns:
            [Batch, Seq_Len, Embed_Dim]
        """
        # Cache for backward pass
        self.x = x

        # Calculate mean and variance along the last dimension (Embed_Dim)
        # axis=-1 means we normalize each vector individually
        self.mean = np.mean(x, axis=-1, keepdims=True)  # [Batch, Seq_Len, 1]
        self.var = np.var(x, axis=-1, keepdims=True)  # [Batch, Seq_Len, 1]

        # Normalize: (x - mean) / sqrt(var + eps)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)

        # Scale and shift using learned parameters
        # [Batch, Seq_Len, Embed_Dim]
        return self.gamma * self.x_norm + self.beta

    def backward(
        self, grad_output: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Args:
            grad_output: [Batch, Seq_Len, Embed_Dim] gradient from next layer
        Returns:
            [Batch, Seq_Len, Embed_Dim] gradient w.r.t. input x, and gradients for parameters
        """
        # grad_output: [Batch, Seq_Len, Embed_Dim]
        batch_size, seq_len, embed_dim = grad_output.shape

        # grad_gamma = sum(grad_output * x_norm, axis=(0, 1))
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=(0, 1))
        # grad_beta = sum(grad_output, axis=(0, 1))
        self.grad_beta = np.sum(grad_output, axis=(0, 1))

        # grad_x_norm = grad_output * gamma
        grad_x_norm = grad_output * self.gamma

        # Backprop through normalization:
        # grad_x = (1 / sqrt(var + eps)) * [grad_x_norm - (1/N) * sum(grad_x_norm * x_norm) - (mean/N) * sum(grad_x_norm)]
        # where N = embed_dim
        N = embed_dim
        sum_grad_x_norm_x_norm = np.sum(
            grad_x_norm * self.x_norm, axis=-1, keepdims=True
        )
        sum_grad_x_norm = np.sum(grad_x_norm, axis=-1, keepdims=True)

        grad_x = (1.0 / np.sqrt(self.var + self.eps)) * (
            grad_x_norm
            - (1.0 / N) * sum_grad_x_norm_x_norm
            - (1.0 / N) * self.mean * sum_grad_x_norm
        )

        return grad_x, self.get_grads()

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {"gamma": self.grad_gamma, "beta": self.grad_beta}
