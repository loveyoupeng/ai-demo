import numpy as np
from typing import Tuple, Dict

class TokenEmbedding:
    """
    Learned token embeddings.
    Maps integer token IDs to continuous vectors.

    Mathematical context:
    The embedding layer is a lookup table $E \in \mathbb{R}^{V \times D}$,
    where $V$ is the vocabulary size and $D$ is the embedding dimension.
    For an input sequence of indices $I \in \mathbb{Z}^{B \times L}$, 
    the output is $X \in \mathbb{R}^{B \times L \times D}$ where 
    $X_{b, l, d} = E_{I_{b, l}, d}$.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Initialize weights randomly
        # Shape: [Vocab_Size, Embed_Dim]
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
        # Output shape: [Batch, Seq_Len, Embed_Dim]
        return self.weights[indices]

    def backward(
        self, grad_output: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Args:
            grad_output: [Batch, Seq_Len, Embed_Dim] gradient from next layer
        Returns:
            dx: [Batch, Seq_Len, Embed_Dim] gradient w.r.t. input indices
            grads: {'weights': [Vocab_Size, Embed_Dim]}
        """
        batch_size, seq_len, embed_dim = grad_output.shape

        # Flatten indices and grad_output to use np.add.at
        rows = self.indices.ravel()
        grad_output_flat = grad_output.reshape(-1, embed_dim)

        # Accumulate gradients for weights
        np.add.at(self.grad_weights, rows, grad_output_flat)

        # Gradient w.r.t. indices is just grad_output for indexing ops
        dx = grad_output
        grads = self.get_grads()
        return dx, grads

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"weights": self.weights}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        if "weights" in params:
            self.weights = params["weights"]

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {"weights": self.grad_weights}


class PositionalEmbedding:
    """
    Fixed sinusoidal positional embeddings.
    Uses sine and cosine functions of different frequencies to encode 
    absolute positions.

    Mathematical context:
    $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
    $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$
    where $pos$ is the position and $i$ is the dimension index.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # [Max_Seq_Len, Embed_Dim]
        pe = np.zeros((max_seq_len, embed_dim))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        
        # Frequency term calculation in log-space
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self) -> np.ndarray:
        """
        Returns the precomputed positional encoding matrix.
        Shape: [Max_Seq_Len, Embed_Dim]
        """
        return self.pe

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Fixed embeddings have zero gradient
        return np.zeros_like(grad_output)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {}


class FeedForward:
    """
    Position-wise Feed-Forward Network (FFN).
    Standard transformer component: $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$.

    Consists of two linear transformations with a ReLU activation.
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
        self.x = x
        # 1. Linear 1: [Batch, Seq_Len, Dim_FF]
        self.z1 = np.dot(x, self.W1) + self.b1
        # 2. ReLU: [Batch, Seq_Len, Dim_FF]
        self.h = np.maximum(0, self.z1)
        # 3. Linear 2: [Batch, Seq_Len, Embed_Dim]
        output = np.dot(self.h, self.W2) + self.b2
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Args:
            grad_output: [Batch, Seq_Len, Embed_Dim]
        Returns:
            dx: [Batch, Seq_Len, Embed_Dim]
        """
        # 1. Backprop through second linear projection
        # h_flat: [Batch * Seq_Len, Dim_FF]
        # grad_output_flat: [Batch * Seq_Len, Embed_Dim]
        h_flat = self.h.reshape(-1, self.dim_feedforward)
        grad_output_flat = grad_output.reshape(-1, self.embed_dim)

        self.grad_W2 = np.dot(h_flat.T, grad_output_flat)
        self.grad_b2 = np.sum(grad_output, axis=(0, 1))

        # 2. Backprop through ReLU
        # grad_h: [Batch, Seq_Len, Dim_FF]
        grad_h = np.dot(grad_output, self.W2.T)
        grad_z1 = grad_h * (self.z1 > 0)

        # 3. Backprop through first linear projection
        self.grad_b1 = np.sum(grad_z1, axis=(0, 1))
        self.grad_W1 = np.dot(
            self.x.reshape(-1, self.embed_dim).T,
            grad_z1.reshape(-1, self.dim_feedforward),
        )

        # 4. Gradient w.r.t. input x
        grad_x = np.dot(grad_z1, self.W1.T)
        return grad_x

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            if k == "W1": self.W1 = v
            elif k == "b1": self.b1 = v
            elif k == "W2": self.W2 = v
            elif k == "b2": self.b2 = v

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.grad_W1, "b1": self.grad_b1,
            "W2": self.grad_W2, "b2": self.grad_b2,
        }


class LayerNorm:
    """
    Layer Normalization.
    Normalizes activations across the feature dimension (last axis).

    Mathematical context:
    $\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
    where $\mu$ is the mean and $\sigma^2$ is the variance of $x$ across the feature dimension.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        self.embed_dim = embed_dim
        self.eps = eps
        # Learnable params: [Embed_Dim]
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
        self.x = x
        # Mean and var over the feature dimension: [Batch, Seq_Len, 1]
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns dx and parameter gradients.
        """
        embed_dim = self.embed_dim
        # grad_output shape: [Batch, Seq_Len, Embed_Dim]
        
        # 1. Gradients w.r.t gamma and beta
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=(0, 1))
        self.grad_beta = np.sum(grad_output, axis=(0, 1))

        # 2. Gradient w.r.t. normalized x
        grad_x_norm = grad_output * self.gamma

        # 3. Gradient through normalization
        N = embed_dim
        sum_grad_x_norm_x_norm = np.sum(grad_x_norm * self.x_norm, axis=-1, keepdims=True)
        sum_grad_x_norm = np.sum(grad_x_norm, axis=-1, keepdims=True)

        grad_x = (1.0 / np.sqrt(self.var + self.eps)) * (
            grad_x_norm
            - (1.0 / N) * sum_grad_x_norm_x_norm
            - (1.0 / N) * self.mean * sum_grad_x_norm
        )

        return grad_x, self.get_grads()

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            if k == "gamma": self.gamma = v
            elif k == "beta": self.beta = v

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {"gamma": self.grad_gamma, "beta": self.grad_beta}

