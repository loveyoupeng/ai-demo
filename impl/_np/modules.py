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
