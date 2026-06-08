from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from core.registry import registry


class PyTorchTokenEmbedding(nn.Module):
    r"""
    Learned token embeddings — PyTorch implementation.

    Maps integer token IDs to dense continuous vectors via a lookup table.

    **Mathematical context**

    The embedding layer defines a parameter matrix :math:`E \in \mathbb{R}^{V \times D}`,
    where :math:`V` is the vocabulary size and :math:`D` is the embedding dimension.
    For an input sequence of indices :math:`I \in \mathbb{Z}^{B \times L}`,
    the output is :math:`X \in \mathbb{R}^{B \times L \times D}` where

    .. math::

        X_{b, l, d} = E_{I_{b, l}, d}

    **Dimension tracking**

    ======================================  ================================
    Symbol                                  Shape
    ======================================  ================================
    Input ``indices``                       [B, L] (Batch \times Seq\_Len)
    ``embedding.weight`` (lookup table)     [V, D] (Vocab\_Size \times Embed\_Dim)
    Output :math:`X`                        [B, L, D]
    ``grad_output``                         [B, L, D]
    ``w.grad``                              [V, D]
    ======================================  ================================

    **How this maps to the NumPy implementation**

    - ``PyTorchTokenEmbedding`` is PyTorch's equivalent of the NumPy
      :class:`TokenEmbedding` in ``src/model/layers.py``.
    - The NumPy version builds ``self.weights`` via ``np.random.randn(vocab_size, embed_dim) * 0.01``
      and looks up via ``self.weights[indices]``.
    - In PyTorch, ``nn.Embedding(vocab_size, embed_dim)`` does the same lookup
      as a fused kernel; gradient accumulation is handled via ``scatter_add`` on
      the weight matrix's gradient.
    - The backward pass in NumPy uses ``np.add.at(self.grad_weights, rows, grad_output_flat)``
      to scatter gradients; PyTorch's ``nn.Embedding`` does this internally,
      and we expose it through ``get_grads()`` which returns the raw gradient tensor.
    - Both ``set_params`` / ``get_params`` / backward interfaces are kept
      identical so that parameter swaps between NumPy and PyTorch are transparent.

    **Tunable points for production**

    ==========  ========   =======  ===============================
    Param       Type       Range    Notes
    ==========  ========   =======  ===============================
    ``vocab_size``  ``int``  ``4–100000+``  Vocabulary size; choose based on tokenizer (BPE, WordPiece, etc.)
    ``embed_dim``   ``int``  ``32–8192``    Embedding dimension; larger → higher expressivity, more memory
    ==========  ========   =======  ===============================

    >>> # Typical small model (toy / experimentation)
    >>> tok = PyTorchTokenEmbedding(vocab_size=1000, embed_dim=64)
    >>> # Typical medium model (GPT-2 scale)
    >>> tok = PyTorchTokenEmbedding(vocab_size=50257, embed_dim=768)
    >>> # Verify forward shape
    >>> import torch
    >>> idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    >>> out = tok(idx)
    >>> out.shape
    torch.Size([2, 3, 64])
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # Embedding table: [V, D]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.indices = None
        registry.register("pytorch", "embedding.weights", "embedding.weight")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings.

        Args:
            indices: Integer token IDs [Batch, Seq_Len]

        Returns:
            Embedding vectors [Batch, Seq_Len, Embed_Dim]
        """
        self.indices = indices
        return self.embedding(indices)

    def backward(
        self, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute gradient w.r.t. input indices and parameter gradients.

        Args:
            grad_output: [Batch, Seq_Len, Embed_Dim] gradient from the next layer

        Returns:
            dx: Gradient w.r.t. input, same shape as forward output [Batch, Seq_Len, Embed_Dim]
            grads: Dictionary with keyed gradient tensor for the embedding weights
        """
        output = self.embedding(self.indices)
        loss = (output * grad_output).sum()
        loss.backward()
        if self.embedding.weight.grad is None:
            raise RuntimeError("embedding.weight.grad is None")
        # For look-up operations, gradient flows through the values so dx = grad_output
        dx = grad_output
        grads = self.get_grads()
        return dx, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        """Return the trainable parameters as a dictionary."""
        return {"embedding.weights": self.embedding.weight}

    def set_params(self, params: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Load parameters from a dictionary. Accepts both NumPy arrays and torch tensors.

        Args:
            params: Dictionary mapping parameter names to tensor/array values
        """
        if "embedding.weights" in params:
            val = params["embedding.weights"]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            with torch.no_grad():
                self.embedding.weight.copy_(val)

    def get_grads(self) -> dict[str, torch.Tensor]:
        """Return the accumulated gradients for trainable parameters."""
        grads = {}
        if self.embedding.weight.grad is not None:
            grads["embedding.weights"] = self.embedding.weight.grad
        return grads


class PyTorchLayerNorm(nn.Module):
    r"""
    Layer Normalization — PyTorch implementation.

    Normalizes activations across the feature (last) dimension with per-channel
    learnable scale and shift parameters.

    **Mathematical context**

    For an input :math:`x \in \mathbb{R}^{B \times L \times D}`, layer norm computes:

    .. math::

        \mu = \frac{1}{D}\sum_{d=1}^{D} x_{b,l,d}, \quad
        \sigma^2 = \frac{1}{D}\sum_{d=1}^{D}(x_{b,l,d} - \mu)^2

    .. math::

        \hat{x}_{b,l} = \frac{x_{b,l} - \mu}{\sqrt{\sigma^2 + \epsilon}}

    .. math::

        y_{b,l} = \gamma \odot \hat{x}_{b,l} + \beta

    **Dimension tracking**

    ======================================  ==============================
    Symbol                                  Shape
    ======================================  ==============================
    Input ``x``                             [B, L, D]
    ``x.mean`` (keep\_dim)                  [B, L, 1]
    ``x.var`` (keep\_dim)                   [B, L, 1]
    ``x_norm`` (normalized)                 [B, L, D]
    ``gamma`` (learnable)                   [D]
    ``beta`` (learnable)                    [D]
    Output                                  [B, L, D]
    ``grad_output``                         [B, L, D]
    ``w.grad`` (weight)                     [D]
    ``b.grad`` (bias)                       [D]
    ======================================  ==============================

    **How this maps to the NumPy implementation**

    - ``PyTorchLayerNorm`` is the PyTorch equivalent of the NumPy
      :class:`LayerNorm` in ``src/model/layers.py``.
    - The NumPy version tracks ``self.gamma`` and ``self.beta`` as plain arrays
      computed manually in ``forward`` and ``backward``.
    - PyTorch's ``nn.Parameter(torch.ones(embed_dim))`` and ``torch.zeros``
      create the same learnable vectors; ``self.eps`` defaults to ``1e-6`` in both.
    - The backward pass manually detaches and re-executes the forward algebra to
      compute exact gradients matching the NumPy formulas from
      :math:`\frac{1}{\sqrt{\sigma^2+\epsilon}}
      (\tilde{g} - \bar{\tilde{g}} - \hat{x}(\tilde{g} \odot \hat{x})\text{mean})`
      which is the well-known LayerNorm gradient (see original paper
      `Ba et al., 2016 <https://arxiv.org/abs/1607.06450>`__).
    - Gradient accumulation uses ``torch.sum(..., dim=tuple(range(x.ndim - 1)))``
      in PyTorch, which is equivalent to ``np.sum(..., axis=(0, 1))`` in NumPy.
    - Both ``backward`` and ``forward`` preserve the computation graph so that
      gradients can be chained through a full transformer stack.

    **Tunable points for production**

    =========  ========   =======  ===============================
    Param      Type       Range    Notes
    =========  ========   =======  ===============================
    ``embed_dim``  ``int``  ``32–8192``  Must match the model dimension; controls vector width
    ``eps``      ``float``  ``1e-6`` (default)  Small constant for numerical stability; rarely changed
    =========  ========   =======  ===============================

    >>> import torch
    >>> # Typical small model
    >>> ln = PyTorchLayerNorm(embed_dim=64)
    >>> x = torch.randn(2, 8, 64)
    >>> out = ln(x)
    >>> out.shape
    torch.Size([2, 8, 64])
    >>> # After forward, gamma is 1.0 and beta is 0.0 by default
    >>> (ln.gamma == 1.0).all().item()
    True
    >>> (ln.beta == 0.0).all().item()
    True
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale: [D]
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        # Learnable shift: [D]
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        registry.register("pytorch", "ln.gamma", "gamma")
        registry.register("pytorch", "ln.beta", "beta")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]

        Returns:
            Normalized and scaled output [Batch, Seq_Len, Embed_Dim]
        """
        self.x = x
        self.x_mean = x.mean(dim=-1, keepdim=True)
        self.x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        self.x_norm = (x - self.x_mean) / torch.sqrt(self.x_var + self.eps)
        # Scale and shift: [D] * [B,L,D] + [D] -> [B,L,D]
        return self.gamma * self.x_norm + self.beta

    def backward(
        self, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Manual backward pass for layer normalization.

        Matches the NumPy LayerNorm backward formula:
        ``d_x = (1 / sqrt(var + eps)) * (d_x_norm - mean(d_x_norm) - x_norm * mean(d_x_norm * x_norm))``

        Args:
            grad_output: Gradient from downstream layers [Batch, Seq_Len, Embed_Dim]

        Returns:
            grad_x: Gradient w.r.t. input [Batch, Seq_Len, Embed_Dim]
            grads: Dictionary with ``"weight"`` (gamma grad) and ``"bias"`` (beta grad)
        """
        # Manual backward to preserve the computation graph for chaining.
        # Matching the NumPy backward computation from
        # https://arxiv.org/abs/1607.06450
        x = self.x
        var = self.x_var
        x_norm = self.x_norm

        eps = self.eps
        gamma = self.gamma

        # Gradient w.r.t. normalized input
        # grad_x_norm shape: [B, L, D]
        grad_x_norm = grad_output * gamma
        mean_grad_x_norm = torch.mean(grad_x_norm, dim=-1, keepdim=True)
        mean_grad_x_norm_x_norm = torch.mean(grad_x_norm * x_norm, dim=-1, keepdim=True)

        # Gradient w.r.t. input x
        # grad_x shape: [B, L, D]
        grad_x = (1.0 / torch.sqrt(var + eps)) * (
            grad_x_norm - mean_grad_x_norm - x_norm * mean_grad_x_norm_x_norm
        )

        # Parameter gradients
        # Sum over batch and sequence dimensions to get [D]
        grads = {
            "weight": torch.sum(grad_output * x_norm, dim=tuple(range(x.ndim - 1))),
            "bias": torch.sum(grad_output, dim=tuple(range(x.ndim - 1))),
        }

        return grad_x, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        """Return trainable parameters as a dictionary."""
        return {"ln.gamma": self.gamma, "ln.beta": self.beta}

    def set_params(self, params: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Load parameters from a dictionary. Accepts both NumPy arrays and torch tensors.

        Args:
            params: Dictionary mapping ``"ln.gamma"`` / ``"ln.beta"`` to values
        """
        if "ln.gamma" in params:
            val = params["ln.gamma"]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            with torch.no_grad():
                self.gamma.copy_(val)
        if "ln.beta" in params:
            val = params["ln.beta"]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            with torch.no_grad():
                self.beta.copy_(val)

    def get_grads(self) -> dict[str, torch.Tensor]:
        """Return the accumulated gradients for trainable parameters."""
        grads = {}
        if self.gamma.grad is not None:
            grads["ln.gamma"] = self.gamma.grad
        if self.beta.grad is not None:
            grads["ln.beta"] = self.beta.grad
        return grads


class PyTorchFeedForward(nn.Module):
    r"""
    Position-wise Feed-Forward Network (FFN) — PyTorch implementation.

    A two-layer fully-connected network with ReLU activation applied identically
    at every position in the sequence (hence "position-wise").

    **Mathematical context**

    For an input :math:`x \in \mathbb{R}^{B \times L \times D}`:

    .. math::

        z_1 = x W_1 + b_1, \quad h = \text{ReLU}(z_1), \quad y = h W_2 + b_2

    Where :math:`W_1 \in \mathbb{R}^{D \times D_{ff}}`, :math:`b_1 \in \mathbb{R}^{D_{ff}}`,
    :math:`W_2 \in \mathbb{R}^{D_{ff} \times D}`, :math:`b_2 \in \mathbb{R}^{D}`.

    **Dimension tracking**

    ==============================  ================================================
    Symbol                          Shape
    ==============================  ================================================
    Input ``x``                     [B, L, D] (Batch \times Seq\_Len \times Embed\_Dim)
    ``z1`` (pre-activation)        [B, L, D\_{ff}] (intermediate hidden dimension)
    ``h`` (ReLU activation)        [B, L, D\_{ff}]
    ``w1``                          [D, D\_{ff}]
    ``b1``                          [D\_{ff}]
    ``w2``                          [D\_{ff}, D]
    ``b2``                          [D]
    Output                          [B, L, D]
    ``grad_output``                 [B, L, D]
    ``grad(w1)``                    [D, D\_{ff}]
    ``grad(b1)``                    [D\_{ff}]
    ``grad(w2)``                    [D\_{ff}, D]
    ``grad(b2)``                    [D]
    ==============================  ================================================

    **How this maps to the NumPy implementation**

    - ``PyTorchFeedForward`` is the PyTorch equivalent of the NumPy
      :class:`FeedForward` in ``src/model/layers.py``.
    - The NumPy version uses ``np.dot(x, self.W1) + self.b1`` for matrix
      multiplication.  PyTorch's ``torch.matmul(x, self.w1) + self.b1``
      is numerically equivalent.
    - ReLU is ``np.maximum(0, z1)`` in NumPy and ``nn.functional.relu(z1)`` in PyTorch.
    - The NumPy backward computes each gradient by reshaping to 2-D
      (``self.h.reshape(-1, self.dim_feedforward)``) and flattening across
      batch + sequence dimensions.  The PyTorch version uses ``autograd``
      on a detached forward pass (``x.detach().requires_grad_(True)`` and
      re-executing the forward algebra) to obtain identical gradient tensors
      without relying on the built-in computation graph from a previous pass.
    - The parameter keys ``ffn.w1``, ``ffn.b1``, ``ffn.w2``, ``ffn.b2``
      follow the same convention used in the NumPy ``get_grads()`` / ``set_params()``
      interface.

    **Tunable points for production**

    ==========   ========   =======  ===============================
    Param        Type       Range    Notes
    ==========   ========   =======  ===============================
    ``embed_dim``   ``int``  ``32–8192``  Model dimension; must match surrounding layers
    ``dim_ff``      ``int``  ``embed_dim * 2 .. 4 * embed_dim``  Intermediate width; common factor is 4 (GPT-style) or 2
    ==========   ========   =======  ===============================

    >>> import torch
    >>> # Typical small model (GPT-2 small: embed=768, ff=3072 = 4x)
    >>> ffn = PyTorchFeedForward(embed_dim=768, dim_ff=3072)
    >>> x = torch.randn(2, 8, 768)
    >>> out = ffn(x)
    >>> out.shape
    torch.Size([2, 8, 768])
    >>> # Medium model (GPT-2 medium: embed=1024, ff=4096)
    >>> ffn = PyTorchFeedForward(embed_dim=1024, dim_ff=4096)
    """

    def __init__(self, embed_dim: int, dim_ff: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_ff = dim_ff
        # W1: [D, D_ff], initialized with small random values
        self.w1 = nn.Parameter(torch.randn(embed_dim, dim_ff) * 0.01)
        # b1: [D_ff], zero-initialized
        self.b1 = nn.Parameter(torch.zeros(dim_ff))
        # W2: [D_ff, D], initialized with small random values
        self.w2 = nn.Parameter(torch.randn(dim_ff, embed_dim) * 0.01)
        # b2: [D], zero-initialized
        self.b2 = nn.Parameter(torch.zeros(embed_dim))
        registry.register("pytorch", "ffn.w1", "w1")
        registry.register("pytorch", "ffn.b1", "b1")
        registry.register("pytorch", "ffn.w2", "w2")
        registry.register("pytorch", "ffn.b2", "b2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Position-wise FFN with ReLU activation.

        Computes: ``output = max(0, x @ w1 + b1) @ w2 + b2``

        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]

        Returns:
            FFN output [Batch, Seq_Len, Embed_Dim]
        """
        self.x = x
        # Linear 1: [B,L,D] @ [D,D_ff] + [D_ff] -> [B,L,D_ff]
        self.z1 = torch.matmul(x, self.w1) + self.b1
        # ReLU: [B,L,D_ff]
        self.h = torch.nn.functional.relu(self.z1)
        # Linear 2: [B,L,D_ff] @ [D_ff,D] + [D] -> [B,L,D]
        self.output = torch.matmul(self.h, self.w2) + self.b2
        return self.output

    def backward(
        self, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute gradients via detached forward pass.

        Args:
            grad_output: Gradient from downstream layers [Batch, Seq_Len, Embed_Dim]

        Returns:
            grad_x: Gradient w.r.t. input [Batch, Seq_Len, Embed_Dim]
            grads: Dictionary with keyed gradients for w1, b1, w2, b2
        """
        x = self.x.detach().requires_grad_(True)
        z1 = torch.matmul(x, self.w1) + self.b1
        h = torch.nn.functional.relu(z1)
        output = torch.matmul(h, self.w2) + self.b2
        loss = (output * grad_output).sum()

        self.zero_grad()
        loss.backward()

        grad_x = x.grad
        assert grad_x is not None, "x.grad should not be None after backward"
        grads = self.get_grads()
        return grad_x, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        """Return trainable parameters as a dictionary."""
        return {
            "ffn.w1": self.w1,
            "ffn.b1": self.b1,
            "ffn.w2": self.w2,
            "ffn.b2": self.b2,
        }

    def set_params(self, params: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Load parameters from a dictionary. Accepts both NumPy arrays and torch tensors.

        Args:
            params: Dictionary with keys ``"ffn.w1"``, ``"ffn.b1"``, ``"ffn.w2"``, ``"ffn.b2"``
        """
        for k in ["w1", "b1", "w2", "b2"]:
            canonical_key = f"ffn.{k}"
            if canonical_key in params:
                val = params[canonical_key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    getattr(self, k).copy_(val)

    def get_grads(self) -> dict[str, torch.Tensor]:
        """Return the accumulated gradients for trainable parameters."""
        grads = {}
        if self.w1.grad is not None:
            grads["ffn.w1"] = self.w1.grad
        if self.b1.grad is not None:
            grads["ffn.b1"] = self.b1.grad
        if self.w2.grad is not None:
            grads["ffn.w2"] = self.w2.grad
        if self.b2.grad is not None:
            grads["ffn.b2"] = self.b2.grad
        return grads


class PyTorchPositionalEmbedding(nn.Module):
    r"""
    Fixed sinusoidal positional embeddings — PyTorch implementation.

    Injects absolute position information into the embedding space using
    sine and cosine functions of geometrically increasing frequency.

    **Mathematical context**

    For position :math:`pos \in [0, L)` and dimension :math:`i \in [0, D)`:

    .. math::

        \text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\,2i / D}}\right)

    .. math::

        \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\,2i / D}}\right)

    The resulting matrix :math:`P \in \mathbb{R}^{L \times D}` is added
    element-wise to the token embeddings, enabling the model to attend to
    relative positions.

    **Dimension tracking**

    ==============================  ================================================
    Symbol                          Shape
    ==============================  ================================================
    ``pe`` (stored buffer)         [Max\_Seq\_Len, D]
    ``pe[:L, :]`` (used slice)     [L, D]
    Input ``x``                     [B, L, D]
    Output ``x + pe``               [B, L, D]
    ``grad_output``                 [B, L, D]
    Gradient w.r.t. pe              [Max\_Seq\_Len, D] (zeros — fixed embedding)
    ==============================  ================================================

    **How this maps to the NumPy implementation**

    - ``PyTorchPositionalEmbedding`` is the PyTorch equivalent of the NumPy
      :class:`PositionalEmbedding` in ``src/model/layers.py``.
    - Both versions compute the positional encoding identically:
      ``position * div_term`` where ``div_term = exp(arange(0, D, 2) * -(log(10000)/D))``.
    - The NumPy version stores ``self.pe`` as a plain ``np.ndarray``;
      PyTorch registers it as a buffer via ``register_buffer("pe", pe)``
      so it moves across devices automatically but is **not** a trainable
      parameter.
    - The backward pass returns zero gradients because positional embeddings
      are fixed and never updated during training.

    **Tunable points for production**

    ================  ========   =======  ===============================
    Param             Type       Range    Notes
    ================  ========   =======  ===============================
    ``max_seq_len``     ``int``  ``512..8192+``  Maximum sequence length; determines PE matrix size
    ``embed_dim``       ``int``  ``32–8192``  Must match the model embedding dimension
    ================  ========   =======  ===============================

    >>> import torch
    >>> # Standard GPT-2 configuration
    >>> pe = PyTorchPositionalEmbedding(max_seq_len=512, embed_dim=768)
    >>> x = torch.randn(2, 8, 768)
    >>> out = pe(x)
    >>> out.shape
    torch.Size([2, 8, 768])
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Positional encoding matrix: [Max_Seq_Len, Embed_Dim]
        pe = torch.zeros((max_seq_len, embed_dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.

        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]

        Returns:
            Token + position embeddings [Batch, Seq_Len, Embed_Dim]
        """
        self.x = x
        pe = self.get_buffer("pe")  # type: ignore[arg-type]
        return x + pe[: x.shape[1], :]

    def backward(
        self, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Positional embeddings are fixed (non-trainable), so gradient w.r.t. input
        passes through unchanged and gradient w.r.t. pe is zero.

        Args:
            grad_output: [Batch, Seq_Len, Embed_Dim]

        Returns:
            grad_output unchanged as input grad
            empty dict for parameter grads
        """
        pe = self.get_buffer("pe")  # type: ignore[arg-type]
        return grad_output, {"pe": torch.zeros_like(pe)}

    def get_params(self) -> dict[str, torch.Tensor]:
        """Return the positional encoding buffer (included for parity, not trainable)."""
        return {"pos.pe": self.get_buffer("pe")}  # type: ignore[return-value]

    def set_params(self, params: dict[str, object]) -> None:
        """Positional embeddings are fixed — no parameters to load."""
        pass

    def get_grads(self) -> dict[str, torch.Tensor]:
        """Return empty dictionary since positional embeddings have no trainable parameters."""
        return {}
