"""CUDAModel — full decoder-only transformer.

Forward: tokens → embedding → stack → layernorm → SwiGLU(output) → output_proj → logits

Architecture:
    Input:  tokens [B, S] (int64)
    │
    ├→ Embedding table lookup       [B, S, D]
    ├→ DecoderStack (n_layers)     [B, S, D]
    ├→ RMSNorm (final_ln)          [B, S, D]
    ├→ SwiGLU (output)             [B, S, D]
    └→ Linear (output_proj)        [B, S, V]
    │
    Output: logits [B, S, V]

Reference
---------
Vaswani et al. "Attention Is All You Need" (2017)
https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import torch

from impl._cuda.ffn import swiglu_ffn
from impl._cuda.layernorm import rmsnorm as _rmsnorm
from impl._cuda.stack import CuDecoderStack


class CUDAModel:
    """CUDA decoder-only transformer — full model with embedding and output projection.

    Uses CuDecoderStack for the transformer blocks and CUDA kernels for all computation.

    This class does NOT inherit from nn.Module — weights are stored as plain tensors
    (attributes) for parity checking against the NumPy implementation.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (number of unique tokens).
    embed_dim : int
        Hidden embedding dimension.
    n_layers : int
        Number of transformer blocks.
    n_heads : int
        Number of attention heads per block.
    n_experts : int
        Number of MoE experts per block.
    ff_dim : int
        Hidden dimension for MoE experts.
    stacking : CuDecoderStack
        Pre-configured decoder stack with all transformer blocks.
    k : int, optional
        Number of top experts to activate per token (default: 2).
    rope_dim : int, optional
        Number of head dimensions for RoPE (0 = disabled, default: 0).
    seed : int, optional
        Random seed for initialization (default: 0).

    Attributes
    ----------
    vocab_size : int
        Vocabulary size.
    embed_dim : int
        Hidden dimension.
    n_layers : int
        Number of blocks.
    n_heads : int
        Number of heads per block.
    n_experts : int
        Number of experts per block.
    ff_dim : int
        Hidden dimension for experts.
    embedding_weights : torch.Tensor, shape (V, D)
        Token embedding weight matrix.
    ln1_gamma : torch.Tensor, shape (D,)
        RMSNorm gamma for attention post-norm.
    ln2_gamma : torch.Tensor, shape (D,)
        RMSNorm gamma for MoE post-norm.
    gate1 : torch.Tensor, shape (1,)
        Learnable gate for attention residual (initialized to 0).
    gate2 : torch.Tensor, shape (1,)
        Learnable gate for MoE residual (initialized to 0).
    Wq, Wk, Wv, Wo : torch.Tensor, shape (D, D)
        Attention projection weights.
    final_ln_gamma : torch.Tensor, shape (D,)
        Final layer normalization gamma.
    output_proj_weights : torch.Tensor, shape (D, V)
        Output projection weight matrix.
    output_proj_bias : torch.Tensor, shape (V,)
        Output projection bias vector.
    output_W1 : torch.Tensor, shape (D, 2D)
        SwiGLU output W1 (gate path).
    output_W3 : torch.Tensor, shape (D, 2D)
        SwiGLU output W3 (proj path).
    output_W2 : torch.Tensor, shape (2D, D)
        SwiGLU output W2 (output projection).
    expert_weights : torch.Tensor, shape (N, D, D)
        MoE expert weight matrices.
    expert_bias : torch.Tensor, shape (N, D)
        MoE expert bias vectors.
    routing_weights : torch.Tensor, shape (N, D)
        Routing score weights for expert selection.

    Forward
    -------
    x : torch.Tensor, shape (batch_size, seq_len), dtype int64, on CUDA

    Returns
    -------
    logits : torch.Tensor, shape (batch_size, seq_len, vocab_size), dtype float32, on CUDA

    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        rope_dim: int = 0,
        seed: int = 0,
        stacking: CuDecoderStack | None = None,
    ) -> None:
        """Initialize CUDAModel with explicit weights.

        For parity checking, all weights should be initialized with the same
        seed as the NumPy/PyTorch reference model.

        Parameters
        ----------
        vocab_size : int
            Vocabulary size.
        embed_dim : int
            Embedding dimension.
        n_layers : int
            Number of transformer blocks.
        n_heads : int
            Number of attention heads.
        n_experts : int
            Number of MoE experts.
        ff_dim : int
            Feed-forward hidden dimension.
        k : int
            Number of top experts to activate.
        rope_dim : int
            Number of head dimensions for RoPE.
        seed : int
            Random seed for initialization.
        stacking : CuDecoderStack, optional
            Pre-configured decoder stack. If None, stack is created internally.

        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.ff_dim = ff_dim
        self.k = k
        self.rope_dim = rope_dim

        # Create or use provided stack
        if stacking is None:
            self.stacking = CuDecoderStack(
                n_layers=n_layers,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_experts=n_experts,
                ff_dim=ff_dim,
                k=k,
                rope_dim=rope_dim,
            )
        else:
            self.stacking = stacking

        # Initialize all weights using seed — same pattern as NumPy/PyTorch models
        self._init_weights(seed)

    def _init_weights(self, seed: int) -> None:
        """Initialize all model weights with a given seed.

        Parameters
        ----------
        seed : int
            Random seed for weight initialization.

        """
        if hasattr(self, "embedding_weights"):
            return  # Already initialized

        import numpy as np

        rng = np.random.default_rng(seed)
        D = self.embed_dim
        V = self.vocab_size

        # Token embedding: (V, D)
        self.embedding_weights = torch.tensor(
            rng.random((V, D)).astype(np.float32),
            dtype=torch.float32,
        )

        # Final RMSNorm gamma: (D,)
        self.final_ln_gamma = torch.ones(D, dtype=torch.float32)

        # Output projection: (D, V) + bias (V,)
        self.output_proj_weights = torch.tensor(
            rng.random((D, V)).astype(np.float32),
            dtype=torch.float32,
        )
        self.output_proj_bias = torch.zeros(V, dtype=torch.float32)

        # Output SwiGLU: W1 (D, 2D), W2 (2D, D), W3 (D, 2D)
        ff_dim_out = D * 2
        self.output_W1 = torch.tensor(
            rng.random((D, ff_dim_out)).astype(np.float32),
            dtype=torch.float32,
        )
        self.output_W3 = torch.tensor(
            rng.random((D, ff_dim_out)).astype(np.float32),
            dtype=torch.float32,
        )
        self.output_W2 = torch.tensor(
            rng.random((ff_dim_out, D)).astype(np.float32),
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete model.

        Parameters
        ----------
        x : torch.Tensor, shape (B, S), dtype int64
            Token IDs. Shape [batch_size, seq_len], dtype int64.

        Returns
        -------
        torch.Tensor, shape (B, S, V)
            Predicted logits for each vocabulary token.

        """
        # Embedding: [B, S] → [B, S, D]
        # Gather token embeddings — shape (B, S) × (V, D) → (B, S, D)
        B, S = x.shape
        x_flat = x.flatten()  # (B*S,)
        x = self.embedding_weights.to(x.device).index_select(0, x_flat.long())  # (B*S, D)
        x = x.view(B, S, x.shape[-1])  # (B, S, D)

        # Decoder stack: [B, S, D] → [B, S, D]
        x = self.stacking.forward(x)

        # Final layer normalization: [B, S, D] → [B, S, D]
        x = _rmsnorm(x, self.final_ln_gamma.to(x.device))

        # SwiGLU output projection: [B, S, D] → [B, S, D]
        x = swiglu_ffn(
            x,
            self.output_W1.to(x.device),
            self.output_W3.to(x.device),
            self.output_W2.to(x.device),
        )

        # Linear projection to vocab: [B, S, D] → [B, S, V]
        logits = x @ self.output_proj_weights.to(x.device) + self.output_proj_bias.to(x.device)

        return logits  # (B, S, V)
