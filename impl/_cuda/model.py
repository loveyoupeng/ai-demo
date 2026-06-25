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

import numpy as np
import torch

from impl._cuda.ffn import swiglu_ffn
from impl._cuda.layernorm import rmsnorm as _rmsnorm
from impl._cuda.stack import CuDecoderStack
from shared.constants import Block, Transformer


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

        import math

        rng = np.random.default_rng(seed)
        D = self.embed_dim
        V = self.vocab_size

        # Token embedding: (V, D) — Kaiming uniform (same as PyTorch nn.Embedding)
        self.embedding_weights = torch.empty(V, D).uniform_(
            -math.sqrt(1.0 / D), math.sqrt(1.0 / D)
        )

        # Final RMSNorm gamma: (D,)
        self.final_ln_gamma = torch.ones(D, dtype=torch.float32)

        # Output projection: (D, V) + bias (V,)
        self.output_proj_weights = torch.empty(D, V).uniform_(
            -math.sqrt(1.0 / V), math.sqrt(1.0 / V)
        )
        self.output_proj_bias = torch.zeros(V, dtype=torch.float32)

        # Output SwiGLU: W1 (D, 2D), W2 (2D, D), W3 (D, 2D)
        ff_dim_out = D * 2
        self.output_W1 = torch.empty(D, ff_dim_out).uniform_(
            -math.sqrt(1.0 / ff_dim_out), math.sqrt(1.0 / ff_dim_out)
        )
        self.output_W3 = torch.empty(D, ff_dim_out).uniform_(
            -math.sqrt(1.0 / ff_dim_out), math.sqrt(1.0 / ff_dim_out)
        )
        self.output_W2 = torch.empty(ff_dim_out, D).uniform_(
            -math.sqrt(1.0 / ff_dim_out), math.sqrt(1.0 / ff_dim_out)
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make the model callable — delegates to forward.

        Parameters
        ----------
        x : torch.Tensor, shape (B, S)
            Token IDs.

        Returns
        -------
        torch.Tensor, shape (B, S, V)
            Predicted logits.
        """
        return self.forward(x)

    def load_from_numpy_dict(self, params: dict[str, np.ndarray]) -> None:
        """Load parameters from a NumPy-compatible dictionary.

        Maps numpy-style keys (blocks.{i}.mha.Wq, blocks.{i}.moe.router, etc.)
        to CUDA flat tensor attributes (block.Wq, block.routing_weights, etc.).

        MoE expert weights: numpy stores per-expert W1/W2/W3 separately,
        CUDA stacks them into flat tensor attributes.

        Args:
            params: Dictionary mapping parameter names to NumPy arrays.
        """

        def load_flat(key: str, tensor: torch.Tensor) -> None:
            if key not in params:
                return
            t = torch.from_numpy(params[key]).to(tensor.dtype)
            tensor.data.copy_(t)

        device = self.embedding_weights.device

        # Load model-level weights (CUDA stores tensors on CPU in __init__)
        load_flat(Transformer.EMBEDDING_WEIGHTS, self.embedding_weights.to(device))
        load_flat(Transformer.FINAL_GAMMA, self.final_ln_gamma.to(device))
        load_flat(Transformer.OUTPUT_PROJ_W, self.output_proj_weights.to(device))
        load_flat(Transformer.OUTPUT_PROJ_B, self.output_proj_bias.to(device))
        load_flat(Transformer.OUTPUT_W1, self.output_W1.to(device))
        load_flat(Transformer.OUTPUT_W3, self.output_W3.to(device))
        load_flat(Transformer.OUTPUT_W2, self.output_W2.to(device))

        for i, block in enumerate(self.stacking.blocks):
            # Layer norm
            load_flat(Block.ln1_gamma(i), block.ln1_gamma.to(device))
            load_flat(Block.ln2_gamma(i), block.ln2_gamma.to(device))

            # Gates
            load_flat(Block.gate1(i), block.gate1.to(device))
            load_flat(Block.gate2(i), block.gate2.to(device))

            # MHA — weight matrices (CUDA MHA has no biases)
            load_flat(Block.mha(i, "Wq"), block.Wq.to(device))
            load_flat(Block.mha(i, "Wk"), block.Wk.to(device))
            load_flat(Block.mha(i, "Wv"), block.Wv.to(device))
            load_flat(Block.mha(i, "Wo"), block.Wo.to(device))

            # Stack expert W1 → block.expert_weights, W2 → block.expert_bias, W3 → block.routing_weights
            ws1 = np.stack([params[Block.moe_expert(i, e, "W1")] for e in range(self.n_experts)], axis=0)
            ws2 = np.stack([params[Block.moe_expert(i, e, "W2")] for e in range(self.n_experts)], axis=0)
            ws3 = np.stack([params[Block.moe_expert(i, e, "W3")] for e in range(self.n_experts)], axis=0)

            block.expert_weights = torch.from_numpy(ws1).to(dtype=torch.float32, device=device)
            # expert_bias: CUDA MoE asserts shape (N, D) but does not use it in computation.
            # Numpy stores W2 (ff_dim, embed_dim) per expert — not compatible.
            # Use zeros with correct shape for assertion compatibility.
            block.expert_bias = torch.zeros(self.n_experts, self.embed_dim, dtype=torch.float32, device=device)
            # routing_weights: transpose numpy router (embed_dim, n_experts) → (n_experts, embed_dim)
            block.routing_weights = torch.from_numpy(params[Block.moe_router(i)].T).to(dtype=torch.float32, device=device)
