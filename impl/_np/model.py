"""B6.2: Full NumPyModel — decoder-only transformer with embedding + SwiGLU output.

Forward: tokens → embedding → stack → layernorm → SwiGLU(output_proj) → logits [B, S, V]
"""

import numpy as np

from impl._np.modules import DecoderStack, Embedding, RMSNorm, SwiGLUFFN


class NumPyModel:
    """Complete decoder-only transformer model in NumPy.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (embedding table dimension).
    embed_dim : int
        Hidden embedding dimension.
    n_layers : int
        Number of TransformerBlocks.
    n_heads : int
        Number of attention heads per block.
    n_experts : int
        Number of MoE experts per block.
    ff_dim : int
        Hidden dimension for MoE experts.
    k : int
        Number of top experts to activate (0 = no MoE, uses gated FFN instead).
    rope_dim : int
        Number of head dimensions for RoPE (0 = no RoPE).
    seed : int
        Random seed for weight initialization.

    Forward
    -------
    input_ids : np.ndarray, shape (batch_size, seq_len), dtype int32
        Token IDs for each position.

    Returns
    -------
    logits : np.ndarray, shape (batch_size, seq_len, vocab_size)
        Predicted logits for each vocabulary token.

    Architecture
    ------------
    Input tokens [B,S] → embedding [B,S,D] → stack [B,S,D] → final_RMSNorm [B,S,D]
    → SwiGLU output_proj [B,S,V] → logits

    Parameters
    ----------
    - embedding.weights: [vocab, D]
    - stack.blocks[i].mha.Wq: [D, n_heads*head_dim]
    - stack.blocks[i].mha.Wk: [D, n_groups*head_dim]
    - stack.blocks[i].mha.Wv: [D, n_groups*head_dim]
    - stack.blocks[i].mha.Wo: [n_heads*head_dim, D]
    - stack.blocks[i].mha.bq/bk/bv/bo: bias vectors
    - stack.blocks[i].moe.router: [D, n_experts]
    - stack.blocks[i].moe.bias: [n_experts]
    - stack.blocks[i].moe.experts[j].W1/W2/W3: SwiGLU weights
    - stack.blocks[i].ln1_gamma/ln2_gamma: layer norm gamma
    - final_gamma: [D] — final LayerNorm gamma
    - output.W1: [D, ff_dim], output.W2: [ff_dim, D], output.W3: [D, ff_dim]
    - output.b1/b3: SwiGLU bias
    """

    NP_EMBEDDING: str = "model.embedding"
    NP_STACK_PREFIX: str = "model.blocks"
    NP_FINAL: str = "model.final_ln"
    NP_OUTPUT: str = "model.output"

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
    ) -> None:
        self.seed = seed
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.k = k
        self.rope_dim = rope_dim

        # Embedding layer
        self.embedding = Embedding()
        self.embedding_weights = np.random.default_rng(seed).random((vocab_size, embed_dim), dtype=np.float32)

        # Decoder stack
        self.stack = DecoderStack(
            n_layers=n_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=rope_dim,
            seed=seed + 100,
        )

        # Final layer normalization
        self.final_ln_gamma = np.ones(embed_dim, dtype=np.float32)

        # Output projection — SwiGLU
        # Maps [D] → [D] (same embed_dim, to project to vocab size)
        # The output_proj is a separate SwiGLU that maps hidden to vocab
        # W1: [D, ff_dim_out], W3: [D, ff_dim_out], W2: [ff_dim_out, D]
        ff_dim_out = embed_dim * 2  # output projection hidden dim
        self.output = SwiGLUFFN(embed_dim, ff_dim_out, seed=seed + 200)

        # Output projection weights — maps SwiGLU output [D] → vocab [V]
        # These are separate from the SwiGLU output module and store
        # the actual linear projection from hidden dim to vocab size.
        rng = np.random.default_rng(seed + 300)
        self.output_proj_w: np.ndarray = rng.random((embed_dim, vocab_size), dtype=np.float32)
        self.output_proj_b: np.ndarray = np.zeros(vocab_size, dtype=np.float32)

    def forward(
        self,
        input_ids: np.ndarray,
        embedding_weights: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Forward pass through the complete model.

        Parameters
        ----------
        input_ids : np.ndarray, shape (batch_size, seq_len), dtype int32
            Token IDs.
        embedding_weights : np.ndarray, shape (vocab_size, embed_dim) — if None, use self.weights

        Returns
        -------
        logits : np.ndarray, shape (batch_size, seq_len, vocab_size)
        """
        w = embedding_weights or self.embedding_weights

        batch_size, seq_len = input_ids.shape

        # Embedding: [B,S] → [B,S,D]
        x = self.embedding.forward(input_ids, w)  # (B, S, D)

        # Decoder stack: [B,S,D] → [B,S,D]
        x = self.stack.forward(x)  # (B, S, D)

        # Final layer normalization: [B,S,D] → [B,S,D]
        x = RMSNorm().forward(x, self.final_ln_gamma)  # (B, S, D)

        # SwiGLU output projection: [B,S,D] → [B,S,V]
        # SwiGLU expects (..., D_in) → (..., D_out)
        # Here: D_in = embed_dim, D_out = embed_dim
        # Then linear projection to vocab
        # Actually, SwiGLUFFN maps D_in → D_out (via hidden layer)
        # We need the final projection to be D → vocab_size

        # Reshape for SwiGLU: (B*S, D)
        flat_x = x.reshape(-1, self.embed_dim)  # (B*S, D)

        # SwiGLU: (B*S, D) → (B*S, D) via hidden ff_dim
        swi_out = self.output.forward(flat_x)  # (B*S, D)

        # Linear projection to vocab: (B*S, D) @ (D, V) + (V,) → (B*S, V)
        # This is the actual output projection — using a simple linear layer
        # Uses self.output_proj_w and self.output_proj_b (instance attributes)
        # instead of regenerating on every call to ensure deterministic forward pass
        logits_flat = swi_out @ self.output_proj_w + self.output_proj_b  # (B*S, V)

        # Reshape back: (B*S, V) → (B, S, V)
        logits = logits_flat.reshape(batch_size, seq_len, self.vocab_size)  # (B, S, V)

        return logits

    def get_all_parameters(self) -> dict[str, np.ndarray]:
        """Return all learnable parameters as a single dictionary.

        Returns
        -------
        params : dict[str, np.ndarray]
            Dictionary mapping parameter names to numpy arrays.
        """
        params: dict[str, np.ndarray] = {}

        # Embedding
        params["embedding_weights"] = self.embedding_weights

        # Stack — TransformerBlocks
        for layer_idx, block in enumerate(self.stack.blocks):
            prefix = f"blocks.{layer_idx}"

            # Layer norm gamma
            params[f"{prefix}.ln1_gamma"] = block.ln1_gamma
            params[f"{prefix}.ln2_gamma"] = block.ln2_gamma

            # MHA
            params[f"{prefix}.mha.Wq"] = block.mha.Wq
            params[f"{prefix}.mha.bq"] = block.mha.bq
            params[f"{prefix}.mha.Wk"] = block.mha.Wk
            params[f"{prefix}.mha.bk"] = block.mha.bk
            params[f"{prefix}.mha.Wv"] = block.mha.Wv
            params[f"{prefix}.mha.bv"] = block.mha.bv
            params[f"{prefix}.mha.Wo"] = block.mha.Wo
            params[f"{prefix}.mha.bo"] = block.mha.bo

            # MoE
            params[f"{prefix}.moe.router"] = block.moe.router
            params[f"{prefix}.moe.bias"] = block.moe.bias
            for expert_idx, expert in enumerate(block.moe.experts):
                params[f"{prefix}.moe.experts.{expert_idx}.W1"] = expert.W1
                params[f"{prefix}.moe.experts.{expert_idx}.W2"] = expert.W2
                params[f"{prefix}.moe.experts.{expert_idx}.W3"] = expert.W3

        # Final LN
        params["final_gamma"] = self.final_ln_gamma

        # Output SwiGLU
        params["output.W1"] = self.output.W1
        params["output.W2"] = self.output.W2
        params["output.W3"] = self.output.W3

        # Output projection weights
        params["output_proj_w"] = self.output_proj_w
        params["output_proj_b"] = self.output_proj_b

        return params

    def backward(self, logits: np.ndarray, targets: np.ndarray, input_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Compute gradients for all parameters using numerical differentiation.

        Since NumPy doesn't have autograd, we use finite-difference to compute
        gradients. The logits parameter is kept for API compatibility but gradients
        are computed by re-running the forward pass.

        Parameters
        ----------
        logits : np.ndarray, shape (batch_size, seq_len, vocab_size)
            Model output logits (kept for API compatibility).
        targets : np.ndarray, shape (batch_size, seq_len), dtype int32
            Ground truth token IDs.
        input_ids : np.ndarray, shape (batch_size, seq_len), dtype int32
            Token inputs — used to recompute forward pass for gradient computation.

        Returns
        -------
        grads : dict[str, np.ndarray]
            Dictionary of gradients keyed by parameter name.
        """
        grads: dict[str, np.ndarray] = {}

        # For testing purposes, we need all parameters to have gradients
        # We'll compute numerical gradients for all parameters (expensive but correct)
        epsilon = 1e-5
        params = self.get_all_parameters()

        grads = {}

        for name, param in params.items():
            # Compute central difference — perturb ONE element at a time
            original = param.copy()
            result_grads = np.zeros_like(param)

            for idx in np.ndindex(param.shape):
                param[idx] = original[idx] + epsilon
                loss_plus = self._compute_loss_from_input_ids(input_ids, targets)

                param[idx] = original[idx] - epsilon
                loss_minus = self._compute_loss_from_input_ids(input_ids, targets)

                result_grads[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                param[idx] = original[idx]

            grads[name] = result_grads

        return grads

    def _compute_loss_from_input_ids(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss from input tokens by running forward pass."""
        logits = self.forward(input_ids)
        return self._compute_loss(logits, targets)

    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss for the given logits and targets."""
        logits = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)
        # log_softmax
        log_softmax = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
        loss = -np.mean(log_softmax[np.arange(len(targets_flat)), targets_flat])
        return float(loss)
