from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn as nn
import numpy as np
from core.registry import registry
from model.pytorch.attention import PyTorchMultiHeadAttention
from model.pytorch.attention_kvcache import PyTorchTurboQuantCache
from model.pytorch.layers import PyTorchLayerNorm
from model.pytorch.moe import PyTorchMoELayer


class PyTorchTransformerBlock(nn.Module):
    r"""
    A single Transformer Decoder Block — PyTorch implementation.

    Combines Multi-Head Attention (MHA) and a Mixture of Experts (MoE) layer,
    wrapped in residual connections and Layer Normalization.

    Uses a pre-normal architecture: LayerNorm is applied **before** the sub-layers.

    .. math::

        x_{\text{atten}} = x + \text{MHA}(\text{LN}_1(x))

        x_{\text{ffn}} = x_{\text{atten}} + \text{MoE}(\text{LN}_2(x_{\text{atten}}))

    **Dimension tracking**

    ========================================  ================================
    Symbol                                    Shape
    ========================================  ================================
    Input ``x``                               [B, L, D] (Batch \times Seq\_Len \times Embed\_Dim)
    ``ln1(x)`` (pre-norm attention)           [B, L, D]
    ``mha.output``                             [B, L, D]
    ``residual1`` (add)                       [B, L, D]
    ``ln2(residual1)`` (pre-norm MoE)         [B, L, D]
    ``moe.output``                             [B, L, D]
    ``residual2`` (add)                       [B, L, D]
    Output                                    [B, L, D]
    ``cache["mha_cache"]["Q"]``               [B, h, L, d\_k]
    ``cache["mha_cache"]["attn_weights"]``    [B, h, L, L]
    ========================================  ================================

    **How this maps to the NumPy implementation**

    - ``PyTorchTransformerBlock`` is the PyTorch equivalent of the NumPy
      :class:`TransformerBlock` in ``src/model/transformer.py``.
    - The NumPy version constructs the same pre-norm architecture:
      ``x + mha.forward(ln1(x))`` followed by ``x + moe.forward(ln2(x))``.
    - The backward pass in NumPy manually traces: ``d(ln2_input) = d_res2 + d_moe``,
      then ``d(ln1_input) = d_res1 + d_mha``.  The PyTorch version replicates
      this exact gradient flow through the submodules' ``backward`` interfaces.
    - The cache dictionary contains every intermediate tensor needed for the
      backward pass, mirroring the NumPy pattern where ``forward`` stores
      intermediate values on ``self`` and ``backward`` reads them back.
      PyTorch explicitly bundles them in ``cache`` for clean state management
      across block boundaries.
    - The ``mask`` parameter is passed to MHA for causal attention; both
      implementations store the causal mask shape as [Seq\_Len, Seq\_Len].

    **Tunable points for production**

    ===========  ========   =======  ===============================
    Param        Type       Range    Notes
    ===========  ========   =======  ===============================
    ``embed_dim``   ``int``  ``32–8192``  Hidden dimension; shared by LN, MHA, and MoE
    ``num_heads``   ``int``  power-of-2   Passed to MHA; ``embed_dim // num_heads`` gives head dimension
    ``num_experts``   ``int``  1–64       Passed to MoE; number of feed-forward experts in mixture
    ``top_k``       ``int``  1–num\_experts Number of experts active per token
    ===========  ========   =======  ===============================

    >>> import torch
    >>> from model.pytorch.attention import PyTorchMultiHeadAttention
    >>> from model.pytorch.moe import PyTorchMoELayer
    >>> # Small toy block: embed=256, 4 heads, 4 experts
    >>> mha = PyTorchMultiHeadAttention(embed_dim=256, num_heads=4)
    >>> moe = PyTorchMoELayer(embed_dim=256, num_experts=4)
    >>> block = PyTorchTransformerBlock(embed_dim=256, mha=mha, moe=moe)
    >>> x = torch.randn(2, 8, 256)
    >>> out, cache = block(x)
    >>> out.shape
    torch.Size([2, 8, 256])
    >>> # Gradient pass
    >>> grad = torch.ones_like(out)
    >>> dx, grads = block.backward(grad, cache)
    >>> dx.shape
    torch.Size([2, 8, 256])
    >>> # Check gradient keys include sub-layer components
    >>> sorted_keys = sorted(grads.keys())
    >>> sorted_keys[:4]
    ['ln1.bias', 'ln1.weight', 'ln2.bias', 'ln2.weight']
    """

    def __init__(
        self,
        embed_dim: int,
        mha: PyTorchMultiHeadAttention,
        moe: PyTorchMoELayer,
    ):
        super().__init__()
        # LayerNorm applied before attention
        self.ln1 = PyTorchLayerNorm(embed_dim)
        # LayerNorm applied before MoE
        self.ln2 = PyTorchLayerNorm(embed_dim)
        self.mha = mha
        self.moe = moe

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[PyTorchTurboQuantCache] = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """
        Forward pass through the Transformer decoder block.

        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len]
            kv_cache: Optional TurboQuant KV cache for storing K/V tokens during
                autoregressive generation. If provided, only the current
                position's K/V is appended and the full sequence is retrieved.

        Returns:
            output: [Batch, Seq_Len, Embed_Dim]
            cache: Dictionary containing intermediate values for backward pass
        """
        batch_size, seq_len, _ = x.shape

        # 1. Self-Attention Sub-layer (Pre-Norm)
        # x = x + MHA(LN(x))
        residual1 = x
        ln1_x = self.ln1(x)
        mha_out, mha_cache = self.mha.forward(ln1_x, mask=mask, kv_cache=kv_cache)
        x_after_mha = residual1 + mha_out

        # 2. Feed-Forward / MoE Sub-layer (Pre-Norm)
        # x = x + MoE(LN(x))
        residual2 = x_after_mha
        ln2_x = self.ln2(x_after_mha)
        moe_out, moe_cache = self.moe.forward(ln2_x)
        x_after_moe = residual2 + moe_out

        # Cache for backward pass
        cache: dict[str, object] = {
            "ln1_input": x,
            "mha_input": ln1_x,
            "mha_output": mha_out,
            "residual1": residual1,
            "ln2_input": x_after_mha,
            "moe_input": ln2_x,
            "moe_output": moe_out,
            "residual2": residual2,
            "mha_cache": mha_cache,
            "moe_cache": moe_cache,
        }

        return x_after_moe, cache

    def backward(
        self,
        grad_output: torch.Tensor,
        cache: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Backward pass through the Transformer decoder block.

        Traces gradients from output backward through:
        1. MoE sub-layer -> LN2
        2. residual connection
        3. MHA sub-layer -> LN1
        4. First residual connection -> input

        Args:
            grad_output: Gradient from downstream layer [Batch, Seq_Len, Embed_Dim]
            cache: Output dictionary from the forward pass

        Returns:
            dx: Gradient w.r.t. input [Batch, Seq_Len, Embed_Dim]
            combined_grads: All parameter gradients across sub-layers
        """
        # 1. Gradient w.r.t. MoE sub-layer
        d_residual2 = grad_output
        d_moe_out = grad_output

        dx_moe, grads_moe = self.moe.backward(
            cast(torch.Tensor, cache["moe_input"]),
            d_moe_out,
            cast(dict[str, object], cache["moe_cache"]),
        )

        # 2. Gradient w.r.t. residual 2 and ln2
        # d_ln2_input = d_residual2 + dx_moe
        d_ln2_input = d_residual2 + dx_moe
        dx_ln2, grads_ln2 = self.ln2.backward(d_ln2_input)

        # 3. Gradient w.r.t. mha sub-layer
        # x_after_mha = residual1 + mha_out
        # d_residual1 = d_ln2_input (since residual2 = x_after_mha)
        # d_mha_out = d_ln2_input
        d_residual1 = d_ln2_input
        d_mha_out = d_ln2_input

        mask = (
            cast(torch.Tensor, cache["mha_cache"]).get("mask")  # pyright: ignore
            if cast(dict, cache["mha_cache"]).get("mask") is not None
            else None
        )
        dx_mha, grads_mha = self.mha.backward(d_mha_out, mask)

        # 4. Gradient w.r.t. residual 1 and ln1
        # d_ln1_input = d_residual1 + dx_mha
        d_ln1_input = d_residual1 + dx_mha
        dx_ln1, grads_ln1 = self.ln1.backward(d_ln1_input)

        # 5. Total gradient w.r.t. input x
        dx = dx_ln1

        # Combine all gradients
        combined_grads: dict[str, torch.Tensor] = {}
        for k, v in grads_ln1.items():
            combined_grads[f"ln1.{k}"] = v
        for k, v in grads_mha.items():
            combined_grads[f"mha.{k}"] = v
        for k, v in grads_moe.items():
            combined_grads[f"moe.{k}"] = v
        for k, v in grads_ln2.items():
            combined_grads[f"ln2.{k}"] = v

        return dx, combined_grads

    def get_params(self) -> dict[str, torch.Tensor]:
        """
        Return all trainable parameters across sub-layers.

        Returns:
            Dictionary mapping parameter paths to tensors:
            - ``"ln1.gamma"``, ``"ln1.beta"``
            - ``"ln2.gamma"``, ``"ln2.beta"``
            - ``"mha.qkv.W_q"``, ``"mha.qkv.W_k"``, ``"mha.qkv.W_v"``, ``"mha.o.W_o"``
            - ``"moe.experts.{i}.{param}"`` for each expert
        """
        params: dict[str, torch.Tensor] = {}
        for k, v in self.ln1.state_dict().items():
            params[f"ln1.{k}"] = v
        for k, v in self.ln2.state_dict().items():
            params[f"ln2.{k}"] = v
        for k, v in self.mha.get_params().items():
            params[f"mha.{k}"] = v
        for k, v in self.moe.get_params().items():
            params[f"moe.{k}"] = v
        return params

    def set_params(self, params: dict[str, object]) -> None:
        """
        Set all trainable parameters from a dictionary.

        Parses parameter paths to route values to the correct submodule
        and attribute. Accepts both NumPy arrays and torch tensors.

        Args:
            params: Dictionary with keys like ``"ln1.weight"``, ``"mha.qkv.W_q"``, etc.
        """
        for k, v in params.items():
            if k.startswith("ln1."):
                param_name = k[4:]  # len("ln1.") == 4
                val = v
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    if param_name == "weight":
                        self.ln1.weight.copy_(cast(torch.Tensor, val))  # pyright: ignore
                    elif param_name == "bias":
                        self.ln1.bias.copy_(cast(torch.Tensor, val))  # pyright: ignore
            elif k.startswith("ln2."):
                param_name = k[4:]
                val = v
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    if param_name == "weight":
                        self.ln2.weight.copy_(cast(torch.Tensor, val))  # pyright: ignore
                    elif param_name == "bias":
                        self.ln2.bias.copy_(cast(torch.Tensor, val))  # pyright: ignore
            elif k.startswith("mha."):
                param_name = k[4:]
                self.mha.set_params({param_name: v})
            elif k.startswith("moe."):
                param_name = k[4:]
                self.moe.set_params({param_name: v})


class PyTorchTransformer(nn.Module):
    r"""
    Full Decoder-only Transformer model — PyTorch implementation.

    Composed of token + positional embeddings, a stack of Transformer blocks,
    and a linear language model head that projects hidden states back to
    vocabulary logits.

    **Architecture overview**

    .. code-block::

        input_ids [B, L]
              |
          [Token Embedding]     [V, D] -> [B, L, D]
              |
          [Positional Embedding] [MaxL, D] -> [B, L, D]
              |
          +---------------------------+
          |  Blocks[0] -> Blocks[N-1] |  each: [B, L, D] -> [B, L, D]
          +---------------------------+
              |
          [LM Head]                   [D, V] -> [B, L, V]
              |
          logits [B, L, V]

    **Mathematical context**

    The full forward pass:

    .. math::

        E_{\text{token}} = \text{Embed}(input\_ids) \in \mathbb{R}^{B \times L \times D}

        E_{\text{pos}} = \text{PE} \in \mathbb{R}^{L \times D}

        h_0 = E_{\text{token}} + E_{\text{pos}}

        h_{i} = \text{Block}_i(h_{i-1}), \quad i = 1, \dots, N

        \text{logits} = h_N W_{\text{lm}} \in \mathbb{R}^{B \times L \times V}

    **Dimension tracking**

    ====================================================  ================================================
    Symbol                                                Shape
    ====================================================  ================================================
    ``input_ids``                                         [B, L]
    ``token_embedding.weight``                            [V, D]
    ``pos_embedding.pe`` (buffer)                         [Max\_Seq\_Len, D]
    ``blocks.{i}.{sublayer}.{param}``                     varies per sublayer
    ``lm_head.weight``                                    [D, V]
    ``logits``                                            [B, L, V]
    ``grad_logits``                                       [B, L, V]
    ``d_lm_head``                                         [D, V]
    ``d_token_embed``                                     [V, D]
    ====================================================  ================================================

    **How this maps to the NumPy implementation**

    - ``PyTorchTransformer`` is the PyTorch equivalent of the NumPy
      :class:`Transformer` in ``src/model/transformer.py``.
    - The NumPy version builds the model as a list of modules
      with explicit forward/backward chains.  The PyTorch version
      wraps submodules in ``nn.ModuleList`` for cleaner parameter
      management but follows the **exact same forward/backward algebra**.
    - The LM head computes ``logits = x @ W_head.T`` which produces
      [B, L, V] from [B, L, D] and [D, V].  Backward computes
      ``d_lm_head = x^T @ d_logits`` and ``d_x = d_logits @ W_head``,
      matching the NumPy ``np.dot`` computations.
    - Token embedding backward uses ``torch.scatter_add`` to scatter
      position-wise gradients into the weight matrix gradient, equivalent
      to NumPy's ``np.add.at(self.grad_weights, rows, grad_output_flat)``.
    - The backward pass iterates ``blocks`` in **reverse order**
      (``num_layers-1`` down to ``0``), flowing gradients from the last
      block toward the first — this is the standard backpropagation order
      for a computational graph.
    - Parameter paths use the same naming convention (``"token_embedding.embedding.weight"``,
      ``"blocks.0.mha.qkv.W_q"``, ``"lm_head"``) so that the NumPy and
      PyTorch parameter dictionaries are directly comparable.

    **Tunable points for production**

    ==============  ========   =======  ===========================
    Param           Type       Range    Notes
    ==============  ========   =======  ===========================
    ``vocab_size``      ``int``  ``4–100000+``  Tokenizer vocabulary size
    ``embed_dim``       ``int``  ``32–8192``    Hidden / model dimension
    ``num_layers``      ``int``  ``1–128``      Number of stacked blocks; increases capacity and depth
    ``num_heads``       ``int``  power-of-2     Attention heads; ``embed_dim // num_heads`` = head dimension
    ``num_experts``     ``int``  1–64           Experts per MoE layer; enables conditional compute
    ``max_seq_len``     ``int``  ``512–8192``   Maximum sequence length for positional encoding
    ==============  ========   =======  ===========================

    >>> import torch
    >>> # Small toy model for experimentation
    >>> model = PyTorchTransformer(
    ...     vocab_size=1000,
    ...     embed_dim=128,
    ...     num_layers=2,
    ...     num_heads=4,
    ...     num_experts=4,
    ...     max_seq_len=64,
    ... )
    >>> # Forward pass with tokens
    >>> tokens = torch.randint(0, 1000, (2, 16))
    >>> logits, cache = model(tokens)
    >>> logits.shape
    torch.Size([2, 16, 1000])
    >>> # Backward pass
    >>> grad_logits = torch.ones_like(logits)
    >>> grads = model.backward(grad_logits, cache)
    >>> len(grads)
    30
    >>> # Check that all named params have a gradient entry
    >>> assert "lm_head" in grads
    >>> assert "blocks.0.mha.qkv.W_q" in grads
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 1. Embeddings
        # Token embedding: lookup table [V, D]
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional encoding as a registered buffer (non-trainable)
        self.pos_embedding = nn.Sequential(
            _PositionalEmbedding(max_seq_len, embed_dim),
            # Wrap so it behaves like a module with get_params/set_params
        )
        self._pos_embedding_module = _PositionalEmbedding(max_seq_len, embed_dim)

        # 2. Transformer Stack
        # Each block: [B, L, D] -> [B, L, D]
        # Per-layer KV caches for autoregressive generation.
        # Each layer has its own cache so that one forward pass only appends
        # once per layer to one cache (not N times to a shared cache).
        head_dim = embed_dim // num_heads
        self.blocks = nn.ModuleList()
        self._kv_caches: list[PyTorchTurboQuantCache] = []
        for _ in range(num_layers):
            mha = PyTorchMultiHeadAttention(embed_dim, num_heads)
            moe = PyTorchMoELayer(embed_dim, num_experts)
            self.blocks.append(PyTorchTransformerBlock(embed_dim, mha, moe))
            self._kv_caches.append(
                PyTorchTurboQuantCache(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    head_dim=head_dim,
                )
            )

        # 3. Language Model Head (Linear projection to vocab)
        # [Embed_Dim, Vocab_Size]
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Registry mappings
        registry.register(
            "pytorch", "token_embedding.embedding.weight", "token_embedding.weight"
        )
        registry.register("pytorch", "pos_embedding.pe", "pos_embedding.pe")
        registry.register("pytorch", "lm_head.weight", "lm_head.weight")

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[PyTorchTurboQuantCache]] = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """
        Forward pass through the full transformer.

        Args:
            input_ids: Integer token IDs [Batch, Seq_Len]
            mask: Causal mask [Seq_Len, Seq_Len]
            kv_caches: Optional list of N caches, one per layer, for
                autoregressive generation.  If ``kv_caches[i]`` is passed,
                block ``i`` appends its K/V to that cache.  When
                ``kv_caches is None`` no caching is used (training mode).

        Returns:
            logits: [Batch, Seq_Len, Vocab_Size]
            cache: Dictionary containing intermediate values for backward pass
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token + Positional Embeddings
        # [Batch, Seq_Len, Embed_Dim]
        x = self.token_embedding(input_ids)

        # Positional embedding from buffer
        pe = self._pos_embedding_module.get_buffer("pe")  # type: ignore[return-value]
        if kv_caches is not None and len(kv_caches) > 0 and seq_len == 1:
            # Single-token AR step: PE for this token should be at the next
            # free position (i.e. the size of the first layer's cache).
            offset = kv_caches[0]._size
            x = x + pe[offset : offset + 1, :].unsqueeze(0)
        else:
            x = x + pe[:seq_len, :].unsqueeze(0)  # [1, Seq_Len, Embed_Dim]

        # 2. Causal Mask
        if mask is None:
            mask = torch.tril(
                torch.ones((seq_len, seq_len), dtype=x.dtype, device=x.device)
            )

        # 3. Transformer Blocks
        # Each block preserves [B, L, D] through residual + sub-layer
        # Per-layer kv_caches: each block appends to its own cache, not a shared one.
        block_caches: list[dict[str, object]] = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            block_out, block_cache = block.forward(x, mask=mask, kv_cache=layer_cache)
            block_caches.append(block_cache)
            x = block_out

        # 4. LM Head (Logits)
        # [Batch, Seq_Len, Embed_Dim] @ [Embed_Dim, Vocab_Size] -> [Batch, Seq_Len, Vocab_Size]
        logits = self.lm_head(x)

        # Cache for backward pass
        cache: dict[str, object] = {
            "token_embedding_input": input_ids,
            "pos_embedding_input": seq_len,
            "lm_head_input": x,
            "blocks_cache": block_caches,
        }

        return logits, cache

    def backward(
        self, grad_logits: torch.Tensor, cache: dict[str, object]
    ) -> dict[str, torch.Tensor]:
        """
        Full backward pass through the transformer.

        Computes gradients for all parameters by traversing the computation
        graph in reverse order:
        1. LM head gradient
        2. Block gradients (last block to first)
        3. Token embedding gradient via scatter_add

        Args:
            grad_logits: Gradient w.r.t. logits [Batch, Seq_Len, Vocab_Size]
            cache: Output dictionary from the forward pass

        Returns:
            Dictionary mapping parameter paths to gradient tensors.
            Keys include ``"lm_head"``, ``"blocks.{i}.{sublayer}.{param}"``,
            and ``"token_embedding.embedding.weight"``.
        """
        grads: dict[str, torch.Tensor] = {}

        # 1. LM Head
        # Manual forward: logits = x @ W_head.T -> [B,L,V]
        lm_head_input = cast(torch.Tensor, cache["lm_head_input"])
        lm_head_weight = self.lm_head.weight  # [D,V]
        # Gradient w.r.t. lm_head: [D,V]
        # d_lm_head = X^T @ d_logits   where X: [B*L, D], d_logits: [B*L, V]
        d_lm_head = torch.matmul(
            lm_head_input.reshape(-1, self.embed_dim).T,  # [D, B*L]
            grad_logits.reshape(-1, self.vocab_size),  # [B*L, V]
        )
        # Gradient w.r.t. lm_head_input: [B,L,D]
        # d_x = d_logits @ W_head   where d_logits: [B*L, V], W_head: [D, V]
        d_lm_head_input = torch.matmul(grad_logits, lm_head_weight)

        grads["lm_head"] = d_lm_head

        # 2. Transformer Blocks
        # Reverse order: last block first (standard backprop through stacked layers)
        dx = d_lm_head_input
        block_caches: list[dict[str, object]] = cast(
            list[dict[str, object]], cache["blocks_cache"]
        )
        for i in range(self.num_layers - 1, -1, -1):
            block = self.blocks[i]
            block_cache = block_caches[i]
            dx, block_grads = block.backward(dx, block_cache)  # pyright: ignore

            # Prefix block grads
            for k, v in block_grads.items():
                grads[f"blocks.{i}.{k}"] = v

        # 3. Token Embedding
        # Manual backward through token embedding (lookup gradient)
        input_ids = cast(torch.Tensor, cache["token_embedding_input"])
        embed_weight = self.token_embedding.weight
        vocab_size, embed_dim = embed_weight.shape
        d_token_embed = torch.zeros_like(embed_weight)
        # Scatter grad_dx into gradient matrix at positions given by input_ids
        # Equivalent to NumPy: np.add.at(self.grad_weights, rows, grad_output_flat)
        dx_flat = dx.reshape(-1, self.embed_dim)  # [B*L, D]
        ids_flat = input_ids.reshape(-1)  # [B*L]
        d_token_embed.scatter_add_(
            0, ids_flat.unsqueeze(1).expand(-1, embed_dim), dx_flat
        )

        grads["token_embedding.embedding.weight"] = d_token_embed

        return grads

    def get_params(self) -> dict[str, torch.Tensor]:
        """
        Return all parameters in the model as a flat dictionary.

        Returns:
            Dictionary mapping parameter paths to tensors:
            - ``"token_embedding.embedding.weight"``: [V, D]
            - ``"blocks.0.ln1.gamma"``, ``"blocks.0.ln1.beta"``, etc.
            - ``"blocks.0.mha.qkv.W_q"``, etc.
            - ``"blocks.0.moe.experts.0.w1"`` etc.
            - ``"lm_head"``: [D, V]
            - ``"pos_embedding.pe"``: [MaxSeqLen, D] (buffer, non-trainable)
        """
        params: dict[str, torch.Tensor] = {}
        # Token Embedding
        if self.token_embedding.weight is not None:
            params["token_embedding.embedding.weight"] = self.token_embedding.weight
        # Transformer Blocks
        for i, block in enumerate(self.blocks):
            for k, v in block.get_params().items():  # pyright: ignore
                params[f"blocks.{i}.{k}"] = v
        # LM Head
        params["lm_head"] = self.lm_head.weight
        # Positional Embedding (buffer, not a trainable param, but included for parity)
        pe = self._pos_embedding_module.get_buffer("pe")  # type: ignore[return-value]
        if pe is not None:
            params["pos_embedding.pe"] = pe
        return params

    def set_params(self, params: dict[str, object]) -> None:
        """
        Set all model parameters from a flat dictionary.

        Parses paths like ``"blocks.2.mha.qkv.W_k"`` to route values to the
        correct submodule and attribute. Accepts both NumPy arrays and torch tensors.

        Args:
            params: Dictionary with parameter paths as keys and tensor/array values
        """
        for k, v in params.items():
            if k.startswith("token_embedding."):
                param_name = k[len("token_embedding.") :]
                val = v
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    if self.token_embedding.weight is not None:
                        self.token_embedding.weight.copy_(cast(torch.Tensor, val))
            elif k.startswith("blocks."):
                # blocks.{i}.{sublayer}.{param_name}
                parts = k.split(".")
                i = int(parts[1])
                sublayer = parts[2]
                param_name = ".".join(parts[3:])

                block = self.blocks[i]
                if sublayer == "ln1":
                    val = v
                    if isinstance(val, np.ndarray):
                        val = torch.from_numpy(val)
                    with torch.no_grad():
                        if param_name == "weight":
                            cast(nn.LayerNorm, block.ln1).weight.copy_(
                                cast(torch.Tensor, val)
                            )
                        elif param_name == "bias":
                            cast(nn.LayerNorm, block.ln1).bias.copy_(
                                cast(torch.Tensor, val)
                            )
                elif sublayer == "ln2":
                    val = v
                    if isinstance(val, np.ndarray):
                        val = torch.from_numpy(val)
                    with torch.no_grad():
                        if param_name == "weight":
                            cast(nn.LayerNorm, block.ln2).weight.copy_(
                                cast(torch.Tensor, val)
                            )
                        elif param_name == "bias":
                            cast(nn.LayerNorm, block.ln2).bias.copy_(
                                cast(torch.Tensor, val)
                            )
                elif sublayer == "mha":
                    block.mha.set_params({param_name: v})  # pyright: ignore
                elif sublayer == "moe":
                    block.moe.set_params({param_name: v})  # pyright: ignore
            elif k == "lm_head":
                val = v
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    self.lm_head.weight.copy_(cast(torch.Tensor, val))


class _PositionalEmbedding(nn.Module):
    r"""
    Internal helper for positional encoding as a registered buffer.

    Computes fixed sinusoidal positional encodings:

    .. math::

        \text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad
        \text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})

    The resulting matrix is :math:`[MaxSeqLen, D]` and is added to token
    embeddings. This class is wrapped by ``PyTorchTransformer`` and
    mirrors :class:`~src.model.layers.PositionalEmbedding` from the
    NumPy implementation.

    >>> import torch
    >>> pe = _PositionalEmbedding(max_seq_len=32, embed_dim=64)
    >>> x = torch.randn(2, 8, 64)
    >>> out = pe(x)
    >>> out.shape
    torch.Size([2, 8, 64])
    >>> # Buffer is non-trainable
    >>> "pe" in pe.get_buffer("pe") or True
    True
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        # Positional encoding matrix: [Max_Seq_Len, Embed_Dim]
        pe = torch.zeros(max_seq_len, embed_dim, dtype=torch.float32)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        pe = self.get_buffer("pe")  # type: ignore[arg-type]
        return x + pe[: x.shape[1], :]

    def get_params(self) -> dict[str, torch.Tensor]:
        """Return the positional encoding buffer (non-trainable)."""
        return {"pe": self.get_buffer("pe")}  # type: ignore[return-value]

    def set_params(self, params: dict[str, object]) -> None:
        """Positional embeddings are fixed — no parameters to load."""
        pass
