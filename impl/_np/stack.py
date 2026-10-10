"""Decoder stack for the NumPy reference implementation.

A stack of TransformerBlocks — the "body" of the decoder.

**Intuition:** the model is depth × width. Every block refines the hidden
stream so the next block can answer a higher-level question. The stack is
uniform — no state persists between blocks other than the hidden stream
itself.
"""

from __future__ import annotations

import numpy as np

from impl._np.block import TransformerBlock
from shared.config import TransformerConfig


class DecoderStack:
    """Stack of n_layers TransformerBlocks (the "body" of the decoder).

    out = block_{n-1}( ... block_1(block_0(x)) ...)

    Shape letters (CONTEXT.md → "Shape notation"): B = batch size (parallel
    sequences; 1 = one generation), S = sequence length (this pass), D =
    embed_dim (model width), H = n_heads, G = n_groups, V = vocab_size,
    E = n_experts, FF = expert_dim, t = KV-cache depth, k = draft length.
    x: (B, S, D) → out: (B, S, D). Every block draws from the SAME frozen
    ``config.seed``, so identically-shaped parameters initialize identically
    across layers; training is what differentiates them.

    Backward: run the blocks' backwards in *reverse* order, threading the
    gradient from the last block back to the input.
    """

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]

    def forward(
        self, x: np.ndarray, positions: np.ndarray | None = None, record: list[dict] | None = None
    ) -> np.ndarray:
        """Run all blocks in order. x: (B, S, D) → out: (B, S, D).

        x: (B, S, D) the embedding output — B sequences, S positions, each
            a D-wide vector.
        positions: (S,) the absolute token indices for RoPE (threaded to
            every block; None → arange(S)).
        record: optional per-block record list (one dict per block, filled
            by each block's forward — the learning-mode capture).

        Contract: the blocks run in index order 0..n_layers-1; the output
        of block i is the input of block i+1 (the residual stream
        accumulates — nothing is overwritten).
        """
        """Run all blocks. x: (B, S, D) → out: (B, S, D).

        record: optional per-block state dicts (one per layer); when given,
            each block fills its own dict with the intermediates (see
            ``TransformerBlock.forward``). The math is identical either way.
        """
        out = x
        for i, _block in enumerate(self.layers):
            out = _block.forward(out, positions, record=record[i] if record is not None else None)
        return out

    def forward_step(
        self, x: np.ndarray, position: int, cache: list[dict], quantize: bool = False, record: list[dict] | None = None
    ) -> np.ndarray:
        """Process ONE new token through all blocks (KV-cached path).

        x: (B, 1, D) the new token's vector — ONE new position per sequence.
        position: the absolute token index (for RoPE; must equal the
            caches' current depth).
        cache: the per-layer cache LIST (one dict per block, in block
            order; mutated in place).
        quantize: TurboQuant switch (threaded to every block's attention).
        record: optional per-block record list.

        Returns: (B, 1, D) the stack output for the new token. Contract:
        equal to the matching slice of ``forward`` over the full sequence
        (the exact KV-step guarantee); every cache ends holding positions
        0..position.
        """
        """Process ONE new token through all blocks (KV-cached path).

        x: (B, 1, D) the new token's vector.
        position: the absolute token index (RoPE).
        cache: per-layer cache dicts (one per block), mutated in place.
        quantize: if True, append the new K/V to each layer's cache in 1-bit
            TurboQuant form and dequantize the full cached tensor before
            attention; if False, append the full-precision K/V (default).
        record: optional per-block state dicts (one per layer) filled with the
            step's intermediates (see ``TransformerBlock.forward_step``).

        Returns: (B, 1, D) the stack output for the new token.
        """
        out = x
        for i, block in enumerate(self.layers):
            out = block.forward_step(
                out, position, cache[i], quantize=quantize, record=record[i] if record is not None else None
            )
        return out

    def forward_chunk(
        self, x: np.ndarray, position: int, cache: list[dict], record: list[dict] | None = None
    ) -> np.ndarray:
        """Process a CHUNK of c tokens through all blocks (KV-cached path).

        Mirrors ``forward_step`` with the attention core swapped for the
        chunk form: one pass appends every chunk token's K/V per layer and
        scores all chunk positions in parallel — the verification pass.

        x: (B, c, D). position: absolute index of the chunk's first token.
        cache: the per-layer cache list (mutated in place). record: optional
        per-block record list (same as ``forward_step``'s).

        Returns: (B, c, D).
        """
        out = x
        for i, block in enumerate(self.layers):
            out = block.forward_chunk(out, position, cache[i], record=record[i] if record is not None else None)
        return out

    def backward(
        self, dout: np.ndarray, x: np.ndarray, positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[dict]]:
        """Analytic backward through the blocks in reverse order.

        dout: (B, S, D) upstream gradient.
        x: (B, S, D) the forward input (recomputes the layer inputs).
        positions: the RoPE positions used in forward (None → arange(S)).

        Returns: (dx, per_layer_grads) where per_layer_grads[i] is the
        gradient dict of block i (see TransformerBlock.backward for the keys).
        """
        if positions is None:
            positions = np.arange(x.shape[1], dtype=np.int32)

        # Forward: record each layer's input so the backward can recompute
        # with the right activations.
        layer_inputs: list[np.ndarray] = []
        layer_out = x
        for block in self.layers:
            layer_inputs.append(layer_out)
            layer_out = block.forward(layer_out, positions)

        # Backward in reverse order.
        per_layer_grads: list[dict] = [{} for _ in self.layers]
        d = dout
        for i in range(len(self.layers) - 1, -1, -1):
            d, per_layer_grads[i] = self.layers[i].backward(d, layer_inputs[i], positions)
        return d, per_layer_grads
