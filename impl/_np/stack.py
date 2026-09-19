"""Decoder stack for the NumPy reference implementation.

A stack of TransformerBlocks — the "body" of the decoder.
"""

from __future__ import annotations

import numpy as np

from impl._np.block import TransformerBlock
from shared.config import TransformerConfig


class DecoderStack:
    """Stack of n_layers TransformerBlocks (the "body" of the decoder).

    out = block_{n-1}( ... block_1(block_0(x)) ...)

    x: (B, S, D) → out: (B, S, D). Each block gets a distinct seed offset so
    the layers do not start with identical weights.

    Backward: run the blocks' backwards in *reverse* order, threading the
    gradient from the last block back to the input.
    """

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]

    def forward(
        self, x: np.ndarray, positions: np.ndarray | None = None, record: list[dict] | None = None
    ) -> np.ndarray:
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
