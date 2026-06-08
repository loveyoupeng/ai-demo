from __future__ import annotations

import numpy as np
import torch
from core.base_backend import BaseTransformerBackend
from model.pytorch.transformer import PyTorchTransformer

# Canonical (NumPy style) -> PyTorch internal name
# Only used for set_params mapping of fixed/lookup keys. Dynamic keys use pattern matching.
_CANONICAL_TO_PT: dict[str, str] = {
    "token_embedding.weights": "token_embedding.weight",
    "lm_head": "lm_head.weight",
    "blocks.0.mha.W_q": "blocks.0.mha.W_q",
    "blocks.0.mha.W_k": "blocks.0.mha.W_k",
    "blocks.0.mha.W_v": "blocks.0.mha.W_v",
    "blocks.0.mha.W_o": "blocks.0.mha.W_o",
    "blocks.0.moe.router.weights": "blocks.0.moe.router.w",
}


def _canonical_to_pytorch(canonical: str) -> str:
    """Map canonical name to PyTorch internal name."""
    if canonical in _CANONICAL_TO_PT:
        return _CANONICAL_TO_PT[canonical]
    if ".ln1.gamma" in canonical or ".ln1.beta" in canonical:
        return canonical
    if ".ln2.gamma" in canonical or ".ln2.beta" in canonical:
        return canonical
    if ".mha.W_" in canonical:
        return canonical
    if ".moe.expert." in canonical:
        parts = canonical.split(".")
        block_idx = parts[1]
        expert_idx = parts[4]
        param = parts[5]
        return f"blocks.{block_idx}.moe.experts.{expert_idx}.{param.lower()}"
    if "moe.router.weights" in canonical:
        parts = canonical.split(".")
        return f"blocks.{parts[1]}.moe.router.w"
    return canonical


class PyTorchBackend(BaseTransformerBackend):
    """
    PyTorch-based backend wrapping PyTorchTransformer for training and inference.
    Converts all I/O to/from NumPy arrays for parity with the NumPy backend.
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
        self.model = PyTorchTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_experts=num_experts,
            max_seq_len=max_seq_len,
        )

    def _to_np_float64(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch.Tensor to numpy float64."""
        return tensor.detach().float().cpu().numpy().astype(np.float64)

    def _canonical(self, key: str) -> str:
        """Map backward/dict keys to canonical parameter names."""
        if key == "token_embedding.embedding.weight":
            return "token_embedding.weights"
        if key == "token_embedding.weight":
            return "token_embedding.weights"
        if key == "lm_head.weight":
            return "lm_head"
        if key == "lm_head":
            return "lm_head"
        if "moe" in key:
            # Handle MoE params: blocks.N.moe.experts.N.w1 -> blocks.N.moe.expert.N.W1
            # Note: b1/b2 must stay lowercase to match NumPy canonical format
            key = key.replace(".moe.experts.", ".moe.expert.")
            key = key.replace(".moe.router.w", ".moe.router.weights")
            key = key.replace(".w1", ".W1").replace(".w2", ".W2")
            # BIAS KEYS: b1/b2 lowercase (matches NumPy canonical), NOT uppercase
            key = key.replace(".b1", ".b1").replace(".b2", ".b2")
            return key
        # blocks.N.ln1.weight/ bias -> ln1.gamma/beta
        if key.endswith(".ln1.weight"):
            return key[:-7] + ".gamma"
        if key.endswith(".ln1.bias"):
            return key[:-5] + ".beta"
        if key.endswith(".ln2.weight"):
            return key[:-7] + ".gamma"
        if key.endswith(".ln2.bias"):
            return key[:-5] + ".beta"
        # blocks.N.mha.qkv.W_q -> blocks.N.mha.W_q
        if ".mha.qkv.W_" in key:
            return key.replace(".mha.qkv.W_", ".mha.W_")
        # blocks.N.mha.o.W_o -> blocks.N.mha.W_o
        if ".mha.o.W_" in key:
            return key.replace(".mha.o.W_", ".mha.W_")
        return key

    def forward(
        self,
        input_ids: np.ndarray,
        mask: np.ndarray | None = None,
        use_cache: bool = False,
        cache_idx: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        tensor_input = torch.from_numpy(input_ids).to(torch.int64)
        tensor_mask = torch.from_numpy(mask) if mask is not None else None

        logits_tensor, cache = self.model.forward(
            tensor_input,
            mask=tensor_mask,
            use_cache=use_cache,
            cache_idx=cache_idx,
        )

        logits = self._to_np_float64(logits_tensor)
        return logits, cache

    def backward(
        self, grad_logits: np.ndarray, cache: dict[str, object]
    ) -> dict[str, np.ndarray]:
        """Backward pass. Returns parameter gradients in canonical format."""
        grad_tensor = torch.from_numpy(grad_logits).float()
        torch_grads = self.model.backward(grad_tensor, cache)

        grads: dict[str, np.ndarray] = {}
        for key, grad in torch_grads.items():
            canon_name = self._canonical(key)
            grads[canon_name] = self._to_np_float64(grad)
        return grads

    def get_params(self) -> dict[str, np.ndarray]:
        """
        Returns all model parameters as NumPy float64 arrays.
        Returns COPY of each parameter.
        """
        params: dict[str, np.ndarray] = {}
        for name, param in self.model.named_parameters():
            canon_name = self._canonical(name)
            np_arr = self._to_np_float64(param)
            # lm_head: PT stores (vocab, embed), canonical is (embed, vocab)
            if canon_name == "lm_head":
                np_arr = np_arr.T
            params[canon_name] = np_arr
        return params

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Sets model parameters from NumPy arrays using canonical parameter names."""
        with torch.no_grad():
            for canonical, values in params.items():
                pt_name = _canonical_to_pytorch(canonical)
                pt_param = self.model.get_parameter(pt_name)  # type: ignore[attr-defined]
                if pt_param is not None:
                    # lm_head: canonical is (embed, vocab), PT stores (vocab, embed)
                    if canonical == "lm_head":
                        tensor = torch.from_numpy(values.T).float()
                    else:
                        tensor = torch.from_numpy(values).float()
                    pt_param.copy_(tensor)
