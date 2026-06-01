import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple
from utils.backend_interface import BaseTransformerBackend
from model.pytorch.transformer import Transformer

class PyTorchBackend(BaseTransformerBackend):
    """
    Implementation of the PyTorch-based backend of the Transformer.
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
        self.model = Transformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            ffn_intermediate_dim=embed_dim * 4, # Standard scaling
        )

    def forward(
        self,
        input_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_cache: bool = False,
        cache_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        device = next(self.model.parameters()).device
        input_ids_torch = torch.from_numpy(input_ids).to(device)
        
        if mask is not None:
            mask_torch = torch.from_numpy(mask).to(device)
        else:
            mask_torch = None

        logits = self.model(input_ids_torch, mask=mask_torch)
        
        # For parity, we need to store input_ids in cache for backward
        cache = {"input_ids": input_ids}
        
        # Convert logits back to numpy for the interface
        return logits.detach().cpu().numpy(), cache

    def backward(
        self, grad_logits: np.ndarray, cache: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        device = next(self.model.parameters()).device
        grad_logits_torch = torch.from_numpy(grad_logits).to(device)
        
        input_ids = cache.get("input_ids")
        if input_ids is None:
            raise ValueError("input_ids must be stored in cache for backward parity test")

        input_ids_torch = torch.from_numpy(input_ids).to(device)
        logits = self.model(input_ids_torch)
        logits.backward(grad_logits_torch)
        
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().cpu().numpy()
        
        # Zero gradients for next time
        self.model.zero_grad()
        
        return grads

    def get_params(self) -> Dict[str, np.ndarray]:
        params = {}
        for name, param in self.model.named_parameters():
            params[name] = param.detach().cpu().numpy()
        return params

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(torch.from_numpy(params[name]))

