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
            num_experts=num_experts,
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
        """
        Returns parameters in the format expected by the ParityTester (NumPy style).
        """
        params = {}
        for name, param in self.model.named_parameters():
            # Map PyTorch internal names to NumPy/External names
            ext_name = name
            if name == "token_embedding.embedding.weight":
                ext_name = "token_embedding.weights"
            elif name == "lm_head.weight":
                ext_name = "lm_head"
            elif name.startswith("layers."):
                # layers.0.ln1.ln.weight -> blocks.0.ln1.gamma
                parts = name.split(".")
                idx = parts[1]
                sub = parts[2]
                if sub == "ln1":
                    if parts[3] == "ln.weight":
                        ext_name = f"blocks.{idx}.ln1.gamma"
                    elif parts[3] == "ln.bias":
                        ext_name = f"blocks.{idx}.ln1.beta"
                elif sub == "ln2":
                    if parts[3] == "ln.weight":
                        ext_name = f"blocks.{idx}.ln2.gamma"
                    elif parts[3] == "ln.bias":
                        ext_name = f"blocks.{idx}.ln2.beta"
                elif sub == "attn":
                    # layers.0.attn.W_q.weight -> blocks.0.mha.W_q
                    weight_name = parts[3]
                    if weight_name == "weight" and len(parts) > 4:
                        ext_name = f"blocks.{idx}.mha.{parts[4]}"
                    else:
                        ext_name = f"blocks.{idx}.mha.{weight_name}"
                elif sub == "moe":
                    # layers.0.moe.router.weights -> blocks.0.moe.router.weights
                    # layers.0.moe.experts.0.ffn.w1.weight -> blocks.0.moe.experts.0.ffn.w1.weight
                    moe_part = ".".join(parts[3:])
                    ext_name = f"blocks.{idx}.moe.{moe_part}"
            elif name.startswith("ln_f."):
                if name == "ln_f.ln.weight":
                    ext_name = "ln_f.gamma"
                elif name == "ln_f.ln.bias":
                    ext_name = "ln_f.beta"

            val = param.detach().cpu().numpy()
            if ext_name == "lm_head":
                val = val.T
            params[ext_name] = val
            
        return params

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """
        Sets parameters from NumPy-style keys to PyTorch internal names.
        """
        with torch.no_grad():
            for ext_name, val in params.items():
                val_torch = torch.from_numpy(val).to(next(self.model.parameters()).device)
                
                if ext_name == "token_embedding.weights":
                    self.model.token_embedding.embedding.weight.copy_(val_torch)
                elif ext_name == "lm_head":
                    self.model.lm_head.weight.copy_(val_torch.T)
                elif ext_name.startswith("blocks."):
                    parts = ext_name.split(".")
                    idx = int(parts[1])
                    sub = parts[2]
                    if sub == "ln1":
                        if parts[3] == "gamma":
                            self.model.layers[idx].ln1.ln.weight.copy_(val_torch)
                        elif parts[3] == "beta":
                            self.model.layers[idx].ln1.ln.bias.copy_(val_torch)
                    elif sub == "ln2":
                        if parts[3] == "gamma":
                            self.model.layers[idx].ln2.ln.weight.copy_(val_torch)
                        elif parts[3] == "beta":
                            self.model.layers[idx].ln2.ln.bias.copy_(val_torch)
                    elif sub == "mha":
                        weight_name = parts[3]
                        if weight_name == "weight" and len(parts) > 4:
                             getattr(self.model.layers[idx].attn, parts[4]).weight.copy_(val_torch)
                        else:
                             getattr(self.model.layers[idx].attn, weight_name).weight.copy_(val_torch)
                    elif sub == "moe":
                        # Convert blocks.0.moe.router.weights -> layers.0.moe.router.weights
                        # py_name: layers.0.moe.router.weights
                        py_name = f"layers.{idx}.moe.{'.'.join(parts[3:])}"
                        parts_moe = py_name.split(".")
                        
                        # Start from the moe module of the layer
                        curr = self.model.layers[idx].moe
                        # parts_moe is ['layers', idx, 'moe', 'router', 'weights']
                        # We start traversing from the 4th element (index 3).
                        for p in parts_moe[3:]:
                            if p.isdigit():
                                curr = curr[int(p)]
                            else:
                                curr = getattr(curr, p)
                        
                        if isinstance(curr, torch.Tensor):
                            curr.copy_(val_torch)
                        elif isinstance(curr, torch.nn.Parameter):
                            curr.copy_(val_torch)
                        elif hasattr(curr, "weight"):
                            curr.weight.copy_(val_torch)
                        elif hasattr(curr, "bias"):
                            curr.bias.copy_(val_torch)
                elif ext_name.startswith("ln_f."):
                    if ext_name == "ln_f.gamma":
                        self.model.ln_f.ln.weight.copy_(val_torch)
                    elif ext_name == "ln_f.beta":
                        self.model.ln_f.ln.bias.copy_(val_torch)
