"""E9: TritonModel — complete transformer with embedding + Triton kernels.

Forward: tokens -> embedding -> DecoderStack -> RMSNorm -> SwiGLU -> Linear -> logits

This module provides the complete model that plugs into the training/inference
scripts identically to impl._torch.layers.TorchModel.
"""

import numpy as np
import torch
import torch.nn as nn

from impl._torch.layers import SwiGLUFFN
from impl._triton.transformer import TritonDecoderStack
from shared.constants import Block, Mha, Transformer


class TritonModel(nn.Module):
    """Complete decoder-only transformer using Triton kernels.

    Forward: tokens -> embedding -> DecoderStack -> RMSNorm -> SwiGLU -> Linear -> logits

    Architecture:
        Input:  tokens [batch, seq_len] (int64)
        |
        +-> Embedding table lookup       [batch, seq_len, embed_dim]
        +-> DecoderStack (Triton n_layers) [batch, seq_len, embed_dim]
        +-> RMSNorm (final_ln)            [batch, seq_len, embed_dim]
        +-> SwiGLU (output)               [batch, seq_len, embed_dim]
        +-> Linear (output_proj)          [batch, seq_len, vocab_size]

    Parameters:
        vocab_size: Vocabulary size.
        embed_dim: Hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads per block.
        n_experts: Number of MoE experts per block.
        ff_dim: Feed-forward hidden dimension per expert.
        k: Number of top experts per token.
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
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.k = k

        # Embedding layer (PyTorch native — no Triton optimization needed)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Decoder stack (Triton kernels) — named 'layers' to match _torch
        self.stack = TritonDecoderStack(
            n_layers=n_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        # Final layer normalization — RMSNorm instance to match _torch
        self.final_ln = nn.RMSNorm(embed_dim, eps=1e-5)

        # Output projection — SwiGLU -> Linear
        # SwiGLU maps D -> D via hidden (D*2), then linear projects D -> V
        ff_dim_out = embed_dim * 2  # output projection hidden dim
        self.output = SwiGLUFFN(embed_dim, ff_dim_out)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.embedding.weight, a=0.01)
        self.output.reset_parameters()

    def _move_to_device(self, x: torch.Tensor) -> None:
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        for _name, param in self.named_parameters():
            if param.device != device or (dtype is not None and param.dtype != dtype):
                param.data = param.data.to(device, dtype=param.dtype if dtype is None else dtype)
        for _name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                module.to(device)
        self.stack._move_to_device(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._move_to_device(x)
        """Forward pass through the complete model.

        Args:
            x: Token IDs. Shape [batch_size, seq_len], dtype int64.

        Returns:
            Predicted logits. Shape [batch_size, seq_len, vocab_size].
        """
        # Embedding: [B,S] -> [B,S,D]
        x = self.embedding(x)  # (B, S, D)

        # Decoder stack: [B,S,D] -> [B,S,D] — all Triton kernels
        x = self.stack(x)

        # Final layer normalization: [B,S,D] -> [B,S,D] — instance layer
        x = self.final_ln(x)  # (B, S, D)

        # SwiGLU output projection: [B,S,D] -> [B,S,D]
        x = self.output(x)  # (B, S, D)

        # Linear projection to vocabulary: [B,S,D] -> [B,S,V]
        logits = self.output_proj(x)

        return logits

    def _get_param(self, key: str) -> torch.Tensor:  # noqa: C901 — nested module walk is complex
        """Get a parameter by key name (for save/load roundtrip verification).

        Maps key names to actual nn.Parameter attributes.

        Args:
            key: Parameter key in save/load format.

        Returns:
            The corresponding tensor.
        """
        if key == Transformer.EMBEDDING_WEIGHTS:
            return self.embedding.weight
        elif key.startswith("blocks."):
            # Handle NumPy/constant-style keys for backwards compatibility
            # Key format: "blocks.N.attr"  e.g. "blocks.0.ln1_gamma", "blocks.0.moe.W_router"
            layer_str, rest = key.split(".", 1)
            parts = rest.split(".", 1)
            layer_idx = int(parts[0])
            attr_name = parts[1]

            block = self.stack.layers[layer_idx]
            first_attr = attr_name.split(".", 1)[0]

            # MoE expert weights: "blocks.N.moe.experts.M.W1"
            if first_attr == "moe" and "experts" in attr_name:
                sub_parts = attr_name.split(".")
                # sub_parts = ['moe', 'experts', 'M', 'W1']
                expert_idx = int(sub_parts[2])
                weight_key = sub_parts[3]
                return getattr(block.moe.experts[expert_idx], weight_key)

            # Direct block attribute mapping
            if first_attr == "ln1_gamma":
                return block.ln1.weight
            elif first_attr == "ln2_gamma":
                return block.ln2.weight
            elif first_attr == "moe_bias":
                return block.moe.b_router  # constant "moe.bias" -> PyTorch "b_router"
            elif first_attr == "moe":
                # MoE direct: "moe.W_router", "moe.bias", "moe.router"
                param_name = attr_name.split(".", 1)[1]
                key_to_attr = {"W_router": "W_router", "router": "W_router", "bias": "b_router"}
                return getattr(block.moe, key_to_attr.get(param_name, param_name))
            elif first_attr == "gate1":
                return block.gate1
            elif first_attr == "gate2":
                return block.gate2
            elif first_attr == "mha":
                # Constant key: "WQ", "WK", "WV", "WO", "BQ", "BK", "BV", "BO" (uppercase)
                param_name = attr_name.split(".", 1)[1]
                # Map uppercase constant name → Triton attribute name
                upper = param_name.upper()
                mapping = {"WQ": "Wq", "WK": "Wk", "WV": "Wv", "WO": "Wo",
                           "BQ": "bq", "BK": "bk", "BV": "bv", "BO": "bo"}
                return getattr(block.mha, mapping.get(upper, param_name))
            else:
                # Direct attribute: e.g. "blocks.0.ln1_gamma" if not handled above
                return getattr(block, first_attr, None)

        elif key.startswith("layers."):
            # Handle layer parameters — keys are PyTorch-style from named_parameters()
            parts = key.split(".")
            # parts[0] = "layers", parts[1] = layer_idx (e.g. "0"), parts[2+] = attribute path
            layer_idx = int(parts[1])
            attr_path = ".".join(parts[2:])
            block = self.stack.layers[layer_idx]

            # Direct attribute access — ln1.weight, mha.Wq, gate1, moe.W_router, etc.
            part = attr_path.split(".", 1)
            first_attr = part[0]
            if first_attr == "ln1":
                if len(part) == 1:
                    return None
                return getattr(getattr(block, first_attr), part[1]) if len(part) > 1 else None
            elif first_attr in ("mha", "moe"):
                if len(part) == 1:
                    return None
                second_attr = part[1].split(".", 1)[0]  # mha.Wq → Wq, moe.experts → experts
                sub = part[1]
                # MoE expert: "moe.experts.0.W1"
                if second_attr == "experts":
                    sub_parts = sub.split(".")
                    # sub_parts = ['moe', 'experts', '0', 'W1']
                    expert_idx = int(sub_parts[2])
                    weight_key = sub_parts[3]
                    return getattr(block.moe.experts[expert_idx], weight_key)
                else:
                    # MoE direct: "moe.W_router", "moe.b_router"
                    return getattr(block.moe, sub)
            else:
                # Direct block attribute: gate1, gate2, ln1, ln2
                return getattr(block, first_attr)
        elif key == Transformer.FINAL_GAMMA:
            return self.final_ln.weight
        elif key == Transformer.OUTPUT_W1:
            return self.output.W1
        elif key == Transformer.OUTPUT_W2:
            return self.output.W2
        elif key == Transformer.OUTPUT_W3:
            return self.output.W3
        elif key == Transformer.OUTPUT_PROJ_W:
            return self.output_proj.weight
        elif key == Transformer.OUTPUT_PROJ_B:
            return self.output_proj.bias
        else:
            raise KeyError(f"Unknown parameter key: {key}")

    def save_as_numpy(self) -> dict[str, np.ndarray]:
        """Save all parameters as a NumPy-compatible dictionary.

        Returns a dict with the same key structure as torch.TorchModel.save_as_numpy(),
        enabling cross-backend parameter exchange for parity testing.

        Returns:
            params: Dictionary mapping parameter names to NumPy arrays.
        """
        result: dict[str, np.ndarray] = {}

        # Embedding
        result[Transformer.EMBEDDING_WEIGHTS] = self.embedding.weight.detach().cpu().numpy()

        # Stack layers
        for layer_idx, block in enumerate(self.stack.layers):

            # Layer norm gamma (now stored as RMSNorm instance weight)
            result[Block.ln1_gamma(layer_idx)] = block.ln1.weight.detach().cpu().numpy()
            result[Block.ln2_gamma(layer_idx)] = block.ln2.weight.detach().cpu().numpy()

            # MHA weights (PyTorch stores weight as [out, in], save as [in, out])
            result[Block.mha(layer_idx, Mha.WQ)] = block.mha.Wq.detach().cpu().numpy()
            result[Block.mha(layer_idx, Mha.BQ)] = block.mha.bq.detach().cpu().numpy()
            result[Block.mha(layer_idx, Mha.WK)] = block.mha.Wk.detach().cpu().numpy()
            result[Block.mha(layer_idx, Mha.BK)] = block.mha.bk.detach().cpu().numpy()
            result[Block.mha(layer_idx, Mha.WV)] = block.mha.Wv.detach().cpu().numpy()
            result[Block.mha(layer_idx, Mha.BV)] = block.mha.bv.detach().cpu().numpy()
            result[Block.mha(layer_idx, Mha.WO)] = block.mha.Wo.detach().cpu().numpy()
            result[Block.mha(layer_idx, Mha.BO)] = block.mha.bo.detach().cpu().numpy()

            # MoE router and expert weights
            result[Block.moe_router(layer_idx)] = block.moe.W_router.detach().cpu().numpy()
            result[Block.moe_bias(layer_idx)] = block.moe.b_router.detach().cpu().numpy()

            for expert_idx, expert in enumerate(block.moe.experts):
                result[Block.moe_expert(layer_idx, expert_idx, "W1")] = expert.W1.detach().cpu().numpy()
                result[Block.moe_expert(layer_idx, expert_idx, "W2")] = expert.W2.detach().cpu().numpy()
                result[Block.moe_expert(layer_idx, expert_idx, "W3")] = expert.W3.detach().cpu().numpy()

        # Final ln
        result[Transformer.FINAL_GAMMA] = self.final_ln.weight.detach().cpu().numpy()

        # Output SwiGLU
        result[Transformer.OUTPUT_W1] = self.output.W1.detach().cpu().numpy()
        result[Transformer.OUTPUT_W2] = self.output.W2.detach().cpu().numpy()
        result[Transformer.OUTPUT_W3] = self.output.W3.detach().cpu().numpy()

        # Output projection — transpose to NumPy convention (in, out) for
        # cross-backend compatibility (matches TorchModel.save_as_numpy())
        output_proj_w_t = self.output_proj.weight.detach().cpu().T.numpy()
        result[Transformer.OUTPUT_PROJ_W] = output_proj_w_t
        result[Transformer.OUTPUT_PROJ_B] = self.output_proj.bias.detach().cpu().numpy()

        return result

    def load_from_numpy_dict(self, params: dict[str, np.ndarray]) -> None:
        """Load parameters from a NumPy-compatible dictionary.

        Args:
            params: Dictionary mapping parameter names to NumPy arrays.
        """
        def load(key: str, tensor: torch.Tensor) -> None:
            np_array = params[key]
            loaded = torch.from_numpy(np_array).to(tensor.dtype)
            tensor.data.copy_(loaded)

        # Embedding
        load(Transformer.EMBEDDING_WEIGHTS, self.embedding.weight)

        # Stack layers
        for layer_idx, block in enumerate(self.stack.layers):

            # Layer norm gamma (now stored as RMSNorm instance weight)
            load(Block.ln1_gamma(layer_idx), block.ln1.weight)
            load(Block.ln2_gamma(layer_idx), block.ln2.weight)

            # MHA weights
            load(Block.mha(layer_idx, Mha.WQ), block.mha.Wq)
            load(Block.mha(layer_idx, Mha.BQ), block.mha.bq)
            load(Block.mha(layer_idx, Mha.WK), block.mha.Wk)
            load(Block.mha(layer_idx, Mha.BK), block.mha.bk)
            load(Block.mha(layer_idx, Mha.WV), block.mha.Wv)
            load(Block.mha(layer_idx, Mha.BV), block.mha.bv)
            load(Block.mha(layer_idx, Mha.WO), block.mha.Wo)
            load(Block.mha(layer_idx, Mha.BO), block.mha.bo)

            # MoE router and expert weights
            load(Block.moe_router(layer_idx), block.moe.W_router)
            load(Block.moe_bias(layer_idx), block.moe.b_router)

            for expert_idx, expert in enumerate(block.moe.experts):
                load(Block.moe_expert(layer_idx, expert_idx, "W1"), expert.W1)
                load(Block.moe_expert(layer_idx, expert_idx, "W2"), expert.W2)
                load(Block.moe_expert(layer_idx, expert_idx, "W3"), expert.W3)

        # Final ln — accept both naming conventions (final_ln_gamma from np, final_ln from torch)
        try:
            load(Transformer.FINAL_GAMMA, self.final_ln.weight)
        except KeyError:
            load("final_gamma", self.final_ln.weight)

        # Output SwiGLU
        load(Transformer.OUTPUT_W1, self.output.W1)
        load(Transformer.OUTPUT_W2, self.output.W2)
        load(Transformer.OUTPUT_W3, self.output.W3)

        # Output projection — transposed from NumPy (in, out) to PyTorch (out, in)
        def load_output_proj(key, tensor):
            return tensor.data.copy_(
                    torch.from_numpy(params[key]).to(tensor.dtype).T
                )
        load_output_proj(Transformer.OUTPUT_PROJ_W, self.output_proj.weight)
        load(Transformer.OUTPUT_PROJ_B, self.output_proj.bias)
