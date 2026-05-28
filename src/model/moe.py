import numpy as np
from typing import Tuple, Dict

class Router:
    """
    The Routing/Gating network.
    Takes a token embedding and decides which experts to use.
    """

    def __init__(self, embed_dim: int, num_experts: int):
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        # Routing weights: [Embed_Dim, Num_Experts]
        self.weights = np.random.randn(embed_dim, num_experts) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
        Returns:
            [Batch, Seq_Len, Num_Experts] probabilities
        """
        # 1. Project to expert space
        # [Batch, Seq_Len, Num_Experts]
        logits = np.dot(x, self.weights)

        # 2. Softmax to get probabilities
        return self._softmax(logits, axis=-1)

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def backward(self, x: np.ndarray, d_logits: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Backward pass for Router.
        
        Args:
            x: Input [Batch, Seq_Len, Embed_Dim]
            d_logits: Gradient of loss w.r.t. logits [Batch, Seq_Len, Num_Experts]
            
        Returns:
            dx: Gradient of loss w.r.t. input x [Batch, Seq_Len, Embed_Dim]
            grads: Dictionary of gradients for parameters (weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # [Embed_Dim, Num_Experts]
        d_weights = np.dot(x.reshape(-1, self.embed_dim).T, d_logits.reshape(-1, self.num_experts))
        
        # [Batch, Seq_Len, Embed_Dim]
        dx = np.dot(d_logits, self.weights.T)
        
        grads = {"weights": d_weights}
        return dx, grads


class Expert:
    """
    An individual expert in the MoE layer.
    Each expert is essentially a Feed-Forward Network.
    """

    def __init__(self, embed_dim: int, dim_ff: int):
        from model.layers import FeedForward

        self.ffn = FeedForward(embed_dim, dim_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.ffn.forward(x)

    def backward(self, x: np.ndarray, d_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self.ffn.backward(x, d_out)


class MoELayer:
    """
    Mixture of Experts (MoE) layer.
    Uses a router to select the top-k experts for each token.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        dim_ff: int = 128,
        num_experts_per_token: int = 2,
    ):
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.k = num_experts_per_token

        self.router = Router(embed_dim, num_experts)
        self.experts = [Expert(embed_dim, dim_ff) for _ in range(num_experts)]

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
        Returns:
            combined_output: [Batch, Seq_Len, Embed_Dim]
            cache: Dictionary containing intermediate values for backward pass
        """
        batch_size, seq_len, _ = x.shape

        # 1. Get routing probabilities
        # [Batch, Seq_Len, Num_Experts]
        routing_weights = self.router.forward(x)

        # 2. Identify top-k experts for each token
        top_k_indices = np.argsort(routing_weights, axis=-1)[..., -self.k :]
        top_k_weights = np.take_along_axis(routing_weights, top_k_indices, axis=-1)

        # Normalize top-k weights
        top_k_weights = top_k_weights / (np.sum(top_k_weights, axis=-1, keepdims=True) + 1e-8)

        # 3. Compute expert outputs
        # [Num_Experts, Batch, Seq_Len, Embed_Dim]
        all_expert_outputs = np.array([exp.forward(x) for exp in self.experts])

        # 4. Weighted combination
        combined_output = np.zeros_like(x)

        for b in range(batch_size):
            for s in range(seq_len):
                for k_idx in range(self.k):
                    expert_idx = top_k_indices[b, s, k_idx]
                    weight = top_k_weights[b, s, k_idx]
                    combined_output[b, s, :] += weight * all_expert_outputs[expert_idx, b, s, :]

        # Cache for backward pass
        cache = {
            "x": x,
            "routing_weights": routing_weights,
            "top_k_indices": top_k_indices,
            "top_k_weights": top_k_weights,
            "all_expert_outputs": all_expert_outputs,
        }

        return combined_output, cache

    def backward(self, x: np.ndarray, d_out: np.ndarray, cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Backward pass for MoE layer.
        """
        batch_size, seq_len, _ = x.shape
        routing_weights = cache["routing_weights"]
        top_k_indices = cache["top_k_indices"]
        top_k_weights = cache["top_k_weights"]
        all_expert_outputs = cache["all_expert_outputs"]

        # 1. Gradient w.r.t. all_expert_outputs
        # [Num_Experts, Batch, Seq_Len, Embed_Dim]
        d_all_expert_outputs = np.zeros_like(all_expert_outputs)
        
        # 2. Gradient w.r.t. top_k_weights and expert outputs
        # d_top_k_weights: [Batch, Seq_Len, k]
        d_top_k_weights = np.zeros_like(top_k_weights)
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k_idx in range(self.k):
                    expert_idx = top_k_indices[b, s, k_idx]
                    weight = top_k_weights[b, s, k_idx]
                    
                    # d_expert_val = d_out * weight
                    d_all_expert_outputs[expert_idx, b, s, :] += d_out[b, s, :] * weight
                    
                    # d_weight = d_out * expert_val
                    d_top_k_weights[b, s, k_idx] = np.dot(d_out[b, s, :], all_expert_outputs[expert_idx, b, s, :])

        # 3. Gradient w.r.t. experts
        expert_grads = []
        dx_from_experts = np.zeros_like(x)
        for i in range(self.num_experts):
            dx_i, grads_i = self.experts[i].backward(x, d_all_expert_outputs[i])
            expert_grads.append(grads_i)
            
            # Filter dx_i to only include contributions where the expert was used
            # mask: [Batch, Seq_Len, k]
            mask = (top_k_indices == i)
            for b in range(batch_size):
                for s in range(seq_len):
                    if np.any(mask[b, s, :]):
                        # Find which k_idx was the expert
                        k_indices = np.where(mask[b, s, :])[0]
                        for ki in k_indices:
                            weight = top_k_weights[b, s, ki]
                            dx_from_experts[b, s, :] += weight * dx_i[b, s, :]

        # 4. Gradient w.r.t. routing weights
        # d_routing_weights: [Batch, Seq_Len, Num_Experts]
        d_routing_weights = np.zeros_like(routing_weights)
        for b in range(batch_size):
            for s in range(seq_len):
                for k_idx in range(self.k):
                    expert_idx = top_k_indices[b, s, k_idx]
                    d_routing_weights[b, s, expert_idx] += d_top_k_weights[b, s, k_idx]

        # 5. Gradient w.r.t. Router
        dx_router, router_grads = self.router.backward(x, d_routing_weights)

        # 6. Total gradient w.r.t. x
        dx = dx_router + dx_from_experts

        # Flatten the grads to return a Dict[str, np.ndarray]
        flattened_grads: Dict[str, np.ndarray] = {}
        for gk, gv in router_grads.items():
            flattened_grads[f"router.{gk}"] = gv
        for i, g in enumerate(expert_grads):
            for gk, gv in g.items():
                flattened_grads[f"expert_{i}.{gk}"] = gv

        return dx, flattened_grads
