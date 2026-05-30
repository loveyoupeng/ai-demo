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
        self.last_routing_weights = self._softmax(logits, axis=-1)
        return self.last_routing_weights

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def backward(self, x: np.ndarray, d_probs: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Backward pass for Router.
        
        Args:
            x: Input [Batch, Seq_Len, Embed_Dim]
            d_probs: Gradient of loss w.r.t. routing probabilities [Batch, Seq_Len, Num_Experts, Embed_Dim]
            
        Returns:
            dx: Gradient of loss w.r.t. input x [Batch, Seq_Len, Embed_Dim]
            grads: Dictionary of gradients for parameters (weights)
        """
        batch_size, seq_len, embed_dim = x.shape
        num_experts = self.num_experts
        
        w = self.last_routing_weights
        d_probs_summed = np.sum(d_probs, axis=-1) 
        # d_probs_summed is [Batch, Seq_Len, Num_Experts]
        # w is [Batch, Seq_Len, Num_Experts]
        
        # Let's check the shapes here
        # d_probs_summed: (2, 3, 4)
        # w: (2, 3, 4)
        # they should be broadcastable.
        term2 = np.sum(d_probs_summed * w, axis=-1, keepdims=True) 
        d_logits = d_probs_summed - (w * term2)
        
        d_weights = np.dot(x.reshape(-1, embed_dim).T, d_logits.reshape(-1, num_experts))
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
        dx = self.ffn.backward(d_out)
        grads = self.ffn.get_grads()
        return dx, grads


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

    def get_params(self) -> Dict[str, np.ndarray]:
        params = {}
        # Router params
        for k, v in self.router.get_params().items():
            params[f"router.{k}"] = v
        # Expert params
        for i, expert in enumerate(self.experts):
            for k, v in expert.get_params().items():
                params[f"expert_{i}.{k}"] = v
        return params
