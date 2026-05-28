import numpy as np


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


class Expert:
    """
    An individual expert in the MoE layer.
    Each expert is essentially a Feed-Forward Network.
    """

    def __init__(self, embed_dim: int, dim_ff: int):
        from src.model.layers import FeedForward

        self.ffn = FeedForward(embed_dim, dim_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.ffn.forward(x)


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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
        Returns:
            [Batch, Seq_Len, Embed_Dim] weighted sum of top-k experts
        """
        batch_size, seq_len, _ = x.shape

        # 1. Get routing probabilities
        # [Batch, Seq_Len, Num_Experts]
        routing_weights = self.router.forward(x)

        # 2. Identify top-k experts for each token
        # argsort gives indices in ascending order; we take the last k
        # [Batch, Seq_Len, k]
        top_k_indices = np.argsort(routing_weights, axis=-1)[..., -self.k :]
        # Get the corresponding weights for these top-k experts
        # [Batch, Seq_Len, k]
        top_k_weights = np.take_along_axis(routing_weights, top_k_indices, axis=-1)

        # Normalize top-k weights so they sum to 1 for each token
        # This ensures the output magnitude stays consistent
        top_k_weights = top_k_weights / (
            np.sum(top_k_weights, axis=-1, keepdims=True) + 1e-8
        )

        # 3. Compute expert outputs
        # For pedagogical simplicity in Numpy, we'll compute all expert outputs
        # and then combine them. (In real sparse MoE, we only compute the top-k).
        # [Num_Experts, Batch, Seq_Len, Embed_Dim]
        all_expert_outputs = np.array([exp.forward(x) for exp in self.experts])

        # 4. Weighted combination
        # Output = Sum_{i in top_k} weight_i * expert_i(x)

        # Initialize output
        combined_output = np.zeros_like(x)

        # For each position in batch and sequence
        for b in range(batch_size):
            for s in range(seq_len):
                for k_idx in range(self.k):
                    expert_idx = top_k_indices[b, s, k_idx]
                    weight = top_k_weights[b, s, k_idx]

                    # Add weighted contribution of the expert
                    combined_output[b, s, :] += (
                        weight * all_expert_outputs[expert_idx, b, s, :]
                    )

        return combined_output
