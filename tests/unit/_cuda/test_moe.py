"""
MoE (Mixture of Experts) kernel tests — top-k routing + weighted sum.

Tests follow Phase F6 of docs/phase_f_plan.md.
"""

import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no CUDA GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")


class TestMoERouting:
    """Top-k expert routing test class."""

    @pytest.mark.timeout(30)
    def test_topk_matches_torch_float32(self):
        """Top-k routing matches PyTorch reference (rtol=1e-3)."""
        skip_if_no_gpu()
        from impl._cuda.moe import moe_forward

        B, S, D, N, K = 2, 8, 64, 4, 2
        torch.manual_seed(42)
        tokens = torch.randn(B, S, D, dtype=torch.float32, device="cuda")
        expert_weights = torch.randn(N, D, D, dtype=torch.float32, device="cuda")
        expert_bias = torch.zeros(N, D, dtype=torch.float32, device="cuda")
        routing_weights = torch.randn(N, D, dtype=torch.float32, device="cuda")

        out_cuda, indices, weights = moe_forward(
            tokens, expert_weights, expert_bias, routing_weights, top_k=K
        )

        # Reference: compute expert outputs, routing scores, top-k
        # expert_outputs[b, s, n, d] = tokens[b, s] @ expert_weights[n] + bias[n][d]
        expert_outputs = torch.stack([
            torch.nn.functional.linear(tokens, expert_weights[n])
            for n in range(N)
        ], dim=2)  # (B, S, N, D)

        # Routing: scores[b,s,n] = tokens[b,s] ⋅ routing_weights[n]
        # F.linear(tokens, routing_weights) where routing_weights is (N, D)
        scores = torch.nn.functional.linear(tokens, routing_weights)  # (B, S, N)

        # Top-k
        topk_scores, topk_idx = torch.topk(scores, K, dim=-1)
        topk_weights = torch.nn.functional.softmax(topk_scores, dim=-1)

        out_ref = torch.zeros_like(tokens)
        for b in range(B):
            for s in range(S):
                for k in range(K):
                    idx = int(topk_idx[b, s, k])
                    out_ref[b, s] += topk_weights[b, s, k] * expert_outputs[b, s, idx]

        assert out_cuda.shape == (B, S, D)
        torch.testing.assert_close(
            out_cuda, out_ref, rtol=1e-3, atol=1e-3,
            msg="CUDA MoE != torch MoE (f32)"
        )

    @pytest.mark.timeout(30)
    def test_topk_matches_torch_float64(self):
        """Top-k routing matches PyTorch reference float64 (rtol=1e-3)."""
        skip_if_no_gpu()
        from impl._cuda.moe import moe_forward

        B, S, D, N, K = 2, 8, 64, 4, 2
        torch.manual_seed(42)
        tokens = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        expert_weights = torch.randn(N, D, D, dtype=torch.float64, device="cuda")
        expert_bias = torch.zeros(N, D, dtype=torch.float64, device="cuda")
        routing_weights = torch.randn(N, D, dtype=torch.float64, device="cuda")

        out_cuda, indices, _w = moe_forward(
            tokens, expert_weights, expert_bias, routing_weights, top_k=K
        )

        # Reference
        expert_outputs = torch.stack([
            torch.nn.functional.linear(tokens, expert_weights[n])
            for n in range(N)
        ], dim=2)  # (B, S, N, D)
        scores = torch.nn.functional.linear(tokens, routing_weights)  # (B, S, N)

        topk_scores, topk_idx = torch.topk(scores, K, dim=-1)
        topk_weights = torch.nn.functional.softmax(topk_scores, dim=-1)

        out_ref = torch.zeros_like(tokens)
        for b in range(B):
            for s in range(S):
                for k in range(K):
                    idx = int(topk_idx[b, s, k])
                    out_ref[b, s] += topk_weights[b, s, k] * expert_outputs[b, s, idx]

        torch.testing.assert_close(
            out_cuda, out_ref, rtol=1e-3, atol=1e-3,
            msg="CUDA MoE != torch MoE (f64)"
        )
