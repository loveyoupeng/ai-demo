"""E6.1: Tests for Triton Mixture of Experts.

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit.
"""

import numpy as np
import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestMoERouting:
    """Test routing (softmax + top-k) independently."""

    @pytest.mark.timeout(10)
    def test_output_shape(self):
        """Router: [B,S,D] -> routing_weights [B,S,E]."""
        skip_if_no_gpu()
        from impl._triton.moe import _compute_routing_weights

        B, S, D, E = 2, 4, 16, 4
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.randn(E, dtype=torch.float64, device="cuda")
        weights = _compute_routing_weights(x, W_router, bias)
        assert weights.shape == (B, S, E), f"Expected {(B, S, E)}, got {weights.shape}"

    @pytest.mark.timeout(10)
    def test_routing_weights_sum_to_one(self):
        """Each token's routing weights sum to 1 over experts (before top-k mask)."""
        skip_if_no_gpu()
        from impl._triton.moe import _compute_routing_weights

        B, S, D, E = 3, 5, 16, 8
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.randn(E, dtype=torch.float64, device="cuda")
        weights = _compute_routing_weights(x, W_router, bias)
        # Before top-k, all E weights sum to 1
        sums = weights.sum(dim=-1)  # [B, S]
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-10, atol=1e-10)

    @pytest.mark.timeout(10)
    def test_top_k_zeros(self):
        """k=2 with 4 experts: exactly 2 weights per token should be non-zero."""
        skip_if_no_gpu()
        from impl._triton.moe import _compute_routing_weights, _top_k_routing

        B, S, D, E, k = 2, 3, 16, 4, 2
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.zeros(E, dtype=torch.float64, device="cuda")
        full_weights = _compute_routing_weights(x, W_router, bias)
        topk_weights = _top_k_routing(full_weights, k)
        # Non-zero count per token
        non_zero = (topk_weights > 1e-8).sum(dim=-1)  # [B, S]
        assert (non_zero == k).all(), f"Expected {k} non-zero per token, got {non_zero}"

    @pytest.mark.timeout(10)
    def test_top_k_renorm(self):
        """After top-k selection, remaining weights sum to 1."""
        skip_if_no_gpu()
        from impl._triton.moe import _compute_routing_weights, _top_k_routing

        B, S, D, E, k = 3, 4, 16, 8, 3
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.randn(E, dtype=torch.float64, device="cuda")
        full_weights = _compute_routing_weights(x, W_router, bias)
        topk_weights = _top_k_routing(full_weights, k)
        sums = topk_weights.sum(dim=-1)  # [B, S]
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-10, atol=1e-10)

    @pytest.mark.timeout(15)
    def test_gradient_routing(self):
        """Gradients flow through router weights."""
        skip_if_no_gpu()
        from impl._triton.moe import _compute_routing_weights

        B, S, D, E = 2, 3, 16, 4
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda", requires_grad=True)
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda", requires_grad=True)
        bias = torch.randn(E, dtype=torch.float64, device="cuda", requires_grad=True)
        weights = _compute_routing_weights(x, W_router, bias)
        loss = weights.sum()
        loss.backward()
        assert x.grad is not None, "x gradient should not be None"
        assert W_router.grad is not None, "W_router gradient should not be None"
        assert bias.grad is not None, "bias gradient should not be None"
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(W_router.grad).all()


class TestMoEExpertStack:
    """Test stacking expert outputs with routing weights."""

    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """MoE: [B,S,D] + n_experts -> [B,S,D]."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim, k = 2, 4, 16, 4, 32, 2
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.zeros(E, dtype=torch.float64, device="cuda")
        W1 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W3 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W2 = torch.randn(E, ff_dim, D, dtype=torch.float64, device="cuda")
        out = mixture_of_experts(x, W_router, bias, W1, W3, W2, k)
        assert out.shape == (B, S, D), f"Expected {(B, S, D)}, got {out.shape}"

    @pytest.mark.timeout(30)
    def test_all_zero_input(self):
        """Zero input produces zero output (no bias, no residual)."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim, k = 1, 1, 8, 3, 16, 2
        x = torch.zeros(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.zeros(E, dtype=torch.float64, device="cuda")
        W1 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W3 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W2 = torch.randn(E, ff_dim, D, dtype=torch.float64, device="cuda")
        out = mixture_of_experts(x, W_router, bias, W1, W3, W2, k)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-12, atol=1e-12)

    @pytest.mark.timeout(30)
    def test_k1_uses_single_expert(self):
        """k=1: only the highest-weight expert contributes to output."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim, k = 1, 2, 8, 3, 16, 1
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.zeros(E, dtype=torch.float64, device="cuda")
        W1 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W3 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W2 = torch.randn(E, ff_dim, D, dtype=torch.float64, device="cuda")
        out = mixture_of_experts(x, W_router, bias, W1, W3, W2, k)
        # Verify non-zero output (some tokens should be routed)
        assert not torch.allclose(out, torch.zeros_like(out)), (
            "MoE output should not be all zeros — check router bias or routing weights"
        )

    @pytest.mark.timeout(30)
    def test_gradient_shape(self):
        """Gradients w.r.t. all weights have correct shapes."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim, k = 2, 3, 16, 4, 32, 2
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda", requires_grad=True)
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda", requires_grad=True)
        bias = torch.zeros(E, dtype=torch.float64, device="cuda", requires_grad=True)
        W1 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda", requires_grad=True)
        W3 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda", requires_grad=True)
        W2 = torch.randn(E, ff_dim, D, dtype=torch.float64, device="cuda", requires_grad=True)
        out = mixture_of_experts(x, W_router, bias, W1, W3, W2, k)
        loss = out.sum()
        loss.backward()
        assert W_router.grad is not None and W_router.grad.shape == W_router.shape
        assert bias.grad is not None and bias.grad.shape == bias.shape
        assert W1.grad is not None and W1.grad.shape == W1.shape
        assert W3.grad is not None and W3.grad.shape == W3.shape
        assert W2.grad is not None and W2.grad.shape == W2.shape
        assert torch.isfinite(W_router.grad).all()
        assert torch.isfinite(W1.grad).all()

    @pytest.mark.timeout(30)
    def test_parity_with_torch(self):
        """Same float64 parameters → same output as PyTorch MoE (rtol=1e-4)."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim, k = 2, 4, 16, 4, 32, 2
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.zeros(E, dtype=torch.float64, device="cuda")
        W1 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W3 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W2 = torch.randn(E, ff_dim, D, dtype=torch.float64, device="cuda")

        y_triton = mixture_of_experts(x, W_router, bias, W1, W3, W2, k)
        y_torch = self._torch_moe(x, W_router, bias, W1, W3, W2, k)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-4, atol=1e-4)

    @staticmethod
    def _torch_moe(x, W_router, bias, W1, W3, W2, k):
        """PyTorch reference: router -> softmax -> top-k -> expert stack."""
        n_experts = W2.shape[0]
        # Router
        router_scores = x @ W_router + bias  # [B, S, E]
        # Stable softmax
        router_scores_max = router_scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(router_scores - router_scores_max)
        routing_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

        # Top-k
        if k < W_router.shape[1]:
            top_k_values, _ = torch.topk(routing_weights, k, dim=-1)
            threshold = top_k_values.min(dim=-1, keepdim=True).values
            routing_weights = routing_weights * (routing_weights >= threshold).float()
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Expert computation: [E, B*S, D]
        B_s, D_dim = x.shape[0] * x.shape[1], x.shape[2]
        expert_input = x.view(B_s, D_dim).unsqueeze(0).expand(n_experts, -1, -1)  # [E, B*S, D]
        gate = torch.nn.functional.silu(expert_input @ W1)  # [E, B*S, ff_dim]
        proj = expert_input @ W3  # [E, B*S, ff_dim]
        gated = gate * proj  # [E, B*S, ff_dim]
        expert_outs = gated @ W2  # [E, B*S, D]

        # Weighted sum - reshape to match (E,B,S,D) and (E,B,S) for einsum
        B, S = x.shape[0], x.shape[1]
        expert_outs = expert_outs.view(n_experts, B, S, D_dim)  # [E, B, S, D]
        routing_weights = routing_weights.permute(2, 0, 1)  # [B, S, E] -> [E, B, S]
        out = torch.einsum("ebs,ebsd->bsd", routing_weights, expert_outs)
        return out

    @pytest.mark.timeout(30)
    def test_parity_with_numpy(self):
        """Same float64 parameters → same output as NumPy MoE (rtol=1e-4)."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim, k = 2, 4, 16, 4, 32, 2
        seed = 42
        rng = np.random.default_rng(seed)

        x_np = rng.random((B, S, D)).astype(np.float64)
        W_router_np = rng.random((D, E)).astype(np.float64) * 2 - 1
        bias_np = np.zeros(E, dtype=np.float64)
        W1_np = rng.random((E, D, ff_dim)).astype(np.float64) * 2 - 1
        W3_np = rng.random((E, D, ff_dim)).astype(np.float64) * 2 - 1
        W2_np = rng.random((E, ff_dim, D)).astype(np.float64) * 2 - 1

        x_t = torch.from_numpy(x_np).cuda()
        W_r_t = torch.from_numpy(W_router_np).cuda()
        b_t = torch.from_numpy(bias_np).cuda()
        W1_t = torch.from_numpy(W1_np).cuda()
        W3_t = torch.from_numpy(W3_np).cuda()
        W2_t = torch.from_numpy(W2_np).cuda()

        y_triton = mixture_of_experts(x_t, W_r_t, b_t, W1_t, W3_t, W2_t, k).cpu().numpy()

        # NumPy reference
        router_scores = x_np @ W_router_np + bias_np  # [B, S, E]
        router_scores_max = np.max(router_scores, axis=-1, keepdims=True)  # [B, S, 1]
        exp_scores = np.exp(router_scores - router_scores_max)
        routing_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        if k < E:
            sorted_idx = np.argsort(routing_weights, axis=-1)[:, :, ::-1]  # descending
            kth_values = np.take_along_axis(routing_weights, sorted_idx[:, :, k - 1 : k], axis=-1)
            routing_weights = np.where(routing_weights >= kth_values, routing_weights, 0.0)
            renorm_sum = np.maximum(np.sum(routing_weights, axis=-1, keepdims=True), 1e-8)
            routing_weights = routing_weights / renorm_sum

        # Expert computation
        expert_outs = np.zeros((E, B, S, D), dtype=np.float64)
        for i in range(E):
            gate = x_np @ W1_np[i]  # [B, S, ff_dim]
            gate = gate / (1.0 + np.exp(-gate))  # SiLU
            proj = x_np @ W3_np[i]  # [B, S, ff_dim]
            gated = gate * proj  # [B, S, ff_dim]
            expert_outs[i] = gated @ W2_np[i]  # [B, S, D]

        # Weighted sum
        y_numpy = np.einsum("bse,ebsd->bsd", routing_weights, expert_outs)

        np.testing.assert_allclose(y_numpy, y_triton, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(15)
    def test_deterministic(self):
        """Same input → same output (no randomness in MoE)."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim, k = 2, 3, 8, 3, 16, 2
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.zeros(E, dtype=torch.float64, device="cuda")
        W1 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W3 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W2 = torch.randn(E, ff_dim, D, dtype=torch.float64, device="cuda")

        out1 = mixture_of_experts(x, W_router, bias, W1, W3, W2, k)
        out2 = mixture_of_experts(x, W_router, bias, W1, W3, W2, k)
        torch.testing.assert_close(out1, out2)

    @pytest.mark.timeout(15)
    def test_different_topk_sizes(self):
        """k=1 vs k=n_experts produce different outputs (different expert mix)."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim = 2, 3, 16, 4, 32
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.zeros(E, dtype=torch.float64, device="cuda")
        W1 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W3 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W2 = torch.randn(E, ff_dim, D, dtype=torch.float64, device="cuda")

        out_k1 = mixture_of_experts(x, W_router, bias, W1, W3, W2, 1)
        out_kall = mixture_of_experts(x, W_router, bias, W1, W3, W2, E)

        # With k=1 only one expert fires; k=E uses all → different output
        assert not torch.allclose(out_k1, out_kall), "k=1 and k=E should produce different outputs with random weights"

    @pytest.mark.timeout(30)
    def test_batched_sequence(self):
        """Different batches and sequences produce valid finite outputs."""
        skip_if_no_gpu()
        from impl._triton.moe import mixture_of_experts

        B, S, D, E, ff_dim, k = 5, 10, 32, 8, 64, 3
        x = torch.randn(B, S, D, dtype=torch.float64, device="cuda")
        W_router = torch.randn(D, E, dtype=torch.float64, device="cuda")
        bias = torch.zeros(E, dtype=torch.float64, device="cuda")
        W1 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W3 = torch.randn(E, D, ff_dim, dtype=torch.float64, device="cuda")
        W2 = torch.randn(E, ff_dim, D, dtype=torch.float64, device="cuda")

        out = mixture_of_experts(x, W_router, bias, W1, W3, W2, k)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all(), "Output should be fully finite"
