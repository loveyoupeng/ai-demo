"""
MoE (Mixture of Experts) kernel debug tests — break down the weighted sum into
small, verifiable invariants to locate the bug.

These tests verify each component of the MoE pipeline independently:
1. expert_outputs layout (PyTorch stack → view)
2. topk_idx / topk_weights shapes and contiguity
3. flat tensor indexing (view(-1) semantics)
4. CUDA weighted sum kernel (direct, manual test data)
5. Full end-to-end (the original failing test, kept as regression)
"""

import pytest
import torch


def skip_if_no_gpu():
    """Skip test if no CUDA GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")


class TestMoEExpertOutputsLayout:
    """Verify that PyTorch expert_outputs layout matches what the kernel expects."""

    @pytest.mark.timeout(30)
    def test_expert_outputs_stack_shape(self):
        """ torch.stack(..., dim=2) produces (B, S, N, D). """
        skip_if_no_gpu()
        B, S, D, N = 2, 3, 4, 5
        tokens = torch.randn(B, S, D, device="cuda")
        expert_weights = torch.randn(N, D, D, device="cuda")
        expert_outputs = torch.stack([
            torch.nn.functional.linear(tokens, expert_weights[n])
            for n in range(N)
        ], dim=2)
        assert expert_outputs.shape == (B, S, N, D)

    @pytest.mark.timeout(30)
    def test_expert_outputs_view_layout(self):
        """ (B, S, N, D).view(B*S, N, D) should give (total_tokens, N, D)
        where view(token_idx, expert_id, value_idx) gives correct data. """
        skip_if_no_gpu()
        B, S, D, N = 2, 3, 4, 5
        tokens = torch.randn(B, S, D, device="cuda")
        expert_weights = torch.randn(N, D, D, device="cuda")
        exp_out = torch.stack([
            torch.nn.functional.linear(tokens, expert_weights[n])
            for n in range(N)
        ], dim=2)  # (B, S, N, D)

        total = B * S
        flat = exp_out.view(total, N, D)
        assert flat.shape == (total, N, D)

        # Verify: exp_out[0, 0, 2, 3] == flat[0, 2, 3]
        assert torch.allclose(exp_out[0, 0, 2, 3], flat[0, 2, 3])
        # Verify: exp_out[1, 2, 4, 0] == flat[5, 4, 0]
        assert torch.allclose(exp_out[1, 2, 4, 0], flat[5, 4, 0])

    @pytest.mark.timeout(30)
    def test_expert_outputs_contiguous(self):
        """ expert_outputs.view(total, N, D) must be contiguous for kernel. """
        skip_if_no_gpu()
        B, S, D, N = 2, 3, 4, 5
        tokens = torch.randn(B, S, D, device="cuda")
        expert_weights = torch.randn(N, D, D, device="cuda")
        exp_out = torch.stack([
            torch.nn.functional.linear(tokens, expert_weights[n])
            for n in range(N)
        ], dim=2)
        flat = exp_out.view(_total_tokens := B * S, N, D)
        assert flat.is_contiguous(), "expert_outputs.view() is not contiguous!"


class TestMoETopkRouting:
    """Verify topk_idx and topk_weights shapes, contiguity, and values."""

    @pytest.mark.timeout(30)
    def test_topk_idx_shape(self):
        skip_if_no_gpu()
        B, S, D, N, K = 2, 3, 4, 5, 2
        tokens = torch.randn(B, S, D, device="cuda")
        routing_weights = torch.randn(N, D, device="cuda")
        scores = torch.nn.functional.linear(tokens, routing_weights)  # (B, S, N)
        _, topk_idx = torch.topk(scores, K, dim=-1)
        assert topk_idx.shape == (B, S, K)

    @pytest.mark.timeout(30)
    def test_topk_weights_sum_to_one(self):
        skip_if_no_gpu()
        B, S, D, N, K = 2, 3, 4, 5, 2
        tokens = torch.randn(B, S, D, device="cuda")
        routing_weights = torch.randn(N, D, device="cuda")
        scores = torch.nn.functional.linear(tokens, routing_weights)
        topk_scores, _ = torch.topk(scores, K, dim=-1)
        topk_weights = torch.nn.functional.softmax(topk_scores, dim=-1)
        # Weights should sum to 1 per (b, s)
        assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(B, S, device="cuda"), atol=1e-5)

    @pytest.mark.timeout(30)
    def test_topk_idx_values_valid(self):
        """ All topk_idx values must be in range [0, n_experts). """
        skip_if_no_gpu()
        B, S, D, N, K = 2, 3, 4, 5, 2
        tokens = torch.randn(B, S, D, device="cuda")
        routing_weights = torch.randn(N, D, device="cuda")
        scores = torch.nn.functional.linear(tokens, routing_weights)
        _, topk_idx = torch.topk(scores, K, dim=-1)
        assert (topk_idx >= 0).all()
        assert (topk_idx < N).all()

    @pytest.mark.timeout(30)
    def test_flat_view_order(self):
        """ (B, S, K).view(-1) should give [topk_idx[0,0,0], topk_idx[0,0,1],
            topk_idx[0,1,0], topk_idx[0,1,1], topk_idx[1,0,0], ...] """
        skip_if_no_gpu()
        B, S, K = 2, 3, 2
        topk_idx = torch.arange(B * S * K, device="cuda").view(B, S, K)
        flat = topk_idx.view(-1)
        expected = torch.arange(B * S * K, device="cuda")
        assert torch.equal(flat, expected)

    @pytest.mark.timeout(30)
    def test_indices_weights_flat_alignment(self):
        """ idx_flat[i * top_k + k] and w_flat[i * top_k + k] should correspond
        to the same (token, expert) pair. """
        skip_if_no_gpu()
        B, S, D, N, K = 2, 3, 4, 5, 2
        tokens = torch.randn(B, S, D, device="cuda")
        routing_weights = torch.randn(N, D, device="cuda")
        scores = torch.nn.functional.linear(tokens, routing_weights)
        topk_scores, topk_idx = torch.topk(scores, K, dim=-1)
        topk_weights = torch.nn.functional.softmax(topk_scores, dim=-1)

        idx_flat = topk_idx.view(-1)
        w_flat = topk_weights.view(-1)

        # For token 0: idx_flat[0] and w_flat[0] are for expert k=0,
        # idx_flat[1] and w_flat[1] are for expert k=1
        assert idx_flat[0].item() == topk_idx[0, 0, 0].item()
        assert w_flat[0].item() == topk_weights[0, 0, 0].item()
        assert idx_flat[1].item() == topk_idx[0, 0, 1].item()
        assert w_flat[1].item() == topk_weights[0, 0, 1].item()

    @pytest.mark.timeout(30)
    def test_flat_contiguous(self):
        """ idx_flat and w_flat must be contiguous for GPU kernel. """
        skip_if_no_gpu()
        B, S, D, N, K = 2, 3, 4, 5, 2
        tokens = torch.randn(B, S, D, device="cuda")
        routing_weights = torch.randn(N, D, device="cuda")
        scores = torch.nn.functional.linear(tokens, routing_weights)
        topk_scores, topk_idx = torch.topk(scores, K, dim=-1)
        topk_weights = torch.nn.functional.softmax(topk_scores, dim=-1)

        assert topk_idx.view(-1).is_contiguous()
        assert topk_weights.view(-1).is_contiguous()


class TestMoEWrtapedSumManual:
    """Test the weighted sum CUDA kernel with known, hand-computed values."""

    @pytest.mark.timeout(30)
    def test_cuda_weighted_sum_two_experts(self):
        """Manually construct expert_outputs, indices, weights and verify CUDA kernel.

        expert_outputs[0] = expert 0 output = [1.0, 2.0]
        expert_outputs[1] = expert 1 output = [3.0, 4.0]
        For token 0:
          k=0: expert=0, weight=0.6 → 0.6 * [1, 2] = [0.6, 1.2]
          k=1: expert=1, weight=0.4 → 0.4 * [3, 4] = [1.2, 1.6]
        out[0] = [1.8, 2.8]
        """
        skip_if_no_gpu()
        from impl._cuda.moe import _launch_moe_weighted_sum_kernel, _MoeKernels

        # 2 tokens, 2 experts, dim=2 → expert_outputs = (2, 2, 2)
        expert_outputs = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],   # token 0: expert 0 = [1,2], expert 1 = [3,4]
            [[5.0, 6.0], [7.0, 8.0]],   # token 1: expert 0 = [5,6], expert 1 = [7,8]
        ], dtype=torch.float32, device="cuda")

        # For token 0: top_k_idx = [0, 1] (expert 0 and expert 1)
        # For token 1: top_k_idx = [0, 1] (expert 0 and expert 1)
        topk_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device="cuda")
        topk_weights = torch.tensor([0.6, 0.4, 0.7, 0.3], dtype=torch.float32, device="cuda")
        output = torch.zeros((2, 2), dtype=torch.float32, device="cuda")

        _MoeKernels.get_weighted_sum_f32_kernel()
        _launch_moe_weighted_sum_kernel(
            expert_outputs, topk_idx, topk_weights, output,
            total_tokens=2, dim=2, n_experts=2, top_k=2,
        )

        # Expected:
        # out[0] = 0.6 * [1,2] + 0.4 * [3,4] = [0.6+1.2, 1.2+1.6] = [1.8, 2.8]
        # out[1] = 0.7 * [5,6] + 0.3 * [7,8] = [3.5+2.1, 4.2+2.4] = [5.6, 6.6]
        expected = torch.tensor([
            [1.8, 2.8],
            [5.6, 6.6],
        ], dtype=torch.float32, device="cuda")

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_cuda_weighted_sum_single_expert(self):
        """top_k=1: only one expert per token, no weighting ambiguity.

        expert_outputs[0] = [[10.0, 20.0], [30.0, 40.0]]
        For token 0: expert=1 (highest), weight=1.0 → 1.0 * [30, 40] = [30, 40]
        """
        skip_if_no_gpu()
        from impl._cuda.moe import _launch_moe_weighted_sum_kernel

        expert_outputs = torch.tensor([
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
        ], dtype=torch.float32, device="cuda")

        topk_idx = torch.tensor([1, 1], dtype=torch.long, device="cuda")  # top_k=1
        topk_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device="cuda")
        output = torch.zeros((2, 2), dtype=torch.float32, device="cuda")

        _launch_moe_weighted_sum_kernel(
            expert_outputs, topk_idx, topk_weights, output,
            total_tokens=2, dim=2, n_experts=2, top_k=1,
        )

        expected = torch.tensor([[30.0, 40.0], [70.0, 80.0]], device="cuda")
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.timeout(30)
    def test_cuda_weighted_sum_zero_weight(self):
        """When one expert has weight 0, it should not contribute."""
        skip_if_no_gpu()
        from impl._cuda.moe import _launch_moe_weighted_sum_kernel

        expert_outputs = torch.tensor([
            [[1.0, 2.0], [100.0, 200.0]],  # expert 1 has huge output
        ], dtype=torch.float32, device="cuda")

        topk_idx = torch.tensor([0, 1], dtype=torch.long, device="cuda")
        topk_weights = torch.tensor([1.0, 0.0], dtype=torch.float32, device="cuda")
        output = torch.zeros((1, 2), dtype=torch.float32, device="cuda")

        _launch_moe_weighted_sum_kernel(
            expert_outputs, topk_idx, topk_weights, output,
            total_tokens=1, dim=2, n_experts=2, top_k=2,
        )

        # Only expert 0 should contribute
        expected = torch.tensor([[1.0, 2.0]], device="cuda")
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestMoEWeightedSumKernelLaunch:
    """Verify that the Python wrapper correctly launches the CUDA kernel."""

    @pytest.mark.timeout(30)
    def test_launch_no_crash(self):
        """Just ensure the kernel launches without errors."""
        skip_if_no_gpu()
        from impl._cuda.moe import _launch_moe_weighted_sum_kernel

        expert_outputs = torch.ones((2, 2, 4), dtype=torch.float32, device="cuda") * 0.5
        topk_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device="cuda")
        topk_weights = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
        output = torch.zeros((2, 4), dtype=torch.float32, device="cuda")

        _launch_moe_weighted_sum_kernel(
            expert_outputs, topk_idx, topk_weights, output,
            total_tokens=2, dim=4, n_experts=2, top_k=2,
        )

        # All values should be 0.5 (0.5*0.5 + 0.5*0.5 = 0.5)
        assert torch.allclose(output, torch.full_like(output, 0.5), atol=1e-4)
        assert torch.isfinite(output).all()

    @pytest.mark.timeout(30)
    def test_output_shape_preserved(self):
        """Output shape must be (total_tokens, dim)."""
        skip_if_no_gpu()
        from impl._cuda.moe import _launch_moe_weighted_sum_kernel

        expert_outputs = torch.randn((6, 4, 8), dtype=torch.float32, device="cuda")  # 6 tokens, 4 experts, dim=8
        topk_idx = torch.zeros((12,), dtype=torch.long, device="cuda")  # 6 * 2 = 12
        topk_weights = torch.ones((12,), dtype=torch.float32, device="cuda") / 2.0  # top_k=2
        output = torch.zeros((6, 8), dtype=torch.float32, device="cuda")

        _launch_moe_weighted_sum_kernel(
            expert_outputs, topk_idx, topk_weights, output,
            total_tokens=6, dim=8, n_experts=4, top_k=2,
        )

        assert output.shape == (6, 8)


class TestMoEE2ERegression:
    """End-to-end regression test — same as original failing test.

    This is the canary that confirms fix. Do not remove.
    """

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

        assert out_cuda.shape == (B, S, D)
        torch.testing.assert_close(
            out_cuda, out_ref, rtol=1e-3, atol=1e-3,
            msg="CUDA MoE != torch MoE (f32)"
        )
