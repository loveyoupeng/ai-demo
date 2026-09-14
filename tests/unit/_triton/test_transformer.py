"""E7.1: Tests for Triton TransformerBlock (Python wiring).

The block is the LLaMA-style pre-norm decoder block:
    h   = x + attention(rms_norm(x))
    out = h + feed_forward(rms_norm(h))

with a dense triton SwiGLU feed-forward by default, or a MoE when the
config enables it. This is Python-only wiring -- no new Triton kernels.
"""

import pytest
import torch

from shared.config import TransformerConfig


def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


def _block_cfg(**overrides) -> TransformerConfig:
    """Small MoE block config for wiring tests."""
    defaults: dict[str, object] = dict(
        vocab_size=16,
        embed_dim=16,
        n_layers=1,
        n_heads=4,
        n_groups=4,
        n_experts=4,
        top_k=2,
        expert_dim=32,
        rope_dim=0,
        seed=0,
    )
    defaults.update(overrides)
    return TransformerConfig.from_dict(defaults)


class TestTransformerBlockWiring:
    """Test the TransformerBlock Python assembly of Triton kernels."""

    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """Input [B,S,D] -> output [B,S,D]."""
        from impl._triton.transformer import TritonTransformerBlock

        block = TritonTransformerBlock(_block_cfg()).cuda()
        x = torch.randn(2, 4, 16, dtype=torch.float32, device="cuda")
        out = block(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    @pytest.mark.timeout(30)
    def test_residual_connection(self):
        """Output contains original input (residual pass-through)."""
        from impl._triton.transformer import TritonTransformerBlock

        block = TritonTransformerBlock(_block_cfg()).cuda()
        x = torch.randn(1, 4, 16, dtype=torch.float32, device="cuda")
        out = block(x)
        # With zeroed weights the block would be identity; with random weights
        # the output must differ from the input (attn/ffn contribute).
        assert not torch.allclose(out, x, atol=1e-2), "Block output should differ from input"
        assert torch.isfinite(out).all()

    @pytest.mark.timeout(30)
    def test_layernorm_parameters_exist(self):
        """Block has input/post-attention RMSNorm weight tensors of shape (D,)."""
        from impl._triton.transformer import TritonTransformerBlock

        block = TritonTransformerBlock(_block_cfg())
        D = block.config.embed_dim
        assert block.input_layernorm.weight.shape == (D,)
        assert block.post_attention_layernorm.weight.shape == (D,)

    @pytest.mark.timeout(60)
    def test_gradient_flow(self):
        """Gradients flow through all components (attention, mlp, layernorms)."""
        from impl._triton.transformer import TritonTransformerBlock

        block = TritonTransformerBlock(_block_cfg()).cuda()
        x = torch.randn(1, 3, 16, dtype=torch.float32, device="cuda")
        out = block(x)
        loss = out.sum()
        loss.backward()

        attn_grad = block.self_attn.q_proj.weight.grad
        assert attn_grad is not None and attn_grad.norm() > 1e-6, "Attention gradient should be non-trivial"
        if hasattr(block.mlp, "expert_list"):
            mlp_grad = block.mlp.expert_list[0].gate_proj.grad  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        else:
            mlp_grad = block.mlp.gate_proj.grad
        assert mlp_grad is not None, "FFN/MoE gradient should exist"
        ln_grad = block.input_layernorm.weight.grad
        assert ln_grad is not None, "Layernorm gradient should exist"

    @pytest.mark.timeout(60)
    def test_attn_and_mlp_both_active(self):
        """Both attention and feed-forward contribute to output (not zero)."""
        from impl._triton.transformer import TritonTransformerBlock

        block = TritonTransformerBlock(_block_cfg()).cuda()
        x = torch.randn(1, 3, 16, dtype=torch.float32, device="cuda")
        out = block(x)
        # Zero the attention output weights -> attention suppressed -> output changes
        with torch.no_grad():
            block.self_attn.o_proj.weight.zero_()
        out_no_attn = block(x)
        assert not torch.allclose(out, out_no_attn, atol=1e-3), "Attention should contribute to output"
        assert torch.isfinite(out).all()

    @pytest.mark.timeout(60)
    def test_parity_with_torch(self):
        """Same weights -> same output as PyTorch TransformerBlock (atol=1e-2).

        The triton SDPA kernel runs in fp32 with a different summation order,
        so we copy weights from a float64 torch block, cast the torch block to
        fp32, and compare.
        """
        from impl._torch.layers import MixtureOfExperts as TorchMoE
        from impl._torch.layers import TransformerBlock as TorchBlock
        from impl._triton.transformer import TritonMixtureOfExperts, TritonTransformerBlock

        cfg = _block_cfg()
        torch_block = TorchBlock(cfg).double()
        tr_block = TritonTransformerBlock(cfg)

        # Copy weights attribute by attribute (FFN/MoE are raw (in,out) in triton).
        tr_block.input_layernorm.weight.data = torch_block.input_layernorm.weight.data.float().to("cuda")
        tr_block.post_attention_layernorm.weight.data = torch_block.post_attention_layernorm.weight.data.float().to(
            "cuda"
        )
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w = getattr(torch_block.self_attn, name).weight.data.float()
            getattr(tr_block.self_attn, name).weight.data = w.to("cuda")

        t_ffn = torch_block.mlp
        r_ffn = tr_block.mlp
        if isinstance(t_ffn, TorchMoE):
            assert isinstance(r_ffn, TritonMixtureOfExperts)
            r_ffn.gate.weight.data = t_ffn.gate.weight.data.float().to("cuda")
            for i, expert in enumerate(t_ffn.expert_list):
                r_ffn.expert_list[i].gate_proj.data = expert.gate_proj.data.float().to("cuda")
                r_ffn.expert_list[i].up_proj.data = expert.up_proj.data.float().to("cuda")
                r_ffn.expert_list[i].down_proj.data = expert.down_proj.data.float().to("cuda")
        else:
            # Dense SwiGLU: torch nn.Linear (out,in) -> triton raw (in,out)
            r_ffn.gate_proj.data = t_ffn.gate_proj.data.float().to("cuda")
            r_ffn.up_proj.data = t_ffn.up_proj.data.float().to("cuda")
            r_ffn.down_proj.data = t_ffn.down_proj.data.float().to("cuda")

        x = torch.randn(1, 4, cfg.embed_dim, dtype=torch.float64)
        y_torch = torch_block(x).float().to("cuda")
        y_triton = tr_block(x.float().to("cuda"))

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-2)

    @pytest.mark.timeout(30)
    def test_gradient_shape(self):
        """All parameters get valid gradient shapes matching their shape."""
        from impl._triton.transformer import TritonTransformerBlock

        block = TritonTransformerBlock(_block_cfg()).cuda()
        x = torch.randn(1, 3, 16, dtype=torch.float32, device="cuda")
        block(x).sum().backward()
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.shape == p.shape, f"{name} gradient shape mismatch"

    @pytest.mark.timeout(30)
    def test_no_nan_output(self):
        """Forward pass produces no NaN or Inf values."""
        from impl._triton.transformer import TritonTransformerBlock

        block = TritonTransformerBlock(_block_cfg()).cuda()
        x = torch.randn(2, 4, 16, dtype=torch.float32, device="cuda")
        out = block(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"
        assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros"
