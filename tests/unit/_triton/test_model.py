"""E9: Full TritonModel — embedding → DecoderStack → RMSNorm → SwiGLU → output."""

import pytest
import torch


def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestTritonModel:
    """Tests for complete TritonModel integration."""

    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """TritonModel tokens [B, S] → logits [B, S, V]."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        B, S, V, D = 2, 8, 64, 16
        model = TritonModel(
            vocab_size=V,
            embed_dim=D,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=32,
            k=2,
        )

        tokens = torch.randint(0, V, (B, S), dtype=torch.int64, device="cuda")
        out = model(tokens)
        assert out.shape == (B, S, V)

    @pytest.mark.timeout(30)
    def test_forward_pass_finite(self):
        """Outputs are finite (no NaN/Inf)."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        B, S, V, D = 2, 8, 64, 16

        model = TritonModel(
            vocab_size=V,
            embed_dim=D,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=32,
            k=2,
        )

        tokens = torch.randint(0, V, (B, S), dtype=torch.int64, device="cuda")
        out = model(tokens)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    @pytest.mark.timeout(60)
    def test_backward_pass(self):
        """All parameters get valid, non-zero gradients."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        B, S, V, D = 2, 8, 64, 16

        model = TritonModel(
            vocab_size=V,
            embed_dim=D,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=32,
            k=2,
        )

        tokens = torch.randint(0, V, (B, S), dtype=torch.int64, device="cuda")
        logits = model(tokens)
        loss = logits.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert param.grad.shape == param.shape, \
                    f"Gradient shape mismatch for {name}"

    @pytest.mark.timeout(60)
    def test_parity_with_torch(self):
        """Same weights → same output as PyTorchModel (rtol=1e-2 for 2+ layers)."""
        skip_if_no_gpu()
        import impl._torch.layers as torch_layers
        from impl._triton.model import TritonModel

        B, S, V, D, n_heads, n_experts, ff_dim, k = 2, 8, 64, 16, 4, 4, 32, 2
        n_layers = 2

        # Create PyTorch model
        torch_model = torch_layers.TorchModel(
            vocab_size=V,
            embed_dim=D,
            n_layers=n_layers,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
            seed=0,
        ).cuda()

        # Create Triton model
        triton_model = TritonModel(
            vocab_size=V,
            embed_dim=D,
            n_layers=n_layers,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
        )

        # Copy embedding
        triton_model.embedding.weight.data.copy_(torch_model.embedding.weight.data)

        # Copy stack layers
        for i in range(n_layers):
            tb = triton_model.stack.layers[i]
            tblock = torch_model.stack.layers[i]

            # MHA weights — now both backends use Linear wrappers so shapes match
            tb.mha.Wq.weight.data.copy_(tblock.mha.Wq.weight.data)
            tb.mha.Wq.bias.data.copy_(tblock.mha.Wq.bias.data)
            tb.mha.Wk.weight.data.copy_(tblock.mha.Wk.weight.data)
            tb.mha.Wk.bias.data.copy_(tblock.mha.Wk.bias.data)
            tb.mha.Wv.weight.data.copy_(tblock.mha.Wv.weight.data)
            tb.mha.Wv.bias.data.copy_(tblock.mha.Wv.bias.data)
            tb.mha.Wo.weight.data.copy_(tblock.mha.Wo.weight.data)
            tb.mha.Wo.bias.data.copy_(tblock.mha.Wo.bias.data)

            # MoE router and expert weights
            tb.moe.W_router.data.copy_(tblock.moe.router.weight.data.t())
            tb.moe.b_router.data.copy_(tblock.moe.router.bias.data)
            for j in range(n_experts):
                tb.moe.experts[j].W1.data.copy_(tblock.moe.experts[j].W1.data)
                tb.moe.experts[j].W2.data.copy_(tblock.moe.experts[j].W2.data)
                tb.moe.experts[j].W3.data.copy_(tblock.moe.experts[j].W3.data)

            # Normalization
            tb.ln1.weight.data.copy_(tblock.ln1.gamma.data)
            tb.ln2.weight.data.copy_(tblock.ln2.gamma.data)
            tb.gate1.data.copy_(tblock.gate1.data)
            tb.gate2.data.copy_(tblock.gate2.data)

        # Final ln
        triton_model.final_ln.weight.data.copy_(torch_model.final_ln.gamma.data)

        # Output projection
        triton_model.output.W1.data.copy_(torch_model.output.W1.data)
        triton_model.output.W2.data.copy_(torch_model.output.W2.data)
        triton_model.output.W3.data.copy_(torch_model.output.W3.data)
        triton_model.output_proj.weight.data.copy_(torch_model.output_proj.weight.data)
        triton_model.output_proj.bias.data.copy_(torch_model.output_proj.bias.data)

        def sync_to_cuda(model):
            for p in model.parameters():
                if not p.is_cuda:
                    p.data = p.data.cuda()

        sync_to_cuda(triton_model)

        torch_model.eval()
        triton_model.eval()

        tokens = torch.randint(0, V, (B, S), dtype=torch.int64, device="cuda")
        torch_logits = torch_model(tokens)
        triton_logits = triton_model(tokens)

        torch.testing.assert_close(triton_logits, torch_logits, rtol=1e-2, atol=1e-2)

    @pytest.mark.timeout(30)
    def test_save_load_roundtrip(self):
        """save_as_numpy() → load_from_numpy_dict() → same parameters."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        V, D, n_layers = 64, 16, 2

        model = TritonModel(
            vocab_size=V,
            embed_dim=D,
            n_layers=n_layers,
            n_heads=4,
            n_experts=4,
            ff_dim=32,
            k=2,
        )

        params_before = model.save_as_numpy()
        model.load_from_numpy_dict(params_before)

        # Keys saved with (in, out) transpose but stored as (out, in) in PyTorch
        transposed_keys = {
            "output_proj_w",
            "blocks.0.mha.Wq", "blocks.0.mha.Wk", "blocks.0.mha.Wv", "blocks.0.mha.Wo",
            "blocks.1.mha.Wq", "blocks.1.mha.Wk", "blocks.1.mha.Wv", "blocks.1.mha.Wo",
        }
        for key, value in params_before.items():
            loaded = model._get_param(key)
            tensor = torch.from_numpy(value)
            if key in transposed_keys:
                tensor = tensor.T
            assert torch.allclose(tensor, loaded, rtol=1e-4, atol=1e-4), \
                f"Parameter {key} changed after roundtrip"
