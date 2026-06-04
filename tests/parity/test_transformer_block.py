from __future__ import annotations

import numpy as np
import torch
from model.attention import MultiHeadAttention as NumPyMHA
from model.numpy.moe import MoELayer as NumPyMoE
from model.pytorch.attention import PyTorchMultiHeadAttention
from model.pytorch.moe import PyTorchMoELayer


class TestTransformerBlockParity:
    """Parity tests comparing NumPy TransformerBlock against PyTorch TransformerBlock."""

    def setup_method(self):
        self.embed_dim = 64
        self.num_heads = 4
        self.num_experts = 4
        self.dim_ff = 128
        self.num_experts_per_token = 2
        self.seq_len = 8
        self.batch_size = 2

        # NumPy components
        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim,
            self.num_experts,
            dim_ff=self.dim_ff,
            num_experts_per_token=self.num_experts_per_token,
        )

        # PyTorch components
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim,
            self.num_experts,
            dim_ff=self.dim_ff,
            num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        # Sync: NumPy -> PyTorch
        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))

        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

    def test_forward_parity(self):
        """Forward pass should match with tolerance rtol=1e-4, atol=1e-4."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        # Import here so we get a clean failure if class doesn't exist yet
        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        numpy_out, _ = block_np.forward(x, mask)
        pytorch_out, _ = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))

        np.testing.assert_allclose(
            numpy_out, pytorch_out.detach().numpy(), rtol=1e-4, atol=1e-4
        )

    def test_backward_ln1_gamma_parity(self):
        """Backward w.r.t. ln1 gamma should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        _, numpy_grads = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            numpy_grads["ln1.gamma"],
            pytorch_grads["ln1.weight"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_backward_ln1_beta_parity(self):
        """Backward w.r.t. ln1 beta should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        _, numpy_grads = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            numpy_grads["ln1.beta"],
            pytorch_grads["ln1.bias"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_backward_ln2_gamma_parity(self):
        """Backward w.r.t. ln2 gamma should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        _, numpy_grads = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            numpy_grads["ln2.gamma"],
            pytorch_grads["ln2.weight"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_backward_ln2_beta_parity(self):
        """Backward w.r.t. ln2 beta should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        _, numpy_grads = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            numpy_grads["ln2.beta"],
            pytorch_grads["ln2.bias"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_backward_mha_W_q_parity(self):
        """Backward w.r.t. MHA W_q should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        _, numpy_grads = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            numpy_grads["mha.W_q"],
            pytorch_grads["mha.qkv.W_q"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_backward_mha_W_o_parity(self):
        """Backward w.r.t. MHA W_o should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        _, numpy_grads = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            numpy_grads["mha.W_o"],
            pytorch_grads["mha.o.W_o"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_backward_moe_router_w_parity(self):
        """Backward w.r.t. MoE router.w should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        _, numpy_grads = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            numpy_grads["moe.router.w"],
            pytorch_grads["moe.router.w"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_backward_moe_expert_0_w1_parity(self):
        """Backward w.r.t. expert.0.w1 should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        _, numpy_grads = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            numpy_grads["moe.expert.0.w1"],
            pytorch_grads["moe.expert.0.w1"].detach().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_backward_input_x_parity(self):
        """Backward w.r.t. input x should match (wider tolerances)."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float64)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(np.float64)

        from model.numpy.transformer import NumPyTransformerBlock
        from model.pytorch.transformer import PyTorchTransformerBlock

        self.mha_np = NumPyMHA(self.embed_dim, self.num_heads)
        self.moe_np = NumPyMoE(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt = PyTorchMultiHeadAttention(self.embed_dim, self.num_heads)
        self.moe_pt = PyTorchMoELayer(
            self.embed_dim, self.num_experts,
            dim_ff=self.dim_ff, num_experts_per_token=self.num_experts_per_token,
        )
        self.mha_pt.double()
        self.moe_pt.double()

        np_params = self.mha_np.get_params()
        for k, v in np_params.items():
            with torch.no_grad():
                getattr(self.mha_pt, k).copy_(torch.from_numpy(v))
        np_moe_params = self.moe_np.get_params()
        for name, param in np_moe_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    k = name.split(".", 1)[1]
                    self.moe_pt.router.set_params({k: torch.from_numpy(param)})
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    idx = int(parts[1])
                    self.moe_pt.experts[idx].set_params({parts[2]: torch.from_numpy(param)})

        block_np = NumPyTransformerBlock(self.embed_dim, self.mha_np, self.moe_np)
        block_pt = PyTorchTransformerBlock(self.embed_dim, self.mha_pt, self.moe_pt)

        _, numpy_cache = block_np.forward(x, mask)
        dx_numpy, _ = block_np.backward(grad_output, numpy_cache)

        _, pytorch_cache = block_pt.forward(torch.from_numpy(x), torch.from_numpy(mask))
        dx_pytorch, _ = block_pt.backward(torch.from_numpy(grad_output), pytorch_cache)

        np.testing.assert_allclose(
            dx_numpy,
            dx_pytorch.detach().numpy(),
            rtol=1e-3, atol=1e-3,
        )
