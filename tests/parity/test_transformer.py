from __future__ import annotations

import numpy as np
import torch


class TestTransformerBackwardLmHeadParity:
    """Parity test for Transformer backward w.r.t. lm_head."""

    def test_backward_lm_head_parity(self):
        """Backward w.r.t. lm_head should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["lm_head"],
            pytorch_grads["lm_head"].detach().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    def test_backward_0_ln1_gamma_parity(self):
        """Backward w.r.t. blocks.0.ln1.gamma should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["blocks.0.ln1.gamma"],
            pytorch_grads["blocks.0.ln1.weight"].detach().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    def test_backward_0_ln1_beta_parity(self):
        """Backward w.r.t. blocks.0.ln1.beta should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["blocks.0.ln1.beta"],
            pytorch_grads["blocks.0.ln1.bias"].detach().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    def test_backward_0_ln2_gamma_parity(self):
        """Backward w.r.t. blocks.0.ln2.gamma should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["blocks.0.ln2.gamma"],
            pytorch_grads["blocks.0.ln2.weight"].detach().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    def test_backward_0_ln2_beta_parity(self):
        """Backward w.r.t. blocks.0.ln2.beta should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["blocks.0.ln2.beta"],
            pytorch_grads["blocks.0.ln2.bias"].detach().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    def test_backward_0_mha_Wq_parity(self):
        """Backward w.r.t. blocks.0.mha.W_q should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["blocks.0.mha.W_q"],
            pytorch_grads["blocks.0.mha.qkv.W_q"].detach().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    def test_backward_0_mha_Wk_parity(self):
        """Backward w.r.t. blocks.0.mha.W_k should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["blocks.0.mha.W_k"],
            pytorch_grads["blocks.0.mha.qkv.W_k"].detach().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    def test_backward_0_moe_expert_0_W1_parity(self):
        """Backward w.r.t. blocks.0.moe.expert.0.W1 should match between NumPy and PyTorch."""
        np.random.seed(42)
        batch_size, seq_len, vocab_size, embed_dim = 2, 8, 64, 64
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float64)
        grad_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float64)

        from model.transformer import Transformer as NumPyTransformer
        from model.pytorch.transformer import PyTorchTransformer as PyTorchTransformerModel

        model_np = NumPyTransformer(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )
        model_pt = PyTorchTransformerModel(
            vocab_size, embed_dim, 2, 4, 4, max_seq_len=512,
        )

        # Convert to double first, then sync to avoid float32 truncation during copy_()
        model_pt.double()
        self._sync_model_params(model_np, model_pt)

        _, numpy_cache = model_np.forward(input_ids, mask)
        _, pytorch_cache = model_pt.forward(
            torch.from_numpy(input_ids).long(), torch.from_numpy(mask)
        )

        numpy_grads = model_np.backward(grad_logits, numpy_cache)
        pytorch_grads = model_pt.backward(
            torch.from_numpy(grad_logits), pytorch_cache
        )

        np.testing.assert_allclose(
            numpy_grads["blocks.0.moe.expert.0.W1"],
            pytorch_grads["blocks.0.moe.expert.0.w1"].detach().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    def _sync_model_params(self, model_np, model_pt):
        """Sync all NumPy model params to PyTorch model."""
        np_params = model_np.get_params()
        for name, param in np_params.items():
            if name.startswith("token_embedding."):
                model_pt.token_embedding.weight.data = torch.from_numpy(param)
            elif name.startswith("blocks."):
                parts = name.split(".", 3)
                i = int(parts[1])
                sublayer = parts[2]
                param_name = parts[3]
                block = model_pt.blocks[i]
                if sublayer == "ln1":
                    if param_name == "gamma":
                        block.ln1.gamma.data = torch.from_numpy(param)
                    elif param_name == "beta":
                        block.ln1.beta.data = torch.from_numpy(param)
                elif sublayer == "ln2":
                    if param_name == "gamma":
                        block.ln2.gamma.data = torch.from_numpy(param)
                    elif param_name == "beta":
                        block.ln2.beta.data = torch.from_numpy(param)
                elif sublayer == "mha":
                    canonical = f"qkv.{param_name}" if param_name in ("W_q", "W_k", "W_v") else f"o.{param_name}"
                    block.mha.set_params({canonical: param})
                elif sublayer == "moe":
                    # NumPy MoE uses uppercase W1, W2; PyTorch uses lowercase w1, w2
                    # b1, b2 are same in both
                    # NumPy router uses "weights" key; PyTorch uses "w"
                    if param_name.startswith("router.weights"):
                        block.moe.router.set_params({"w": param})
                    elif param_name.startswith("router"):
                        block.moe.router.set_params({param_name.replace("weights", "w"): param})
                    elif param_name.startswith("expert."):
                        canonical = param_name.replace(".W", ".w")
                        block.moe.set_params({canonical: param})
            elif name == "lm_head":
                model_pt.lm_head.weight.data = torch.from_numpy(param.T)
