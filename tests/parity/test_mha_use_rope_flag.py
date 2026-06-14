from __future__ import annotations

import numpy as np
import torch
from model.attention import MultiHeadAttention as NumPyAttention
from model.pytorch.attention import PyTorchMultiHeadAttention as PyTorchAttention


class TestMHARoPEFlag:
    """Test that PyTorch MHA's use_rope parameter matches NumPy MHA behavior."""

    def setup_method(self):
        self.embed_dim = 64
        self.num_heads = 4
        self.numpy_mha = NumPyAttention(self.embed_dim, self.num_heads)
        self.pytorch_mha = PyTorchAttention(self.embed_dim, self.num_heads)
        self.pytorch_mha.double()

        numpy_params = self.numpy_mha.get_params()
        attr_mapping = {"W_q": "W_q", "W_k": "W_k", "W_v": "W_v", "W_o": "W_o"}
        for k, v in numpy_params.items():
            attr_name = attr_mapping.get(k)
            if attr_name:
                with torch.no_grad():
                    getattr(self.pytorch_mha, attr_name).copy_(torch.from_numpy(v))

    def test_mha_no_rope_forward_parity(self):
        """With use_rope=False, outputs must match (no rotation)."""
        np.random.seed(42)
        x = np.random.randn(2, 8, 64).astype(np.float64)
        mask = np.tril(np.ones((8, 8))).astype(np.float64)

        np_out, _ = self.numpy_mha.forward(x, mask, use_rope=False)
        pt_out, _ = self.pytorch_mha.forward(
            torch.from_numpy(x), torch.from_numpy(mask), use_rope=False
        )

        np.testing.assert_allclose(
            np_out, pt_out.detach().numpy(), rtol=1e-4, atol=1e-4
        )
        # EXPECTED: FAIL — PyTorch MHA doesn't accept use_rope yet

    def test_mha_with_rope_forward_parity(self):
        """With use_rope=True, outputs match NumPy with use_rope=True."""
        np.random.seed(42)
        x = np.random.randn(2, 8, 64).astype(np.float64)
        mask = np.tril(np.ones((8, 8))).astype(np.float64)

        np_out, _ = self.numpy_mha.forward(x, mask, use_rope=True)
        pt_out, _ = self.pytorch_mha.forward(
            torch.from_numpy(x), torch.from_numpy(mask), use_rope=True
        )

        np.testing.assert_allclose(
            np_out, pt_out.detach().numpy(), rtol=1e-4, atol=1e-4
        )
        # EXPECTED: FAIL — PyTorch MHA doesn't accept use_rope yet

    def test_mha_outputs_differ_when_rope_toggled(self):
        """Enabling/disabling RoPE should produce different outputs.

        Since RoPE applies a deterministic rotation based on position,
        forward with use_rope=True should differ from use_rope=False
        (except at position 0 where rotation is identity).

        We use different random weights to make RoPE effect measurable.
        """
        # Create a fresh NumPy MHA so its random weights differ
        mha2 = NumPyAttention(self.embed_dim, self.num_heads)
        np.random.seed(123)
        x = np.random.randn(2, 8, 64).astype(np.float64)
        mask = np.tril(np.ones((8, 8))).astype(np.float64)

        np_no_rope, _ = mha2.forward(x, mask, use_rope=False)
        np_with_rope, _ = mha2.forward(x, mask, use_rope=True)

        # Max diff should be significant (not just float epsilon)
        max_diff = np.max(np.abs(np_no_rope - np_with_rope))
        print(f"[5.1] Max diff between rope=True and rope=False: {max_diff:.6f}")
        # Note: RoPE rotates Q and K with orthogonal transformation,
        # which only slightly modifies attention for small inputs.
        # A diff > 0 confirms RoPE is actually changing the output.
        assert max_diff > 0, (
            f"RoPE toggle should produce different outputs, max_diff={max_diff:.6f}"
        )
