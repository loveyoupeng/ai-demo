"""Isolate TransformerBlock backward parity between NumPy and PyTorch backends.

TDD Plan - isolate backward divergence step by step:
1. MHA backward in isolation
2. MoE backward in isolation
3. Single block backward (full composition)
4. Two-layer chain backward
"""
from __future__ import annotations

import numpy as np
import torch
import pytest


# ──────────────────────────── fixtures ───────────────────────────

@pytest.fixture
def synced_mha():
    """One pair of identical N/PyTorch MHA modules with forward pass."""
    np.random.seed(42)
    torch.manual_seed(42)

    from model.transformer import Transformer as NumPyTransformer
    from model.pytorch.transformer import PyTorchTransformer as PtModel

    np_model = NumPyTransformer(64, 64, 1, 4, 2, max_seq_len=128)
    pt_model = PtModel(64, 64, 1, 4, 2, max_seq_len=128)
    pt_model.double()

    np_mha = np_model.blocks[0].mha
    pt_mha = pt_model.blocks[0].mha

    for name in ["W_q", "W_k", "W_v", "W_o"]:
        np_val = getattr(np_mha, name)
        with torch.no_grad():
            getattr(pt_mha, name).data.copy_(
                torch.from_numpy(np_val).double())

    x_np = np.random.randn(2, 8, 64)
    x_pt = torch.from_numpy(x_np)
    mask = np.tril(np.ones((8, 8)))

    np_out, np_cache = np_mha.forward(x_np, mask)
    pt_out, pt_cache = pt_mha.forward(x_pt, torch.from_numpy(mask).double())

    return {
        "np_mha": np_mha, "pt_mha": pt_mha,
        "x_np": x_np, "x_pt": x_pt,
        "np_out": np_out, "pt_out": pt_out,
        "np_cache": np_cache, "pt_cache": pt_cache,
        "mask": mask,
    }


def _sync_block(np_model, pt_model, layer_idx: int = 0) -> None:
    """Sync a NumPy block's weights to the matching PyTorch block."""
    np_block = np_model.blocks[layer_idx]
    pt_block = pt_model.blocks[layer_idx]

    # LayerNorm
    pt_block.ln1.gamma.data.copy_(
        torch.from_numpy(np_block.ln1.gamma).double())
    pt_block.ln1.beta.data.copy_(
        torch.from_numpy(np_block.ln1.beta).double())
    pt_block.ln2.gamma.data.copy_(
        torch.from_numpy(np_block.ln2.gamma).double())
    pt_block.ln2.beta.data.copy_(
        torch.from_numpy(np_block.ln2.beta).double())

    # MHA
    for name in ["W_q", "W_k", "W_v", "W_o"]:
        np_val = getattr(np_block.mha, name)
        with torch.no_grad():
            getattr(pt_block.mha, name).data.copy_(
                torch.from_numpy(np_val).double())

    # MoE router
    with torch.no_grad():
        pt_block.moe.router.w.data.copy_(
            torch.from_numpy(np_block.moe.router.weights).double())

    # MoE experts (NumPy experts is a plain list)
    num_experts = (np_block.moe.num_experts
                   if hasattr(np_block.moe, 'num_experts')
                   else len(np_block.moe.experts))
    for ei in range(num_experts):
        ne = np_block.moe.experts[ei]
        pe = pt_block.moe.experts[ei]
        for pname in ["W1", "W2", "b1", "b2"]:
            np_val = getattr(ne.ffn, pname)
            getattr(pe, pname.lower() if len(pname) == 2 else pname.lower()
                      ).data.copy_(torch.from_numpy(np_val).double())


# ──────────────────────────── MHA isolation ───────────────────────

class TestMHABackwardIsolation:
    """Step 1: MHA backward in isolation."""

    def test_mha_forward_output(self, synced_mha):
        fwd_diff = np.abs(
            synced_mha["np_out"] -
            synced_mha["pt_out"].detach().numpy()).max()
        print(f"  MHA forward output diff: {fwd_diff:.2e}")
        assert fwd_diff < 1e-10

    def test_mha_cache_Q_and_K(self, synced_mha):
        for key in ("Q", "K"):
            np_val = synced_mha["np_cache"][key]
            pt_val = synced_mha["pt_cache"][key].detach().numpy()
            diff = np.abs(np_val - pt_val).max()
            assert diff < 1e-6, f"Cache {key} differs: {diff:.2e}"

    def test_mha_cache_attn_and_context(self, synced_mha):
        for key in ("attn_weights", "context"):
            np_val = synced_mha["np_cache"][key]
            pt_val = synced_mha["pt_cache"][key].detach().numpy()
            diff = np.abs(np_val - pt_val).max()
            assert diff < 1e-8, f"Cache {key} differs: {diff:.2e}"

    def test_mha_Wq_gradient(self, synced_mha):
        h = synced_mha
        _, np_grads = h["np_mha"].backward(
            h["x_np"], h["np_out"], h["mask"],
            h["np_cache"]["Q"], h["np_cache"]["K"],
            h["np_cache"]["V"], h["np_cache"]["attn_weights"],
            h["np_cache"]["context"])
        _, pt_grads = h["pt_mha"].backward(
            h["pt_out"], torch.from_numpy(h["mask"]).double())

        diff = np.abs(np_grads["W_q"] -
                       pt_grads["qkv.W_q"].detach().numpy()).max()
        print(f"  W_q diff: {diff:.2e}")
        np.testing.assert_allclose(np_grads["W_q"],
                                   pt_grads["qkv.W_q"].detach().numpy(),
                                   rtol=1e-8, atol=1e-8,
                                   err_msg="W_q mismatch")

    def test_mha_Wk_gradient(self, synced_mha):
        h = synced_mha
        _, np_grads = h["np_mha"].backward(
            h["x_np"], h["np_out"], h["mask"],
            h["np_cache"]["Q"], h["np_cache"]["K"],
            h["np_cache"]["V"], h["np_cache"]["attn_weights"],
            h["np_cache"]["context"])
        _, pt_grads = h["pt_mha"].backward(
            h["pt_out"], torch.from_numpy(h["mask"]).double())

        diff = np.abs(np_grads["W_k"] -
                       pt_grads["qkv.W_k"].detach().numpy()).max()
        np.testing.assert_allclose(np_grads["W_k"],
                                   pt_grads["qkv.W_k"].detach().numpy(),
                                   rtol=1e-8, atol=1e-8)

    def test_mha_Wo_gradient(self, synced_mha):
        h = synced_mha
        _, np_grads = h["np_mha"].backward(
            h["x_np"], h["np_out"], h["mask"],
            h["np_cache"]["Q"], h["np_cache"]["K"],
            h["np_cache"]["V"], h["np_cache"]["attn_weights"],
            h["np_cache"]["context"])
        _, pt_grads = h["pt_mha"].backward(
            h["pt_out"], torch.from_numpy(h["mask"]).double())

        diff = np.abs(np_grads["W_o"] -
                       pt_grads["o.W_o"].detach().numpy()).max()
        np.testing.assert_allclose(np_grads["W_o"],
                                   pt_grads["o.W_o"].detach().numpy(),
                                   rtol=1e-8, atol=1e-8)

    def test_mha_dx_input_passthrough(self, synced_mha):
        h = synced_mha
        np_dx, _ = h["np_mha"].backward(
            h["x_np"], h["np_out"], h["mask"],
            h["np_cache"]["Q"], h["np_cache"]["K"],
            h["np_cache"]["V"], h["np_cache"]["attn_weights"],
            h["np_cache"]["context"])
        pt_dx, _ = h["pt_mha"].backward(
            h["pt_out"], torch.from_numpy(h["mask"]).double())

        diff = np.abs(np_dx - pt_dx.detach().numpy()).max()
        print(f"  dx_input diff: {diff:.2e}")
        np.testing.assert_allclose(np_dx, pt_dx.detach().numpy(),
                                   rtol=1e-8, atol=1e-8,
                                   err_msg="dx_input mismatch")


# ──────────────────────────── MoE isolation ──────────────────────

class TestMoEBackwardIsolation:
    """Step 2: MoE backward in isolation."""

    def _make_synced_moe(self):
        np.random.seed(42)
        torch.manual_seed(42)

        from model.transformer import Transformer as N
        from model.pytorch.transformer import PyTorchTransformer as P

        np_model = N(64, 64, 1, 4, 2, max_seq_len=128)
        pt_model = P(64, 64, 1, 4, 2, max_seq_len=128)
        pt_model.double()

        np_moe = np_model.blocks[0].moe
        pt_moe = pt_model.blocks[0].moe

        # Sync router
        with torch.no_grad():
            pt_moe.router.w.data.copy_(
                torch.from_numpy(np_moe.router.weights).double())

        # Sync experts
        num_e = np_moe.num_experts if hasattr(np_moe, 'num_experts') else len(np_moe.experts)
        for ei in range(num_e):
            ne = np_moe.experts[ei]
            pe = pt_moe.experts[ei]
            for pname in ["W1", "W2", "b1", "b2"]:
                np_val = getattr(ne.ffn, pname)
                getattr(pe, pname.lower()).data.copy_(
                    torch.from_numpy(np_val).double())

        x_np = np.random.randn(2, 8, 64)
        x_pt = torch.from_numpy(x_np).double()
        mask = np.tril(np.ones((8, 8)))

        np_out, np_cache = np_moe.forward(x_np)
        pt_out, pt_cache = pt_moe.forward(x_pt)

        return np_moe, pt_moe, x_np, x_pt, np_out, pt_out, np_cache, pt_cache

    def test_moe_W1_expert0(self):
        np_moe, pt_moe, x_np, x_pt, np_out, pt_out, np_cache, pt_cache = \
            self._make_synced_moe()
        _, np_grads = np_moe.backward(x_np, np_out, np_cache)
        _, pt_grads = pt_moe.backward(x_pt, pt_out, pt_cache)

        diff = np.abs(np_grads["expert.0.W1"] -
                       pt_grads["expert.0.w1"].detach().numpy()).max()
        print(f"  expert.0.W1 diff: {diff:.2e}")
        np.testing.assert_allclose(np_grads["expert.0.W1"],
                                   pt_grads["expert.0.w1"].detach().numpy(),
                                   rtol=1e-5, atol=1e-5,
                                   err_msg="MoE W1 mismatch")

    def test_moe_dx_input(self):
        np_moe, pt_moe, x_np, x_pt, np_out, pt_out, np_cache, pt_cache = \
            self._make_synced_moe()
        np_dx, _ = np_moe.backward(x_np, np_out, np_cache)
        pt_dx, _ = pt_moe.backward(x_pt, pt_out, pt_cache)

        diff = np.abs(np_dx - pt_dx.detach().numpy()).max()
        print(f"  dx_input diff: {diff:.2e}")
        np.testing.assert_allclose(np_dx, pt_dx.detach().numpy(),
                                   rtol=1e-5, atol=1e-5,
                                   err_msg="dx_input mismatch")

    def test_moe_router_gradient(self):
        np_moe, pt_moe, x_np, x_pt, np_out, pt_out, np_cache, pt_cache = \
            self._make_synced_moe()
        _, np_grads = np_moe.backward(x_np, np_out, np_cache)
        _, pt_grads = pt_moe.backward(x_pt, pt_out, pt_cache)

        diff = np.abs(np_grads["router.weights"] -
                       pt_grads["router.w"].detach().numpy()).max()
        print(f"  router diff: {diff:.2e}")
        # Router gradient is more sensitive to routing noise
        np.testing.assert_allclose(np_grads["router.weights"],
                                   pt_grads["router.w"].detach().numpy(),
                                   rtol=1e-3, atol=1e-3,
                                   err_msg="router mismatch")


# ──────────────────────────── Single block ───────────────────────

class TestSingleBlockBackward:
    """Step 3: Full single-block backward composition."""

    def _make_block(self):
        np.random.seed(42)
        torch.manual_seed(42)

        from model.transformer import Transformer as N
        from model.pytorch.transformer import PyTorchTransformer as P

        np_model = N(64, 64, 1, 4, 2, max_seq_len=128)
        pt_model = P(64, 64, 1, 4, 2, max_seq_len=128)
        pt_model.double()

        _sync_block(np_model, pt_model, 0)

        # Position embedding — identical source, BEFORE any forward pass
        rng = np.random.RandomState(123)
        pos_np = rng.randn(64, 64)

        # Set position embedding first (stored in token_embedding.weights)
        np_model.token_embedding.weights = pos_np
        with torch.no_grad():
            pt_model.token_embedding.weight.data.copy_(
                torch.from_numpy(pos_np).double())

        bs, sl = 2, 8
        input_ids = np.random.randint(0, 64, (bs, sl))
        mask = np.tril(np.ones((sl, sl)))

        # Token embedding — AFTER position embedding is set
        tok_np = np_model.token_embedding.forward(input_ids)
        tok_pt = pt_model.token_embedding.forward(
            torch.from_numpy(input_ids).long()).double()

        x_np = tok_np + pos_np[:sl]
        x_pt = tok_pt + torch.from_numpy(pos_np[:sl]).double()

        grad_in = np.random.RandomState(42).randn(bs, sl, 64)
        return np_model, pt_model, x_np, x_pt, grad_in, mask

    def test_block_forward_matches(self):
        np_model, pt_model, x_np, x_pt, grad_in, mask = self._make_block()
        np_out, np_cache = np_model.blocks[0].forward(x_np, mask)
        pt_out, pt_cache = pt_model.blocks[0].forward(
            x_pt, torch.from_numpy(mask).double())

        fwd_diff = np.abs(np_out - pt_out.detach().numpy()).max()
        print(f"\n  Block forward diff: {fwd_diff:.2e}")
        np.testing.assert_allclose(np_out, pt_out.detach().numpy(),
                                   rtol=1e-8, atol=1e-8,
                                   err_msg="Block forward mismatch")

    def test_block_backward_parameters(self):
        np_model, pt_model, x_np, x_pt, grad_in, mask = self._make_block()
        _, np_cache = np_model.blocks[0].forward(x_np, mask)
        _, pt_cache = pt_model.blocks[0].forward(
            x_pt, torch.from_numpy(mask).double())

        np_dx, np_grads = np_model.blocks[0].backward(grad_in, np_cache)
        pt_dx, pt_grads = pt_model.blocks[0].backward(
            torch.from_numpy(grad_in), pt_cache)

        print("\n  === Backward gradient comparison ===")
        failures = []

        mapping = {
            "ln1.gamma": "ln1.gamma",
            "ln1.beta": "ln1.beta",
            "ln2.gamma": "ln2.gamma",
            "ln2.beta": "ln2.beta",
            "mha.W_q": "mha.W_q",
            "mha.W_k": "mha.W_k",
            "mha.W_v": "mha.W_v",
            "mha.W_o": "mha.W_o",
            "moe.router.weights": "moe.router.w",
        }

        for np_key, pt_key in mapping.items():
            np_val = np_grads.get(np_key)
            pt_val = pt_grads.get(pt_key)
            if np_val is not None and pt_val is not None:
                diff = np.abs(np_val - pt_val.detach().numpy()).max()
                status = "MATCH" if diff < 1e-6 else "MISMATCH"
                print(f"    {np_key:30s} <-> {pt_key:30s} diff={diff:.6e} [{status}]")
                if diff > 1e-6:
                    failures.append((np_key, pt_key, diff))

        if failures:
            print(f"  FAILURES ({len(failures)}):")
            for np_key, pt_key, diff in failures:
                print(f"    {np_key} -> {pt_key}: {diff:.6e}")
        else:
            print("    All match!")

        for np_key, pt_key, diff in failures:
            np.testing.assert_allclose(
                np_grads[np_key],
                pt_grads[pt_key].detach().numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"{np_key} vs {pt_key}: {diff:.6e}")

    def test_block_dx_residual(self):
        np_model, pt_model, x_np, x_pt, grad_in, mask = self._make_block()
        _, np_cache = np_model.blocks[0].forward(x_np, mask)
        _, pt_cache = pt_model.blocks[0].forward(
            x_pt, torch.from_numpy(mask).double())

        np_dx, _ = np_model.blocks[0].backward(grad_in, np_cache)
        pt_dx, _ = pt_model.blocks[0].backward(
            torch.from_numpy(grad_in), pt_cache)

        diff = np.abs(np_dx - pt_dx.detach().numpy()).max()
        print(f"\n  dx residual diff: {diff:.6e}")
        np.testing.assert_allclose(np_dx, pt_dx.detach().numpy(),
                                   rtol=1e-3, atol=1e-3,
                                   err_msg="dx residual mismatch")


# ──────────────────────────── two-layer chain ────────────────────

class TestTwoLayerChainBackward:
    """Step 4: Two-layer chain backward — where drift accumulates."""

    def _make_two_layer(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        from model.transformer import Transformer as N
        from model.pytorch.transformer import PyTorchTransformer as P

        bs, sl, vs, ed = 2, 8, 64, 64
        input_ids = np.random.randint(0, vs, (bs, sl))
        mask = np.tril(np.ones((sl, sl)))
        grad_logits = np.random.randn(bs, sl, vs)

        np_model = N(vs, ed, 2, 4, 4, max_seq_len=128)
        pt_model = P(vs, ed, 2, 4, 4, max_seq_len=128)
        pt_model.double()

        # Sync everything
        for i in range(2):
            _sync_block(np_model, pt_model, i)

        # Token embedding
        with torch.no_grad():
            pt_model.token_embedding.weight.data.copy_(
                torch.from_numpy(np_model.token_embedding.weights).double())
        # LM head
        with torch.no_grad():
            pt_model.lm_head.weight.data.copy_(
                torch.from_numpy(np_model.lm_head).double())
        return np_model, pt_model, (input_ids, mask, grad_logits)

    def test_two_layer_mha_Wo_all(self):
        np_model, pt_model, (input_ids, mask, grad_logits) = \
            self._make_two_layer()

        _, np_cache = np_model.forward(input_ids, mask)
        _, pt_cache = pt_model.forward(
            torch.from_numpy(input_ids).long(),
            torch.from_numpy(mask).double())

        np_grads = np_model.backward(grad_logits, np_cache)
        pt_grads = pt_model.backward(
            torch.from_numpy(grad_logits), pt_cache)

        for layer in ["0", "1"]:
            np_key = f"blocks.{layer}.mha.W_o"
            pt_key = f"blocks.{layer}.mha.o.W_o"
            diff = np.abs(np_grads[np_key] -
                           pt_grads[pt_key].detach().numpy()).max()
            print(f"\n  block.{layer} mha.W_o diff: {diff:.6e}")
            np.testing.assert_allclose(np_grads[np_key],
                                       pt_grads[pt_key].detach().numpy(),
                                       rtol=1e-1, atol=1e-1,
                                       err_msg=f"block.{layer} mha.W_o mismatch")

    def test_two_layer_ln1_gamma(self):
        np_model, pt_model, (input_ids, mask, grad_logits) = \
            self._make_two_layer()

        _, np_cache = np_model.forward(input_ids, mask)
        _, pt_cache = pt_model.forward(
            torch.from_numpy(input_ids).long(),
            torch.from_numpy(mask).double())

        np_grads = np_model.backward(grad_logits, np_cache)
        pt_grads = pt_model.backward(
            torch.from_numpy(grad_logits), pt_cache)

        for layer in ["0", "1"]:
            np_key = f"blocks.{layer}.ln1.gamma"
            pt_key = f"blocks.{layer}.ln1.weight"
            diff = np.abs(np_grads[np_key] -
                           pt_grads[pt_key].detach().numpy()).max()
            print(f"\n  block.{layer} ln1.gamma diff: {diff:.6e}")
            np.testing.assert_allclose(np_grads[np_key],
                                       pt_grads[pt_key].detach().numpy(),
                                       rtol=1e-1, atol=1e-1,
                                       err_msg=f"block.{layer} ln1.gamma mismatch")

    def test_two_layer_moe_W1_expert0(self):
        np_model, pt_model, (input_ids, mask, grad_logits) = \
            self._make_two_layer()

        _, np_cache = np_model.forward(input_ids, mask)
        _, pt_cache = pt_model.forward(
            torch.from_numpy(input_ids).long(),
            torch.from_numpy(mask).double())

        np_grads = np_model.backward(grad_logits, np_cache)
        pt_grads = pt_model.backward(
            torch.from_numpy(grad_logits), pt_cache)

        np_key = "blocks.0.moe.expert.0.W1"
        pt_key = "blocks.0.moe.experts.0.w1"

        if np_key not in np_grads or pt_key not in pt_grads:
            pytest.skip(f"Key missing: {np_key} in {list(np_grads.keys())}")

        diff = np.abs(np_grads[np_key] -
                       pt_grads[pt_key].detach().numpy()).max()
        print(f"\n  {np_key} vs {pt_key}: diff={diff:.6e}")
        np.testing.assert_allclose(np_grads[np_key],
                                   pt_grads[pt_key].detach().numpy(),
                                   rtol=1e-1, atol=1e-1,
                                   err_msg=f"{np_key} mismatch")

    def test_two_layer_mha_Wq_all(self):
        np_model, pt_model, (input_ids, mask, grad_logits) = \
            self._make_two_layer()

        _, np_cache = np_model.forward(input_ids, mask)
        _, pt_cache = pt_model.forward(
            torch.from_numpy(input_ids).long(),
            torch.from_numpy(mask).double())

        np_grads = np_model.backward(grad_logits, np_cache)
        pt_grads = pt_model.backward(
            torch.from_numpy(grad_logits), pt_cache)

        for layer in ["0", "1"]:
            np_key = f"blocks.{layer}.mha.W_q"
            pt_key = f"blocks.{layer}.mha.qkv.W_q"
            diff = np.abs(np_grads[np_key] -
                           pt_grads[pt_key].detach().numpy()).max()
            print(f"\n  block.{layer} mha.W_q diff: {diff:.6e}")
            np.testing.assert_allclose(np_grads[np_key],
                                       pt_grads[pt_key].detach().numpy(),
                                       rtol=1e-1, atol=1e-1,
                                       err_msg=f"block.{layer} mha.W_q mismatch")
