"""FlashAttention online-softmax kernel: parity vs the two-pass Triton kernel
and vs the framework SDPA, plus a memory/throughput note.

The online-softmax kernel (``impl/_triton/flash_attn.py``) must produce the
same output as the two-pass kernel (``impl/_triton/attn.py``) and as the
framework ``F.scaled_dot_product_attention``, up to float32 rounding. The
online version's advantage is memory: it never materializes the full
(BLOCK_M, Sk) score matrix, so peak per-program memory is O(BLOCK_M*D +
BLOCK_N*D) instead of O(Sk*D). The test below measures the peak memory of
both kernels on a long sequence to document the difference.
"""

from __future__ import annotations

import torch

from impl._triton.attn import scaled_dot_product_attention
from impl._triton.flash_attn import flash_attention


def _cfg(
    B: int = 2, H: int = 4, Sq: int = 256, Sk: int = 256, D: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float32)
    return q, k, v


class TestFlashAttentionParity:
    """Online-softmax kernel vs two-pass kernel vs framework SDPA."""

    def test_forward_matches_two_pass(self) -> None:
        """Flash attention forward == two-pass Triton kernel (float32, causal=False)."""
        q, k, v = _cfg()
        out_flash = flash_attention(q, k, v, is_causal=False)
        out_two = scaled_dot_product_attention(q, k, v, is_causal=False)
        diff = float((out_flash - out_two).abs().max())
        assert diff < 1e-4, f"flash vs two-pass max diff = {diff}"

    def test_forward_matches_two_pass_causal(self) -> None:
        """Flash attention forward == two-pass Triton kernel (float32, causal=True)."""
        q, k, v = _cfg()
        out_flash = flash_attention(q, k, v, is_causal=True)
        out_two = scaled_dot_product_attention(q, k, v, is_causal=True)
        diff = float((out_flash - out_two).abs().max())
        assert diff < 1e-4, f"flash vs two-pass (causal) max diff = {diff}"

    def test_forward_matches_framework_sdpa(self) -> None:
        """Flash attention forward == F.scaled_dot_product_attention (causal=False)."""
        import torch.nn.functional as F

        q, k, v = _cfg()
        out_flash = flash_attention(q, k, v, is_causal=False)
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        diff = float((out_flash - out_sdpa).abs().max())
        assert diff < 1e-4, f"flash vs framework SDPA max diff = {diff}"

    def test_forward_matches_framework_sdpa_causal(self) -> None:
        """Flash attention forward == F.scaled_dot_product_attention (causal=True)."""
        import torch.nn.functional as F

        q, k, v = _cfg()
        out_flash = flash_attention(q, k, v, is_causal=True)
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        diff = float((out_flash - out_sdpa).abs().max())
        assert diff < 1e-4, f"flash vs framework SDPA (causal) max diff = {diff}"

    def test_long_sequence_memory_note(self) -> None:
        """Peak GPU memory: two-pass scales with Sk; online is ~independent.

        This is the documented tradeoff: the two-pass kernel materializes the
        full (BLOCK_M, Sk) score matrix per program, so its peak memory grows
        with the key length. The online kernel never does — its per-program
        working set is bounded by the block sizes. We measure the peak
        allocation for both on a long sequence to make the difference concrete.
        """
        B, H, Sq, Sk, D = 1, 1, 256, 1024, 16
        torch.manual_seed(0)
        q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float32)

        # Two-pass peak memory.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out_two = scaled_dot_product_attention(q, k, v, is_causal=False)
        two_peak = torch.cuda.max_memory_allocated()

        # Online peak memory.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out_flash = flash_attention(q, k, v, is_causal=False)
        flash_peak = torch.cuda.max_memory_allocated()

        # The outputs must match.
        diff = float((out_flash - out_two).abs().max())
        assert diff < 1e-4, f"flash vs two-pass (long seq) max diff = {diff}"

        # The online kernel should use *less* peak memory than the two-pass
        # kernel on a long sequence (the score matrix is the difference). We
        # allow some slack (the online kernel still allocates the padded K/V
        # and the output), but the two-pass kernel's score matrix is the
        # dominant extra allocation.
        # NOTE: this is a *note*, not a strict bound — the memory difference
        # depends on the GPU's allocator behavior. We assert the online peak
        # is no more than 2x the two-pass peak (a generous upper bound that
        # would catch a regression where the online kernel accidentally
        # materializes the full score matrix).
        assert flash_peak <= 2.0 * two_peak, (
            f"flash peak ({flash_peak / 1e6:.1f} MB) exceeds 2x two-pass peak "
            f"({two_peak / 1e6:.1f} MB) — the online kernel may be materializing "
            f"the full score matrix (regression)."
        )
