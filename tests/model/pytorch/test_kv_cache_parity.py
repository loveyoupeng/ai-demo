"""Test KV cache autoregressive output parity with full-sequence output.

Per the plan:
13-E: `tests/test_turboquant_cache.py` or new file — KV cache autoregressive output
   closely matches full-sequence output (tier-2 tolerance rtol=1e-2)

The test compares autoregressive generation (feed 1 token at a time, append KV cache)
vs feed the full sequence at once.  They should match closely because the KV cache
correctly stores all past K/V tokens for attention.
"""

from __future__ import annotations

import torch

from model.pytorch.attention_kvcache import PyTorchTurboQuantCache
from model.pytorch.transformer import PyTorchTransformer


def _make_model(vocab_size, embed_dim, num_layers, num_heads, num_experts, max_seq_len):
    return PyTorchTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )


# ---------------------------------------------------------------------------
# TDD: Failing test first
# ---------------------------------------------------------------------------


# Step 1: Write a minimal test to check if single-token AR matches full seq
def test_kv_cache_single_token_matches_full():
    """Feed 1 token at a time vs feed full seq → logits at [0,pos] should match."""
    model = _make_model(
        vocab_size=50,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        num_experts=2,
        max_seq_len=10,
    )
    # Copy weights for the AR model
    model_ar = _make_model(
        vocab_size=50,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        num_experts=2,
        max_seq_len=10,
    )
    for p, q in zip(model.parameters(), model_ar.parameters()):
        q.data.copy_(p.data)

    seq_len = 8
    seq = torch.randint(0, 50, (1, seq_len))
    model.eval()
    model_ar.eval()

    # Full batch run — produces logits for all positions in one shot
    mask_full = torch.tril(torch.ones((seq_len, seq_len)))
    with torch.no_grad():
        logits_full, _ = model(seq, mask=mask_full)

    # Autoregressive: feed 1 token at a time using KV cache
    caches = [
        PyTorchTurboQuantCache(
            embed_dim=32,
            num_heads=4,
            max_seq_len=seq_len,
            head_dim=8,
            batch_size=1,
        )
        for _ in range(2)
    ]

    # Feed all tokens via single-token-at-a-time approach
    all_logits_ar = []
    for step in range(seq_len):
        x = seq[:, step : step + 1]
        # Full mask of history (token at step can attend to 0..step)
        cur_mask = torch.tril(torch.ones((step + 1, step + 1)))
        with torch.no_grad():
            logits_step, _ = model_ar(x, mask=cur_mask, kv_caches=caches)
        all_logits_ar.append(logits_step[:, -1, :])  # take last token

    # Compare all positions
    max_diffs = []
    for i in range(seq_len):
        full_at_i = logits_full[0, i, :]
        ar_logits = all_logits_ar[i].squeeze(0)
        diff = torch.abs(full_at_i - ar_logits).max().item()
        max_diffs.append(diff)

    print(f"Max diffs per position: {[round(d, 6) for d in max_diffs]}")

    # Tier-2 tolerance for component-in-chain: rtol=1e-2, atol=1e-2
    for i, diff in enumerate(max_diffs):
        full_val = logits_full[0, i, :].abs().mean().item()
        rtol = diff / (full_val + 1e-8)
        print(
            f"Position {i}: diff={diff:.6f}, full_mean={full_val:.6f}, rtol={rtol:.6f}"
        )
        assert rtol <= 1e-2 or diff <= 1e-2, (
            f"KV cache PARITY FAILED at position {i}: "
            f"diff={diff:.6f}, rtol={rtol:.6f} > 1e-2"
        )
