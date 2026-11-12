# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Multi-head attention for the NumPy reference implementation.

Scaled dot-product attention with GQA and RoPE, with the analytic backward.

**Intuition:** attention answers "for this token's current question
(query), which previous tokens (keys) are relevant?". QK^T / sqrt(d) is a
soft semantic lookup — higher scores get more weight in the value mix.
The causal mask stops tokens from peeking at the future, which is what
makes next-token prediction possible. Softmax turns raw scores into a
probability distribution per query position.
"""

from __future__ import annotations

import numpy as np

from impl._np.init import xavier_uniform
from impl._np.rope import RoPE
from impl._np.softmax import stable_softmax
from shared.result import Result


class MultiHeadAttention:
    """Scaled dot-product multi-head attention with Grouped-Query Attention.

    Reference: Vaswani et al. 2017 (§3.2.1/§3.2.2); GQA: Ainslie et al. 2023.

    The model width D is split into H query heads of width hd = D // H.
    Queries get one projection per head; keys and values get one projection
    per *group* (G groups, G <= H). When G == H this is ordinary MHA; when
    G < H it is GQA and each K/V head is shared by H // G query heads — the
    KV cache shrinks by a factor of H // G at inference.

    The (B, S, D) input — where each axis comes from
    ------------------------------------------------
    A language model never sees one string; it sees a rectangular block of
    token ids — B sequences of S token positions, each position a
    D-wide vector. Where each axis comes from:

    - **S** = sequence length = HOW MANY tokens are in THIS forward pass:
      training reads a full context window (a fixed slice, e.g. 128
      positions of a chunked story); prefill reads the whole prompt;
      a decode step reads only the ONE newest token (S = 1, see
      ``forward_step`` — the past lives in the KV cache, so it is not
      re-read).
    - **B** = batch size = how many INDEPENDENT sequences are stacked and
      processed side by side. They never interact: every attention matrix,
      every residual value is computed per sequence — the batch axis
      exists so one matmul processes all B sequences instead of B
      separate loops. Training stacks many sequences per step (e.g. 32
      padded story chunks from ``make_batch`` in
      scripts/train_demo_model_v3.py, each truncated/padded to the same
      S, so one weight update averages their gradients); inference on
      the learning page generates one sequence (B = 1). The whole
      attention math below runs identically for every B row — nothing
      crosses the batch axis.
    - **D** = embed_dim = the model width: each token position is one
      D-dimensional vector (D = 64 in the learning model). LN1 produced
      the ``x`` this operator receives: one (D,) vector per (B, S) cell.

    So ``x: (B, S, D)`` reads: "for each of the B sequences, for each of
    its S token positions, the current D-vector of the residual stream."

    Shape letters (defined once in CONTEXT.md → "Shape notation"; the
    learning page's model-info panel shows the actual numbers):
    B = batch size (independent sequences in parallel; 1 = one generation),
    S = sequence length (token positions in THIS pass), D = embed_dim (the
    model width), H = n_heads, G = n_groups/KV heads, hd = D // H,
    V = vocab_size, E = n_experts, FF = expert_dim, t = KV-cache depth.

    Forward (x: (B, S, D)):

        q = x @ Wq        (B, S, H*hd) → (B, H, S, hd)   [transpose]
        k = x @ Wk        (B, S, G*hd) → (B, G, S, hd)   [transpose]
        v = x @ Wv        (B, S, G*hd) → (B, G, S, hd)   [transpose]
        q, k = RoPE(q), RoPE(k)                       (B, ·, S, hd)
        if G < H: repeat k, v along the head axis so head h uses group h // (H // G)
                  k, v: (B, G, S, hd) → (B, H, S, hd)
        scores = q @ k^T / sqrt(hd)                   (B, H, S, S)
        scores = where(row < col, -inf, scores)      (B, H, S, S)   [causal mask]
        attn   = softmax(scores, axis=-1)             (B, H, S, S)
        ctx    = attn @ v                             (B, H, S, hd)
        out    = ctx @ Wo   (after un-permute to (B, S, H*hd))  (B, S, D)

    Why divide by sqrt(hd)? If q and k entries are ~N(0, 1), the dot product
    over hd dimensions has variance hd, so scores grow with hd and the
    softmax saturates (gradients vanish). Dividing by sqrt(hd) keeps the
    score variance at 1.

    Weights:
        Wq: (D, H*hd)   Wk: (D, G*hd)   Wv: (D, G*hd)   Wo: (H*hd, D)

    Backward (chain rule in reverse order)
    --------------------------------------
    Given dout (B, S, D) and recomputing the forward intermediates:

        1. dctx = dout @ Wo^T           (B, S, H*hd) → (B, H, S, hd) [un-permute]
           dWo  = ctx^T @ dout          (H*hd, D)
        2. dv    = attn^T @ dctx        (B, H, S, hd)
           d_attn = dctx @ v^T          (B, H, S, S)
        3. Softmax backward (p = attn): dscores = p ⊙ (d_attn − p·d_attn) / scale
           (one division: scores = (q·k^T)/scale in the forward).
        4. dq = dscores @ k             (B, H, S, hd)
           dk = dscores^T @ q           (B, H, S, hd)
        5. GQA un-repeat: a K/V group receives the *sum* of the gradients of
           the H // G query heads that share it.
        6. RoPE backward (inverse rotations) on dq and dk.
        7. dWq = x^T @ q_pre, dWk = x^T @ k_pre, dWv = x^T @ v_pre, where
           *_pre are the (pre-RoPE, permuted-back to (B, S, ·)) projections.
        8. dx = q_pre @ Wq^T + k_pre @ Wk^T + v_pre @ Wv^T
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_groups: int,
        rope_dim: int,
        seed: int = 0,
        swa_window: int = 0,
        mla_latent_dim: int = 0,
        q_proj: np.ndarray | None = None,
        k_proj: np.ndarray | None = None,
        v_proj: np.ndarray | None = None,
        o_proj: np.ndarray | None = None,
        kv_down_proj: np.ndarray | None = None,
        k_up_proj: np.ndarray | None = None,
        v_up_proj: np.ndarray | None = None,
    ) -> None:
        """Create the projections.

        Classic mode (mla_latent_dim=0): the four (in, out) projections
        q/k/v/o. MLA mode (mla_latent_dim=L>0, ADR 0009): k/v_proj are
        replaced by the low-rank latent factorization — kv_down (D, L),
        k_up (L, G*hd), v_up (L, G*hd).

        All optional arrays are trained weights injected **at
        construction** (copied in place over the seed-initialized storage;
        storage identity never changes). None (default) keeps the
        seed-initialized weights (the training path).
        """
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_dim = embed_dim // n_heads
        self.rope_dim = rope_dim
        self.swa_window = swa_window
        self.mla_latent_dim = mla_latent_dim
        rng = np.random.default_rng(seed)
        hd = self.head_dim
        # (D, H*hd) query projection
        self.q_proj = xavier_uniform(rng, embed_dim, n_heads * hd)
        if mla_latent_dim > 0:
            # MLA (ADR 0009): K/V are cached as one latent c per token and
            # reconstructed from it — the cache shrinks from 2*G*hd to L
            # floats per token.
            L = mla_latent_dim
            self.kv_down_proj = xavier_uniform(rng, embed_dim, L)  # (D, L)
            self.k_up_proj = xavier_uniform(rng, L, n_groups * hd)  # (L, G*hd)
            self.v_up_proj = xavier_uniform(rng, L, n_groups * hd)  # (L, G*hd)
            self.k_proj = np.zeros((embed_dim, n_groups * hd))  # unused in MLA mode
            self.v_proj = np.zeros((embed_dim, n_groups * hd))
        else:
            # (D, G*hd) key / value projections — smaller when G < H (GQA)
            self.k_proj = xavier_uniform(rng, embed_dim, n_groups * hd)
            self.v_proj = xavier_uniform(rng, embed_dim, n_groups * hd)
            self.kv_down_proj = np.zeros((embed_dim, 1))
            self.k_up_proj = np.zeros((1, n_groups * hd))
            self.v_up_proj = np.zeros((1, n_groups * hd))
        # (H*hd, D) output projection (mixes the heads back into one vector)
        self.o_proj = xavier_uniform(rng, n_heads * hd, embed_dim)
        for target, injected in (
            (self.q_proj, q_proj),
            (self.k_proj, k_proj),
            (self.v_proj, v_proj),
            (self.o_proj, o_proj),
            (self.kv_down_proj, kv_down_proj),
            (self.k_up_proj, k_up_proj),
            (self.v_up_proj, v_up_proj),
        ):
            if injected is not None:
                target[...] = injected

    def forward_state(self, x: np.ndarray, position: int, causal: bool = True) -> Result[np.ndarray, dict]:
        """Run the attention forward; return THE result + the captured context.

        Returns a ``Result`` (shared/result.py): ``.value`` is THE answer
        (the attention output); ``.ctx`` is the captured context,
        sub-grouped by consumer — ``.ctx.backward`` = what the analytic
        backward recomputes from (the gradient chain's inputs);
        ``.ctx.display`` = what ONLY the learning-mode record shows (never
        touched by gradients). Access is typed: ``.unwrap()`` for
        the value, ``.unwrap_with()`` for the flat (value, merged-ctx)
        shape.

        Invariant/contract: ``.value`` is bit-identical whether or not the
        context is captured (the record is an overlay, not a second code
        path); this pass's tokens sit at the CONSECUTIVE absolute
        positions position, position+1, …, position+S-1 (RoPE builds the
        array internally — callers pass the int, never the array); GQA
        repeat (when G < H)
        happens BEFORE the score computation, so the returned ``k`` is
        post-repeat (B, H, S, hd) while ``k_group`` stays per-group
        (B, G, S, hd) — the cache-ready view.

        Parameters
        ----------
        x : np.ndarray, shape (B, S, D)
            The block's input vectors — the post-LN1 stream. Reading the
            axes: the OUTERMOST B is the batch (independent sequences
            stacked so one matmul processes them side by side; they never
            interact — every score/weight below is computed per sequence);
            the MIDDLE S is this pass's token count (training: a full
            context window; prefill: the whole prompt; decode: 1); the
            INNER D is the model width — each (S, D) cell is one token
            position's current residual-stream vector. A concrete slice:
            x[b, s] is "sequence b, token s" as a (D,) vector; x[b] alone
            is that sequence's (S, D) matrix — the matrix the attention
            math actually runs on (B of them in lockstep).
        position : int
            The ABSOLUTE token index of this pass's FIRST position
            (0-based) — the ONLY position information in the model. RoPE
            rotates q and k by angle ∝ (position + s). Positions are
            consecutive by construction: training reads a context window
            (position 0), prefill reads the prompt (position 0), a decode
            step reads the newest token (position = the cache depth), a
            verification chunk reads the draft block (position = the
            cache depth).
        causal : bool, default True
            Mask the future (True required for autoregressive generation —
            position i attends only to j <= i). False = bidirectional
            within the sequence (used only by the DSpark drafter's parallel
            backbone, where order comes from learned position embeddings).

        Returns
        -------
        Result[np.ndarray, dict]
            .value — the attention output, shape (B, S, D) (the
            residual-stream delta, pre-add).
            .ctx.backward — the backward's inputs:
                q_pre, k_pre, v_pre : (B, S, ·) projections before head permute
                q_rope_in, k_rope_in: pre-RoPE q/k in head layout
                q, k, v             : (B, H, S, hd) after RoPE / GQA repeat
                attn                : (B, H, S, S) attention weights
                ctx                 : (B, S, H*hd) merged context (pre Wo)
                scale               : float, sqrt(hd) — the scores divisor
                rope                : RoPE state (freqs/angles/cos/sin)
            .ctx.display — the record-only captures:
                k_group, v_group    : (B, G, S, hd) per-group K/V (cache-ready)
                scores, scores_masked, causal_mask : the score-matrix history
        """
        batch_size, seq_len, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        # Q projection: (B, S, D) @ (D, H*hd) → (B, S, H*hd)
        q_pre = x @ self.q_proj  # (B, S, H*hd)

        # K/V projections. Classic: k_pre/v_pre = x @ Wk/Wv (B, S, G*hd).
        # MLA (ADR 0009): the CACHE stores one latent c = x @ W_kv_down
        # (B, S, L); K/V are reconstructed as c @ W_k_up / c @ W_v_up.
        # We compute both views: the latent (cache-ready) and the
        # reconstructed per-group K/V (attention-ready).
        if self.mla_latent_dim > 0:
            L = self.mla_latent_dim
            c_pre = x @ self.kv_down_proj  # (B, S, L) — the latent
            k_pre = c_pre @ self.k_up_proj  # (B, S, G*hd) — reconstructed K
            v_pre = c_pre @ self.v_up_proj  # (B, S, G*hd) — reconstructed V
            c_group = c_pre.reshape(batch_size, seq_len, 1, L).transpose(0, 2, 1, 3)  # (B, 1, S, L)
        else:
            k_pre = x @ self.k_proj  # (B, S, G*hd)
            v_pre = x @ self.v_proj  # (B, S, G*hd)
            c_pre = None
            c_group = None

        # Split the width into heads: (B, S, H*hd) → (B, H, S, hd)
        q_heads = q_pre.reshape(batch_size, seq_len, H, hd).transpose(0, 2, 1, 3)  # (B, H, S, hd)
        k_heads = k_pre.reshape(batch_size, seq_len, G, hd).transpose(0, 2, 1, 3)  # (B, G, S, hd)
        v = v_pre.reshape(batch_size, seq_len, G, hd).transpose(0, 2, 1, 3)  # (B, G, S, hd)

        # RoPE rotates q and k (position information). RoPE's contract shape
        # is (B, S, H, D), so permute, rotate, permute back.
        rope_res = RoPE().forward_state(q_heads.transpose(0, 2, 1, 3), position, rope_dim=self.rope_dim)
        q, rope = rope_res.unwrap_with()
        q = q.transpose(0, 2, 1, 3)  # (B, H, S, hd)
        k = RoPE().forward(k_heads.transpose(0, 2, 1, 3), position, rope_dim=self.rope_dim)
        k = k.transpose(0, 2, 1, 3)  # (B, G, S, hd)
        k_group, v_group = k, v  # (B, G, S, hd) per-group: RoPE'd K / un-rotated V

        # GQA: broadcast each K/V group to its H // G query heads.
        # Repeat group g H // G times (consecutively): (B, G, S, hd) → (B, H, S, hd)
        if G != H:
            k = np.repeat(k, H // G, axis=1)  # (B, H, S, hd)
            v = np.repeat(v, H // G, axis=1)  # (B, H, S, hd)

        # Scaled dot-product attention.
        scale = float(np.sqrt(hd))
        # PROD: production uses an online-softmax flash-attention kernel that never
        #       materializes the (B, H, S, S) score matrix (Dao et al. 2022)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # (B, H, S, S)

        # Causal mask: position i may attend only to positions j <= i.
        # A lower-triangular mask (broadcast over B and H) sets the strictly
        # upper triangle to -inf; the stable softmax below turns those into
        # exactly-zero attention weight. causal=False (the DSpark drafter's
        # parallel backbone — see impl/_np/drafters.py) keeps the full
        # matrix: every block position sees every other, position
        # information comes from the learned block-position embeddings.
        if causal:
            rows = np.arange(seq_len)[:, None]  # (S, 1) local row index i
            cols = np.arange(seq_len)[None, :]  # (1, S) local col index j
            causal_mask = cols > rows  # (S, S), True where j > i
            if self.swa_window > 0:
                # SWA (ADR 0008): a query at absolute p+i also masks keys at
                # absolute positions < p+i-w — in pass-local coords that is
                # (i - j > w), so the mask becomes causal OR beyond-window.
                causal_mask = causal_mask | ((rows - cols) > self.swa_window)
            scores_masked = np.where(causal_mask, -np.inf, scores)  # (B, H, S, S)
        else:
            causal_mask = np.zeros((seq_len, seq_len), dtype=bool)
            scores_masked = scores

        # Softmax over the key axis → attention weights (one shared helper;
        # the -inf mask entries become exactly 0).
        attn = stable_softmax(scores_masked, axis=-1)  # (B, H, S, S)

        # Weighted sum of values: (B, H, S, S) @ (B, H, S, hd) → (B, H, S, hd)
        ctx = attn @ v  # (B, H, S, hd)

        # Merge heads back: (B, H, S, hd) → (B, S, H*hd)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, H * hd)  # (B, S, H*hd)
        out = ctx @ self.o_proj  # (B, S, D)
        return Result.capture(
            out,
            backward={
                "q_pre": q_pre,
                "k_pre": k_pre,
                "v_pre": v_pre,
                "c_pre": c_pre,
                "q_rope_in": q_heads,
                "k_rope_in": k_heads,
                "q": q,
                "k": k,
                "v": v,
                "attn": attn,
                "ctx": ctx,
                "scale": scale,
                "rope": rope,
            },
            display={
                "k_group": k_group,
                "v_group": v_group,
                "c_group": c_group,
                "scores": scores,
                "scores_masked": scores_masked,
                "causal_mask": causal_mask,
                "swa_window": self.swa_window,
            },
        )

    def forward(self, x: np.ndarray, position: int = 0, causal: bool = True) -> np.ndarray:
        """Multi-head attention forward pass (the plain path — no cache).

        x: (B, S, D) → out: (B, S, D). Contract: bit-identical to the
        cached step path over the same tokens (pinned by the parity tests).
        position: the absolute index of this pass's first token (0-based).
        causal: mask the future (default True — required for autoregressive
            generation); False = bidirectional within the sequence (used by
            the DSpark drafter's parallel backbone).
        """
        return self.forward_state(x, position, causal=causal).unwrap()

    def forward_step(
        self, x: np.ndarray, position: int, cache: dict, quantize: bool = False, record: dict | None = None
    ) -> np.ndarray:
        """Process ONE new token against the cached K/V (per-token inference path).

        x: (B, 1, D) the new token's embedding — ONE new position per
            sequence (S = 1). B = the batch (the learning page: 1
            generation); the token's D-vector came from the embedding
            lookup. The KV cache is what makes S = 1 possible: the PAST
            tokens' K/V live in the cache, so this pass forwards only the
            new position — the "context" is not re-read as tokens at all.
        position: the absolute token index (for RoPE; 0-based) — must
            equal the cache's current depth (the cache holds positions
            0..position-1; the new token becomes position `position`).
        cache: per-layer dict. Two forms:
            Naive (quantize=False): {"k": (B, G, t, hd), "v": (B, G, t, hd)}
                K/V *per group* (GQA keeps the cache small: G heads, not H).
            TurboQuant (quantize=True):
                {"bits_k": (B, H, t, hd) int8, "scales_k": (B, H, t, 1) float,
                 "bits_v": (B, H, t, hd) int8, "scales_v": (B, H, t, 1) float}
                1-bit compressed K/V *per head* (H heads, after GQA repeat);
                each scale is one scalar per (batch, head) per token-write.
            The dict is mutated in place: the new token's K/V are appended
            at the correct position before attention runs.

        quantize: if False (default), append the full-precision K/V (naive
            path, exact). If True, 1-bit quantize the new K/V (sign bits plus
            one scalar scale per (batch, head) per token-write), append the (bits, scale) pair, then dequantize
            the full cached tensor before attention — so the step attends
            against the lossy cache. This is the documented TurboQuant
            compression experiment; it degrades outputs (see the parity-
            budget test) while shrinking the KV memory by ≈4x (int8 bits
            plus one float scale per (batch, head) per token-write).
        record: optional dict; when given it is filled with the same
            intermediate keys as ``forward_state``'s state (the
            learning-mode record) — the math is bit-identical either way.

        Returns
        -------
        out : np.ndarray, shape (B, 1, D)
            The attention output for the new token (the residual-stream
            delta for this position, pre-add). Contract: with quantize=
            False, equal to the matching slice of ``forward`` over the
            same sequence (the exact KV-step guarantee, pinned by tests);
            the cache is left holding positions 0..position (mutated in
            place).

        Math (identical to the slice of ``forward`` that attends the new
        token to all cached tokens; only the new K/V are computed):
            q = (x @ Wq) → (B, H, 1, hd), RoPE at ``position``
            k = (x @ Wk) → (B, G, 1, hd), RoPE at ``position``
            v = (x @ Wv) → (B, G, 1, hd)
            [naive]    append k, v to cache → (B, G, t+1, hd)
            [turbo]    quantize k, v → bits/scale; append → (B, H, t+1, hd);
                       dequantize full cache → k_r, v_r (B, H, t+1, hd)
            [naive]    GQA repeat: k_r, v_r = repeat(cache, H // G, axis=1)
            scores = q @ k_r^T / sqrt(hd)             (B, H, 1, t+1)
            attn   = stable softmax(scores, axis=-1)
            ctx    = attn @ v_r                       (B, H, 1, hd)
            out    = (ctx un-permuted) @ W_o          (B, 1, D)
        """
        batch_size, seq_len, _ = x.shape  # seq_len == 1 for a single step
        assert seq_len == 1, "forward_step expects exactly one token"
        # Fail fast on a cache/schema mismatch (the schema is the caller's
        # quantize flag, carried out-of-band): a naive cache handed to a
        # quantize=True step (or vice versa) would KeyError deep inside
        # attention — here it gets a named error instead.
        if quantize:
            missing = [k for k in ("bits_k", "scales_k", "bits_v", "scales_v") if k not in cache]
            assert not missing, f"quantize=True step needs a TurboQuant cache (missing keys: {missing})"
        elif self.mla_latent_dim > 0:
            missing = [k for k in ("c",) if k not in cache]
            assert not missing, f"MLA step needs a latent {{'c'}} cache (missing keys: {missing})"
        else:
            missing = [k for k in ("k", "v") if k not in cache]
            assert not missing, f"quantize=False step needs a naive {{'k','v'}} cache (missing keys: {missing})"
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        q_pre = x @ self.q_proj  # (B, 1, H*hd)
        if self.mla_latent_dim > 0:
            L = self.mla_latent_dim
            c_pre = x @ self.kv_down_proj  # (B, 1, L) — the latent write
            k_pre = c_pre @ self.k_up_proj  # (B, 1, G*hd) — reconstructed K
            v_pre = c_pre @ self.v_up_proj  # (B, 1, G*hd)
        else:
            c_pre = None
            k_pre = x @ self.k_proj  # (B, 1, G*hd)
            v_pre = x @ self.v_proj  # (B, 1, G*hd)

        q_heads = q_pre.reshape(batch_size, 1, H, hd).transpose(0, 2, 1, 3)  # (B, H, 1, hd)
        k_heads = k_pre.reshape(batch_size, 1, G, hd).transpose(0, 2, 1, 3)  # (B, G, 1, hd)
        v = v_pre.reshape(batch_size, 1, G, hd).transpose(0, 2, 1, 3)  # (B, G, 1, hd)

        q, rope_state = (
            RoPE().forward_state(q_heads.transpose(0, 2, 1, 3), position, rope_dim=self.rope_dim).unwrap_with()
        )
        q = q.transpose(0, 2, 1, 3)  # (B, H, 1, hd)
        k = RoPE().forward_state(k_heads.transpose(0, 2, 1, 3), position, rope_dim=self.rope_dim).unwrap()
        k = k.transpose(0, 2, 1, 3)  # (B, G, 1, hd)

        if quantize:
            # TurboQuant: 1-bit quantize the new K/V (per head, after GQA
            # repeat) and append the (bits, scale) pair to the cache.
            k_r = np.repeat(k, H // G, axis=1) if G != H else k  # (B, H, 1, hd)
            v_r = np.repeat(v, H // G, axis=1) if G != H else v  # (B, H, 1, hd)
            k_bits, k_scale = self._quantize_turbo(k_r)
            v_bits, v_scale = self._quantize_turbo(v_r)
            cache["bits_k"] = np.concatenate([cache["bits_k"], k_bits], axis=2)  # (B, H, t+1, hd)
            cache["scales_k"] = np.concatenate([cache["scales_k"], k_scale], axis=2)  # (B, H, t+1, 1)
            cache["bits_v"] = np.concatenate([cache["bits_v"], v_bits], axis=2)  # (B, H, t+1, hd)
            cache["scales_v"] = np.concatenate([cache["scales_v"], v_scale], axis=2)  # (B, H, t+1, 1)
            # Dequantize the full cached tensor for attention (broadcast the
            # per-head scale over the head-dim axis).
            k_r = self._dequantize_turbo(cache["bits_k"], cache["scales_k"])  # (B, H, t+1, hd)
            v_r = self._dequantize_turbo(cache["bits_v"], cache["scales_v"])  # (B, H, t+1, hd)
        else:
            # Naive: append the new K/V (per group) — or, in MLA mode, the
            # latent c only; K/V are reconstructed from the whole cache.
            if self.mla_latent_dim > 0:
                assert c_pre is not None
                L = self.mla_latent_dim
                cache["c"] = np.concatenate([cache["c"], c_pre.reshape(batch_size, 1, 1, L)], axis=2)  # (B, 1, t+1, L)
                # Reconstruct per-group K/V from the WHOLE cached latent, then
                # RoPE the K rows by their absolute positions (0..t). RoPE's
                # contract is (B, S, H, hd), so go through the (B, t+1, G, hd)
                # view: permute to (B, t+1, G, hd) → rotate → back.
                c_cache = cache["c"].transpose(0, 2, 1, 3)  # (B, t+1, 1, L)
                # c_cache @ W_up: (B, t+1, 1, G*hd) → split into heads:
                # (B, t+1, G, hd) → (B, G, t+1, hd).
                k_g = (c_cache @ self.k_up_proj).reshape(batch_size, -1, G, hd).transpose(0, 2, 1, 3)  # (B, G, t+1, hd)
                v_g = (c_cache @ self.v_up_proj).reshape(batch_size, -1, G, hd).transpose(0, 2, 1, 3)  # (B, G, t+1, hd)
                # RoPE the cached K rows by their absolute positions (0..t):
                # RoPE's contract is (B, S, H, hd), so permute → rotate → back.
                k_g = RoPE().forward(k_g.transpose(0, 2, 1, 3), 0, rope_dim=self.rope_dim).transpose(0, 2, 1, 3)
                k, v = k_g, v_g
            else:
                cache["k"] = np.concatenate([cache["k"], k], axis=2)  # (B, G, t+1, hd)
                cache["v"] = np.concatenate([cache["v"], v], axis=2)  # (B, G, t+1, hd)
                k, v = cache["k"], cache["v"]
            # GQA repeat: broadcast each group to its H // G query heads.
            if G != H:
                k_r = np.repeat(k, H // G, axis=1)  # (B, H, t+1, hd)
                v_r = np.repeat(v, H // G, axis=1)
            else:
                k_r, v_r = k, v

        scale = float(np.sqrt(hd))
        # SWA (ADR 0008) decode: the query at absolute `position` sees only
        # cache rows with absolute position >= position - w. Slice the LAST
        # w+1 rows instead of masking — O(w) work instead of O(t).
        if self.swa_window > 0:
            w = self.swa_window
            start = max(0, k_r.shape[2] - (w + 1))
            k_vis, v_vis = k_r[:, :, start:], v_r[:, :, start:]
            swa_offset = start  # first visible cache row (for the record mask)
        else:
            k_vis, v_vis = k_r, v_r
            swa_offset = 0
        scores = (q @ k_vis.transpose(0, 1, 3, 2)) / scale  # (B, H, 1, t_vis)
        # Nothing is masked in decode: every visible cached row is in the
        # past of the new token, so the causal mask is all zeros.
        attn = stable_softmax(scores, axis=-1)  # (B, H, 1, t_vis) — no mask needed in decode

        ctx = attn @ v_vis  # (B, H, 1, hd)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(batch_size, 1, H * hd)  # (B, 1, H*hd)
        out = ctx @ self.o_proj  # (B, 1, D)

        # Learning-mode record: the same intermediates the dense path keeps,
        # taken from THIS step (the cache after the append is what the query
        # attended to). The dense path's state is filled by forward_state;
        # here we fill the parallel keys from the step-local variables.
        if record is not None:
            # The step record's k/v mirror the torch display path: the FULL
            # GQA-expanded cache (B, H, t, hd) — same as k_cache/v_cache.
            k_exp = k_r  # (B, H, t, hd) — the cache after this append
            v_exp = v_r  # (B, H, t, hd)
            record.update(
                {
                    "q_pre": q_pre,
                    "k_pre": k_pre,
                    "v_pre": v_pre,
                    "c_pre": c_pre,
                    "q_rope_in": q_heads,
                    "k_rope_in": k_heads,
                    "q": q,
                    "k": k_exp,
                    "v": v_exp,
                    "k_group": k,
                    "v_group": v,
                    "c_group": c_pre.reshape(batch_size, 1, 1, self.mla_latent_dim).transpose(0, 2, 1, 3)
                    if c_pre is not None
                    else None,
                    "attn": attn,
                    "ctx": ctx,
                    "scale": scale,
                    "rope": rope_state,
                    "scores": scores,
                    "scores_masked": scores,
                    # SWA decode: rows before swa_offset are outside the
                    # window — record the mask over the FULL cache so the
                    # page can show what was sliced away.
                    "causal_mask": np.concatenate(
                        [np.ones((1, swa_offset), dtype=bool), np.zeros((1, k_r.shape[2] - swa_offset), dtype=bool)],
                        axis=1,
                    ),
                    "k_cache": k_r,
                    "v_cache": v_r,
                    "c_cache": cache.get("c"),
                    "swa_window": self.swa_window,
                }
            )
        return out

    def forward_chunk(self, x: np.ndarray, position: int, cache: dict, record: dict | None = None) -> np.ndarray:
        """Process a CHUNK of c tokens against the cached K/V (verification path).

        The speculative-decoding workhorse: one call appends all c tokens'
        K/V and scores every chunk position in parallel — the target's
        verification pass over a draft block (see ADR 0003). Naive cache
        only (TurboQuant is a single-token write path; verification demands
        full precision so the lossless contract holds).

        x: (B, c, D) the chunk's embeddings — c = the draft block's
            length (k candidate tokens proposed in one speculative round;
            ADR 0003). Each row is one draft token's D-vector, IN ORDER —
            row j is the (j+1)-th token after the cache's last position.
        position: the absolute index of the chunk's FIRST token (0-based;
            = the cache's current depth — the cache holds positions
            0..position-1).
        cache: per-layer dict {"k": (B, G, t, hd), "v": (B, G, t, hd)};
            mutated in place — the chunk's K/V are appended before attention.
        record: optional dict; when given it is filled with the same keys as
            ``forward_step``'s record (the learning-mode record); the math
            is bit-identical either way.

        Returns
        -------
        out : np.ndarray, shape (B, c, D)
            The attention output for every chunk position (pre-residual-
            add). Contract: equal to the matching c slices of ``forward``
            over the full sequence (chunk parity, pinned by tests); the
            cache is left holding positions 0..position+c-1 (mutated in
            place) — the caller trims it on rollback.

        Math (the slice of ``forward`` where positions [position, position+c)
        attend to everything at or before them):
            q = (x @ Wq) → (B, H, c, hd), RoPE at position..position+c-1
            k = (x @ Wk) → (B, G, c, hd), RoPE at the same positions
            v = (x @ Wv) → (B, G, c, hd)
            append k, v to cache → (B, G, t+c, hd)
            k_r, v_r = GQA repeat(cache) → (B, H, t+c, hd)
            scores = q @ k_r^T / sqrt(hd)         (B, H, c, t+c)
            causal  = block-causal (c, t+c) mask — chunk token i attends to
                      all t cache rows plus chunk rows j <= i
            attn    = stable softmax(scores_masked, axis=-1)
            ctx     = attn @ v_r                   (B, H, c, hd)
            out     = (ctx un-permuted) @ W_o      (B, c, D)
        """
        B, c, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        q_pre = x @ self.q_proj  # (B, c, H*hd)
        if self.mla_latent_dim > 0:
            L = self.mla_latent_dim
            c_pre = x @ self.kv_down_proj  # (B, c, L) — the latent write
            k_pre = c_pre @ self.k_up_proj  # (B, c, G*hd)
            v_pre = c_pre @ self.v_up_proj  # (B, c, G*hd)
        else:
            c_pre = None
            k_pre = x @ self.k_proj  # (B, c, G*hd)
            v_pre = x @ self.v_proj  # (B, c, G*hd)

        q_heads = q_pre.reshape(B, c, H, hd).transpose(0, 2, 1, 3)  # (B, H, c, hd)
        k_heads = k_pre.reshape(B, c, G, hd).transpose(0, 2, 1, 3)  # (B, G, c, hd)
        v = v_pre.reshape(B, c, G, hd).transpose(0, 2, 1, 3)  # (B, G, c, hd)

        # RoPE the whole chunk at once (the operator's (B, S, H, hd) view).
        q, rope_state = (
            RoPE().forward_state(q_heads.transpose(0, 2, 1, 3), position, rope_dim=self.rope_dim).unwrap_with()
        )
        q = q.transpose(0, 2, 1, 3)  # (B, H, c, hd)
        k = RoPE().forward_state(k_heads.transpose(0, 2, 1, 3), position, rope_dim=self.rope_dim).unwrap()
        k = k.transpose(0, 2, 1, 3)  # (B, G, c, hd)

        # Append the chunk's K/V (or latent c, MLA mode) to the cache.
        if self.mla_latent_dim > 0:
            L = self.mla_latent_dim
            assert c_pre is not None
            cache["c"] = np.concatenate([cache["c"], c_pre.reshape(B, 1, c, L)], axis=2)  # (B, 1, t+c, L)
            c_cache = cache["c"].transpose(0, 2, 1, 3)  # (B, t+c, 1, L)
            k = (c_cache @ self.k_up_proj).reshape(B, -1, G, hd).transpose(0, 2, 1, 3)  # (B, G, t+c, hd), pre-RoPE
            v = (c_cache @ self.v_up_proj).reshape(B, -1, G, hd).transpose(0, 2, 1, 3)  # (B, G, t+c, hd)
            # RoPE every cached K row by its absolute position (0..t+c-1).
            k = RoPE().forward(k.transpose(0, 2, 1, 3), 0, rope_dim=self.rope_dim).transpose(0, 2, 1, 3)
        else:
            cache["k"] = np.concatenate([cache["k"], k], axis=2)  # (B, G, t+c, hd)
            cache["v"] = np.concatenate([cache["v"], v], axis=2)  # (B, G, t+c, hd)
            k, v = cache["k"], cache["v"]
        if G != H:
            k_r = np.repeat(k, H // G, axis=1)  # (B, H, t+c, hd)
            v_r = np.repeat(v, H // G, axis=1)  # (B, H, t+c, hd)
        else:
            k_r, v_r = k, v

        scale = float(np.sqrt(hd))
        # (B, H, c, t+c) — chunk queries against every cached + chunk key.
        scores = q @ k_r.transpose(0, 1, 3, 2) / scale
        t = k_r.shape[2] - c
        # Block-causal mask: cache rows (0..t-1) are always in the past of the
        # chunk; chunk row t+j is visible to chunk token i iff j <= i.
        # SWA (ADR 0008): a chunk token at absolute position+i may also NOT
        # see cache/chunk columns at absolute position < position+i-w.
        causal = np.triu(np.ones((c, c), dtype=bool), k=1)  # (c, c), True where j > i
        mask = np.zeros((c, t + c), dtype=bool)  # (c, t+c), True = masked
        mask[:, t:] = causal
        if self.swa_window > 0:
            row_abs = position + np.arange(c)[:, None]  # (c, 1) absolute row positions
            col_abs = np.arange(t + c)[None, :]  # (1, t+c) absolute col positions
            mask = mask | ((row_abs - col_abs) > self.swa_window)
        scores_masked = np.where(mask, -np.inf, scores)  # (B, H, c, t+c)

        # Softmax over the key axis → attention weights.
        attn = stable_softmax(scores_masked, axis=-1)  # (B, H, c, t+c)

        ctx = attn @ v_r  # (B, H, c, t+c) @ (B, H, t+c, hd) → (B, H, c, hd)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, c, H * hd)  # (B, c, H*hd)
        out = ctx @ self.o_proj  # (B, c, D)

        if record is not None:
            if self.mla_latent_dim > 0:
                k_exp = np.repeat(k, H // G, axis=1) if G != H else k  # (B, H, t+c, hd)
                v_exp = np.repeat(v, H // G, axis=1) if G != H else v
                k_grp, v_grp, c_grp = None, None, cache["c"]
            else:
                k_exp = np.repeat(cache["k"], H // G, axis=1) if G != H else cache["k"]  # (B, H, t+c, hd)
                v_exp = np.repeat(cache["v"], H // G, axis=1) if G != H else cache["v"]
                k_grp, v_grp, c_grp = cache["k"], cache["v"], None
            record.update(
                {
                    "q_pre": q_pre,
                    "k_pre": k_pre,
                    "v_pre": v_pre,
                    "c_pre": c_pre,
                    "q_rope_in": q_heads,  # pre-RoPE q in head layout
                    "k_rope_in": k_heads,  # pre-RoPE k
                    "q": q,
                    "k": k_exp,
                    "v": v_exp,
                    "k_group": k_grp,
                    "v_group": v_grp,
                    "c_group": c_grp,
                    "attn": attn,
                    "ctx": ctx,
                    "scale": scale,
                    "rope": rope_state,
                    "scores": scores,
                    "scores_masked": scores_masked,
                    "causal_mask": mask,
                    "k_cache": k_r,
                    "v_cache": v_r,
                    "c_cache": cache.get("c"),
                    "swa_window": self.swa_window,
                }
            )
        return out

    @staticmethod
    def _quantize_turbo(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """1-bit quantize a K or V tensor (per head, per token).

        x: (B, H, 1, hd) — one token's K or V (after GQA repeat).
        Returns:
            bits: (B, H, 1, hd) int8 — 1 for positive, 0 for non-positive.
            scale: (B, H, 1, 1) float — one scalar per (batch, head) per
                token-write: mean(|x|) over the (1, hd) token vector.
        Dequantize: bits.astype(float) * scale.
        """
        absolute_values = np.abs(x)  # (B, H, 1, hd)
        scale = np.mean(absolute_values, axis=(-2, -1), keepdims=True)  # (B, H, 1, 1)
        bits = (x > 0).astype(np.int8)  # (B, H, 1, hd)
        return bits, scale

    @staticmethod
    def _quantize_turbo_full(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """1-bit quantize a K or V tensor over a WHOLE cached block (prefill).

        The batched form of ``_quantize_turbo``: the same per-token write
        rule — sign bits plus one scalar scale per (batch, head) per token,
        mean(|x|) over that token's head-dim vector — applied to every
        cached position at once instead of one append at a time.

        x: (B, H, S, hd) — S tokens' K or V (after GQA repeat).
        Returns:
            bits: (B, H, S, hd) int8 — 1 for positive, 0 for non-positive.
            scale: (B, H, S, 1) float.
        Dequantize: bits.astype(float) * scale (broadcast over hd) —
        identical to what per-token appends would have stored.
        """
        absolute_values = np.abs(x)  # (B, H, S, hd)
        scale = np.mean(absolute_values, axis=-1, keepdims=True)  # (B, H, S, 1)
        bits = (x > 0).astype(np.int8)  # (B, H, S, hd)
        return bits, scale

    @staticmethod
    def _dequantize_turbo(bits: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Dequantize 1-bit storage back to float.

        bits: (B, H, t, hd) int8 (0 or 1). scale: (B, H, t, 1) float (one
        scalar per head, broadcast over the head-dim axis).
        Returns: bits.astype(float) * scale → (B, H, t, hd) float.
        """
        return bits.astype(scale.dtype) * scale

    def backward(self, dout: np.ndarray, x: np.ndarray, position: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Analytic backward (derivation in the class docstring).

        Invariant/contract: the forward intermediates are RECOMPUTED from
        ``x`` (stateless — no cached activations), so ``x`` and
        ``position`` must be exactly the forward's inputs; the gradient
        w.r.t. each weight has the weight's own shape, and ``dx`` carries
        the full upstream (the residual is the caller's concern). Verified
        against finite differences (float64, ~1e-10) by
        ``tests/unit/_np/test_gradient_check.py``.

        Parameters
        ----------
        dout : np.ndarray, shape (B, S, D)
            The upstream gradient of the loss w.r.t. this operator's
            output (the residual add is the caller's job).
        x : np.ndarray, shape (B, S, D)
            The forward input (the post-LN1 stream) — recomputes every
            intermediate from it.
        position : int, optional
            The same first-token index the forward used.

        Returns
        -------
        dx : np.ndarray, shape (B, S, D)
            The gradient w.r.t. the input x.
        dparams : dict[str, np.ndarray]
            Gradients keyed by attribute-path local names:
            {"q_proj": (D, H*hd), "k_proj": (D, G*hd), "v_proj": (D, G*hd),
             "o_proj": (H*hd, D)}.
        """
        st = self.forward_state(x, position).ctx.merged()
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        B, S, _D = x.shape
        scale = st["scale"]
        q, k, v = st["q"], st["k"], st["v"]  # (B, H, S, hd), k/v post-repeat (B, H, S, hd)
        attn = st["attn"]  # (B, H, S, S)
        ctx = st["ctx"]  # (B, S, H*hd)

        # 1. Output projection: out = ctx @ Wo
        dctx = dout @ self.o_proj.T  # (B, S, H*hd)
        dWo = ctx.reshape(-1, ctx.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])  # (H*hd, D)
        dctx_heads = dctx.reshape(B, S, H, hd).transpose(0, 2, 1, 3)  # (B, H, S, hd)

        # 2. Value-weighted sum: ctx = attn @ v
        dv = attn.transpose(0, 1, 3, 2) @ dctx_heads  # (B, H, S, hd)
        d_attn = dctx_heads @ v.transpose(0, 1, 3, 2)  # (B, H, S, S)

        # 3. Softmax backward + scale.
        #    s = (q @ k^T) / scale.  dL/ds = p ⊙ (d_attn − p·d_attn)  (softmax jacobian).
        #    dL/d(q @ k^T) = dL/ds / scale.  This single /scale is the only one:
        #    the dot product q·k is scaled by sqrt(hd) once in the forward.
        p_d = np.sum(attn * d_attn, axis=-1, keepdims=True)  # (B, H, S, 1)
        dA = attn * (d_attn - p_d) / scale  # (B, H, S, S) = dL/d(q @ k^T)

        # 4. Score gradient: A = q @ k^T → dq = dA @ k, dk = dA^T @ q.
        dq = dA @ k  # (B, H, S, hd)
        dk = dA.transpose(0, 1, 3, 2) @ q  # (B, H, S, hd)

        # 5. GQA un-repeat: sum each group's H // G query-head gradients.
        if G != H:
            dk = dk.reshape(B, G, H // G, S, hd).sum(axis=2)  # (B, G, S, hd)
            dv = dv.reshape(B, G, H // G, S, hd).sum(axis=2)  # (B, G, S, hd)

        # 6. RoPE backward (inverse rotations) — permute to RoPE's contract shape.
        #    Feed the *pre-RoPE* head tensors (same shape, clean provenance).
        rope = RoPE()
        q_in = st["q_rope_in"]  # (B, H, S, hd)
        k_in = st["k_rope_in"]  # (B, G, S, hd)
        dq = rope.backward(
            dq.transpose(0, 2, 1, 3), q_in.transpose(0, 2, 1, 3), position, rope_dim=self.rope_dim
        ).transpose(0, 2, 1, 3)  # (B, H, S, hd)
        dk = rope.backward(
            dk.transpose(0, 2, 1, 3), k_in.transpose(0, 2, 1, 3), position, rope_dim=self.rope_dim
        ).transpose(0, 2, 1, 3)  # (B, G, S, hd)

        # 7. Projection gradients and input gradient (linear layer rule).
        dq_flat = dq.transpose(0, 2, 1, 3).reshape(-1, H * hd)  # (T, H*hd)
        dk_flat = dk.transpose(0, 2, 1, 3).reshape(-1, G * hd)  # (T, G*hd)
        dv_flat = dv.transpose(0, 2, 1, 3).reshape(-1, G * hd)  # (T, G*hd)
        x_flat = x.reshape(-1, x.shape[-1])  # (T, D)
        dWq = x_flat.T @ dq_flat  # (D, H*hd)
        dx = dq_flat @ self.q_proj.T  # (T, D)
        dparams: dict[str, np.ndarray] = {"q_proj": dWq, "o_proj": dWo}
        if self.mla_latent_dim > 0:
            # MLA (ADR 0009) backward — chain rule through the low-rank
            # factorization k_pre = c_pre @ W_k_up, c_pre = x @ W_kv_down:
            #   dW_k_up = c_pre^T @ dk_flat        (L, G*hd)
            #   dW_v_up = c_pre^T @ dv_flat        (L, G*hd)
            #   dc_pre  = dk_flat @ W_k_up^T + dv_flat @ W_v_up^T   (T, L)
            #   dW_kv_down = x_flat^T @ dc_pre     (D, L)
            #   dx += dc_pre @ W_kv_down^T         (T, D)
            c_pre = st["c_pre"]  # (B, S, L)
            c_flat = c_pre.reshape(-1, self.mla_latent_dim)  # (T, L)
            dWk_up = c_flat.T @ dk_flat  # (L, G*hd)
            dWv_up = c_flat.T @ dv_flat  # (L, G*hd)
            dc_flat = dk_flat @ self.k_up_proj.T + dv_flat @ self.v_up_proj.T  # (T, L)
            dWkv_down = x_flat.T @ dc_flat  # (D, L)
            dx = dx + dc_flat @ self.kv_down_proj.T  # (T, D)
            dparams["kv_down_proj"] = dWkv_down
            dparams["k_up_proj"] = dWk_up
            dparams["v_up_proj"] = dWv_up
        else:
            dWk = x_flat.T @ dk_flat  # (D, G*hd)
            dWv = x_flat.T @ dv_flat  # (D, G*hd)
            dx = dx + dk_flat @ self.k_proj.T + dv_flat @ self.v_proj.T  # (T, D)
            dparams["k_proj"] = dWk
            dparams["v_proj"] = dWv

        return dx.reshape(x.shape), dparams
