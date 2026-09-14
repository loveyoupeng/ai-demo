"""Multi-head attention for the NumPy reference implementation.

Scaled dot-product attention with GQA and RoPE, with the analytic backward.
"""

from __future__ import annotations

import numpy as np

from impl._np.init import xavier_uniform
from impl._np.rope import RoPE


class MultiHeadAttention:
    """Scaled dot-product multi-head attention with Grouped-Query Attention.

    Reference: Vaswani et al. 2017 (§3.2.1/§3.2.2); GQA: Ainslie et al. 2023.

    The model width D is split into H query heads of width hd = D // H.
    Queries get one projection per head; keys and values get one projection
    per *group* (G groups, G <= H). When G == H this is ordinary MHA; when
    G < H it is GQA and each K/V head is shared by H // G query heads — the
    KV cache shrinks by a factor of H // G at inference.

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
    ) -> None:
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_dim = embed_dim // n_heads
        self.rope_dim = rope_dim
        rng = np.random.default_rng(seed)
        hd = self.head_dim
        # (D, H*hd) query projection
        self.q_proj = xavier_uniform(rng, embed_dim, n_heads * hd)
        # (D, G*hd) key / value projections — smaller when G < H (GQA)
        self.k_proj = xavier_uniform(rng, embed_dim, n_groups * hd)
        self.v_proj = xavier_uniform(rng, embed_dim, n_groups * hd)
        # (H*hd, D) output projection (mixes the heads back into one vector)
        self.o_proj = xavier_uniform(rng, n_heads * hd, embed_dim)

    def _forward_state(self, x: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run the forward and return the intermediates the backward needs.

        Returns (out, state) where state holds:
            q_pre, k_pre, v_pre : (B, S, ·) projections before head permute
            q, k, v             : (B, H|G, S, hd) after RoPE / GQA repeat
            attn                : (B, H, S, S) attention weights
            ctx                 : (B, S, H*hd) merged context (pre Wo)
            scale               : float, 1/sqrt(hd)
        """
        batch_size, seq_len, _ = x.shape
        H, G, hd = self.n_heads, self.n_groups, self.head_dim

        # Q, K, V projections: (B, S, D) @ (D, H*hd) → (B, S, H*hd)
        q_pre = x @ self.q_proj  # (B, S, H*hd)
        k_pre = x @ self.k_proj  # (B, S, G*hd)
        v_pre = x @ self.v_proj  # (B, S, G*hd)

        # Split the width into heads: (B, S, H*hd) → (B, H, S, hd)
        q_heads = q_pre.reshape(batch_size, seq_len, H, hd).transpose(0, 2, 1, 3)  # (B, H, S, hd)
        k_heads = k_pre.reshape(batch_size, seq_len, G, hd).transpose(0, 2, 1, 3)  # (B, G, S, hd)
        v = v_pre.reshape(batch_size, seq_len, G, hd).transpose(0, 2, 1, 3)  # (B, G, S, hd)

        # RoPE rotates q and k (position information). RoPE's contract shape
        # is (B, S, H, D), so permute, rotate, permute back.
        q = RoPE().forward(q_heads.transpose(0, 2, 1, 3), positions, rope_dim=self.rope_dim).transpose(0, 2, 1, 3)
        k = RoPE().forward(k_heads.transpose(0, 2, 1, 3), positions, rope_dim=self.rope_dim).transpose(0, 2, 1, 3)

        # GQA: broadcast each K/V group to its H // G query heads.
        # Repeat group g H // G times (consecutively): (B, G, S, hd) → (B, H, S, hd)
        if G != H:
            k = np.repeat(k, H // G, axis=1)  # (B, H, S, hd)
            v = np.repeat(v, H // G, axis=1)  # (B, H, S, hd)

        # Scaled dot-product attention.
        scale = float(np.sqrt(hd))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # (B, H, S, S)

        # Causal mask: position i may attend only to positions j <= i.
        # A lower-triangular mask (broadcast over B and H) sets the strictly
        # upper triangle to -inf; the stable softmax below turns those into
        # exactly-zero attention weight.
        causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)  # (S, S), True where j > i
        scores = np.where(causal, -np.inf, scores)  # (B, H, S, S)

        # Numerically stable softmax over the key axis: subtract the row max
        # before exponentiating so exp() never overflows.
        scores = scores - np.max(scores, axis=-1, keepdims=True)  # (B, H, S, S)
        exp_scores = np.exp(scores)  # (B, H, S, S)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (B, H, S, S)

        # Weighted sum of values: (B, H, S, S) @ (B, H, S, hd) → (B, H, S, hd)
        ctx = attn @ v  # (B, H, S, hd)

        # Merge heads back: (B, H, S, hd) → (B, S, H*hd)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, H * hd)  # (B, S, H*hd)
        out = ctx @ self.o_proj  # (B, S, D)
        state = {
            "q_pre": q_pre,
            "k_pre": k_pre,
            "v_pre": v_pre,
            "q_rope_in": q_heads,
            "k_rope_in": k_heads,
            "q": q,
            "k": k,
            "v": v,
            "attn": attn,
            "ctx": ctx,
            "scale": scale,
        }
        return out, state

    def forward(self, x: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
        """Multi-head attention forward pass.

        x: (B, S, D) → out: (B, S, D).
        positions: (S,) token indices for RoPE; default arange(S).
        """
        if positions is None:
            positions = np.arange(x.shape[1], dtype=np.int32)
        out, _state = self._forward_state(x, positions)
        return out

    def forward_step(self, x: np.ndarray, position: int, cache: dict, quantize: bool = False) -> np.ndarray:
        """Process ONE new token against the cached K/V (per-token inference path).

        x: (B, 1, D) the new token's embedding (a single step).
        position: the absolute token index (for RoPE; 0-based).
        cache: per-layer dict. Two forms:
            Naive (quantize=False): {"k": (B, G, t, hd), "v": (B, G, t, hd)}
                K/V *per group* (GQA keeps the cache small: G heads, not H).
            TurboQuant (quantize=True):
                {"bits_k": (B, H, t, hd) int8, "scales_k": (B, H, t, hd) float,
                 "bits_v": (B, H, t, hd) int8, "scales_v": (B, H, t, hd) float}
                1-bit compressed K/V *per head* (H heads, after GQA repeat).
            The dict is mutated in place: the new token's K/V are appended
            at the correct position before attention runs.

        quantize: if False (default), append the full-precision K/V (naive
            path, exact). If True, 1-bit quantize the new K/V (sign + per-
            channel scale), append the (bits, scale) pair, then dequantize
            the full cached tensor before attention — so the step attends
            against the lossy cache. This is the documented TurboQuant
            compression experiment; it degrades outputs (see the parity-
            budget test) while shrinking the KV memory by ~32x.

        Returns: out (B, 1, D) — the attention output for the new token.

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
        H, G, hd = self.n_heads, self.n_groups, self.head_dim
        positions = np.array([position], dtype=np.int32)

        q_pre = x @ self.q_proj  # (B, 1, H*hd)
        k_pre = x @ self.k_proj  # (B, 1, G*hd)
        v_pre = x @ self.v_proj  # (B, 1, G*hd)

        q_heads = q_pre.reshape(batch_size, 1, H, hd).transpose(0, 2, 1, 3)  # (B, H, 1, hd)
        k_heads = k_pre.reshape(batch_size, 1, G, hd).transpose(0, 2, 1, 3)  # (B, G, 1, hd)
        v = v_pre.reshape(batch_size, 1, G, hd).transpose(0, 2, 1, 3)  # (B, G, 1, hd)

        q = RoPE().forward(q_heads.transpose(0, 2, 1, 3), positions, rope_dim=self.rope_dim).transpose(0, 2, 1, 3)
        k = RoPE().forward(k_heads.transpose(0, 2, 1, 3), positions, rope_dim=self.rope_dim).transpose(0, 2, 1, 3)

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
            # Naive: append the full-precision K/V (per group).
            cache["k"] = np.concatenate([cache["k"], k], axis=2)  # (B, G, t+1, hd)
            cache["v"] = np.concatenate([cache["v"], v], axis=2)  # (B, G, t+1, hd)
            # GQA repeat: broadcast each group to its H // G query heads.
            if G != H:
                k_r = np.repeat(cache["k"], H // G, axis=1)  # (B, H, t+1, hd)
                v_r = np.repeat(cache["v"], H // G, axis=1)
            else:
                k_r, v_r = cache["k"], cache["v"]

        scale = float(np.sqrt(hd))
        scores = (q @ k_r.transpose(0, 1, 3, 2)) / scale  # (B, H, 1, t+1)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (B, H, 1, t+1)

        ctx = attn @ v_r  # (B, H, 1, hd)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(batch_size, 1, H * hd)  # (B, 1, H*hd)
        return ctx @ self.o_proj  # (B, 1, D)

    @staticmethod
    def _quantize_turbo(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """1-bit quantize a K or V tensor (per head, per token).

        x: (B, H, 1, hd) — one token's K or V (after GQA repeat).
        Returns:
            bits: (B, H, 1, hd) int8 — 1 for positive, 0 for non-positive.
            scale: (B, H, 1, hd) float — per-channel mean(|x|) per (B, H).
        Dequantize: bits.astype(float) * scale.
        """
        absolute_values = np.abs(x)  # (B, H, 1, hd)
        scale = np.mean(absolute_values, axis=(-2, -1), keepdims=True)  # (B, H, 1, 1)
        bits = (x > 0).astype(np.int8)  # (B, H, 1, hd)
        return bits, scale

    @staticmethod
    def _dequantize_turbo(bits: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Dequantize 1-bit storage back to float.

        bits: (B, H, t, hd) int8 (0 or 1). scale: (B, H, t, 1) float (one
        scalar per head, broadcast over the head-dim axis).
        Returns: bits.astype(float) * scale → (B, H, t, hd) float.
        """
        return bits.astype(scale.dtype) * scale

    def backward(
        self, dout: np.ndarray, x: np.ndarray, positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Analytic backward (derivation in the class docstring).

        dout: (B, S, D) upstream gradient.
        x: (B, S, D) the forward input (recomputes the intermediates).
        positions: the same RoPE positions used in forward.

        Returns: (dx, dparams) with local names {"q_proj", "k_proj", "v_proj", "o_proj"}.
        """
        if positions is None:
            positions = np.arange(x.shape[1], dtype=np.int32)
        _out, st = self._forward_state(x, positions)
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
            dq.transpose(0, 2, 1, 3), q_in.transpose(0, 2, 1, 3), positions, rope_dim=self.rope_dim
        ).transpose(0, 2, 1, 3)  # (B, H, S, hd)
        dk = rope.backward(
            dk.transpose(0, 2, 1, 3), k_in.transpose(0, 2, 1, 3), positions, rope_dim=self.rope_dim
        ).transpose(0, 2, 1, 3)  # (B, G, S, hd)

        # 7. Projection gradients and input gradient (linear layer rule).
        dq_flat = dq.transpose(0, 2, 1, 3).reshape(-1, H * hd)  # (T, H*hd)
        dk_flat = dk.transpose(0, 2, 1, 3).reshape(-1, G * hd)  # (T, G*hd)
        dv_flat = dv.transpose(0, 2, 1, 3).reshape(-1, G * hd)  # (T, G*hd)
        x_flat = x.reshape(-1, x.shape[-1])  # (T, D)
        dWq = x_flat.T @ dq_flat  # (D, H*hd)
        dWk = x_flat.T @ dk_flat  # (D, G*hd)
        dWv = x_flat.T @ dv_flat  # (D, G*hd)
        dx = dq_flat @ self.q_proj.T + dk_flat @ self.k_proj.T + dv_flat @ self.v_proj.T  # (T, D)

        return dx.reshape(x.shape), {
            "q_proj": dWq,
            "k_proj": dWk,
            "v_proj": dWv,
            "o_proj": dWo,
        }
