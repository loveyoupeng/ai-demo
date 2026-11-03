# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Gated DeltaNet for the NumPy reference implementation (ADR 0006).

A **linear attention** layer: instead of scoring every past token against the
query (full attention's O(S²) score matrix), DeltaNet keeps ONE recurrent
state matrix per head,

    S_t = (1 - β_t) · S_{t-1} (I - k_t k_tᵀ)  +  β_t · k_t v_tᵀ
    o_t = S_t q_t

with L2-normalized q_t, k_t per head. Reading S_t q_t is O(hd²) — no token ever
"looks back": each token only *writes* into S. The delta rule subtracts the
key's old value before adding the new one (k_t k_tᵀ normalizes away the key's
own contribution), so the state remembers values, not keys.

**Gated** (Yang et al., "Gated DeltaNet", 2024/2025): the forget scalar is
per-head and input-dependent,

    β_t = sigmoid(a · x_t + b)          a = a_proj(x), b = dt_bias

and an output gate modulates the read-out,

    out_t = α_t ⊙ RMSNorm(o_t)          α_t = sigmoid(gate_proj(x_t))

then ``out = out @ o_proj`` mixes heads back to width D.

Forward paths (mirroring ``MultiHeadAttention``):
    forward_state(x, position, record)  — full sequence, captures state
    forward(x, position)                — plain path (no cache)
    forward_step(x, position, cache)    — ONE new token against the recurrent S
    forward_chunk(x, position, cache)   — a c-token chunk; the scan is
        SEQUENTIAL within the chunk (S is a function of all preceding tokens),
        so chunk verification loses the intra-chunk parallelism attention
        has — but stays exactly equal to the full forward (chunk parity).

Cache contract: a DeltaNet mixer's cache dict is
    {"s": (B, H, hd, hd),               # the recurrent state
     "s_hist": list of (B, H, hd, hd)}  # per-position snapshots for rollback
Attention mixers keep the {"k","v"} dict — ``make_cache`` picks the kind per
mixer from ``config.mixer_kind(m)``.

Backward: backpropagation through time over the recurrence (reverse scan);
stateless — every intermediate is recomputed from x (same convention as the
rest of the NumPy track), verified against finite differences.
"""

from __future__ import annotations

import numpy as np

from impl._np.init import xavier_uniform
from impl._np.layernorm import RMSNorm
from shared.constants import DeltaNet, Keys
from shared.result import Result


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize along ``axis`` (the DeltaNet q/k normalization)."""
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / norm


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class GatedDeltaNet:
    """Gated DeltaNet mixing sublayer (linear attention, recurrent state S).

    Shape letters: B = batch, S = sequence length this pass, D = embed_dim,
    H = n_heads, hd = head_dim, t = cached position count, c = chunk length.

    Forward (per head h, position t):
        q_t, k_t, v_t = x_t @ Wq/Wk/Wv  → heads (B, H, S, hd)
        q̂, k̂ = L2norm(q), L2norm(k)                (no RoPE — linear layers
                                                     have no absolute position)
        β_t = sigmoid(a_t + dt_bias)     a_t = x @ a_proj → (B, H, S)
        S_t = (1-β_t)·S_{t-1}·(I - k̂_t k̂_tᵀ) + β_t·k̂_t v_tᵀ
        o_t = S_t q̂_t                                   (B, H, S, hd)
        α_t = sigmoid(x @ gate_proj)                    (B, S, D)
        out = (α ⊙ RMSNorm(o merged to D)) @ o_proj

    Weights (all (in, out) layout, checkpoint keys under ``Keys.deltanet``):
        q_proj, k_proj, v_proj: (D, H*hd)   o_proj: (H*hd, D)
        a_proj: (D, H)          dt_bias: (H,)
        gate_proj: (D, D)       norm: RMSNorm gamma (D,)
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        seed: int = 0,
        layer_index: int | None = None,
        mixer_index: int | None = None,
        params=None,
    ) -> None:
        """Create the projections + gates, injecting this mixer's checkpoint slice.

        layer_index/mixer_index/params: the block's position, this mixing
        sublayer's index within the block, and the flat registry-keyed
        checkpoint; when all are given every tensor is born with its trained
        array (read from ``Keys.deltanet(layer_index, mixer_index, ...)``),
        copied in place over the seed-initialized storage. None (default)
        keeps the seeded init (the training path).
        """
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        rng = np.random.default_rng(seed)
        D, H, hd = embed_dim, n_heads, self.head_dim
        self.q_proj = xavier_uniform(rng, D, H * hd)
        self.k_proj = xavier_uniform(rng, D, H * hd)
        self.v_proj = xavier_uniform(rng, D, H * hd)
        self.o_proj = xavier_uniform(rng, H * hd, D)
        self.a_proj = xavier_uniform(rng, D, H)
        self.dt_bias = np.zeros(H, dtype=np.float32)
        self.gate_proj = xavier_uniform(rng, D, D)
        self.norm = RMSNorm(D)
        if params is not None and layer_index is not None and mixer_index is not None:
            i, m = layer_index, mixer_index
            self.q_proj[...] = params[Keys.deltanet(i, m, DeltaNet.Q_PROJ)]
            self.k_proj[...] = params[Keys.deltanet(i, m, DeltaNet.K_PROJ)]
            self.v_proj[...] = params[Keys.deltanet(i, m, DeltaNet.V_PROJ)]
            self.o_proj[...] = params[Keys.deltanet(i, m, DeltaNet.O_PROJ)]
            self.a_proj[...] = params[Keys.deltanet(i, m, DeltaNet.A_PROJ)]
            self.dt_bias[...] = params[Keys.deltanet(i, m, DeltaNet.DT_BIAS)]
            self.gate_proj[...] = params[Keys.deltanet(i, m, DeltaNet.GATE_PROJ)]
            self.norm.gamma[...] = params[Keys.deltanet(i, m, DeltaNet.NORM)]

    # ------------------------------------------------------------------ core
    def _scan(
        self,
        x: np.ndarray,
        s_init: np.ndarray,
        record: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Run the recurrence over S positions starting from state ``s_init``.

        x: (B, S, D). s_init: (B, H, hd, hd).
        Returns (out (B, S, D), s_final (B, H, hd, hd), intermediates).
        """
        B, S, D = x.shape
        H, hd = self.n_heads, self.head_dim

        q = (x @ self.q_proj).reshape(B, S, H, hd).transpose(0, 2, 1, 3)  # (B,H,S,hd)
        k = (x @ self.k_proj).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        v = (x @ self.v_proj).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        qn = _l2_normalize(q, axis=-1)
        kn = _l2_normalize(k, axis=-1)

        a = x @ self.a_proj  # (B, S, H)
        beta = _sigmoid(a.transpose(0, 2, 1)[..., None] + self.dt_bias[None, :, None, None])  # (B,H,S,1)
        alpha = _sigmoid(x @ self.gate_proj)  # (B, S, D)

        s = s_init.copy()  # (B,H,hd,hd)
        outs: list[np.ndarray] = []
        betas: list[np.ndarray] = []
        ss: list[np.ndarray] = []
        for t in range(S):
            kt = kn[:, :, t, :]  # (B,H,hd)
            vt = v[:, :, t, :]  # (B,H,hd)
            qt = qn[:, :, t, :]
            bt = beta[:, :, t, :, None]  # (B,H,1,1)
            kk = kt[..., None] * kt[:, :, None, :]  # (B,H,hd,hd)
            kv = kt[..., None] * vt[:, :, None, :]  # (B,H,hd,hd)
            # S = (1-b) S (I - k kᵀ) + b k vᵀ
            s = (1.0 - bt) * (s - s @ kk) + bt * kv
            # o_t = S_t q_t
            o = np.einsum("bhij,bhj->bhi", s, qt)  # (B,H,hd)
            outs.append(o)
            betas.append(bt[..., 0])  # store (B,H,1); reassembled as (B,H,S,1)
            ss.append(s.copy())
        o_all = np.stack(outs, axis=2)  # (B,H,S,hd)
        beta_all = np.stack(betas, axis=2)  # (B,H,S,1)
        s_hist = np.stack(ss, axis=2)  # (B,H,S,hd,hd)

        # merge heads → (B,S,H*hd) → D, then output gate + norm
        o_merged = o_all.transpose(0, 2, 1, 3).reshape(B, S, H * hd)  # (B,S,H*hd)
        pre_o = o_merged @ self.o_proj  # (B,S,D)
        normed = self.norm.forward(pre_o)  # (B,S,D)
        out = alpha * normed  # (B,S,D)

        inter = {
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "qn": qn,
            "kn": kn,
            "beta": beta_all,  # (B,H,S,1)
            "alpha": alpha,  # (B,S,D)
            "o": o_all,  # (B,H,S,hd)
            "pre_o": pre_o,  # (B,S,D) before norm+gate
            "s_hist": s_hist,  # (B,H,S,hd,hd) state after each position
            "s_init": s_init,
            "s_final": s,
        }
        if record is not None:
            record.update(inter)
        return out, s, inter

    # ------------------------------------------------------------- forward
    def forward_state(self, x: np.ndarray, position: int = 0, record: dict | None = None) -> Result[np.ndarray, dict]:
        """Full-sequence forward from a ZERO state (fresh sequence / prefill pass).

        x: (B, S, D) → out (B, S, D). position is unused (no RoPE) but kept
        for interface parity with attention. The returned ``.ctx.display``
        carries the state the cache backfill needs (``s_final``, ``s_hist``).
        """
        B = x.shape[0]
        H, hd = self.n_heads, self.head_dim
        s0 = np.zeros((B, H, hd, hd), dtype=x.dtype)
        out, s_final, inter = self._scan(x, s0, record=record)
        return Result.capture(
            out,
            backward=inter,
            display={"s_final": s_final, "s_hist": inter["s_hist"]},
        )

    def forward(self, x: np.ndarray, position: int = 0) -> np.ndarray:
        """Plain full-sequence forward (no cache). x: (B, S, D) → (B, S, D)."""
        return self.forward_state(x, position).unwrap()

    # ------------------------------------------------- cached paths (step/chunk)
    def forward_step(self, x: np.ndarray, position: int, cache: dict, record: dict | None = None) -> np.ndarray:
        """Process ONE new token against the recurrent state.

        x: (B, 1, D). cache: this layer's dict {"s": (B,H,hd,hd), "s_hist": [...]},
        mutated in place: s becomes the state after this token, and the new
        snapshot is appended to s_hist (so spec rollback can restore it).
        Returns out (B, 1, D). Contract: equal to the matching slice of
        ``forward`` over the full sequence.
        """
        s = cache["s"]
        out, s_new, inter = self._scan(x, s, record=record)
        cache["s"] = s_new
        cache["s_hist"].append(s_new.copy())
        if record is not None:
            record["s_cache"] = s_new
        return out

    def forward_chunk(self, x: np.ndarray, position: int, cache: dict, record: dict | None = None) -> np.ndarray:
        """Process a CHUNK of c tokens (verification path). Sequential scan.

        x: (B, c, D). cache mutated in place (s + one snapshot per chunk
        position). Returns out (B, c, D). Chunk parity: equal to the matching
        c slices of ``forward`` over the whole sequence.
        """
        s = cache["s"]
        out, s_new, inter = self._scan(x, s, record=record)
        cache["s"] = s_new
        for j in range(x.shape[1]):
            cache["s_hist"].append(inter["s_hist"][:, :, j])
        if record is not None:
            record["s_cache"] = s_new
        return out

    # ------------------------------------------------------------- backward
    def backward(self, dout: np.ndarray, x: np.ndarray, position: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Analytic backward (BPTT through the recurrence).

        Recomputes the forward intermediates from ``x`` (stateless), then runs
        the reverse-time recurrence for dS alongside the per-position
        parameter gradients. Verified against finite differences.

        Returns (dx (B,S,D), dparams keyed by DeltaNet attr names).
        """
        B, S, D = x.shape
        H, hd = self.n_heads, self.head_dim
        s0 = np.zeros((B, H, hd, hd), dtype=x.dtype)
        _, _, st = self._scan(x, s0)

        q, k, v = st["q"], st["k"], st["v"]
        qn, kn = st["qn"], st["kn"]
        beta = st["beta"]  # (B,H,S,1)
        alpha = st["alpha"]  # (B,S,D)
        s_hist = st["s_hist"]  # (B,H,S,hd,hd) state AFTER each position
        pre_o = st["pre_o"]

        # 1. output: out = alpha * norm(pre_o);  pre_o = o_merged @ o_proj
        #    recompute norm(pre_o) once, share it between both branches.
        normed = self.norm.forward(pre_o)  # (B,S,D)
        d_alpha = dout * normed  # (B,S,D)
        d_normed = dout * alpha
        d_pre_o, d_norm_gamma = self.norm.backward(d_normed, pre_o)  # (B,S,D), (D,)

        # alpha = sigmoid(x @ gate_proj)
        dgate_pre = d_alpha * alpha * (1.0 - alpha)  # (B,S,D)
        d_gate_proj = x.reshape(-1, D).T @ dgate_pre.reshape(-1, D)  # (D,D)

        # 2. pre_o = o_merged @ o_proj
        d_o_merged = d_pre_o @ self.o_proj.T  # (B,S,H*hd)
        d_o_proj = pre_o.reshape(-1, D).T @ d_pre_o.reshape(-1, D)  # WRONG shape; fix below
        d_o_proj = st["o"].transpose(0, 2, 1, 3).reshape(-1, H * hd).T @ d_pre_o.reshape(-1, D)  # (H*hd, D)
        d_o = d_o_merged.reshape(B, S, H, hd).transpose(0, 2, 1, 3)  # (B,H,S,hd)

        # 3. BPTT: o_t = S_t q_t → dS_t = d_o_t q_tᵀ ; dq_t = S_tᵀ d_o_t
        #    recurrence: S_t = (1-b_t) S_{t-1} (I - k_t k_tᵀ) + b_t k_t v_tᵀ
        dq = np.zeros_like(qn)
        dk = np.zeros_like(kn)
        dv = np.zeros_like(v)
        dbeta = np.zeros((B, H, S, 1))
        ds_next = np.zeros((B, H, hd, hd))  # dS flowing from the future
        for t in range(S - 1, -1, -1):
            s_t = s_hist[:, :, t]  # state after position t
            s_prev = s_hist[:, :, t - 1] if t > 0 else s0
            kt = kn[:, :, t, :]
            vt = v[:, :, t, :]
            bt = beta[:, :, t, :, None]  # (B,H,1,1)

            # dS_t from the read-out plus from the future
            ds = ds_next + np.einsum("bhi,bhj->bhij", d_o[:, :, t, :], qn[:, :, t, :])
            # dq_t = S_tᵀ d_o_t
            dq[:, :, t, :] = np.einsum("bhji,bhj->bhi", s_t, d_o[:, :, t, :])
            # backward through S_t = (1-b) S_prev (I - k kᵀ) + b k vᵀ
            #   dS_prev = (1-b) dS_t (I - k kᵀ)
            kk = kt[..., None] * kt[:, :, None, :]
            ds_prev = (1.0 - bt) * (ds - ds @ kk)
            #   dk from the (I - k kᵀ) term:  dL/dk = -2 (1-b) (dS_tᵀ S_prev) k ... plus b v term
            #   full: S_t = (1-b) S_prev - (1-b) S_prev k kᵀ + b k vᵀ
            #   d/dk: -(1-b)(S_prevᵀ dS + (S_prevᵀ dS)ᵀ) k  + b dSᵀ v
            # dL/dk from S_t = (1-b) S_prev - (1-b) S_prev k kᵀ + b k vᵀ:
            #   term2 grad = -(1-b)(S_prevᵀ dS + dSᵀ S_prev) k
            #   term3 grad =  b dS v   (since tr(dSᵀ du vᵀ) = vᵀ dSᵀ du → dS v)
            b1 = bt[..., 0]  # (B,H,1)
            m1 = np.einsum("bhji,bhjk->bhik", s_prev, ds)  # S_prevᵀ dS  (bhji = transpose of bhij)
            dkt = -(1.0 - b1) * np.einsum("bhij,bhj->bhi", m1 + m1.transpose(0, 1, 3, 2), kt) + b1 * np.einsum(
                "bhij,bhj->bhi", ds, vt
            )
            dk[:, :, t, :] = dkt
            #   dv: dL/dv = b dSᵀ k
            dv[:, :, t, :] = b1 * np.einsum("bhji,bhj->bhi", ds, kt)  # dSᵀ k
            #   dbeta: dS/db = -S_prev (I - k kᵀ) + k vᵀ ; contract with dS
            dbeta[:, :, t, 0] = np.sum(
                ds * (-(s_prev - s_prev @ kk) + kt[..., None] * vt[:, :, None, :]), axis=(-2, -1)
            )
            ds_next = ds_prev

        # 4. L2-norm backward: qn = q / ||q||
        def _l2norm_backward(dn, x):
            n = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + 1e-8)
            return (dn - x * np.sum(dn * x, axis=-1, keepdims=True) / (n**2)) / n

        dq_raw = _l2norm_backward(dq, q)
        dk_raw = _l2norm_backward(dk, k)

        # 5. beta = sigmoid(a + dt_bias); a = x @ a_proj
        dbeta_pre = dbeta * beta * (1.0 - beta)  # (B,H,S,1)
        d_a = dbeta_pre[:, :, :, 0].transpose(0, 2, 1)  # (B,S,H)
        d_a_proj = x.reshape(-1, D).T @ d_a.reshape(-1, H)  # (D,H)
        d_dt_bias = dbeta_pre.sum(axis=(0, 2)).reshape(H)  # (H,) — dbeta_pre is (B,H,S,1)

        # 6. projections: q/k/v = x @ W
        x_flat = x.reshape(-1, D)
        d_q_pre = dq_raw.transpose(0, 2, 1, 3).reshape(-1, H * hd)
        d_k_pre = dk_raw.transpose(0, 2, 1, 3).reshape(-1, H * hd)
        d_v_pre = dv.transpose(0, 2, 1, 3).reshape(-1, H * hd)
        d_q_proj = x_flat.T @ d_q_pre
        d_k_proj = x_flat.T @ d_k_pre
        d_v_proj = x_flat.T @ d_v_pre
        dx = (d_q_pre @ self.q_proj.T + d_k_pre @ self.k_proj.T + d_v_pre @ self.v_proj.T).reshape(B, S, D)
        dx = dx + (d_a.reshape(-1, H) @ self.a_proj.T).reshape(B, S, D)
        dx = dx + (dgate_pre.reshape(-1, D) @ self.gate_proj.T).reshape(B, S, D)

        return dx, {
            "q_proj": d_q_proj,
            "k_proj": d_k_proj,
            "v_proj": d_v_proj,
            "o_proj": d_o_proj,
            "a_proj": d_a_proj,
            "dt_bias": d_dt_bias,
            "gate_proj": d_gate_proj,
            "norm": d_norm_gamma,
        }
