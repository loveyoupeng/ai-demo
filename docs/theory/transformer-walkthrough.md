# The Forward Pass, Stage by Stage

A guided tour of one forward pass through this repo's decoder-only transformer —
the same math the [learning page](../../impl/_np/web/index.html) renders interactively.
Each stage gives the equation (KaTeX), the *why*, and the code that implements it.
Shapes use the conventions `B` batch, `S` sequence, `D` embed dim, `H` query heads,
`G` KV groups, `hd = D / H` head dim, `V` vocab size.

All pointers are in the NumPy track (`impl/_np/`) — the reference implementation with
hand-rolled math and analytic backward. The PyTorch/Triton/CUDA tracks compute the
same thing with production ops (`F.scaled_dot_product_attention`, fused kernels).

```text
tokens → Embedding → [ RMSNorm → Attention → + → RMSNorm → FFN/MoE → + ] × L
       → final RMSNorm → lm_head → logits → (softmax → next token)
```

---

## 1. Embedding — text becomes vectors

Each token id selects a row of the embedding matrix:

$$E \in \mathbb{R}^{V \times D}, \qquad x = E[\text{ids}] \in \mathbb{R}^{B \times S \times D}$$

No positional information yet — position enters via RoPE inside attention (§4).
The embedding table is the only place the vocab size appears on the input side;
the *output* side has its own projection, `lm_head` (§7).

- Code: `impl/_np/embedding.py` → `Embedding`
- Why a lookup (not a learned function of the id)? Tokens are discrete symbols;
  a learned per-token vector is the standard, parameter-efficient encoding.

## 2. The block — pre-norm with a plain additive residual

Each of the `L` layers is:

$$h = x + \underbrace{\mathrm{Attn}\big(\mathrm{RMSNorm}(x)\big)}_{\text{attention sublayer}},
\qquad
\mathrm{out} = h + \underbrace{\mathrm{FFN}\big(\mathrm{RMSNorm}(h)\big)}_{\text{feed-forward sublayer}}$$

The residual stream `x` passes through **unchanged**; each sublayer sees a normalized
copy and contributes an *update*. This pre-norm + additive residual is exactly the
LLaMA block.

> This repo originally used a non-standard "gated residual" (a learned scalar
> multiplying the stream). It was abandoned for the standard form —
> see [ADR-0001](../adr/0001-gated-residual-abandonment.md). The standard form is
> what every paper and reference implementation documents, so the math here is
> web-searchable.

- Code: `impl/_np/block.py` → `TransformerBlock`
- Stack of `L` blocks: `impl/_np/stack.py` → `DecoderStack`

## 3. RMSNorm — scale without centering

Root-mean-square normalization normalizes each row by its RMS, then scales by a
learned per-dimension vector `γ`:

$$\mathrm{rms}(x) = \sqrt{\tfrac{1}{D}\textstyle\sum_{d} x_d^{2} + \varepsilon},
\qquad
y = \frac{x}{\mathrm{rms}(x)} \cdot \gamma$$

Compared to LayerNorm there is **no mean subtraction** (no bias/β) — fewer ops, no
mean bookkeeping, and (empirically) equivalent quality. This is the normalization
in LLaMA and most modern LLMs.

- Code: `impl/_np/layernorm.py` → `RMSNorm`
- Paper: Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019), arXiv:1910.07467.

## 4. Attention — where tokens talk

### 4.1 Q, K, V projections

Three linear maps (no bias) turn the normalized input into queries, keys and values:

$$q = x W_q \in \mathbb{R}^{B \times S \times H \cdot hd}, \quad
k = x W_k, \quad v = x W_v$$

Then the width is split into heads: `q` into `H` heads, `k`/`v` into `G` groups.

- Code: `impl/_np/attention.py` → `MultiHeadAttention` (projections + head split)

### 4.2 RoPE — position by rotation

Rotary Position Embedding rotates each pair of head dimensions by an angle
proportional to the token's position `pos` and the pair index `m`:

$$\theta_m = 10000^{-2m / d}, \qquad
\begin{pmatrix} y_{2m} \\ y_{2m+1} \end{pmatrix}
=
\begin{pmatrix} \cos \theta_m \, pos & -\sin \theta_m \, pos \\
                \sin \theta_m \, pos & \cos \theta_m \, pos \end{pmatrix}
\begin{pmatrix} x_{2m} \\ x_{2m+1} \end{pmatrix}$$

RoPE is applied to `q` and `k` *before* scoring. Because a rotation's angle
difference depends only on the *relative* position, the dot product
`q_i · k_j` encodes how far apart `i` and `j` are — absolute positions never
appear as extra vectors (unlike the learned position embeddings in Vaswani §3.2.3).

- Code: `impl/_np/rope.py` → `RoPE` (contract: `(B, S, H, d)` in and out)
- Paper: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021), arXiv:2104.09864.

### 4.3 GQA — fewer KV heads than query heads

In grouped-query attention the `H` query heads share `G < H` key/value groups —
head `h` uses group `h // (H // G)`. The KV cache shrinks by `H / G` (a large
memory saving at inference) with negligible quality loss, and it generalizes both
multi-head (`G = H`) and multi-query (`G = 1`) attention.

- Code: `impl/_np/attention.py` → `np.repeat(k, H // G, axis=1)`
- Paper: Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023), arXiv:2305.13245.

### 4.4 Scaled dot-product + causal mask

Scores are the rotated dot products, divided by `sqrt(hd)` (so the softmax stays
in a well-conditioned range as `hd` grows); the causal mask zeroes every future
position `j > i`, so a token can only attend to its past:

$$\mathrm{attn}_{i,:} = \mathrm{softmax}\!\left(\frac{q_i k_{\le i}^{T}}{\sqrt{hd}}\right),
\qquad
\mathrm{ctx}_i = \mathrm{attn}_{i,:} \, v_{\le i}$$

The context is projected back to width `D`: `out = ctx W_o`.

- Code: `impl/_np/attention.py` → `MultiHeadAttention` (scores, mask, stable softmax, `ctx @ W_o`)
- Paper: Vaswani et al., *Attention Is All You Need* (2017) §3.2.1 (attention), §3.2.2 (multi-head), §3.2.3 (position, replaced here by RoPE).
- PROD note: the NumPy path materializes the full `(B, H, S, S)` score matrix.
  Production uses a flash-attention kernel that never materializes it —
  `impl/_triton/flash_attn.py` → `flash_attention` (Dao et al. 2022, arXiv:2205.14135).

## 5. FFN — where the "knowledge" sits

The dense feed-forward layer is a **SwiGLU**: two parallel projections `W_gate`,
`W_up` into a wider hidden space (`FF`), one of them passed through the
`SiLU` (swish) gate, element-wise product, then `W_down` back to `D`:

$$\mathrm{SwiGLU}(h) = \big(\mathrm{SiLU}(h W_{\text{gate}}) \odot (h W_{\text{up}})\big) W_{\text{down}},
\qquad \mathrm{SiLU}(z) = z \cdot \sigma(z)$$

The gate lets the network *modulate* the message instead of just transforming it.
In the MoE configuration the same block is replicated as `E` experts and a
**router** picks which experts compute per token (§6).

- Code: `impl/_np/ffn.py` → `SwiGLUFFN`, `silu`
- Paper: Dai et al., *GLU Variants Improve Transformer* (2019), arXiv:1910.07487 (§SwiGLU);
  Vaswani 2017 §3.3 (the plain FFN it refines).

## 6. MoE — capacity without the compute

A router (one linear layer) scores each expert per token, softmaxes over all `E`
experts, keeps only the top-`k` (renormalized so the weights still sum to 1), and
adds the weighted expert outputs:

$$\mathrm{score} = h W_r, \quad p = \mathrm{softmax}(\mathrm{score}), \quad
w_i = \frac{p_i}{\sum_{j \in \mathrm{top}\text{-}k} p_j} \cdot \mathbb{1}[i \in \mathrm{top}\text{-}k],
\qquad
\mathrm{out} = \textstyle\sum_i w_i \, \mathrm{Expert}_i(h)$$

Total parameters grow with `E`, but each token only pays for `k` experts —
decoupling model *capacity* from per-token *FLOPs*.

- Code: `impl/_np/moe.py` → `MixtureOfExperts` (router, top-k selection, weighted sum)
- PROD note: the reference computes **all** experts and masks with zeros;
  production gathers tokens per expert and runs only the selected top-k.
- Paper: Fedus et al., *Switch Transformers* (2021), arXiv:2101.03961; *Mixtral of Experts* (2023), arXiv:2401.04624.

## 7. LM head + softmax — back to the vocab

A final RMSNorm, then a linear projection to the vocabulary:

$$\mathrm{logits} = \mathrm{RMSNorm}(\mathrm{stack\_out}) \, W_{\text{lm}} \in \mathbb{R}^{B \times S \times V}$$

`softmax(logits)` is the model's distribution over the next token at each position;
during training the same logits feed the cross-entropy loss (Vaswani 2017 §3.5 —
the loss is computed over *all* positions, each predicting its successor).

- Code: `impl/_np/model.py` → `NumPyModel` (final norm + `lm_head`),
  `impl/_np/cross_entropy.py` → `CrossEntropyLoss`

## 8. Inference — KV cache, prefill, decode

Generation is autoregressive: at step `t` we need the logits at the *last* position,
which needs the full K/V of every past position. The **KV cache** stores those so
each step recomputes only the new token's K/V (one row appended per block):

- **Prefill** — one full forward over the prompt fills the cache and produces the
  first logits (the prompt's positions are absolute: `position_offset`).
- **Decode** — per step, embed only the new token, append its K/V, attend to the
  whole cache. No causal mask is needed: every cached row is in the past.

- Code: `impl/_np/kv_cache.py` → `NaiveKVCache`; `impl/_np/model.py` →
  `forward_prefill` / `forward_step`; sampling in `impl/_np/inference.py` → `TextGenerator`.
- Note: with a finite context window, the reference keeps the *full history* in the
  cache and only the last `context_length` tokens in the softmax window at prefill;
  the learning page documents this window → full-history semantics explicitly.

## Paper → repo map

| Concept | Paper (section) | Code |
| --- | --- | --- |
| Scaled dot-product attention | Vaswani 2017 §3.2.1 | `impl/_np/attention.py` |
| Multi-head attention | Vaswani 2017 §3.2.2 | `impl/_np/attention.py` (head split) |
| Position (learned) | Vaswani 2017 §3.2.3 | *replaced by RoPE* — `impl/_np/rope.py` |
| Feed-forward block | Vaswani 2017 §3.3 | `impl/_np/ffn.py` (as SwiGLU) |
| Cross-entropy training | Vaswani 2017 §3.5 | `impl/_np/cross_entropy.py` |
| RMSNorm | Zhang & Sennrich 2019 | `impl/_np/layernorm.py` |
| SwiGLU | Dai et al. 2019 | `impl/_np/ffn.py` |
| RoPE | Su et al. 2021 | `impl/_np/rope.py` |
| GQA | Ainslie et al. 2023 | `impl/_np/attention.py` (group repeat) |
| Switch-style MoE | Fedus et al. 2022 / Mixtral 2023 | `impl/_np/moe.py` |
| Flash attention (PROD) | Dao et al. 2022 | `impl/_triton/flash_attn.py` |

## Design decisions in this repo

- **Standard additive residual** (not gated) — [ADR-0001](../adr/0001-gated-residual-abandonment.md).
- **Four equivalent tracks** (NumPy reference, PyTorch, Triton, CUDA) share one
  `TransformerConfig` and one flat checkpoint key scheme (`shared/registry.py`),
  so any track can load any checkpoint — see [design.md](../design.md).
- **Records live in the track.** Learning mode (`impl/_np/learning.py`,
  `impl/_torch/learning.py`) does not re-implement the model: it calls each
  track's own `_forward_state` hooks and serializes the intermediates those paths
  already compute, so the page's numbers are the track's actual forward pass.
