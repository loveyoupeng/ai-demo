# Domain Glossary

This project implements a decoder-only transformer in four equivalent tracks
(NumPy, PyTorch, Triton, CUDA) for educational purposes. The glossary below
defines the project's vocabulary. Use these terms consistently in code,
docs, and tests.

## Architecture

- **Decoder-only transformer**: a transformer that generates text
  autoregressively, where each position attends only to previous positions
  (causal mask). No encoder, no cross-attention.

- **Block** (a.k.a. "layer"): one unit of the decoder stack. The standard
  pre-norm layout is `h = x + attn(ln1(x)); out = h + mlp(ln2(h))`. The
  residual stream `x` is added to the attention and FFN outputs; the layer
  norms are applied *before* each sublayer (pre-norm), not after.

- **Pre-norm**: normalization (RMSNorm) applied to the input *before* the
  sublayer (attention or FFN). The alternative is post-norm (normalize the
  output). Pre-norm trains more stably and is the standard in modern LLMs
  (LLaMA, GPT-3).

- **Residual connection**: the additive skip connection `out = x + f(x)`
  that lets gradients flow directly through the block. The "stream" is the
  `x` that bypasses the sublayer.

- **KV cache**: at inference, the K/V tensors from previous positions are
  cached so that each new token only needs to compute its own Q/K/V and
  attend to the cached K/V. This makes generation O(1) per token instead of
  O(T) per token.

- **GQA** (Grouped-Query Attention): the K/V head count G is smaller than the
  query head count H; each K/V head is shared by H//G query heads. Shrinks
  the KV cache by a factor of H//G. G == H is ordinary multi-head attention.

- **RoPE** (Rotary Position Embedding): injects position by rotating pairs of
  head dimensions by an angle proportional to position. Preserves vector
  norms; makes the q·k inner product depend only on relative position.

- **SwiGLU**: the feed-forward activation `SiLU(x @ gate) * (x @ up)`,
  followed by `@ down`. A variant of GLU (Gated Linear Unit) that uses the
  SiLU activation instead of ReLU.

- **MoE** (Mixture of Experts): the FFN is replaced by a router that selects
  top-k experts (each a SwiGLU) per token. The router is a linear layer
  (D → E) followed by softmax + top-k mask + renormalize.

- **Shared expert** (a.k.a. global expert): an always-active expert that
  processes every token without routing — `out = E_shared(x) + Σ_j wⱼ·E_j(x)`.
  Ungated by the router (DeepSeek-V2/V3 style), so it captures common
  features while routed experts specialize. Enabled with `n_shared_experts`;
  0 disables it (pure top-k routing). See `docs/adr/0002-shared-expert-moe.md`.

- **lm_head**: the final linear layer (D → V) that maps the last hidden state
  to logits over the vocabulary. No weight tying with the embedding in this
  project (separate parameters).

## Attention

- **Scaled dot-product attention**: `softmax(QK^T / sqrt(d)) @ V`. The
  `1/sqrt(d)` scaling keeps the score variance at 1 when Q, K entries are
  ~N(0,1), so the softmax does not saturate.

- **Causal mask**: the lower-triangular mask that sets the strictly upper
  triangle of the score matrix to -inf before the softmax. Position i
  attends only to positions j ≤ i. Required for autoregressive generation.

- **SDPA**: `torch.nn.functional.scaled_dot_product_attention` — the
  framework's fused attention that dispatches to flash/efficient kernels on
  GPU. The PyTorch track uses this; the Triton/CUDA tracks use custom
  kernels.

- **Online softmax** (FlashAttention): a single-pass algorithm that computes
  softmax without materializing the full score matrix. Updates a running
  max/sum/accumulator per key block. Peak memory is O(BLOCK_M*D +
  BLOCK_N*D) instead of O(Sk*D). See `impl/_triton/flash_attn.py`.

- **Two-pass attention**: the standard algorithm that (1) computes the full
  score matrix, (2) softmaxes it, (3) multiplies by V. Materializes the
  (BLOCK_M, Sk) score matrix. See `impl/_triton/attn.py`.

## Norms

- **RMSNorm**: `x / sqrt(mean(x^2) + eps) * gamma`. No mean-centering
  (unlike LayerNorm); scales each row to unit root-mean-square, then applies
  the learned gain `gamma`. The `eps` (from `config.norm_eps`) prevents
  division by zero.

- **LayerNorm**: `x / sqrt(var(x) + eps) * gamma + beta`. Mean-centers and
  scales. Not used in this project (RMSNorm is the standard in modern LLMs).

## Training

- **Analytic backward**: the chain rule applied symbolically — each operator
  has a `backward(dout, x) -> (dinput, dparams)` method that computes
  gradients in closed form. The NumPy track implements this; the PyTorch/
  Triton/CUDA tracks use autograd.

- **Gradient check**: compare the analytic backward against finite
  differences to verify correctness. The NumPy track's
  `check_model_gradients` does this (float64, per-parameter worst relative
  error).

- **AdamW**: the optimizer with decoupled weight decay. The NumPy track has
  a hand-rolled implementation (the teaching artifact); the PyTorch track
  uses `torch.optim.AdamW`.

- **Gradient clipping**: scale the gradients so their global L2 norm is at
  most `max_norm`. Prevents exploding gradients.

- **Cross-entropy loss**: the standard language-modeling loss. Compares the
  logits to the target token IDs; ignores `ignore_index` positions.

- **SFT (supervised fine-tuning)**: post-training on instruction-following
  data — the model learns to *answer* the prompt, not just continue it.
  Unlike pre-training (next-token over raw text), SFT data are
  (prompt, response) pairs and the **loss is masked**: only the response
  tokens contribute to the loss and gradient (the prompt positions are
  excluded via ``ignore_index``). Foundation model baseline + a small SFT
  step = a usable coder.

- **Prompt masking**: the mechanism that tells the loss function "these
  tokens are context, not candidates — compute next-token loss only on
  the response side." Implementation: build ``targets`` where prompt
  positions are ``ignore_index`` (=-100) and response positions hold the
  real next-token IDs.

- **Tool calling**: a model output format where the assistant's response
  is not prose, but a JSON `tool_calls` array (function name +
  arguments). The training data carries the tool schema and example
  traces (assistant calls → tool result → final answer) so the model
  learns the JSON shape. Real setup is OpenAI-style messages.

- **BPE tokenizer**: byte-pair-encoding tokenizer — builds a vocabulary
  of sub-word tokens by merging the most frequent byte pairs until the
  target vocab size. The script trains one on the SFT dataset itself so
  the tokenizer merges common code patterns (e.g. ``"def"``, ``"tool_calls"``)
  into single tokens.

## Inference

- **Greedy decoding**: pick the argmax token at each step. Deterministic.

- **Sampled decoding**: sample a token from the (temperature-scaled, top-k
  filtered) softmax distribution. Stochastic; uses a seeded RNG for
  reproducibility.

- **Temperature**: the scalar that scales the logits before softmax.
  T → 0 makes the distribution sharper (greedy); T → ∞ makes it flatter
  (uniform).

- **Top-k filtering**: keep only the k largest logits; set the rest to -inf
  before softmax. Constrains the sampling distribution to the top-k tokens.

- **TurboQuant**: the 1-bit quantized KV cache. Stores K/V as (sign,
  per-channel scale) pairs; dequantizes before attention. Lossy but ~32x
  smaller than the full-precision cache. See
  `impl/_np/turboquant_kv_cache.py` and the parity-budget test.

## Learning Mode

- **Learning mode**: the web visualizer for interactive inference —
  architecture diagram, the actual numbers at every step with
  click-to-inspect detail, and downloadable inference records. Served by the
  standalone entry point `scripts/learning.py` on top of
  `impl/_np/learning_server.py`; backend-agnostic (NumPy default, PyTorch
  via `--backend`), since a record is the same JSON shape from either
  adapter. Off by default; it has no impact on any track.

- **Inference record**: the per-token capture of every forward intermediate — embedding through each block's
  attention/FFN (or MoE) tensors, final norm, logits, and the sampled token — serialized as JSON for the page and
  for download.

- **Instrumented forward**: the overlay that runs the model's own components and recomputes each component's math
  from its public parameters to capture intermediates, without modifying the NumPy track.

- **Demo model**: the pretrained toy model — char-level vocabulary (V=20), D=8, H=4, L=3, E=3 MoE — exported to
  `resource/models/learning_demo/` (checkpoint + `vocab.json`). The learning mode's default model.

- **Entry-point contract**: every per-track `cli.py` (e.g.
  `python -m impl._torch.cli`) is a single-track smoke demo on random
  weights — fast, dependency-minimal, answers "does this track run?".
  Everything checkpoint-based or cross-backend lives in `scripts/`
  (`train.py`, `infer.py`, `learning.py`, `verify_equivalence.py`) with a
  `--backend` selector. New user-facing entry points go in `scripts/`;
  `cli.py` files stay thin demos and never learn checkpoint loading.

- **KV-step interface**: the shared decode contract all four tracks
  satisfy (`make_cache` / `forward_prefill` / `forward_step` on every
  model class). Prefill runs the full prompt once and backfills each
  block's per-group K/V cache from the attention state it already
  computed; each subsequent step appends exactly one token's K/V and
  attends against the cached tensors — O(1) per token instead of O(T).
  The NumPy track's reference implementation is the template the other
  three tracks mirror. The plain `forward` stays as the parity oracle.

- **Shared generator** (`shared/generator.py`): the single deep
  `TextGenerator` implementation behind the KV-step interface —
  sampling math (temperature, top-k, entropy) lives inside. Used by the
  torch/triton/cuda tracks; the NumPy track keeps its own teaching
  generator (`impl/_np/inference.py`) alongside it, so there's no seam
  between "the reference" and "the production path" at generation time.

- **Binding** (registry): the mapping from each registry key to the
  owning storage object in a track's model (`impl/_np.model.NumPyModel.
  _param_arrays`, `impl/_torch.layers.TorchModel._param_tensors`, etc.).
  `ParameterRegistry.bind(binding)` materializes it as the full
  key→param dict in entry order and is what every save/load path walks;
  a drifted binding (stale/extra keys on a track) fails there, not at
  save-load time.

## Tracks

- **NumPy track** (`impl/_np/`): the math reference. Hand-rolled
  implementations of every operator (forward + analytic backward), with
  shape comments and formula references. The teaching artifact.

- **PyTorch track** (`impl/_torch/`): the production reference. Uses
  `nn.Module`, autograd, `F.scaled_dot_product_attention`, and
  `torch.optim`. The idiom reference.

- **Triton track** (`impl/_triton/`): the kernel reference. A PyTorch model
  with the attention core and FFN activation on Triton kernels. Depends on
  the PyTorch track for the model skeleton (see
  `docs/seam_triton_to_torch.md`).

- **CUDA track** (`impl/_cuda/`): the bare-metal reference. Custom CUDA
  kernels for RMSNorm, RoPE, attention, and FFN, launched via NVRTC. The
  lowest-level track.

## Cross-cutting

- **Keys scheme**: the checkpoint key convention
  (`shared.constants.Keys`). One flat dict per model, keyed by
  HF-Llama-style names (`model.layers.{i}.self_attn.q_proj.weight`, etc.).
  All four tracks use the same scheme; the parameter registry
  (`shared/registry.py`) is the single owner of the checkpoint format.

- **Parity**: the property that all four tracks produce (approximately)
  identical outputs for the same input and weights. The cross-backend tests
  verify this at three tolerance tiers (see AGENTS.md rule 2).

- **Shared seam**: the documented one-way dependency from the Triton track
  to the PyTorch track (RoPE). See `docs/seam_triton_to_torch.md`.
