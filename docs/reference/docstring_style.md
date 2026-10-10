# Docstring Style

All docstrings in this project follow the **numpydoc** convention, with one
addition for the math reference: **shape comments** on every matrix
operation.

## Numpydoc sections

Use these sections in this order (omit any that don't apply):

1. **Summary** (one or two sentences) — what the function/class does.
2. **Parameters** — one entry per parameter, `name : type` followed by a
   description.
3. **Returns** — one entry per return value.
4. **Raises** — if the function raises exceptions.
5. **Notes** / **Example** — optional; for math derivations, edge cases, or
   usage examples.

Example:

```python
def forward(self, x: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
    """Multi-head attention forward pass.

    Parameters
    ----------
    x : np.ndarray, shape (B, S, D)
        Input embeddings.
    positions : np.ndarray, shape (S,), optional
        RoPE positions (None → arange(S)).

    Returns
    -------
    np.ndarray, shape (B, S, D)
        Attention output.
    """
```

## NumPy-track class docstrings (extension)

NumPy-track operator classes (the math reference) add three teaching elements
to the numpydoc skeleton — see `impl/_np/rope.py` → `RoPE` for the canonical
example:

1. A bold-marked **Intuition** paragraph (or an equivalent
   formula-with-*why* block) right after the summary — the math stated, then
   one paragraph on why it has the property the model needs.
2. Custom `Forward` / `Backward` sections (numpydoc-style headers with
   `-------` underlines) that give the input/output shapes as short prose
   lines, followed by a "Step shapes:" list — one line per intermediate,
   `name : shape — meaning`.
3. Formula references cite the paper inline in the summary
   (`Su et al., 2021, "RoFormer"`).

```python
class RoPE:
    """Rotary Position Embedding (Su et al., 2021, "RoFormer").

    ...formula block + one intuition paragraph...

    Forward
    -------
    x : (B, S, H, D)   (query or key tensor)

    Step shapes:
        freqs   : (D//2,)         — one theta per pair
        angles  : (B, S, D//2)    — pos * theta per pair
        output  : re-assembled to (B, S, H, D)

    Backward
    --------
    R is orthonormal, so its inverse is its transpose: ...
    """
```

## Shape convention

Every line that performs a matrix operation (projection, matmul, reshape,
transpose, element-wise op that changes the logical shape) carries a
trailing comment showing the input and output shapes:

```python
q = x @ self.q_proj  # (B, S, D) @ (D, H*hd) → (B, S, H*hd)
# (B, S, H*hd) → (B, H, S, hd)
q = q.view(B, S, H, hd).permute(0, 2, 1, 3)
```

The arrow `→` separates input from output. When the line is an
assignment, the comment goes on the same line (trailing) if it fits, or
above the line if it doesn't. When a single expression is split as a
statement (not an assignment), the comment goes above the line.

For the NumPy track (the math reference), the shape comments are **required**
— they are the primary teaching artifact. For the other tracks, they are
**encouraged** on non-obvious operations (GQA repeat, RoPE permute, the
online-softmax rescale) and optional on obvious ones (a plain
`nn.Linear` call).

## When to write a docstring

- Every public class and method (no `_` prefix) **must** have a docstring.
- Private methods (`_` prefix) **should** have a docstring if they encode a
  non-obvious invariant or a math step.
- Test methods **should** have a one-line docstring stating what they
  verify.

## What NOT to write

- Don't restate the type hints (`x : int  # the integer x`).
- Don't write "This function does X" when the name already says it.
- Don't leave a docstring that describes the *old* behavior after a rename.

## Production-optimization notes (`# PROD:`)

Where the teaching implementation deliberately takes the *readable* path
instead of the *production* one, mark the spot with a single-line comment:

```python
# PROD: production uses an online-softmax flash kernel that never materializes
#       the (B, H, S, S) score matrix — see impl/_triton/flash_attn.py
scores = (q @ k.T) / scale  # (B, H, S, S)
```

Rules:

- One line (two only if the second points at the in-repo production
  example). Never a paragraph.
- State what production would do and *why* the demo doesn't (readability,
  step-by-step inspection).
- If the repo contains the production version (flash kernel, KV-cache step
  path, fused SDPA), point at it by path.
- Do not mark a spot that is *already* the production choice.
