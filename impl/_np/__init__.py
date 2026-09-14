"""NumPy reference implementation of the decoder-only transformer.

One module per operator (each with forward + analytic backward):

- ``embedding``   — token lookup table
- ``layernorm``   — RMSNorm
- ``rope``        — rotary position embedding
- ``ffn``         — SwiGLU feed-forward (dense)
- ``attention``   — multi-head attention (dense + GQA)
- ``moe``         — mixture of experts (router + SwiGLU experts)
- ``block``       — one pre-norm transformer block
- ``stack``       — the N-layer decoder stack
- ``model``       — NumPyModel (embedding + stack + final norm + lm head)
- ``cross_entropy`` — loss with optional shift / padding mask
- ``gradcheck``   — finite-difference gradient checker (test reference)
"""

__all__ = []
