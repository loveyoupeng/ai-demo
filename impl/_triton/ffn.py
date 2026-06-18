import torch


def swiglu_ffn(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU Feed-Forward Network with gating.

    Computes: gate * proj @ w2  where
        gate = silu(x @ w1)   — smooth gating signal
        proj = x @ w3         — parallel projection

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
        Input activations.
    w1 : torch.Tensor, shape (D, ff_dim)
        First linear projection weights.
    w3 : torch.Tensor, shape (D, ff_dim)
        Third linear projection weights (gating path).
    w2 : torch.Tensor, shape (ff_dim, D)
        Output projection weights.

    Returns
    -------
    torch.Tensor, shape (..., D)
        Gated feedforward output.

    Notes
    -----
    SwiGLU formula:
      gate = silu(x @ w1)          → (..., ff_dim)
      proj = x @ w3                → (..., ff_dim)
      gated = gate * proj          → (..., ff_dim)
      out = gated @ w2             → (..., D)

    where x is (..., D), w1/w3 are (D, ff_dim), w2 is (ff_dim, D).
    """
    # x:        (..., D)
    # x @ w1:   (..., ff_dim) — projection to inner dim
    # silu:     (..., ff_dim) — element-wise activation
    # x @ w3:   (..., ff_dim) — gating signal
    # gated:    (..., ff_dim) — silu(xW1) * xW3
    # out:      (..., D)       — gated @ W2

    gate = torch.nn.functional.silu(x @ w1)  # (..., ff_dim)
    proj = x @ w3  # (..., ff_dim)
    gated = gate * proj  # (..., ff_dim)
    return gated @ w2  # (..., D)
