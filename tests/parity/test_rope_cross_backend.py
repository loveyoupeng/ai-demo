from __future__ import annotations

import numpy as np
import torch

from model.rope import apply_rope as _apply_rope_np, compute_theta as _compute_theta_np
from model.pytorch.rope import apply_rope as _apply_rope_pt, compute_theta as _compute_theta_pt


def test_pt_rope_matches_np_rope_3d():
    """PyTorch apply_rope output must match NumPy apply_rope for [B, L, D]."""
    np.random.seed(42)
    B, L, D = 2, 8, 8

    x_np = np.random.randn(B, L, D).astype(np.float64)
    x_pt = torch.from_numpy(x_np)

    theta_np = _compute_theta_np(np.arange(L, dtype=np.float64), D)
    theta_pt = _compute_theta_pt(torch.arange(L, dtype=torch.float64), D)

    result_np = _apply_rope_np(x_np, theta_np)
    result_pt = _apply_rope_pt(x_pt, theta_pt.to(torch.float64))

    max_abs_err = np.max(np.abs(result_np - result_pt.detach().numpy()))
    print(f"[4.0] max_abs_err: {max_abs_err:.2e}")
    assert max_abs_err < 1e-5, f"PyTorch apply_rope ≠ NumPy: max_diff={max_abs_err:.2e}"
