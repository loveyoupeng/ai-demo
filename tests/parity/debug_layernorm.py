from __future__ import annotations

import numpy as np
import torch
import pytest
from model.numpy.layers import NumPyLayerNorm
from model.pytorch.layers import PyTorchLayerNorm


def to_np(val):
    """Convert torch tensor to numpy, leave numpy arrays alone."""
    if isinstance(val, torch.Tensor):
        return val.cpu().detach().numpy()
    return val


def pretty(val, width=14):
    """Format a numpy array value nicely for comparison."""
    val = to_np(val)
    if val.ndim == 0:
        return f"{val.item():>{width}.10f}"
    if val.size == 1:
        return f"{val.item():>{width}.6f}"
    first = val.reshape(-1)[0]
    last = val.reshape(-1)[-1]
    return f"[{first:.10f} ... {last:.10f}]"


def debug_forward(numpy_ln: NumPyLayerNorm, pytorch_ln: PyTorchLayerNorm, x: np.ndarray):
    """Run forward pass for both and compare all intermediates."""
    print(f"\n{'='*70}")
    print(f"FORWARD PASS COMPARISON")
    print(f"  input shape: {x.shape}, dtype: {x.dtype}")
    print(f"  eps: numpy={numpy_ln.eps}, pytorch={pytorch_ln.eps}")
    print(f"{'='*70}")

    numpy_ln.forward(x)
    pytorch_ln.forward(torch.from_numpy(x))

    for attr, pt_attr_name in [("mean", "x_mean"), ("var", "x_var"), ("x_norm", "x_norm")]:
        np_attr = getattr(numpy_ln, attr)
        pt_attr = getattr(pytorch_ln, pt_attr_name).numpy()

        diff = np.abs(np_attr - pt_attr).max()
        status = "OK" if diff < 1e-10 else "MISMATCH"
        print(f"\n  [{attr}] max_diff={diff:.2e} [{status}]")
        print(f"    numpy  : {pretty(np_attr)}")
        print(f"    pytorch: {pretty(pt_attr)}")

    # output
    np_out = numpy_ln.output
    pt_out = to_np(pytorch_ln.forward(torch.from_numpy(x)))
    diff = np.abs(np_out - pt_out).max()
    status = "OK" if diff < 1e-10 else "MISMATCH"
    print(f"\n  [output] max_diff={diff:.2e} [{status}]")
    print(f"    numpy  : {pretty(np_out)}")
    print(f"    pytorch: {pretty(pt_out)}")

    return numpy_ln, pytorch_ln


def debug_backward_detailed(
    numpy_ln: NumPyLayerNorm,
    pytorch_ln: PyTorchLayerNorm,
    x: np.ndarray,
    grad_output: np.ndarray,
):
    """Manually trace backward step by step, print everything."""
    eps = numpy_ln.eps
    N_feature = x.shape[-1]  # This is the embedding dimension = 16

    print(f"\n{'='*70}")
    print(f"BACKWARD PASS STEP-BY-STEP")
    print(f"  input shape: {x.shape}, eps={eps}, N_feature={N_feature}")
    print(f"{'='*70}")

    # ---- forward intermediates ----
    np_xn = numpy_ln.x_norm
    np_var = numpy_ln.var
    np_gamma = numpy_ln.gamma

    pt_xn = to_np(pytorch_ln.x_norm)
    pt_var = to_np(pytorch_ln.x_var)
    pt_gamma = to_np(pytorch_ln.gamma)

    go = grad_output

    # ==========================================
    # STEP 1: grad_x_norm = grad_output * gamma
    # ==========================================
    np_gxn = go * np_gamma
    pt_gxn = go * pt_gamma
    diff = np.abs(np_gxn - pt_gxn).max()
    status = "OK" if diff < 1e-10 else "MISMATCH"
    print(f"\n  [1] grad_x_norm = go * gamma")
    print(f"       max_diff={diff:.2e} [{status}]")
    print(f"       numpy  : {pretty(np_gxn)}")
    print(f"       pytorch: {pretty(pt_gxn)}")

    # ==========================================
    # STEP 2: sum/mean computations
    # ==========================================
    # NumPy uses np.sum(..., axis=0) -> sum over batch, keeps feature dim
    np_sum_gxn = np.sum(np_gxn, axis=0, keepdims=True)           # (1, 16) -> sum over 4 elements
    pt_sum_gxn = np.sum(pt_gxn, axis=0, keepdims=True)           # same

    np_gxn_sum_gxn = np_gxn * np_xn
    np_sum_gxn_xn = np.sum(np_gxn_sum_gxn, axis=0, keepdims=True)
    pt_sum_gxn_xn = np.sum(pt_gxn * pt_xn, axis=0, keepdims=True)

    diff_sum = np.abs(np_sum_gxn - pt_sum_gxn).max()
    diff_sum_xn = np.abs(np_sum_gxn_xn - pt_sum_gxn_xn).max()
    status_sum = "OK" if diff_sum < 1e-10 else "MISMATCH"
    status_sum_xn = "OK" if diff_sum_xn < 1e-10 else "MISMATCH"
    print(f"\n  [2a] sum_of_grad_x_norm (axis=0)  max_diff={diff_sum:.2e} [{status_sum}]")
    print(f"       numpy  : {pretty(np_sum_gxn)}")
    print(f"       pytorch: {pretty(pt_sum_gxn)}")
    print(f"\n  [2b] sum(grad_x_norm * x_norm)    max_diff={diff_sum_xn:.2e} [{status_sum_xn}]")
    print(f"       numpy  : {pretty(np_sum_gxn_xn)}")
    print(f"       pytorch: {pretty(pt_sum_gxn_xn)}")

    # ==========================================
    # STEP 3: inv_sqrt(var + eps)
    # ==========================================
    np_inv_sqrt = 1.0 / np.sqrt(np_var + eps)
    pt_inv_sqrt = 1.0 / np.sqrt(pt_var + eps)
    diff = np.abs(np_inv_sqrt - pt_inv_sqrt).max()
    status = "OK" if diff < 1e-10 else "MISMATCH"
    print(f"\n  [3] inv_sqrt(var + eps) max_diff={diff:.2e} [{status}]")
    print(f"      numpy  : {pretty(np_inv_sqrt)}")
    print(f"      pytorch: {pretty(pt_inv_sqrt)}")

    # ==========================================
    # STEP 4: The inner formula for grad_x
    # ==========================================
    # NumPy backward:
    #   (N * grad_x_norm - sum(grad_x_norm) - x_norm * sum(grad_x_norm * x_norm)) / N
    #   where N = last_dim = embed_dim = 16   <-- THIS IS THE BUG
    #
    # My manual reference (correct):
    #   grad_x_norm - mean(grad_x_norm) - x_norm * mean(grad_x_norm * x_norm)
    #   where mean = sum / batch_size = 4

    # NumPy uses N_feature (16) in the formula
    np_inner_divby_Nfeature = (
        N_feature * np_gxn - np_sum_gxn - np_xn * np_sum_gxn_xn
    ) / N_feature

    # Correct: divide by batch_size
    N_batch = x.shape[0]  # = 4
    correct_inner = (
        np_gxn - np_sum_gxn / N_batch - np_xn * np_sum_gxn_xn / N_batch
    )
    # Actually let me check what pytorch does in its backward
    pt_mean_gxn = np.mean(pt_gxn, axis=-1, keepdims=True)
    pt_mean_gxn_xn = np.mean(pt_gxn * pt_xn, axis=-1, keepdims=True)
    print(f"\n  [4a] pytorch uses mean(..., axis=-1):")
    print(f"       mean_gxn = {pretty(pt_mean_gxn)}")
    print(f"       mean_gxn_xn = {pretty(pt_mean_gxn_xn)}")

    print(f"\n  [4b] numpy 'inner' (dividing by N_feature={N_feature} after multiplying by N_feature):")
    print(f"       inner = {pretty(np_inner_divby_Nfeature)}")
    print(f"       equivalent to: grad_x_norm - sum/N_feature - x_norm*sum_xn/N_feature")

    print(f"\n  [4c] correct formula (dividing by N_batch={N_batch}):")
    print(f"       inner = {pretty(correct_inner)}")
    print(f"       equivalent to: grad_x_norm - mean - x_norm*mean_xn")

    inner_diff = np.abs(np_inner_divby_Nfeature - correct_inner).max()
    print(f"\n  [4d] difference between numpy formula and correct formula: max_diff={inner_diff:.6f}")

    # PyTorch backward inner comparison
    pt_backward_inner = pt_gxn - pt_mean_gxn - pt_xn * pt_mean_gxn_xn
    print(f"\n  [4e] pytorch backward inner (grad_x_norm - mean_gxn - x_norm*mean_gxn_xn):")
    print(f"       = {pretty(pt_backward_inner)}")

    pt_vs_correct = np.abs(pt_backward_inner - correct_inner).max()
    print(f"       vs correct = {correct_inner} : max_diff={pt_vs_correct:.2e}")

    np_vs_correct = np.abs(np_inner_divby_Nfeature - correct_inner).max()
    print(f"       numpy formula vs correct : max_diff={np_vs_correct:.6f}")

    # ==========================================
    # STEP 5: grad_x = inv_sqrt * inner
    # ==========================================
    np_grad_x_formula = np_inv_sqrt * np_inner_divby_Nfeature

    # Check what PyTorch's backward actually produces
    _, pt_actual_grads = pytorch_ln.backward(torch.from_numpy(go))
    pt_grad_x_actual = to_np(pt_actual_grads.get("dx", torch.zeros_like(torch.from_numpy(go))))

    # Manual pytorch backward (same formula)
    pt_grad_x_manual = pt_inv_sqrt * pt_backward_inner

    diff_manual = np.abs(np_grad_x_formula - pt_grad_x_manual).max()
    print(f"\n  [5] grad_x comparison (formula):")
    print(f"      numpy formula vs pytorch manual: max_diff={diff_manual:.2e}")
    print(f"      numpy  : {pretty(np_grad_x_formula)}")
    print(f"      pytorch: {pretty(pt_grad_x_manual)}")

    diff_actual = np.abs(np_grad_x_formula - pt_grad_x_actual).max()
    print(f"      numpy formula vs pytorch.backward() result: max_diff={diff_actual:.2e}")

    # ==========================================
    # STEP 6: gamma_grad
    # ==========================================
    np_gamma_grad = np.sum(go * np_xn, axis=0)
    pt_gamma_grad = np.sum(go * pt_xn, axis=0, keepdims=True).flatten()

    diff = np.abs(np_gamma_grad - pt_gamma_grad).max()
    status = "OK" if diff < 1e-10 else "MISMATCH"
    print(f"\n  [6] gamma_grad = sum(go * x_norm, axis=0)")
    print(f"      max_diff={diff:.2e} [{status}]")
    print(f"      numpy  : {pretty(np_gamma_grad)}")
    print(f"      pytorch: {pretty(pt_gamma_grad)}")

    # ==========================================
    # STEP 7: beta_grad
    # ==========================================
    np_beta_grad = np.sum(go, axis=0)
    pt_beta_grad = np.sum(go, axis=0, keepdims=True).flatten()

    diff = np.abs(np_beta_grad - pt_beta_grad).max()
    status = "OK" if diff < 1e-10 else "MISMATCH"
    print(f"\n  [7] beta_grad = sum(go, axis=0)")
    print(f"      max_diff={diff:.2e} [{status}]")
    print(f"      numpy  : {pretty(np_beta_grad)}")
    print(f"      pytorch: {pretty(pt_beta_grad)}")

    # ==========================================
    # ACTUAL backward results
    # ==========================================
    print(f"\n{'='*70}")
    print(f"ACTUAL BACKWARD() RESULTS")
    print(f"{'='*70}")

    _, np_all_grads = numpy_ln.backward(go)
    np_grad_x_actual, _ = numpy_ln.backward(go)
    np_gamma_actual = np_all_grads["gamma"]
    np_beta_actual = np_all_grads["beta"]

    _, pt_all_grads = pytorch_ln.backward(torch.from_numpy(go))
    pt_grad_x_actual = to_np(pt_all_grads.get("dx", torch.zeros_like(torch.from_numpy(go))))
    pt_gamma_actual = to_np(pt_all_grads["weight"])
    pt_beta_actual = to_np(pt_all_grads["bias"])

    dx_diff = np.abs(np_grad_x_actual - pt_grad_x_actual).max()
    gamma_diff = np.abs(np_gamma_actual - pt_gamma_actual).max()
    beta_diff = np.abs(np_beta_actual - pt_beta_actual).max()

    print(f"\n  [dx]         max_diff={dx_diff:.6f} {'OK' if dx_diff < 1e-10 else 'MISMATCH'}")
    print(f"  [gamma_grad] max_diff={gamma_diff:.6f} {'OK' if gamma_diff < 1e-10 else 'MISMATCH'}")
    print(f"  [beta_grad]  max_diff={beta_diff:.6f} {'OK' if beta_diff < 1e-10 else 'MISMATCH'}")

    # ==========================================
    # ROOT CAUSE ANALYSIS
    # ==========================================
    print(f"\n{'='*70}")
    print(f"ROOT CAUSE ANALYSIS")
    print(f"{'='*70}")

    # The key issue: NumPy backward uses N = shape[-1] = 16 as the divisor
    # Correct should use batch_size = 4
    # Let me show the exact math
    print(f"\n  NumPy backward line:")
    print(f"    N = self.x.shape[-1] = {N_feature}  <-- embed dimension, NOT batch size!")
    print(f"    grad_x = (N * grad_x_norm - sum - x_norm * sum_xn) / N")
    print(f"    This is equivalent to: grad_x_norm - sum/{N_feature} - x_norm * sum_xn/{N_feature}")
    print(f"    So NumPy divides by {N_feature} (embed_dim) instead of {N_batch} (batch_size)")

    print(f"\n  PyTorch backward (and correct formula):")
    print(f"    grad_x = grad_x_norm - mean(grad_x_norm) - x_norm * mean(grad_x_norm * x_norm)")
    print(f"    mean(a) = sum(a) / batch_size = sum(a) / {N_batch}")
    print(f"    So PyTorch divides by {N_batch} (batch_size)")

    print(f"\n  The gap for each element in grad_x:")
    print(f"    NumPy: ... - sum/{N_feature} ...")
    print(f"    Correct: ... - sum/{N_batch} ...")
    print(f"    Ratio = N_feature / N_batch = {N_feature} / {N_batch} = {N_feature / N_batch}")

    print(f"\n  Summary:")
    print(f"    grad_x max_diff:  {dx_diff:.6f}")
    print(f"    gamma_grad diff:  {gamma_diff:.6f}  <-- gamma_grad uses sum(axis=0), no division")
    print(f"    beta_grad diff:   {beta_diff:.6f}  <-- beta_grad uses sum(axis=0), no division")

    return {
        "dx_diff": dx_diff,
        "gamma_grad_diff": gamma_diff,
        "beta_grad_diff": beta_diff,
        "N_feature_used": N_feature,
        "N_batch": N_batch,
    }


class DebugLayerNorm:
    """Wraps the actual implementations to compare step-by-step"""
    def __init__(self, numpy_ln, pytorch_ln, x, grad_output):
        self.numpy_ln = numpy_ln
        self.pytorch_ln = pytorch_ln
        self.x = x
        self.grad_output = grad_output


def test_debug_layernorm_backward_parity():
    """Debug test: identify exactly where NumPy and PyTorch LayerNorm diverge in backward pass."""
    np.random.seed(42)
    embed_dim = 16
    eps = 1e-5

    numpy_ln = NumPyLayerNorm(embed_dim, eps)
    pytorch_ln = PyTorchLayerNorm(embed_dim, eps)
    numpy_ln.set_params(numpy_ln.get_params())
    pytorch_ln.set_params(numpy_ln.get_params())

    x = np.random.randn(4, embed_dim).astype(np.float64)
    go = np.random.randn(4, embed_dim).astype(np.float64)

    # Step 1: Forward pass comparison
    debug_forward(numpy_ln, pytorch_ln, x)

    # Step 2: Backward pass - detailed step-by-step
    result = debug_backward_detailed(numpy_ln, pytorch_ln, x, go)

    print(f"\n{'='*70}")
    print(f"FINAL DIAGNOSIS")
    print(f"{'='*70}")
    print(f"  N (embed_dim) in NumPy backward = {result['N_feature_used']}")
    print(f"  B (batch_size) actual           = {result['N_batch']}")
    print(f"  N/B ratio                       = {result['N_feature_used'] / result['N_batch']}")
    print(f"  grad_x max diff:  {result['dx_diff']:.6f}")
    print(f"  gamma_diff:       {result['gamma_grad_diff']:.6f}")
    print(f"  beta_diff:        {result['beta_grad_diff']:.6f}")
    print(f"{'='*70}"
)

    # This test intentionally fails to show the divergence
    assert False, "Debug test - see output for analysis"


if __name__ == "__main__":
    test_debug_layernorm_backward_parity()
