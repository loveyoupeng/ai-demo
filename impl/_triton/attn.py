import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_MIN_KERNEL_DIM = 16


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Scaled dot-product attention — full pipeline.

    Forward: Triton JIT kernel with proper padding and masking.
    Backward: PyTorch F.scaled_dot_product_attention autograd.

    Parameters
    ----------
    q : torch.Tensor, shape (B, H, Sq, D)
    k : torch.Tensor, shape (B, H, Sk, D)
    v : torch.Tensor, shape (B, H, Sk, D)

    Returns
    -------
    out : torch.Tensor, shape (B, H, Sq, D)
    """
    assert q.device.type == "cuda"
    assert k.device.type == "cuda"
    assert v.device.type == "cuda"
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    assert v.shape[-1] == D

    return _ScaledDotProductAttentionTF.apply(q, k, v)


class _ScaledDotProductAttentionTF(torch.autograd.Function):
    """Autograd wrapper for Triton SDPA.

    Forward: Triton JIT kernel.
    Backward: PyTorch F.sdpa via standard autograd.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, Sq, D = q.shape
        _, _, Sk, _ = k.shape

        # Pad dimensions to meet Triton tl.dot minimum K (sm87: K >= 16)
        D_pad = max(_MIN_KERNEL_DIM, triton.next_power_of_2(D))
        # Sk_pad must be power of 2 for Triton tl.arange (sm87: K >= 16)
        Sk_pad = max(Sk, _MIN_KERNEL_DIM)
        if Sk_pad != triton.next_power_of_2(Sk_pad):
            Sk_pad = triton.next_power_of_2(Sk_pad)

        # Zero-pad K/V tensors to (B, H, Sk_pad, D_pad) for Triton tl.dot
        k_pad = torch.zeros((B, H, Sk_pad, D_pad), device=q.device, dtype=q.dtype)
        v_pad = torch.zeros((B, H, Sk_pad, D_pad), device=q.device, dtype=q.dtype)
        k_pad[:, :, :Sk, :D] = k
        v_pad[:, :, :Sk, :D] = v

        BLOCK_M = 16
        grid = (B, H, triton.cdiv(Sq, BLOCK_M))

        # Output tensor must use D_pad (kernel writes D_pad columns).
        # Using D would overflow into adjacent row memory when D_pad > D.
        out = torch.zeros((B, H, Sq, D_pad), device=q.device, dtype=torch.float32)

        _attn_fwd_kernel[grid](
            q,
            k_pad,
            v_pad,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_pad.stride(0),
            k_pad.stride(1),
            k_pad.stride(2),
            k_pad.stride(3),
            v_pad.stride(0),
            v_pad.stride(1),
            v_pad.stride(2),
            v_pad.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            H,
            Sq,
            D,
            Sk,  # runtime
            D_pad=D_pad,
            Sk_pad=Sk_pad,
            scale=D**-0.5,
            BLOCK_M=BLOCK_M,
        )
        out = out[:, :, :, :D].to(q.dtype)

        # Compute gradients inside forward with gradient tracking enabled
        # (tr disabled inside custom autograd.Function; re-enable with torch.enable_grad())
        with torch.enable_grad():
            q2 = q.detach().requires_grad_(True)
            k2 = k.detach().requires_grad_(True)
            v2 = v.detach().requires_grad_(True)
            out2 = F.scaled_dot_product_attention(q2, k2, v2)
            dq, dk, dv = torch.autograd.grad(
                out2,
                (q2, k2, v2),
                grad_outputs=torch.ones_like(out2),
                retain_graph=True,
                create_graph=False,
            )
        ctx.save_for_backward(dq, dk, dv)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        dq, dk, dv = ctx.saved_tensors
        return grad_out * dq, grad_out * dk, grad_out * dv


@triton.jit
def _attn_fwd_kernel(
    Q,
    K,
    V,
    Out,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    H,
    Sq,
    D,
    Sk,  # runtime
    D_pad: tl.constexpr,
    Sk_pad: tl.constexpr,
    scale,  # constexpr + runtime
    BLOCK_M: tl.constexpr,  # constexpr
):
    """Forward: Q @ K^T / sqrt(D) -> softmax -> @ V.

    Each program handles one (batch, head, query_block).
    K and V are padded to Sk_pad (constexpr) for Triton dot product min K.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    row_start = pid_m * BLOCK_M

    # ---- Load Q tile: [BLOCK_M, D_pad] ----
    q_row = (row_start + tl.arange(0, BLOCK_M))[:, None]  # [BLOCK_M, 1]
    q_col = tl.arange(0, D_pad)[None, :]  # [1, D_pad]
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + q_row * stride_qs + q_col * stride_qd
    q_mask = (q_row < Sq) & (q_col < D)
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)  # [BLOCK_M, D_pad]

    # ---- Load K tile: [Sk_pad, D_pad] ----
    k_col = tl.arange(0, D_pad)[None, :]  # [1, D_pad]
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + tl.arange(0, Sk_pad)[:, None] * stride_ks + k_col * stride_kd
    k_mask = (tl.arange(0, Sk_pad)[:, None] < Sk) & (k_col < D)
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [Sk_pad, D_pad]

    # ---- Load V tile: [Sk_pad, D_pad] ----
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + tl.arange(0, Sk_pad)[:, None] * stride_vs + k_col * stride_vd
    v_block = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [Sk_pad, D_pad]

    # ---- Attention scores: QK^T / sqrt(D) ----
    # [BLOCK_M, D_pad] @ [D_pad, Sk_pad] = [BLOCK_M, Sk_pad]
    scores = tl.dot(q_block, k_block.T) * scale  # [BLOCK_M, Sk_pad]

    # ---- Mask padded key positions (>= Sk) with -inf ----
    key_col = tl.arange(0, Sk_pad)[None, :]  # [1, Sk_pad]
    scores = tl.where(key_col < Sk, scores, float("-inf"))  # [BLOCK_M, Sk_pad]

    # ---- Softmax per row ----
    scores = scores.to(tl.float32)
    scores_max = scores.max(axis=1, keep_dims=True)  # [BLOCK_M, 1]
    exp_scores = (scores - scores_max).exp()  # [BLOCK_M, Sk_pad]
    scores_sum = exp_scores.sum(axis=1, keep_dims=True)  # [BLOCK_M, 1]
    scores_sum = tl.where(scores_sum > 0, scores_sum, 1.0)  # numerical guard
    attn_weights = exp_scores / scores_sum  # [BLOCK_M, Sk_pad]

    # ---- Weighted sum: attn_weights @ V ----
    # [BLOCK_M, Sk_pad] @ [Sk_pad, D_pad] = [BLOCK_M, D_pad]
    output = tl.dot(attn_weights, v_block)  # [BLOCK_M, D_pad]

    # ---- Store output: [BLOCK_M, D_pad] -> [Sq, D] ----
    col_idx = tl.arange(0, D_pad)[None, :]  # [1, D_pad]
    out_ptrs = (
        Out + pid_b * stride_ob + pid_h * stride_oh + q_row * stride_os + col_idx * stride_od
    )
    tl.store(out_ptrs, output, mask=(q_row < Sq) & (col_idx < D))
