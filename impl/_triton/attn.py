import torch


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Scaled dot-product attention — Triton wrapper (uses PyTorch cuBLAS for now).

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
    B, H, Sq, D_q = q.shape
    _, _, Sk, D_k = k.shape
    assert D_q == D_k == v.shape[-1]

    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)
