"""Device/dtype resolution for the torch-family models (one deep helper).

"Where does this model compute, and in what dtype?" is ONE fact per model —
but the three torch-family tracks spell it differently:

- **TorchModel** — an ``nn.Module``: ``parameters()`` first tensor carries
  the device/dtype (CPU float64 on the record path, float32 when built
  fresh; the caller decides).
- **TritonModel** — also an ``nn.Module`` (same probe works), always
  CUDA-resident at inference.
- **CUDAModel** — NOT an ``nn.Module`` (the bare-metal track): raw tensor
  attributes that rest on CPU with ``requires_grad``; the NVRTC kernels
  follow the INPUT's device, so compute happens wherever the prompt is
  sent — and the kernels themselves are GPU-only, so the prompt must land
  on CUDA.

That last quirk is the load-bearing knowledge: probing the *weights'* device
gives the wrong answer for CUDAModel (CPU) while the engine's hidden states
live on CUDA. This module owns the resolution once; every consumer (the
learning server, the record adapters, the CLI) asks here instead of
re-deriving it — the bug class this session (four separate device crashes)
came from exactly that leakage.

Also owns the shared embedding/lm_head accessor: the three tracks spell the
shared weights differently (TorchModel/TritonModel: ``.embedding`` /
``.lm_head`` nn modules; CUDAModel: ``.embedding_weights`` /
``.lm_head_weight`` raw tensors), and the nn.Linear-backed lm_head stores
its weight transposed ((V, D) per nn.Linear convention) while the drafter
contract wants the NumPy track's (D, V).
"""

from __future__ import annotations

import torch


def compute_device(model) -> torch.device:
    """The device this model's COMPUTE happens on (the prompt's device).

    nn.Module tracks (torch/triton): the first parameter's device.
    CUDAModel (follow-the-input, NVRTC kernels GPU-only): CUDA when
    available, else CPU (the caller will see the kernels fail on CPU with
    a clear error — never a silent wrong answer).
    """
    if hasattr(model, "parameters"):
        p = next(model.parameters(), None)
        if p is not None:
            return p.device
    if hasattr(model, "embedding_weights") and not hasattr(model, "embedding"):
        # CUDAModel: follow-the-input — kernels are GPU-only.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def shared_embedding_and_head(model) -> tuple[torch.Tensor, torch.Tensor]:
    """The (embedding (V, D), lm_head (D, V)) pair in the model's dtype/device.

    The lm_head is returned in the NumPy track's (D, V) layout — the
    drafter contract — regardless of how the track stores it (nn.Linear
    stores (V, D); the transpose happens here, once).
    """
    with torch.no_grad():
        if hasattr(model, "embedding_weights") and not hasattr(model, "embedding"):
            emb = model.embedding_weights.detach().clone()  # already (V, D)
            lm = model.lm_head_weight.detach().clone()  # already (D, V)
        else:
            emb = model.embedding.weight.detach().clone()  # (V, D)
            lm = model.lm_head.weight.detach().clone().T.contiguous()  # (V, D) → (D, V)
        device = compute_device(model)
        return emb.to(device), lm.to(device)
