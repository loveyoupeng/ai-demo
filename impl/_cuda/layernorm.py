"""RMSNorm — nvrtc compiled + PyTorch dispatch.

Computes Root Mean Square Layer Normalization:
    out = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * gamma

This implements a warp-reduction kernel in CUDA C, demonstrating:
- Block-wide tree reduction using shared memory
- shfl_down for intra-warp communication
- Synchronization barriers for multi-warp coordination
- Shared memory for intermediate reduction results

The kernel processes each row independently, computing RMS normalization
over the last dimension (embed_dim), then scaling by learnable gamma.

Learning objectives:
- Warp-level reduction (shfl_down intrinsic)
- Shared memory for multi-warp coordination
- Synchronization patterns (__syncthreads)
- 2D kernel layout (rows × features)
"""

from __future__ import annotations

import ctypes
from typing import Any

import torch
from cuda import cuda as _cuda_lib

from impl._cuda.compiler import compile_and_load, get_kernel_handle

# ---------------------------------------------------------------------------
# CUDA kernel source — loaded from companion .cu file
# ---------------------------------------------------------------------------

_KERNEL_SOURCE_PATH = __file__.rsplit("/", 1)[0] + "/kernels/layernorm.cu"


def _load_kernel_source() -> str:
    """LoadCUDA kernel source from file.

    Returns
    -------
    str
        Raw CUDA C source code.
    """
    with open(_KERNEL_SOURCE_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Kernel compiler — nvrtc compile → PTX → module → kernel handle
# ---------------------------------------------------------------------------


class _RmsNormKernels:
    """Compile and cache RMSNorm kernels.

    The compilation pipeline:
    1. Load CUDA C source from file
    2. nvrtcCompileProgram → PTX bytecode
    3. cuModuleLoadData → CUDA runtime module
    4. cuModuleGetFunction → kernel handles
    5. Cache handles for the lifetime of the process

    Module lifetime
    ---------------
    The CUDA module stays loaded until the process exits. Kernel handles
    are valid for the lifetime of the module.
    """

    _forward_kernel = None
    _forward_f64_kernel = None
    _module = None
    _ptx_data = None

    @classmethod
    def compile(cls) -> None:
        """Compile CUDA source and store kernel handles."""
        source = _load_kernel_source()
        module, ptx_data = compile_and_load(source)
        cls._module = module
        cls._ptx_data = ptx_data
        cls._forward_kernel = get_kernel_handle(module, "rmsnorm_forward_kernel", ptx_data)
        cls._forward_f64_kernel = get_kernel_handle(module, "rmsnorm_forward_f64_kernel", ptx_data)

    @classmethod
    def get_forward_kernel(cls) -> Any:
        """Get or compile the float32 forward kernel."""
        if cls._forward_kernel is None:
            cls.compile()
        return cls._forward_kernel

    @classmethod
    def get_forward_f64_kernel(cls) -> Any:
        """Get or compile the float64 forward kernel."""
        if cls._forward_f64_kernel is None:
            cls.compile()
        return cls._forward_f64_kernel


# ---------------------------------------------------------------------------
# Kernel launcher — PyTorch tensors → device pointers → kernel launch
# ---------------------------------------------------------------------------


def _launch_kernel(kernel: Any, params: list, grid_x: int, block_x: int = 256, shared_mem: int = 0) -> None:  # noqa: C901
    """Launch a compiled CUDA kernel using the CUDA driver API.

    This bridges Python/PyTorch with raw CUDA kernel execution. It handles
    tensor pointer extraction, parameter type resolution, block/grid setup,
    and stream management.

    Parameters
    ----------
    kernel : Any
        Kernel handle from cuModuleGetFunction.
    params : list
        List of torch.Tensor, int, or ctypes.c_void_p to pass as arguments.
    grid_x : int
        Number of blocks (one per row).
    block_x : int
        Threads per block (typically = cols / block_size, up to 256).

    Notes
    -----
    The shared memory size for the reduction is computed dynamically:
        shared_mem = num_warps * sizeof(float) = ceil(block_x/32) * 4 bytes
    """
    values, types = [], []
    for p in params:
        if isinstance(p, torch.Tensor):
            values.append(ctypes.c_void_p(p.data_ptr()))
            types.append(ctypes.c_void_p)
        elif isinstance(p, ctypes.c_void_p):
            values.append(p)
            types.append(ctypes.c_void_p)
        elif isinstance(p, ctypes.c_int):
            values.append(p)
            types.append(ctypes.c_int)
        elif isinstance(p, int):
            values.append(ctypes.c_int(p))
            types.append(ctypes.c_int)
        elif isinstance(p, ctypes.c_float):
            values.append(p)
            types.append(ctypes.c_float)
        elif isinstance(p, ctypes.c_double):
            values.append(p)
            types.append(ctypes.c_double)
        elif isinstance(p, float):
            values.append(ctypes.c_float(p) if not isinstance(p, float) else ctypes.c_float(p))
            types.append(ctypes.c_float)
        else:
            raise TypeError(f"Unsupported param type: {type(p)}")

    stream_ret = _cuda_lib.cuStreamCreate(0)
    if stream_ret[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to create CUDA stream: {stream_ret}")
    stream = stream_ret[1]

    kernel_args = (tuple(values), tuple(types))

    try:
        status = _cuda_lib.cuLaunchKernel(
            kernel,
            grid_x,
            1,
            1,  # 1D grid: one block per row
            block_x,
            1,
            1,  # 1D block: threads per row (up to 256)
            shared_mem,  # shared memory: enough for warp reduction storage
            stream,
            kernel_args,
            0,  # extra launch attributes — 0, not None on Jetson/L4T
        )
        if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel failed: {status}")
    finally:
        _cuda_lib.cuStreamDestroy(stream)


# ---------------------------------------------------------------------------
# Custom autograd function — forward in CUDA, backward in PyTorch
# ---------------------------------------------------------------------------


class _RmsNormCudaFunction(torch.autograd.Function):
    """RMSNorm forward in CUDA, backward in PyTorch.

    Forward pass:
    1. Compute RMS norm via warp-reduction CUDA kernel
    2. Each row processed in parallel by one block of 256 threads

    Backward pass:
    1. Gradients flow through PyTorch's built-in operations
    2. No CUDA backward kernel needed

    Usage
    -----
    >>> x = torch.randn(B, S, D, device="cuda")
    >>> gamma = torch.ones(D, device="cuda")
    >>> y = _RmsNormCudaFunction.apply(x, gamma)
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Forward: out = x / rms_norm(x) * gamma.

        Parameters
        ----------
        ctx : Any
            Autograd context — saved tensors needed for backward.
        input : torch.Tensor
            Input tensor (..., embed_dim) on device.
        gamma : torch.Tensor
            Learnable scale parameter (embed_dim,).

        Returns
        -------
        torch.Tensor
            Normalized, scaled output — same shape as input.
        """
        ctx.save_for_backward(input, gamma)
        ctx.original_shape = input.shape
        is_float64 = input.dtype == torch.float64

        # input: (..., embed_dim) -> flatten to (N, D)
        original_shape = input.shape
        N = input.numel() // input.shape[-1]
        D = input.shape[-1]

        # Flatten inputs to (N, D) for kernel launch
        x_flat = input.view(N, D) if input.dim() > 2 else input

        # Output: (N, D)
        output_flat = torch.empty_like(x_flat)

        block_size = min(256, D)  # clamp to D if D < 256
        if block_size == 0:
            raise ValueError("embed_dim must be > 0")

        # Shared memory: one float64 per warp (8 bytes), one float32 per warp (4 bytes)
        num_warps = max(1, (block_size + 31) // 32)
        shared_mem = num_warps * (8 if is_float64 else 4)

        eps = 1e-6

        if is_float64:
            x_ptr = ctypes.c_void_p(x_flat.data_ptr())
            g_ptr = ctypes.c_void_p(gamma.data_ptr())
            o_ptr = ctypes.c_void_p(output_flat.data_ptr())
            _launch_kernel(
                _RmsNormKernels.get_forward_f64_kernel(),
                [x_ptr, g_ptr, o_ptr, ctypes.c_int(N), ctypes.c_int(D), ctypes.c_double(eps)],
                grid_x=N,
                block_x=block_size,
                shared_mem=shared_mem,
            )
        else:
            _launch_kernel(
                _RmsNormKernels.get_forward_kernel(),
                [x_flat, gamma, output_flat, N, D, ctypes.c_float(eps)],
                grid_x=N,
                block_x=block_size,
                shared_mem=shared_mem,
            )

        # Reshape back to original shape
        output = output_flat.view(original_shape)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Backward pass for RMSNorm.

        Gradient computation:
            d_input = gamma * inv_rms * (d_output - mean(d_output * normalized_input) * normalized_input)

        Parameters
        ----------
        ctx : Any
            Autograd context with saved input and gamma.
        grad_output : torch.Tensor
            Gradient from the loss.

        Returns
        -------
        tuple
            Gradient w.r.t. input and gamma (for weight).
        """
        input, gamma = ctx.saved_tensors

        # Reshape to (N, D) for computation
        N = input.numel() // input.shape[-1]
        D = input.shape[-1]

        if input.dim() > 2:
            x = input.view(N, D)
            normalized = input / (torch.sqrt(torch.mean(input**2, dim=-1, keepdim=True) + 1e-6))
        else:
            x = input
            normalized = x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6))

        grad_output = grad_outputs[0].view(N, D)

        # RMSNorm gradient (standard derivation):
        # d_x = gamma * inv_rms * (d_out - mean(d_out * x * inv_rms) * (x * inv_rms))
        # where x * inv_rms is the normalized (before scaling)
        eps = 1e-6
        inv_rms = 1.0 / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        x_norm = x * inv_rms  # normalized (before gamma scaling)

        # d_out_scaled = d_out * gamma
        d_out_scaled = grad_output * gamma.unsqueeze(0)

        # mean(d_out_scaled * x_norm) — per-row mean
        mean_val = torch.mean(d_out_scaled * x_norm, dim=-1, keepdim=True)

        # d_x = gamma * inv_rms * (d_out - mean(d_out * x_norm) * x_norm)
        grad_input = gamma.unsqueeze(0) * inv_rms * (d_out_scaled - mean_val * x_norm)

        # Gradient for gamma: sum over all dimensions except last (feature dimension)
        # normalized has same shape as input (B, S, D) or (N, D)
        # grad_output is reshaped to (N, D) for backward computation
        # We must flatten normalized to match grad_output's (N, D) shape before multiplication
        normalized_flat = normalized.view(N, D)
        if input.dim() == 2:
            grad_gamma = torch.sum(grad_output * normalized_flat, dim=0)
        else:  # input is 3D, sum over batch dimensions to get (D,)
            grad_gamma = torch.sum(grad_output * normalized_flat, dim=0)

        return grad_input.view(ctx.original_shape) if input.dim() > 2 else grad_input, grad_gamma


# ---------------------------------------------------------------------------
# Public API — user-facing function
# ---------------------------------------------------------------------------


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Compute RMSNorm: x / sqrt(mean(x^2, dim=-1) + eps) * gamma.

    This kernel uses CUDA warp reduction to compute the RMS normalization
    factor in a fully parallel manner. Each row of the input is processed
    independently by one block of threads.

    The kernel demonstrates:
    - Warp-level reduction using shfl_down
    - Shared memory for multi-warp coordination
    - Synchronization barriers (__syncthreads)
    - 2D thread mapping: 1D grid (rows) × 1D block (features)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (..., embed_dim). Any leading batch dimensions
        are allowed. Shape can be 1D, 2D, 3D, etc.
    gamma : torch.Tensor
        Learnable scale parameter of shape (embed_dim,). Must be on the
        same device as x.

    Returns
    -------
    torch.Tensor
        RMS-normalized output with the same shape as x. The output is
        scaled by gamma so that each output vector has unit RMS.

    Example
    -------
    >>> x = torch.randn(2, 4, 128, device="cuda")  # (batch, seq, embed)
    >>> gamma = torch.ones(128, device="cuda")
    >>> y = rmsnorm(x, gamma)  # (2, 4, 128) — each feature vector normalized

    Kernel configuration
    --------------------
    - Block size: min(256, embed_dim) threads
    - Grid size: batch_size * seq_len blocks (one per normalization row)
    - Shared memory: ceil(block_size / 32) * 4 bytes (warp reduction storage)
    - Memory access: fully coalesced (consecutive threads access consecutive elements)
    """
    return _RmsNormCudaFunction.apply(x, gamma)
