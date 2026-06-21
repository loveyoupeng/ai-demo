"""Rotary Position Embedding (RoPE) — nvrtc compiled + PyTorch dispatch.

This module implements RoPE kernel using:
1. nvrtc compiled CUDA C kernel (kernels/rope.cu)
2. PyTorch autograd function for forward/backward dispatch
3. Automatic cos/sin table generation

RoPE encodes absolute position information into each token's embedding
by applying a position-dependent rotation in 2D subspaces:

  For each (odd, even) dimension pair (x_{2m}, x_{2m+1}) at position p:

    [x'_{2m},   x'_{2m+1}] = [cos(p*θ_m), -sin(p*θ_m)] [x_{2m}]
                            [sin(p*θ_m),  cos(p*θ_m)] [x_{2m+1}]

  where θ_m = 10000^(-2m/d_model) is the frequency for dimension pair m.

Learning objectives:
- CUDA C kernel for 2D rotation with position-dependent angles
- Precomputing and broadcasting cos/sin tables
- PyTorch autograd integration for gradient computation
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

_KERNEL_SOURCE_PATH = __file__.rsplit("/", 1)[0] + "/kernels/rope.cu"


def _load_kernel_source() -> str:
    """Load CUDA kernel source from file for nvrtc compilation.

    Returns
    -------
    str
        Raw CUDA C source code.
    """
    with open(_KERNEL_SOURCE_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Kernel compiler — nvrtc compile → PTX → module → kernel handles
# ---------------------------------------------------------------------------


class _RoPEKernels:
    """Compile and cache RoPE forward and backward kernels.

    This class manages the nvrtc compilation pipeline for RoPE:
    1. Load CUDA C source from file
    2. Compile with nvrtc → PTX bytecode
    3. Load PTX as runtime CUDA module
    4. Extract kernel handles via cuModuleGetFunction

    The compilation happens once on first use and kernels are cached
    for the lifetime of the process.
    """

    _forward_kernel = None
    _forward_f64_kernel = None
    _backward_kernel = None
    _backward_f64_kernel = None
    _module = None
    _ptx_data = None

    @classmethod
    def compile(cls) -> None:
        """Compile CUDA source and store kernel handles.

        This follows the compilation pipeline:
        1. nvrtcCreateProgram + nvrtcCompileProgram → PTX bytecode
        2. cuModuleLoadDataEx → CUDA runtime module
        3. cuModuleGetFunction → kernel function handles

        The handles are stored in class attributes and cached for
        the lifetime of the process.
        """
        source = _load_kernel_source()
        module, ptx_data = compile_and_load(source)
        cls._module = module
        cls._ptx_data = ptx_data

        cls._forward_kernel = get_kernel_handle(module, "rope_fwd_f32", ptx_data)
        cls._forward_f64_kernel = get_kernel_handle(module, "rope_fwd_f64", ptx_data)
        cls._backward_kernel = get_kernel_handle(module, "rope_bwd_f32", ptx_data)
        cls._backward_f64_kernel = get_kernel_handle(module, "rope_bwd_f64", ptx_data)

    @classmethod
    def get_forward_kernel(cls) -> Any:
        """Get or compile the forward kernel.

        Returns
        -------
        Any
            Forward kernel handle.
        """
        if cls._forward_kernel is None:
            cls.compile()
        return cls._forward_kernel

    @classmethod
    def get_forward_f64_kernel(cls) -> Any:
        """Get or compile the float64 forward kernel.

        Returns
        -------
        Any
            Float64 forward kernel handle.
        """
        if cls._forward_f64_kernel is None:
            cls.compile()
        return cls._forward_f64_kernel

    @classmethod
    def get_backward_kernel(cls) -> Any:
        """Get or compile the backward kernel.

        Returns
        -------
        Any
            Backward kernel handle.
        """
        if cls._backward_kernel is None:
            cls.compile()
        return cls._backward_kernel

    @classmethod
    def get_backward_f64_kernel(cls) -> Any:
        """Get or compile the float64 backward kernel.

        Returns
        -------
        Any
            Float64 backward kernel handle.
        """
        if cls._backward_f64_kernel is None:
            cls.compile()
        return cls._backward_f64_kernel


# ---------------------------------------------------------------------------
# Cos/Sin table computation — geometric frequency schedule
# ---------------------------------------------------------------------------


def _compute_rope_tables(
    head_dim: int,
    max_position: int,
    rope_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cos/sin tables for all positions and dimension pairs.

    Generates the geometric frequency schedule and precomputes cos/sin
    values for all positions. This is done ONCE per model configuration.

    Parameters
    ----------
    head_dim : int
        Head dimension (D, must be even).
    max_position : int
        Maximum position index (context length).
    rope_dim : int
        Number of dimensions to rotate (must be even, <= head_dim).
    device : torch.device
        CUDA device.
    dtype : torch.dtype
        Output dtype (default float64 for precision).

    Returns
    -------
    cos : torch.Tensor, shape (max_position, rope_dim // 2)
        Per-position cosine values for each dimension pair.
    sin : torch.Tensor, shape (max_position, rope_dim // 2)
        Per-position sine values for each dimension pair.

    Notes
    -----
    Frequency formula: θ_m = 10000^(-2m / head_dim) for m = 0..D/2-1

    The 10000 base creates a range of frequencies from period 2π (slowest)
    to period 2π/10000 (fastest), allowing the model to detect both
    local and distant position relations.
    """
    pair_dim = rope_dim // 2

    # freqs: (pair_dim,)
    freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, device=device, dtype=dtype) * 2.0 / head_dim))

    # positions: (max_position,)
    positions = torch.arange(max_position, device=device, dtype=dtype)

    # angles: (max_position, pair_dim)
    # Each row is the angle in radians for each dimension pair at that position
    angles = positions[:, None] * freqs[None, :]

    return torch.cos(angles), torch.sin(angles)


# ---------------------------------------------------------------------------
# Kernel launcher — grid/block configuration
# ---------------------------------------------------------------------------


def _launch_rope_kernel(
    kernel: Any,
    input_tensor: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    out_tensor: torch.Tensor,
    total_tokens: int,
    S: int,
    D: int,
    rope_dim: int,
    block_size: int = 256,
) -> None:
    """Launch a RoPE kernel with the given configuration.

    Parameters
    ----------
    kernel : Any
        Kernel handle from cuModuleGetFunction.
    input_tensor : torch.Tensor
        Input tensor (tokens, D) flattened.
    cos_table : torch.Tensor
        Cosine table (max_pos, pair_dim).
    sin_table : torch.Tensor
        Sine table (max_pos, pair_dim).
    out_tensor : torch.Tensor
        Output tensor (tokens, D) flattened.
    total_tokens : int
        Total number of tokens (B * S * H).
    S : int
        Sequence length (used for position computation).
    D : int
        Head dimension.
    rope_dim : int
        Number of dimensions to rotate.
    block_size : int
        Number of threads per block (default 256).
    """
    grid_size = (total_tokens + block_size - 1) // block_size

    # Build kernel parameters: f32/f64: x, cos, sin, x_out, total_tokens, S, D, rope_dim
    params = [
        input_tensor,  # const float* / const double*
        cos_table,  # const float* / const double*
        sin_table,  # const float* / const double*
        out_tensor,  # float* / double*
        ctypes.c_int(total_tokens),
        ctypes.c_int(S),
        ctypes.c_int(D),
        ctypes.c_int(rope_dim),
    ]

    # Build parameter tuple: ((values...), (types...))
    values, types = [], []
    for p in params:
        if isinstance(p, torch.Tensor):
            values.append(ctypes.c_void_p(p.data_ptr()))
            types.append(ctypes.c_void_p)
        else:
            values.append(p)
            types.append(ctypes.c_int)

    kernel_args = (tuple(values), tuple(types))

    # Launch kernel
    status = _cuda_lib.cuLaunchKernel(
        kernel,
        grid_size,  # grid x
        1,  # grid y
        1,  # grid z
        block_size,  # block x
        1,  # block y
        1,  # block z
        0,  # shared memory
        None,  # stream
        kernel_args,
        0,  # extra
    )
    if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {status}")


# ---------------------------------------------------------------------------
# Custom autograd function — forward/backward in CUDA
# ---------------------------------------------------------------------------


class _RoPECudaFunction(torch.autograd.Function):
    """RoPE forward and backward pass in CUDA.

    This implements Rotary Position Embedding with:
    - Forward: CUDA kernel applies position-dependent 2D rotations
    - Backward: CUDA kernel applies transpose rotation to gradients

    The forward pass applies orthogonal rotation matrices parameterized
    by position. The backward pass applies the transpose (inverse) of
    these matrices to compute gradients.

    Usage
    -----
    >>> x = torch.randn(2, 8, 4, 16, device="cuda")  # (B, S, H, D)
    >>> positions = torch.arange(8, device="cuda")
    >>> y = _RoPECudaFunction.apply(x, positions)
    >>> y.sum().backward()  # gradients flow through CUDA
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: apply RoPE rotation to input tensor.

        Parameters
        ----------
        ctx : Any
            Autograd context — stores tensors needed for backward pass.
        x : torch.Tensor
            Input tensor on device, shape (B, S, H, D).
        positions : torch.Tensor
            Position indices, shape (S,) or (B, S) — flattened to (S).

        Returns
        -------
        torch.Tensor
            Rotated tensor with same shape and dtype as input.

        CUDA kernel configuration
        -------------------------
        - Block size: 256 threads (one thread per token)
        - Grid size: ceil(n_tokens / 256)
        - Shared memory: none (element-wise operation)
        - Memory access: coalesced reads/writes to input/output
        """
        # Save inputs for backward
        ctx.save_for_backward(x, positions)
        ctx.x_dtype = x.dtype

        # Flatten to (tokens, head_dim)
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        n_tokens = x_flat.shape[0]

        # Get per-token positions from positions tensor
        # positions is (S,) or (B, S); we use (token_idx // H) % S
        # For simplicity in CUDA kernel, we pass a full positions vector of length S

        rope_dim = x.shape[-1]  # All dimensions are rotated

        # Compute cos/sin tables
        max_pos = int(positions.max()) + 1
        cos_table, sin_table = _compute_rope_tables(
            head_dim=x.shape[-1],
            max_position=max_pos,
            rope_dim=rope_dim,
            device=x.device,
            dtype=x.dtype,
        )

        # Create output tensor
        output_flat = torch.empty_like(x_flat)

        # Launch kernel
        if x.dtype == torch.float64:
            _launch_rope_kernel(
                _RoPEKernels.get_forward_f64_kernel(),
                x_flat,
                cos_table,
                sin_table,
                output_flat,
                n_tokens,
                x.shape[1],  # S
                x.shape[-1],  # D
                rope_dim,
            )
        else:
            _launch_rope_kernel(
                _RoPEKernels.get_forward_kernel(),
                x_flat,
                cos_table,
                sin_table,
                output_flat,
                n_tokens,
                x.shape[1],  # S
                x.shape[-1],  # D
                rope_dim,
            )

        return output_flat.view(original_shape)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Backward pass: apply inverse rotation to gradient tensor.

        The backward of RoPE is the transpose of the forward rotation:
        if R = [[cos, -sin], [sin, cos]], then R^T = [[cos, sin], [-sin, cos]]

        Parameters
        ----------
        ctx : Any
            Autograd context containing saved input tensor and positions.
        grad_output : torch.Tensor
            Gradient from upstream (same shape as forward output).

        Returns
        -------
        tuple
            (grad_x, None, None) — only x has requires_grad, positions doesn't.
        """
        input_tensor, positions = ctx.saved_tensors
        rope_dim = input_tensor.shape[-1]

        # Flatten gradient
        grad_flat = grad_outputs[0].reshape(-1, input_tensor.shape[-1])
        n_tokens = grad_flat.shape[0]

        # Compute cos/sin tables (same as forward)
        max_pos = int(positions.max()) + 1
        cos_table, sin_table = _compute_rope_tables(
            head_dim=input_tensor.shape[-1],
            max_position=max_pos,
            rope_dim=rope_dim,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

        # Create output for gradient
        grad_x_flat = torch.empty_like(grad_flat)

        # Launch backward kernel (applies inverse rotation)
        if input_tensor.dtype == torch.float64:
            _launch_rope_kernel(
                _RoPEKernels.get_backward_f64_kernel(),
                grad_flat,
                cos_table,
                sin_table,
                grad_x_flat,
                n_tokens,
                input_tensor.shape[1],  # S
                input_tensor.shape[-1],  # D
                rope_dim,
            )
        else:
            _launch_rope_kernel(
                _RoPEKernels.get_backward_kernel(),
                grad_flat,
                cos_table,
                sin_table,
                grad_x_flat,
                n_tokens,
                input_tensor.shape[1],  # S
                input_tensor.shape[-1],  # D
                rope_dim,
            )

        return grad_x_flat.reshape(input_tensor.shape), None


# ---------------------------------------------------------------------------
# Public API — user-facing function
# ---------------------------------------------------------------------------


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_dim: int = 0,
) -> torch.Tensor:
    """Apply Rotary Positional Embedding via CUDA kernel.

    RoPE encodes absolute position information into each query/key vector
    by applying a position-dependent rotation in 2D subspaces. Tokens at
    different positions rotate by different angles, creating a rich
    positional signal that attention can exploit.

    Algorithm
    ---------
    For each (odd, even) dimension pair at position p:

      [x'_m,     x'_{m+1}] = [cos(p·θ_m), -sin(p·θ_m)] [x_m]
                            [sin(p·θ_m),  cos(p·θ_m)] [x_{m+1}]

    where θ_m = 10000^(-2m/d) is the frequency for pair m, and p
    is the token position. The frequency decreases geometrically across
    pairs, creating multi-scale positional encoding.

    Parameters
    ----------
    x : torch.Tensor, shape (B, S, H, D)
        Q or K tensor. B=batch, S=sequence, H=heads, D=dim_per_head.
    positions : torch.Tensor, shape (S,)
        Position indices for each sequence element (0 to S-1).
    rope_dim : int, optional
        Number of head dimensions to rotate. 0 = rotate all D dims.
        Must be divisible by 2.

    Returns
    -------
    torch.Tensor, shape (B, S, H, D)
        Rotated tensor, same shape as input.

    Notes
    -----
    - The kernel processes dimensions in (odd, even) pairs: (0,1), (2,3), ...
    - Each pair rotates at a different frequency determined by the pair index
    - Position p controls the rotation angle for that pair
    - The rotation preserves vector norms (orthogonal transformation)
    - Non-rotated dimensions (if rope_dim < D) pass through unchanged (future)

    Example
    -------
    >>> import torch
    >>> x = torch.randn(2, 8, 4, 16, device="cuda")  # (B, S, H, D)
    >>> positions = torch.arange(8, device="cuda")
    >>> y = apply_rope(x, positions)
    >>> y.shape
    torch.Size([2, 8, 4, 16])
    >>> torch.allclose(y.norm(dim=-1), x.norm(dim=-1))  # Norm preserved
    True

    Reference
    ---------
    Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """
    return _RoPECudaFunction.apply(x, positions)
