"""SiLU activation — nvrtc compiled + PyTorch dispatch.

This is the first CUDA kernel in the project. It demonstrates the complete
nVRTC compilation pipeline:

1. Write CUDA C kernel (.cu source)
2. Compile at runtime with nvrtc → PTX bytecode
3. Load PTX as a CUDA runtime module
4. Extract kernel handle via cuModuleGetFunction
5. Launch kernel via cuLaunchKernel with PyTorch tensor pointers

The backward pass computes gradients using PyTorch's built-in operations,
showing both the CUDA kernel dispatch pattern and the hybrid approach.

Learning objectives:
- nVRTC compilation pipeline (create → compile → get PTX)
- PTX-based module loading and kernel handle extraction
- PyTorch tensor → device pointer → kernel parameter passing
- Grid-stride loop pattern for arbitrary input sizes
- Backward pass computation (gradient of SiLU)
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

_KERNEL_SOURCE_PATH = (
    __file__.rsplit("/", 1)[0]  # directory of this file
    + "/kernels/activation.cu"  # kernel source file
)


def _load_kernel_source() -> str:
    """Load CUDA kernel source from file. Used for nvrtc compilation.

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


class _SiluKernels:
    """Compile and cache SiLU forward and backward kernels.

    This class follows the compilation pipeline:
    1. Load CUDA C source from file
    2. Compile with nvrtc → PTX bytecode
    3. Load PTX as runtime CUDA module
    4. Extract kernel handles via cuModuleGetFunction
    5. Cache handles (kernel handles are valid for the lifetime of the program)

    The compilation happens only once (singletons cached on first access).
    All subsequent calls reuse the compiled handles.

    Module lifetime
    ---------------
    The CUDA module stays loaded for the entire program lifetime. Kernel
    handles are valid until the module is unloaded (which we never do).
    This is the correct pattern for production use — avoid recompiling
    on every call.
    """

    _forward_kernel = None
    _forward_f64_kernel = None
    _backward_kernel = None
    _backward_f64_kernel = None
    _module = None
    _ptx_data = None

    @classmethod
    def compile(cls) -> None:
        """Compile CUDA source and store forward and backward kernel handles.

        This follows the three-step compilation pipeline:
        1. nvrtcCreateProgram + nvrtcCompileProgram → PTX bytecode
        2. cuModuleLoadDataEx → CUDA runtime module
        3. cuModuleGetFunction → kernel function handles

        The handles are stored in cls._forward_kernel and cls._backward_kernel
        and cached for the lifetime of the process.
        """
        source = _load_kernel_source()
        module, ptx_data = compile_and_load(source)
        cls._module = module
        cls._ptx_data = ptx_data

        # Extract kernel handles (handles are valid until module is unloaded)
        cls._forward_kernel = get_kernel_handle(module, "silu_forward_kernel", ptx_data)
        cls._forward_f64_kernel = get_kernel_handle(module, "silu_forward_f64_kernel", ptx_data)
        cls._backward_kernel = get_kernel_handle(module, "silu_backward_kernel", ptx_data)
        cls._backward_f64_kernel = get_kernel_handle(module, "silu_backward_f64_kernel", ptx_data)

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
# Kernel launcher — PyTorch tensors → device pointers → kernel launch
# ---------------------------------------------------------------------------


def _ensure_cuda_context() -> None:
    """Ensure a CUDA context is active before launching kernels.

    The CUDA driver requires an active context for module loading and
    kernel launching. PyTorch creates one on first GPU use, but we
    must ensure it exists before calling cuLaunchKernel.

    This is a one-time initialization — cuInit is safe to call multiple
    times (it initializes the driver if not already initialized).
    """
    status = _cuda_lib.cuInit(0)
    if status[0] not in (
        _cuda_lib.CUresult.CUDA_SUCCESS,
    ):
        raise RuntimeError(f"Failed to initialize CUDA driver: {status}")


def _resolve_kernel_param(p: Any) -> tuple[int, Any]:
    """Resolve a single kernel parameter to (kind, value) for cuda-python.

    Parameters
    ----------
    p : Any
        Raw parameter: tensor, int, ctypes wrapper, or None.

    Returns
    -------
    tuple
        (kind, value) — kind is 0 for pointer, 1 for int, 2 for size_t,
        3 for bytes, -1 for error. Value is the resolved ctypes object.
    """
    if isinstance(p, torch.Tensor):
        return (0, ctypes.c_void_p(p.data_ptr()))
    if isinstance(p, ctypes.c_void_p):
        return (0, p)
    if isinstance(p, ctypes.c_int):
        return (1, p)
    if isinstance(p, int):
        return (1, ctypes.c_int(p))
    if p is None:
        return (0, ctypes.c_void_p(0))
    if isinstance(p, ctypes.c_size_t):
        return (2, p)
    if isinstance(p, (bytes, bytearray)):
        return (3, ctypes.addressof(ctypes.create_string_buffer(p if isinstance(p, bytes) else bytes(p))))
    raise TypeError(
        f"Unsupported param type: {type(p)} — expected torch.Tensor, int, ctypes.c_void_p, or None"
    )


def _launch_kernel(kernel: Any, params: list, grid_x: int, block_x: int = 256) -> None:
    """Launch a compiled CUDA kernel using the CUDA driver API.

    This is the core kernel launch mechanism that bridges Python/PyTorch
    with the CUDA driver API. It handles:
    - Converting torch tensor .data_ptr() to device pointers (c_void_p)
    - Passing parameter pointers via (values, types) tuple — required by cuda-python
    - Grid and block dimension configuration
    - Stream creation for this platform (required on Jetson/L4T)

    Parameters
    ----------
    kernel : Any
        Kernel handle from cuModuleGetFunction.
    params : list
        List of torch tensors, integers, or None values to pass as kernel parameters.
        For torch tensors, their .data_ptr() is used as the device pointer.
        For integers (like size), they are cast to int directly.
    grid_x : int
        Number of blocks in x dimension (typically num_elements / block_size).
    block_x : int
        Number of threads per block (default 256).

    Example
    -------
    >>> x = torch.randn(1024, dtype=torch.float32, device="cuda")
    >>> y = torch.empty_like(x)
    >>> _launch_kernel(forward_kernel, [x, y, 1024], grid_x=4, block_x=256)
    """
    # Build parameter tuple: ((values...), (types...))
    values, types = [], []
    for p in params:
        kind, resolved = _resolve_kernel_param(p)
        if kind == 0:
            types.append(ctypes.c_void_p)
        elif kind == 1:
            types.append(ctypes.c_int)
        elif kind == 2:
            types.append(ctypes.c_size_t)
        elif kind == 3:
            types.append(ctypes.c_void_p)
        values.append(resolved)

    # Create stream — required on this platform for cuLaunchKernel to work
    stream_ret = _cuda_lib.cuStreamCreate(0)
    if stream_ret[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to create CUDA stream: {stream_ret}")
    stream = stream_ret[1]

    # Build the kernel arguments tuple for cuda-python's HelperKernelParams
    kernel_args = (tuple(values), tuple(types))

    try:
        # Launch the kernel — Note: extra must be 0 (not None) on this platform
        status = _cuda_lib.cuLaunchKernel(
            kernel,
            grid_x,  # grid x — 1D grid with one block per element chunk
            1,       # grid y — single row of blocks (1D execution)
            1,       # grid z — single layer of blocks
            block_x, # block x — 256 threads per block
            1,       # block y — 1 thread dimension (1D thread block)
            1,       # block z — 1 thread dimension
            0,       # shared memory size — none for this kernel
            stream,  # stream handle (not None — required on this platform)
            kernel_args,  # (values, types) tuple — required by cuda-python HelperKernelParams
            0,       # extra launch attributes — must be 0, NOT None on Jetson/L4T
        )
        if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel failed: {status}")
    finally:
        # Clean up stream
        _cuda_lib.cuStreamDestroy(stream)


# ---------------------------------------------------------------------------
# Custom autograd function — forward in CUDA, backward in PyTorch
# ---------------------------------------------------------------------------


class _SiluCudaFunction(torch.autograd.Function):
    """SiLU activation forward pass in CUDA, backward pass in PyTorch.

    This follows the hybrid pattern recommended for the platform:
    - Forward pass: manual CUDA kernel launch (learning the bare-metal API)
    - Backward pass: PyTorch autograd (automatic, correct gradients)

    The forward pass demonstrates:
    - Loading compiled kernels from nvrtc
    - Grid-stride loop configuration (num_elements / block_size)
    - Passing torch tensor pointers to kernel parameters
    - Memory layout: row-major, contiguous tensors

    The backward pass demonstrates:
    - How PyTorch builds computation graphs
    - Gradient computation as element-wise operations
    - Integration of manual CUDA with framework autograd

    Usage
    -----
    >>> x = torch.randn(1024, device="cuda")
    >>> y = _SiluCudaFunction.apply(x)
    >>> y.sum().backward()  # gradients flow through PyTorch
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        """Forward pass: output = input * sigmoid(input).

        Parameters
        ----------
        ctx : Any
            Autograd context — stores tensors needed for backward pass.
        input : torch.Tensor
            Input tensor on device (any shape, any dtype float).

        Returns
        -------
        torch.Tensor
            Output tensor with same shape and dtype as input.

        CUDA execution
        --------------
        1. Compute grid size: ceil(num_elements / block_size)
        2. Launch silu_forward_kernel (f32) or silu_forward_f64_kernel (f64)
        3. Each thread processes one element (grid-stride loop)
        """
        # Save input for backward pass (autograd context is the right place)
        ctx.save_for_backward(input)

        # Create output tensor (PyTorch manages its memory)
        output = torch.empty_like(input)
        is_float64 = input.dtype == torch.float64

        # Calculate grid dimensions (1D grid, each element gets one thread)
        num_elements = input.numel()
        block_size = 256  # 256 threads per block (standard size)
        grid_size = (num_elements + block_size - 1) // block_size  # ceiling division

        # Select kernel by dtype: float32 or float64
        if is_float64:
            # Float64: kernel expects const double*, double*, int
            input_ptr = ctypes.c_void_p(input.data_ptr())
            output_ptr = ctypes.c_void_p(output.data_ptr())
            _launch_kernel(
                _SiluKernels.get_forward_f64_kernel(),
                [input_ptr, output_ptr, ctypes.c_int(num_elements)],
                grid_x=grid_size,
                block_x=block_size,
            )
        else:
            _launch_kernel(
                _SiluKernels.get_forward_kernel(),
                [input, output, num_elements],
                grid_x=grid_size,
                block_x=block_size,
            )

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, ...]:
        """Backward pass: grad_input = grad_output * d/dx(SiLU(x)).

        This demonstrates PyTorch autograd integration. The backward
        function receives the gradient from the loss and computes the
        gradient with respect to the input.

        The gradient of SiLU(x) is:
            d/dx SiLU(x) = sigmoid(x) + SiLU(x) * (1 - sigmoid(x))
                         = sigmoid(x) * (1 + SiLU(x))

        Parameters
        ----------
        ctx : Any
            Autograd context containing saved input tensor.
        grad_output : torch.Tensor
            Gradient from the loss (same shape as forward output).

        Returns
        -------
        torch.Tensor
            Gradient with respect to input (same shape as input).
        """
        input, = ctx.saved_tensors  # Retrieve saved input (gradients not computed here)
        grad_output = grad_outputs[0]  # Get the gradient output

        # Compute gradient using PyTorch operations (automatic differentiation)
        # This follows the hybrid approach: CUDA for forward, PyTorch for backward
        sigmoid = 1.0 / (1.0 + torch.exp(-input))
        silu = input * sigmoid
        grad_input = grad_output * (sigmoid + silu * (1.0 - sigmoid))

        return grad_input  # Only input has requires_grad


# ---------------------------------------------------------------------------
# Public API — user-facing function
# ---------------------------------------------------------------------------


def silu(x: torch.Tensor) -> torch.Tensor:
    """Compute SiLU (Swish) activation: output = x * sigmoid(x).

    This is the first CUDA kernel in the project. It implements:

    SiLU(x) = x / (1 + exp(-x))

    For large positive x: SiLU(x) ≈ x (near-identity)
    For large negative x: SiLU(x) ≈ 0 (suppressed)
    For x = 0: SiLU(0) = 0

    The implementation uses nvrtc-compiled CUDA C for the forward pass
    and PyTorch autograd for the backward pass. This demonstrates both:
    - Manual CUDA kernel launch with device pointer passing
    - Integration with framework automatic differentiation

    Parameters
    ----------
    x : torch.Tensor
        Input tensor on CUDA device. Shape can be any size (1D, 2D, 3D, etc.).

    Returns
    -------
    torch.Tensor
        Output tensor with same shape and dtype as input.

    Example
    -------
    >>> x = torch.randn(1024, device="cuda")
    >>> y = silu(x)  # Forward pass in CUDA
    >>> y.sum().backward()  # Backward pass in PyTorch

    CUDA kernel configuration
    -------------------------
    - Block size: 256 threads/block
    - Grid size: ceil(num_elements / 256)
    - Memory access: coalesced (contiguous tensor, consecutive threads)
    - No shared memory (element-wise operation, no cross-thread communication)
    """
    return _SiluCudaFunction.apply(x)
