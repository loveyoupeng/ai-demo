"""CUDA bare-metal compiler — compiles .cu source to runtime CUDA modules.

This module handles the full compilation pipeline for CUDA C kernels:
1. Read .cu source file
2. Compile with nvrtc (NVIDIA Runtime Compilation)
3. Load compiled PTX as a CUDA runtime module

All compiled modules are cached to avoid recompilation overhead.
The cache lives in impl/_cuda/.cache/ directory.

This is the foundation for all CUDA kernels in this project — every
kernel goes through the same compile -> load -> launch pipeline.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from typing import Any

# CUDA runtime and compilation libraries
from cuda import cuda as _cuda_lib
from cuda import nvrtc as _nvrtc_lib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_DIR.mkdir(exist_ok=True)

# Compile option for compute capability 8.0 (Orin GPUs)
_ARCH_FLAG = b"-arch=compute_80"

# Compile options shared across all kernels
_COMPILE_OPTIONS = [_ARCH_FLAG]


# ---------------------------------------------------------------------------
# Cached compilation
# ---------------------------------------------------------------------------


def _cache_key(source: bytes) -> str:
    """Generate a cache filename from the source code hash.

    Parameters
    ----------
    source : bytes
        Raw CUDA C source code.

    Returns
    -------
    str
        Cache filename (hex hash).
    """
    combined = source + b"|".join(_COMPILE_OPTIONS)
    return hashlib.sha256(combined).hexdigest()[:16]


def _ensure_cuda_context() -> None:
    """Ensure a CUDA context is active before loading modules.

    The CUDA driver requires an active context to load and manage modules.
    When using cuda-python alongside PyTorch, PyTorch creates a context but
    cuda-python may not see it as the current thread's context. We need to
    explicitly create/bind a context.

    This function is idempotent with respect to CUDA driver initialization:
    cuInit can be called once, but cuCtxCreate may fail if a context already
    exists (which is fine — we just skip).
    """
    status = _cuda_lib.cuInit(0)
    if status[0] not in (
        _cuda_lib.CUresult.CUDA_SUCCESS,
    ):
        raise RuntimeError(f"Failed to initialize CUDA driver: {status}")

    # Try to get current context — 0x0 context is not valid
    current = _cuda_lib.cuCtxGetCurrent()
    current_ctx = current[1]
    # int() gives the raw address value; 0 means no real context
    if current_ctx is not None and int(current_ctx) != 0:
        return  # Valid context already exists

    # No valid context — create one on device 0
    status, device = _cuda_lib.cuDeviceGet(0)
    if status != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA device: {status}")

    try:
        status, _ = _cuda_lib.cuCtxCreate(0, device)
        if status != _cuda_lib.CUresult.CUDA_SUCCESS:
            if status == _cuda_lib.CUresult.CUDA_ERROR_UNKNOWN:
                # Context might already exist — this is OK
                return
            raise RuntimeError(f"Failed to create CUDA context: {status}")
    except (OSError, RuntimeError):
        # If context creation fails because one exists, that's fine
        pass


_PTCL_BUFFER: bytes | None = None


def _load_module_from_ptx(ptx_data: bytes) -> Any:
    """Load PTX bytecode as a CUDA runtime module.

    Parameters
    ----------
    ptx_data : bytes
        PTX bytecode to load.

    Returns
    -------
    Any
        CUDA module handle.
    """
    # Ensure a CUDA context is active before loading modules
    _ensure_cuda_context()

    # Keep the buffer alive to prevent GC from freeing memory
    global _PTCL_BUFFER
    _PTCL_BUFFER = ptx_data

    # Create a C string buffer from PTX data and pass as raw pointer
    ptx_buffer = ctypes.create_string_buffer(ptx_data)
    ptx_ptr = ctypes.addressof(ptx_buffer)

    # Keep buffer alive
    status, module = _cuda_lib.cuModuleLoadDataEx(ptx_ptr, 0, None, None)
    if status != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to load CUDA module: {status}")

    return module


def _compile_and_cache(source: bytes, key: str) -> tuple[Any, bytes]:
    """Compile CUDA source with nvrtc and cache the result.

    Compilation pipeline:
    1. nvrtcCreateProgram — create compilation unit
    2. nvrtcCompileProgram — compile with options
    3. nvrtcGetPTX — retrieve PTX bytecode
    4. Load and cache PTX

    Parameters
    ----------
    source : bytes
        Raw CUDA C source code.
    key : str
        Cache filename.

    Returns
    -------
    tuple
        (loaded_module, ptx_bytes)
    """
    # Create NVRTC program
    status, prog = _nvrtc_lib.nvrtcCreateProgram(source, b"kernel", 0, None, None)
    if status != _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram failed: {status}")

    try:
        # Compile with options — returns (status,)
        compile_status = _nvrtc_lib.nvrtcCompileProgram(
            prog, len(_COMPILE_OPTIONS), _COMPILE_OPTIONS
        )
        if compile_status[0] != _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS:
            # Get log for debugging
            log_size_status, log_size = _nvrtc_lib.nvrtcGetProgramLogSize(prog)
            log_buf = bytearray(log_size + 1)
            _nvrtc_lib.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(
                f"nvrtcCompileProgram failed:\n{bytes(log_buf).decode('utf-8', errors='ignore')}"
            )

        # Get PTX size and data
        ptx_size_status, ptx_size = _nvrtc_lib.nvrtcGetPTXSize(prog)
        ptx_buf = bytearray(ptx_size)
        ptx_status = _nvrtc_lib.nvrtcGetPTX(prog, ptx_buf)
        if ptx_status[0] != _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcGetPTX failed: {ptx_status}")

        ptx_data = bytes(ptx_buf)

        # Cache the PTX
        _CACHE_DIR.mkdir(exist_ok=True)
        cache_path = _CACHE_DIR / f"{key}.ptx"
        with open(cache_path, "wb") as f:
            f.write(ptx_data)

        # Load as CUDA module
        module = _load_module_from_ptx(ptx_data)
        return module, ptx_data

    finally:
        _nvrtc_lib.nvrtcDestroyProgram(prog)


def _load_cached(key: str) -> tuple[Any, bytes]:
    """Load a pre-compiled PTX from cache.

    Parameters
    ----------
    key : str
        Cache filename.

    Returns
    -------
    tuple
        (loaded_module, ptx_bytes)
    """
    cache_path = _CACHE_DIR / f"{key}.ptx"
    with open(cache_path, "rb") as f:
        ptx_data = f.read()
    module = _load_module_from_ptx(ptx_data)
    return module, ptx_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_and_load(source: str | bytes) -> tuple[Any, bytes]:
    """Compile CUDA C source and load as a runtime CUDA module.

    This is the main entry point for the CUDA compilation pipeline:
    1. Compile the CUDA source with nvrtc
    2. Load the resulting module (with cache lookup)

    Parameters
    ----------
    source : str | bytes
        CUDA C source code as a string or bytes.

    Returns
    -------
    tuple
        (module, ptx_bytes) — loaded CUDA module and its PTX bytecode.
    """
    source_bytes = source.encode("utf-8") if isinstance(source, str) else source
    key = _cache_key(source_bytes)

    # Check cache first
    cache_path = _CACHE_DIR / f"{key}.ptx"
    if cache_path.exists():
        try:
            return _load_cached(key)
        except OSError:
            pass

    # Compile and cache
    return _compile_and_cache(source_bytes, key)


def get_kernel_handle(module: Any, kernel_name: str, ptx_data: bytes) -> Any:
    """Retrieve a kernel handle from a loaded CUDA module.

    Because newer PTX versions mangle kernel names (C++ name mangling),
    we need to resolve the unmangled name to the mangled name using
    the PTX source.

    Parameters
    ----------
    module : Any
        CUDA module handle loaded via compile_and_load.
    kernel_name : str
        Unmangled name of the kernel as written in the source.
    ptx_data : bytes
        Raw PTX bytecode for name resolution.

    Returns
    -------
    Any
        Kernel function handle for cuLaunchKernel.
    """
    # Extract the lowered (mangled) name from PTX
    # Look for: .entry _Z{len}kernel_name{mangling}
    pattern = f"_Z{len(kernel_name)}{kernel_name}".encode()
    lowered = ptx_data.find(pattern)
    if lowered == -1:
        raise RuntimeError(
            f"Could not find kernel '{kernel_name}' in PTX source. "
            f"Looking for pattern: {pattern}"
        )
    # Extract until ')' or '(' — that's the full lowered name
    end = lowered
    while end < len(ptx_data) and ptx_data[end : end + 1] not in (b"(", b")", b"\n", b" "):
        end += 1
    lowered_name = ptx_data[lowered:end].decode("utf-8")

    status, kernel = _cuda_lib.cuModuleGetFunction(module, lowered_name.encode("utf-8"))
    if status != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleGetFunction failed: {status}")
    return kernel


def launch_kernel(
    kernel: Any,
    grid_x: int,
    grid_y: int = 1,
    grid_z: int = 1,
    block_x: int = 256,
    block_y: int = 1,
    block_z: int = 1,
    shared_mem: int = 0,
    stream: Any = None,
    kernel_params: list[Any] | None = None,
) -> None:
    """Launch a compiled CUDA kernel with the given configuration.

    Parameters
    ----------
    kernel : Any
        Kernel handle from cuModuleGetFunction.
    grid_x : int
        Number of blocks in x dimension.
    grid_y : int
        Number of blocks in y dimension.
    grid_z : int
        Number of blocks in z dimension.
    block_x : int
        Number of threads per block in x dimension.
    block_y : int
        Number of threads per block in y dimension.
    block_z : int
        Number of threads per block in z dimension.
    shared_mem : int
        Dynamic shared memory size in bytes.
    stream : Any
        CUDA stream for asynchronous execution. None for default.
    kernel_params : list or None
        Array of pointers to kernel parameters. Integers or ctypes pointers.
    """
    params = list(kernel_params) if kernel_params else []

    # Create a ctypes c_void_p array from the params list
    if params:
        param_array_type = ctypes.c_void_p * len(params)
        ctypes_params = param_array_type(*[ctypes.c_void_p(p) for p in params])
    else:
        ctypes_params = None

    status = _cuda_lib.cuLaunchKernel(
        kernel,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_mem,
        stream,
        ctypes_params,
        None,
    )
    if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {status}")
