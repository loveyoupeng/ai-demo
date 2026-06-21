"""Comprehensive test suite for CUDA Python API on Jetson platform.

Validates the bare-metal CUDA driver API through cuda-python:
1. CUDA initialization and device management
2. NVRTC compilation pipeline (create -> compile -> PTX)
3. Module loading and kernel retrieval
4. Kernel launch with ctypes kernel parameters
5. Full pipeline integration
6. Compilation caching

Learning objectives:
- CUDA driver API initialization (cuInit, cuDeviceGet, cuCtxCreate)
- NVRTC compilation pipeline (nvrtcCreateProgram, nvrtcCompileProgram, nvrtcGetPTX)
- PTX-based module loading (cuModuleLoadData, cuModuleGetFunction)
- Kernel parameter passing with ctypes type resolution
- Grid-stride kernel launch configuration
"""

import ctypes
from pathlib import Path

import pytest
import torch
from cuda import cuda as _cuda_lib
from cuda import nvrtc as _nvrtc_lib

# ---------------------------------------------------------------------------
# CUDA kernel source - vector addition: c[i] = a[i] + b[i]
# ---------------------------------------------------------------------------

_KERNEL_SOURCE = """
__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_cuda_context() -> None:
    """Ensure a CUDA context is active before loading modules."""
    ret = _cuda_lib.cuInit(0)
    if ret[0] not in (_cuda_lib.CUresult.CUDA_SUCCESS,):
        raise RuntimeError(f"Failed to initialize CUDA driver: {ret}")

    current = _cuda_lib.cuCtxGetCurrent()
    ctx = current[1]
    if ctx is None or int(ctx) == 0:
        ret = _cuda_lib.cuDeviceGet(0)
        if ret[0] not in (_cuda_lib.CUresult.CUDA_SUCCESS,):
            raise RuntimeError(f"Failed to get CUDA device: {ret}")
        device = ret[1]
        try:
            ret = _cuda_lib.cuCtxCreate(0, device)
            if ret[0] not in (
                _cuda_lib.CUresult.CUDA_SUCCESS,
                _cuda_lib.CUresult.CUDA_ERROR_UNKNOWN,
            ):
                raise RuntimeError(f"Failed to create CUDA context: {ret}")
        except (OSError, RuntimeError):
            pass


def _nvrtc_compile(source: str, arch: str = "sm_87") -> tuple[int, bytes]:
    """Compile CUDA source with NVRTC and return program_id + PTX bytes.

    Parameters
    ----------
    source : str
        Raw CUDA C source code.
    arch : str
        GPU architecture flag (default: sm_87 for Orin).

    Returns
    -------
    tuple
        (program_id, ptx_data)
    """
    ret = _nvrtc_lib.nvrtcCreateProgram(source.encode("utf-8"), b"kernel.cu", 0, None, None)
    assert ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcCreateProgram failed: {ret}"
    prog_id = ret[1]

    compile_opts = [f"-arch={arch}".encode()]
    compile_ret = _nvrtc_lib.nvrtcCompileProgram(prog_id, len(compile_opts), compile_opts)
    assert compile_ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcCompileProgram failed: {compile_ret}"

    size_ret = _nvrtc_lib.nvrtcGetPTXSize(prog_id)
    assert size_ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcGetPTXSize failed: {size_ret}"
    size = size_ret[1]

    ptx_buf = bytearray(size)
    ptx_ret = _nvrtc_lib.nvrtcGetPTX(prog_id, ptx_buf)
    assert ptx_ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcGetPTX failed: {ptx_ret}"

    return prog_id, bytes(ptx_buf)


def _resolve_kernel_name(ptx: bytes, kernel_name: str) -> str:
    """Resolve C++ mangled kernel name from PTX source.

    Parameters
    ----------
    ptx : bytes
        PTX bytecode.
    kernel_name : str
        Human-readable kernel name.

    Returns
    -------
    str
        Mangled kernel name found in PTX.
    """
    pattern = ("_Z" + str(len(kernel_name)) + kernel_name).encode("utf-8")
    lowered = ptx.find(pattern)
    assert lowered != -1, f"Could not find kernel name pattern in PTX: {pattern}"
    end = lowered
    while end < len(ptx) and ptx[end : end + 1] not in (b"(", b")", b"\n", b" "):
        end += 1
    return ptx[lowered:end].decode("utf-8")


def _load_module_from_ptx(ptx_data: bytes) -> int:
    """Load PTX bytecode as a CUDA runtime module.

    Parameters
    ----------
    ptx_data : bytes
        PTX bytecode to load.

    Returns
    -------
    int
        Module handle.
    """
    ptx_buffer = ctypes.create_string_buffer(ptx_data)
    ptx_ptr = ctypes.addressof(ptx_buffer)
    ret = _cuda_lib.cuModuleLoadDataEx(ptx_ptr, 0, None, None)
    assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuModuleLoadDataEx failed: {ret}"
    return ret[1]


def _get_kernel_handle(module: int, lowered_name: str) -> int:
    """Get kernel handle from module.

    Parameters
    ----------
    module : int
        Module handle.
    lowered_name : str
        Mangled kernel name.

    Returns
    -------
    int
        Kernel handle.
    """
    ret = _cuda_lib.cuModuleGetFunction(module, lowered_name.encode("utf-8"))
    assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuModuleGetFunction failed: {ret}"
    return ret[1]


def _make_kernel_args(a, b, c, n) -> tuple:
    """Create kernel parameter tuple for cuda-python HelperKernelParams.

    HelperKernelParams requires a 2-tuple of (values, types) for kernels with
    parameters. This is the only format that works reliably on Jetson/L4T.

    Parameters
    ----------
    a, b, c : torch.Tensor or int
        Device pointers or integers.
    n : int
        Element count.

    Returns
    -------
    tuple
        (values_tuple, types_tuple) for cuLaunchKernel kernelParams param.
    """
    vals = (
        ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else ctypes.c_void_p(a),
        ctypes.c_void_p(b.data_ptr()) if isinstance(b, torch.Tensor) else ctypes.c_void_p(b),
        ctypes.c_void_p(c.data_ptr()) if isinstance(c, torch.Tensor) else ctypes.c_void_p(c),
        ctypes.c_int(n),
    )
    types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    return (vals, types)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCUDAInit:
    """Validate CUDA driver API initialization."""

    @pytest.mark.timeout(30)
    def test_cuInit(self):
        """cuInit(0) initializes the CUDA driver."""
        ret = _cuda_lib.cuInit(0)
        assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuInit failed: {ret}"

    @pytest.mark.timeout(30)
    def test_cuDeviceGet(self):
        """cuDeviceGet(0) returns a valid device handle."""
        _cuda_lib.cuInit(0)  # Must initialize before cuDeviceGet
        ret = _cuda_lib.cuDeviceGet(0)
        assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuDeviceGet failed: {ret}"
        assert ret[1] is not None, "Device handle is None"
        assert int(ret[1]) >= 0, f"Invalid device number: {int(ret[1])}"

    @pytest.mark.timeout(30)
    def test_cuCtxCreate(self):
        """cuCtxCreate creates a valid CUDA context."""
        _ensure_cuda_context()
        ret = _cuda_lib.cuDeviceGet(0)
        assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuDeviceGet failed: {ret}"
        device = ret[1]

        ret = _cuda_lib.cuCtxCreate(0, device)
        assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuCtxCreate failed: {ret}"
        ctx = ret[1]
        assert ctx is not None, "Context handle is None"
        assert int(ctx) != 0, "Context is invalid (zero)"


class TestNvrtcCompileOnly:
    """Validate NVRTC compilation pipeline."""

    @pytest.mark.timeout(60)
    def test_nvrtcCreateProgram(self):
        """nvrtcCreateProgram creates a valid program from CUDA source."""
        ret = _nvrtc_lib.nvrtcCreateProgram(_KERNEL_SOURCE.encode("utf-8"), b"kernel.cu", 0, None, None)
        assert ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcCreateProgram failed: {ret}"
        prog_id = ret[1]
        assert prog_id is not None, "Program ID is None"
        assert int(prog_id) != 0, "Program ID is zero (invalid)"

    @pytest.mark.timeout(60)
    def test_nvrtcCompileProgram(self):
        """nvrtcCompileProgram with --gpu-architecture=sm_87 compiles successfully."""
        ret = _nvrtc_lib.nvrtcCreateProgram(_KERNEL_SOURCE.encode("utf-8"), b"kernel.cu", 0, None, None)
        assert ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcCreateProgram failed: {ret}"
        prog_id = ret[1]

        try:
            arch_flag = b"-arch=sm_87"
            ret = _nvrtc_lib.nvrtcCompileProgram(prog_id, 1, [arch_flag])
            assert ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcCompileProgram failed: {ret}"
        finally:
            _nvrtc_lib.nvrtcDestroyProgram(prog_id)

    @pytest.mark.timeout(60)
    def test_nvrtcGetPTX(self):
        """nvrtcGetPTX returns valid PTX bytecode."""
        ret = _nvrtc_lib.nvrtcCreateProgram(_KERNEL_SOURCE.encode("utf-8"), b"kernel.cu", 0, None, None)
        assert ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcCreateProgram failed: {ret}"
        prog_id = ret[1]

        try:
            ret = _nvrtc_lib.nvrtcCompileProgram(prog_id, 1, [b"-arch=sm_87"])
            assert ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcCompileProgram failed: {ret}"

            size_ret = _nvrtc_lib.nvrtcGetPTXSize(prog_id)
            assert size_ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcGetPTXSize failed: {size_ret}"
            size = size_ret[1]

            ptx_buf = bytearray(size)
            ptx_ret = _nvrtc_lib.nvrtcGetPTX(prog_id, ptx_buf)
            assert ptx_ret[0] == _nvrtc_lib.nvrtcResult.NVRTC_SUCCESS, f"nvrtcGetPTX failed: {ptx_ret}"
            ptx = bytes(ptx_buf)

            assert isinstance(ptx, bytes), f"PTX is not bytes: {type(ptx)}"
            assert len(ptx) > 0, "PTX data is empty"

            ptx_str = ptx.decode("utf-8", errors="replace")
            assert ".version" in ptx_str, "PTX missing .version directive"
            assert "vec_add_kernel" in ptx_str, "PTX missing kernel name vec_add_kernel"
        finally:
            _nvrtc_lib.nvrtcDestroyProgram(prog_id)


class TestModuleLoad:
    """Validate module loading and kernel retrieval.

    Note: Tests in this class require PyTorch CUDA initialization first
    because cuModuleLoadDataEx on this platform needs the driver state
    set up by PyTorch context creation.
    """

    @pytest.mark.timeout(60)
    def test_module_load_data(self):
        """cuModuleLoadData loads PTX as a runtime module."""
        torch.randn(1, dtype=torch.float32, device="cuda")
        _ensure_cuda_context()

        _, ptx = _nvrtc_compile(_KERNEL_SOURCE, "sm_87")

        # Keep buffer reference alive to prevent GC during cuModuleLoadDataEx
        ptx_buf = ctypes.create_string_buffer(ptx)
        module_ret = _cuda_lib.cuModuleLoadDataEx(ctypes.addressof(ptx_buf), 0, None, None)
        assert module_ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuModuleLoadDataEx failed: {module_ret}"
        module = module_ret[1]
        assert module is not None, "Module handle is None"
        assert int(module) != 0, "Module handle is zero (invalid)"

    @pytest.mark.timeout(60)
    def test_module_get_function(self):
        """cuModuleGetFunction retrieves a valid kernel handle."""
        torch.randn(1, dtype=torch.float32, device="cuda")
        _ensure_cuda_context()

        # Full pipeline: compile -> load -> get kernel
        prog_id, ptx = _nvrtc_compile(_KERNEL_SOURCE, "sm_87")
        module = _load_module_from_ptx(ptx)

        lowered_name = _resolve_kernel_name(ptx, "vec_add_kernel")
        kernel = _get_kernel_handle(module, lowered_name)

        print(f"Kernel handle: {kernel} (int value: {int(kernel)})")
        assert kernel is not None, "Kernel handle is None"
        assert int(kernel) != 0, "Kernel handle is zero (invalid)"

    @pytest.mark.timeout(60)
    def test_kernel_handle_print(self):
        """Print and verify kernel handle is not None."""
        torch.randn(1, dtype=torch.float32, device="cuda")
        _ensure_cuda_context()

        prog_id, ptx = _nvrtc_compile(_KERNEL_SOURCE, "sm_87")
        module = _load_module_from_ptx(ptx)

        lowered_name = _resolve_kernel_name(ptx, "vec_add_kernel")
        kernel = _get_kernel_handle(module, lowered_name)

        print(f"Kernel handle: {kernel} (int value: {int(kernel)})")
        assert kernel is not None, "Kernel handle should not be None"
        assert int(kernel) != 0, "Kernel handle should be non-zero"


class TestKernelLaunch:
    """Validate kernel launch with ctypes kernel parameters.

    This is the critical test that verifies bare-metal kernel execution
    on the Jetson platform using only CUDA driver API (no torch.cuda functions).
    """

    @pytest.mark.timeout(90)
    def test_kernel_launch_and_verify(self):
        """Launch vec_add_kernel (via ptx) -> verify c == a + b."""
        # Initialize PyTorch CUDA context first (required for module loading)
        torch.randn(1, dtype=torch.float32, device="cuda")
        _ensure_cuda_context()

        # Compile NVRTC program
        prog_id, ptx = _nvrtc_compile(_KERNEL_SOURCE, "sm_87")
        assert int(prog_id) != 0, f"NVRTC program is invalid: {prog_id}"

        # Load PTX as module
        module = _load_module_from_ptx(ptx)

        # Get kernel handle
        lowered_name = _resolve_kernel_name(ptx, "vec_add_kernel")
        ret = _cuda_lib.cuModuleGetFunction(module, lowered_name.encode("utf-8"))
        assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuModuleGetFunction failed: {ret}"
        kernel = ret[1]

        # Create stream - required for cuLaunchKernel on this platform
        ret = _cuda_lib.cuStreamCreate(0)
        assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuStreamCreate failed: {ret}"
        stream = ret[1]

        # Create test data on GPU
        a = torch.randn(1024, dtype=torch.float32, device="cuda")
        b = torch.randn(1024, dtype=torch.float32, device="cuda")
        c = torch.empty_like(a)

        # Use tuple format for kernelParams - required by cuda-python HelperKernelParams
        # Format: ((value1, value2, ...), (type1, type2, ...))
        kernel_args = _make_kernel_args(a, b, c, 1024)

        # Launch: grid = (4, 1, 1), block = (256, 1, 1) = 1024 total threads
        # Note: extra must be integer (0) not None on this platform
        status = _cuda_lib.cuLaunchKernel(
            kernel,
            4,
            1,
            1,
            256,
            1,
            1,
            0,
            stream,
            kernel_args,
            0,
        )

        # Clean up NVRTC program
        _nvrtc_lib.nvrtcDestroyProgram(prog_id)

        # Verify the launch status
        assert status[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuLaunchKernel returned: {status}"

        # Synchronize and verify results
        torch.cuda.synchronize()

        expected = a + b
        torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)


class TestKernelLaunchWithNvrtc:
    """Full pipeline: nvrtc compile -> module load -> get function -> launch -> verify."""

    @pytest.mark.timeout(120)
    def test_full_compile_to_launch(self):
        """Complete pipeline: kernel source -> compile -> load -> launch -> verify."""
        # Initialize PyTorch CUDA context first
        torch.randn(1, dtype=torch.float32, device="cuda")
        _ensure_cuda_context()

        # Step 1: NVRTC compile
        prog_id, ptx = _nvrtc_compile(_KERNEL_SOURCE, "sm_87")
        assert int(prog_id) != 0, "NVRTC program should be non-zero"
        assert isinstance(ptx, bytes) and len(ptx) > 0, "PTX should be valid bytes"

        # Step 2: Load module from PTX
        module = _load_module_from_ptx(ptx)
        assert int(module) != 0, "Module handle should be non-zero"

        # Step 3: Get kernel handle
        lowered_name = _resolve_kernel_name(ptx, "vec_add_kernel")
        ret = _cuda_lib.cuModuleGetFunction(module, lowered_name.encode("utf-8"))
        assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"Get function failed: {ret}"
        kernel = ret[1]
        assert int(kernel) != 0, "Kernel handle should be non-zero"

        # Create stream - required for cuLaunchKernel on this platform
        ret = _cuda_lib.cuStreamCreate(0)
        assert ret[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"cuStreamCreate failed: {ret}"
        stream = ret[1]

        # Step 4: Launch kernel
        a = torch.randn(1024, dtype=torch.float32, device="cuda")
        b = torch.randn(1024, dtype=torch.float32, device="cuda")
        c = torch.empty_like(a)

        # Use tuple format for kernelParams - required by cuda-python HelperKernelParams
        kernel_args = _make_kernel_args(a, b, c, 1024)

        status = _cuda_lib.cuLaunchKernel(
            kernel,
            4,
            1,
            1,
            256,
            1,
            1,
            0,
            stream,
            kernel_args,
            0,
        )

        # Clean up NVRTC program
        _nvrtc_lib.nvrtcDestroyProgram(prog_id)

        assert status[0] == _cuda_lib.CUresult.CUDA_SUCCESS, f"Launch failed: {status}"
        torch.cuda.synchronize()

        # Step 5: Verify result
        expected = a + b
        torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)


class TestCaching:
    """Validate compilation caching behavior."""

    _CACHE_DIR = Path(__file__).parent / ".cache"
    _TEST_SOURCE = _KERNEL_SOURCE

    @pytest.mark.timeout(90)
    def test_compile_and_cache_file(self):
        """First compilation creates cache file on disk."""
        torch.randn(1, dtype=torch.float32, device="cuda")
        _ensure_cuda_context()

        # Remove existing cache to ensure fresh compile
        cache_path = self._CACHE_DIR / "test_cache.ptx"
        if cache_path.exists():
            cache_path.unlink()

        # Compile fresh
        prog_id, ptx = _nvrtc_compile(self._TEST_SOURCE, "sm_87")
        assert prog_id is not None

        # Load module
        _load_module_from_ptx(ptx)

        # Create cache file to test caching
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_bytes(ptx)

        # Verify cache exists and has content
        assert cache_path.exists(), "Cache file should exist on disk"
        cached_data = cache_path.read_bytes()
        assert len(cached_data) > 0, "Cached PTX should have content"
        assert cached_data[:20] == ptx[:20], "Cached PTX should match compiled PTX"

    @pytest.mark.timeout(90)
    def test_second_call_uses_cache(self):
        """Second load from cache returns valid module."""
        torch.randn(1, dtype=torch.float32, device="cuda")
        _ensure_cuda_context()

        cache_path = self._CACHE_DIR / "test_cache.ptx"
        cache_path.parent.mkdir(exist_ok=True)

        # Get a valid cache file (from previous test or create one)
        if not cache_path.exists():
            _, ptx = _nvrtc_compile(self._TEST_SOURCE, "sm_87")
            cache_path.write_bytes(ptx)

        # Load from cache
        cached_ptx = cache_path.read_bytes()
        assert len(cached_ptx) > 0, "Cached PTX should be non-empty"

        module = _load_module_from_ptx(cached_ptx)
        assert module is not None
        assert int(module) != 0
