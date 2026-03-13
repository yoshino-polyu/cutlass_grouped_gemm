"""JIT compilation spec for MX FP8 RC grouped GEMM wrapper."""
import functools
import sys
from pathlib import Path

WRAPPER_DIR = Path(__file__).resolve().parent
REPO_DIR = WRAPPER_DIR.parent

# Ensure repo root is on sys.path so `jit_utils` is importable
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from jit_utils import gen_jit_spec, sm100a_nvcc_flags


def gen_module():
    """Create a JitSpec for the MX FP8 RC grouped GEMM wrapper."""
    sources = [
        WRAPPER_DIR / "launcher.cu",
        WRAPPER_DIR / "binding.cu",
    ]
    extra_cuda_cflags = sm100a_nvcc_flags + [
        "--expt-relaxed-constexpr",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
    ]
    extra_include_paths = [
        REPO_DIR / "include",           # CUTLASS headers
        REPO_DIR / "csrc",              # tvm_ffi_utils.h (vendored)
        WRAPPER_DIR,                     # kernel_traits.cuh, preprocess_kernels.cuh
    ]
    return gen_jit_spec(
        name="mxfp8_rc_grouped_gemm",
        sources=[str(s) for s in sources],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(p) for p in extra_include_paths],
    )


@functools.cache
def get_module():
    """Build (if needed) and load the compiled module."""
    return gen_module().build_and_load()
