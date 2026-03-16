"""JIT compilation spec for block_A layout parity check."""
import functools
import sys
from pathlib import Path

SUBDIR = Path(__file__).resolve().parent
WRAPPER_DIR = SUBDIR.parent
REPO_DIR = WRAPPER_DIR.parent

# Ensure repo root is on sys.path so `jit_utils` is importable
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from jit_utils import gen_jit_spec, sm100a_nvcc_flags


def gen_module():
    """Create a JitSpec for the layout check module."""
    sources = [
        SUBDIR / "layout_check.cu",
        SUBDIR / "layout_check_binding.cu",
        SUBDIR / "sfa_check.cu",
        SUBDIR / "sfa_check_binding.cu",
        SUBDIR / "gemm_check.cu",
        SUBDIR / "gemm_check_binding.cu",
    ]
    extra_cuda_cflags = sm100a_nvcc_flags + [
        "--expt-relaxed-constexpr",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
    ]
    extra_include_paths = [
        REPO_DIR / "include",           # CUTLASS headers
        REPO_DIR / "csrc",              # tvm_ffi_utils.h
        WRAPPER_DIR,
    ]
    return gen_jit_spec(
        name="layout_check",
        sources=[str(s) for s in sources],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(p) for p in extra_include_paths],
    )


@functools.cache
def get_layout_check_module():
    """Build (if needed) and load the layout check module."""
    return gen_module().build_and_load()
