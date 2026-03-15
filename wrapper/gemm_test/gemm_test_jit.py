"""JIT compilation spec for gemm_test wrapper."""
import functools
import sys
from pathlib import Path

SUBDIR = Path(__file__).resolve().parent
WRAPPER_DIR = SUBDIR.parent
REPO_DIR = WRAPPER_DIR.parent

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from jit_utils import gen_jit_spec, sm100a_nvcc_flags


def gen_module():
    sources = [
        SUBDIR / "gemm_test.cu",
        SUBDIR / "gemm_test_binding.cu",
    ]
    extra_cuda_cflags = sm100a_nvcc_flags + [
        "--expt-relaxed-constexpr",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
    ]
    extra_include_paths = [
        REPO_DIR / "include",
        REPO_DIR / "csrc",
        WRAPPER_DIR,
    ]
    return gen_jit_spec(
        name="gemm_test",
        sources=[str(s) for s in sources],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(p) for p in extra_include_paths],
    )


@functools.cache
def get_gemm_test_module():
    return gen_module().build_and_load()
