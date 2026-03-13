"""JIT environment paths — simplified from FlashInfer.

Strips out AOT, cubin, jit-cache, and FlashInfer-version-based paths.
Points CUTLASS include dirs at the local repo's include/ tree.
"""

import os
import pathlib

from .compilation_context import CompilationContext

# Repo root = parent of this file's directory
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Cache directory for compiled .so files
_BASE_DIR = pathlib.Path(
    os.getenv("FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)
_CACHE_DIR = _BASE_DIR / ".cache" / "flashinfer"


def _get_workspace_dir_name() -> pathlib.Path:
    ctx = CompilationContext()
    arch = "_".join(
        f"{major}{minor}"
        for major, minor in sorted(ctx.TARGET_CUDA_ARCHS)
    )
    return _CACHE_DIR / "sm100_grouped_gemm" / arch


FLASHINFER_WORKSPACE_DIR: pathlib.Path = _get_workspace_dir_name()
FLASHINFER_JIT_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "cached_ops"

# Include dirs — point at *our* repo's CUTLASS/CuTe headers, not FlashInfer's
CUTLASS_INCLUDE_DIRS: list[pathlib.Path] = [
    _REPO_ROOT / "include",
]

# csrc dir for tvm_ffi_utils.h — we vendor it locally
FLASHINFER_CSRC_DIR: pathlib.Path = _REPO_ROOT / "csrc"
FLASHINFER_INCLUDE_DIR: pathlib.Path = _REPO_ROOT / "include"
