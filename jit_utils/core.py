"""JitSpec and gen_jit_spec — extracted from FlashInfer.

Stripped: AOT paths, cubin loader, FlashInfer-specific logger, registry.
Kept: JitSpec (build/load via ninja + tvm_ffi), gen_jit_spec, nvcc flag presets.
"""

import dataclasses
import os
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Union

import tvm_ffi
from filelock import FileLock

from .compilation_context import CompilationContext
from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

os.makedirs(jit_env.FLASHINFER_WORKSPACE_DIR, exist_ok=True)

# ---- NVCC flag presets ----
common_nvcc_flags = [
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
]
sm100a_nvcc_flags = ["-gencode=arch=compute_100a,code=sm_100a"] + common_nvcc_flags

current_compilation_context = CompilationContext()


def _check_cuda_arch():
    eligible = False
    for major, minor in current_compilation_context.TARGET_CUDA_ARCHS:
        if major >= 8:
            eligible = True
        elif major == 7 and minor.isdigit() and int(minor) >= 5:
            eligible = True
    if not eligible:
        raise RuntimeError("Requires GPUs with sm75 or higher")


def _get_tmpdir() -> Path:
    tmpdir = jit_env.FLASHINFER_JIT_DIR / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


@dataclasses.dataclass
class JitSpec:
    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[Path]]
    needs_device_linking: bool = False

    @property
    def ninja_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / "build.ninja"

    @property
    def build_dir(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name

    @property
    def jit_library_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / f"{self.name}.so"

    @property
    def is_compiled(self) -> bool:
        return self.jit_library_path.exists()

    @property
    def lock_path(self) -> Path:
        return _get_tmpdir() / f"{self.name}.lock"

    @property
    def is_ninja_generated(self) -> bool:
        return self.ninja_path.exists()

    def write_ninja(self) -> None:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
            needs_device_linking=self.needs_device_linking,
        )
        write_if_different(self.ninja_path, content)

    def build(self, verbose: bool, need_lock: bool = True) -> None:
        lock = (
            FileLock(self.lock_path, thread_local=False) if need_lock else nullcontext()
        )
        with lock:
            if not self.is_ninja_generated:
                self.write_ninja()
            run_ninja(self.build_dir, self.ninja_path, verbose)

    def load(self, so_path: Path):
        return tvm_ffi.load_module(str(so_path))

    def build_and_load(self):
        with FileLock(self.lock_path, thread_local=False):
            so_path = self.jit_library_path
            verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
            self.build(verbose, need_lock=False)
            return self.load(so_path)


def gen_jit_spec(
    name: str,
    sources: Sequence[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
    needs_device_linking: bool = False,
) -> JitSpec:
    _check_cuda_arch()

    debug_env = os.environ.get("FLASHINFER_JIT_DEBUG")
    verbose_env = os.environ.get("FLASHINFER_JIT_VERBOSE", "0")
    debug = (debug_env if debug_env is not None else verbose_env) == "1"

    cflags_has_std = extra_cflags is not None and any(
        f.startswith("-std=") for f in extra_cflags
    )
    cuda_cflags_has_std = extra_cuda_cflags is not None and any(
        f.startswith("-std=") for f in extra_cuda_cflags
    )

    cflags = ["-Wno-switch-bool"]
    if not cflags_has_std:
        cflags.insert(0, "-std=c++17")

    cuda_cflags = [
        f"--threads={os.environ.get('FLASHINFER_NVCC_THREADS', '1')}",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    if not cuda_cflags_has_std:
        cuda_cflags.insert(0, "-std=c++17")

    if debug:
        cflags += ["-O0", "-g"]
        cuda_cflags += ["-g", "-O0", "-G", "-lineinfo", "--ptxas-options=-v"]
    else:
        cuda_cflags += ["-DNDEBUG", "-O3"]
        cflags += ["-O3"]

    if os.environ.get("FLASHINFER_JIT_LINEINFO", "0") == "1":
        cuda_cflags += ["-lineinfo"]

    if extra_cflags is not None:
        cflags += extra_cflags
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    return JitSpec(
        name=name,
        sources=[Path(x) for x in sources],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_dirs=(
            [Path(x) for x in extra_include_paths]
            if extra_include_paths is not None
            else None
        ),
        needs_device_linking=needs_device_linking,
    )
