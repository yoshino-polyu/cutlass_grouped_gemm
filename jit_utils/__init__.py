"""Minimal JIT compilation utilities extracted from FlashInfer.

Only the pieces needed to compile CUDA .cu files into .so modules
via ninja + nvcc and load them through tvm_ffi.

External pip dependencies:
  apache-tvm-ffi   (tvm_ffi.load_module, tvm_ffi.libinfo)
  filelock
  torch
  packaging
  ninja             (system binary)
"""

from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags
