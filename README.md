# Blackwell MoE Grouped GEMM — Block-Scaled Ragged Contiguous

A self-contained, minimal extract of CUTLASS example `92_blackwell_moe_gemm_blockscaled_rcgrouped`.
This implements a **Ragged Contiguous (RC) Grouped GEMM** using Blackwell SM100 TMA + warp-specialized
kernels with **MX (microscaling) FP8 block-scaled** narrow-precision operands.

## Getting Started — Step by Step

This section walks through everything needed to go from a fresh clone to a passing
`test_wrapper.py` run. Follow the steps in order.

### Step 0: Verify Hardware and CUDA

You need a Blackwell GPU (SM100) and CUDA 12.8+.

```bash
# Check GPU
nvidia-smi          # Should show B200, B300, or similar Blackwell GPU

# Check CUDA toolkit version (must be >= 12.8)
nvcc --version

# Check that PyTorch sees the GPU
python -c "import torch; print(torch.cuda.get_device_capability())"
# Expected: (10, 0) for B200/B300
```

### Step 1: Install pip Dependencies

The wrapper needs four lightweight pip packages. None of them bring CUTLASS headers
(that's the whole point of our extraction — see [Architecture](#architecture) below).

```bash
pip install apache-tvm-ffi filelock packaging ninja
```

| Package | Why it's needed |
|---|---|
| `apache-tvm-ffi` | `tvm_ffi.load_module()` loads the compiled `.so` into Python; also provides DLPack include paths for the C++ build |
| `filelock` | Prevents concurrent ninja builds from corrupting each other |
| `packaging` | Parses CUDA version strings (e.g. `Version("12.8")`) |
| `ninja` | The build system that compiles `.cu` → `.o` → `.so` |

`torch` is assumed already installed with CUDA support.

### Step 2: Verify the Import Chain

Before compiling anything, confirm Python can find all modules:

```bash
cd /path/to/sm100-grouped-gemm

# Test the JIT utilities (no GPU needed for this)
python -c "from jit_utils import gen_jit_spec, sm100a_nvcc_flags; print('OK')"

# Test the wrapper module (imports api.py → jit_module.py → jit_utils)
python -c "from wrapper.api import mxfp8_rc_grouped_gemm; print('OK')"
```

Both should print `OK`. If not, check that your working directory is the repo root.

### Step 3: Run the Test

```bash
cd /path/to/sm100-grouped-gemm
python -m wrapper.test_wrapper
```

**What happens on the first run:**

1. Python imports `wrapper.api` → `wrapper.jit_module` → `jit_utils`
2. `jit_utils` detects your GPU architecture (e.g. `10.0a`) via `torch.cuda`
3. A `build.ninja` file is generated under
   `~/.cache/flashinfer/sm100_grouped_gemm/100a/cached_ops/mxfp8_rc_grouped_gemm/`
4. `ninja` is invoked to compile `launcher.cu` + `binding.cu` with nvcc
   (flags: `-gencode=arch=compute_100a,code=sm_100a -std=c++17 -O3 --expt-relaxed-constexpr`)
5. The resulting `mxfp8_rc_grouped_gemm.so` is loaded via `tvm_ffi.load_module()`
6. The test generates random FP8 data, runs both the CUTLASS kernel and an FP32 reference,
   and compares outputs

**First compilation takes ~1–2 minutes.** Subsequent runs (even in new Python processes)
skip compilation because ninja sees the `.so` is up-to-date.

**If compilation fails**, enable verbose output to see the exact nvcc commands:

```bash
FLASHINFER_JIT_VERBOSE=1 python -m wrapper.test_wrapper
```

### Step 4: Use in Your Own Code

```python
import torch
from wrapper import mxfp8_rc_grouped_gemm

# Prepare inputs (same interface as Triton Mgemm_mxfp8)
x_fp8   = ...  # [M, K]     float8_e4m3fn — activations
x_scale = ...  # [M, K//32] uint8 (E8M0)  — activation block scales
w_fp8   = ...  # [E, N, K]  float8_e4m3fn — expert weights
w_scale = ...  # [E, N, K//32] uint8 (E8M0) — weight block scales
cnt     = ...  # [E+1]      int32         — cumulative token counts per expert

# Run grouped GEMM
y = mxfp8_rc_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt, mma_sm=1)
# y: [M, N] bfloat16
```

For benchmarking, the first call pays the compilation cost (or skips it if already cached).
All subsequent calls go straight to the GPU kernel:

```python
# Warm up (triggers JIT compilation if needed + CUDA context init)
_ = mxfp8_rc_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt)
torch.cuda.synchronize()

# Benchmark
import time
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    y = mxfp8_rc_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100
print(f"{elapsed*1e3:.2f} ms per call")
```

### Environment Variables Reference

All optional. The defaults work on a standard single-GPU setup.

```bash
# Show full nvcc commands during compilation (default: quiet)
export FLASHINFER_JIT_VERBOSE=1

# Skip torch.cuda GPU detection, hardcode architecture
export FLASHINFER_CUDA_ARCH_LIST="10.0a"

# Override CUDA toolkit location
export CUDA_HOME=/usr/local/cuda

# Override host C++ compiler
export CXX=g++

# Override nvcc binary path
export FLASHINFER_NVCC=/usr/local/cuda/bin/nvcc

# Limit ninja parallelism (useful on memory-constrained machines)
export MAX_JOBS=4
```

### Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'tvm_ffi'` | `pip install apache-tvm-ffi` |
| `ModuleNotFoundError: No module named 'filelock'` | `pip install filelock` |
| `ModuleNotFoundError: No module named 'jit_utils'` | Make sure you `cd` into the repo root before running |
| `RuntimeError: Requires GPUs with sm75 or higher` | No CUDA GPU detected — check `nvidia-smi` and `torch.cuda.is_available()` |
| `Ninja build failed` | Run with `FLASHINFER_JIT_VERBOSE=1` to see the actual nvcc error. Common causes: missing CUDA 12.8+, wrong `CUDA_HOME` |
| `CUTLASS can_implement failed` | Problem dimensions may not meet alignment requirements (K must be divisible by 128, N by 16) |
| Compilation is slow | Set `MAX_JOBS=4` or higher. The `.so` is cached — only the first run compiles |
| Want to force recompilation | Delete `~/.cache/flashinfer/sm100_grouped_gemm/` and re-run |

---

## Architecture

### Repository Structure

```
sm100-grouped-gemm/
├── 92_blackwell_moe_gemm_blockscaled_rcgrouped.cu   # Original CUTLASS example
├── include/                  # CUTLASS + CuTe headers (vendored, not from FlashInfer)
│   ├── cutlass/
│   └── cute/
├── csrc/
│   └── tvm_ffi_utils.h      # TVM-FFI C++ utilities (vendored from FlashInfer)
├── jit_utils/                # JIT build infrastructure (extracted from FlashInfer)
│   ├── __init__.py           #   exports gen_jit_spec, sm100a_nvcc_flags
│   ├── compilation_context.py#   GPU arch detection via torch.cuda
│   ├── core.py               #   JitSpec class + gen_jit_spec factory
│   ├── cpp_ext.py            #   ninja build file generation + run_ninja
│   ├── env.py                #   path config (points at our include/, not FlashInfer's)
│   └── utils.py              #   write_if_different helper
└── wrapper/                  # The CUTLASS wrapper
    ├── __init__.py            #   re-exports mxfp8_rc_grouped_gemm
    ├── api.py                 #   Python API (workspace mgmt, output alloc, module call)
    ├── jit_module.py          #   JitSpec creation + cached build_and_load
    ├── kernel_traits.cuh      #   CUTLASS type aliases (Gemm1SM, Gemm2SM, etc.)
    ├── preprocess_kernels.cuh #   CUDA kernels: transpose, SF reshuffle, gather
    ├── launcher.cu            #   C++ entry point (7-step GPU pipeline)
    ├── binding.cu             #   TVM_FFI_DLL_EXPORT_TYPED_FUNC
    ├── test_wrapper.py        #   End-to-end test vs FP32 reference
    └── DESIGN.md              #   Detailed design doc (SfAtom math, role swap, etc.)
```

### Why `jit_utils/` Exists (Instead of `pip install flashinfer`)

Installing `flashinfer-python` would work, but it vendors its own CUTLASS headers
inside `flashinfer/data/cutlass/`. Those headers can conflict with the ones in our
`include/` (we need SM100 block-scaled GEMM headers that may be newer or patched).

So we extracted **only the JIT build machinery** — 5 Python files (~600 lines total)
that know how to: detect GPU arch → generate `build.ninja` → run nvcc → load `.so`.
No CUTLASS headers come along.

| `jit_utils/` file | FlashInfer source | What changed |
|---|---|---|
| `compilation_context.py` | `flashinfer/compilation_context.py` | Verbatim |
| `env.py` | `flashinfer/jit/env.py` | **Rewritten** — points includes at our `include/` and `csrc/`. Removed AOT, cubin, jit-cache, version checks |
| `utils.py` | `flashinfer/jit/utils.py` | Verbatim (just `write_if_different`) |
| `cpp_ext.py` | `flashinfer/jit/cpp_ext.py` | Removed FlashInfer-specific includes (spdlog, nvshmem, FlashInfer data dirs) |
| `core.py` | `flashinfer/jit/core.py` | Removed `JitSpecRegistry`, AOT paths, `FlashInferJITLogger`. Kept `JitSpec`, `gen_jit_spec`, `sm100a_nvcc_flags` |

`csrc/tvm_ffi_utils.h` was copied from `flashinfer/csrc/tvm_ffi_utils.h`. It provides
`TensorView`, `get_current_stream()`, and DLPack dtype constants for the C++ side.

### Dependency Graph

```
User code
  │
  ▼
wrapper/api.py                         Python API (user-facing)
  │  allocates output [M,N] bf16
  │  manages workspace buffer
  │  calls module.mxfp8_rc_grouped_gemm(...)
  │
  ▼
wrapper/jit_module.py                  Creates JitSpec, calls build_and_load()
  │
  ▼
jit_utils/core.py                      JitSpec + gen_jit_spec + sm100a_nvcc_flags
  ├── jit_utils/cpp_ext.py             Generates build.ninja, runs ninja
  │     ├── jit_utils/env.py           Path config → our include/ and csrc/
  │     └── tvm_ffi (pip)              tvm_ffi.load_module(), dlpack includes
  ├── jit_utils/compilation_context.py GPU arch detection via torch.cuda
  ├── filelock (pip)                   Concurrent build safety
  └── packaging (pip)                  CUDA version parsing
           │
           │  ninja compiles at runtime:
           ▼
wrapper/launcher.cu + binding.cu  ──►  mxfp8_rc_grouped_gemm.so
  ├── wrapper/kernel_traits.cuh        CUTLASS Gemm1SM/Gemm2SM type aliases
  ├── wrapper/preprocess_kernels.cuh   Transpose B, reshuffle SFs, gather output
  ├── include/cutlass/...              CUTLASS headers (vendored in this repo)
  └── csrc/tvm_ffi_utils.h             TVM-FFI C++ utilities (vendored)
```

### How JIT Compilation Works (Detailed)

1. `wrapper/api.py` calls `get_module()` which is `@functools.cache`-decorated
2. On first call, `gen_module()` creates a `JitSpec` with:
   - Sources: `wrapper/launcher.cu`, `wrapper/binding.cu`
   - NVCC flags: `-gencode=arch=compute_100a,code=sm_100a`, `--expt-relaxed-constexpr`,
     `-std=c++17`, `-O3`, `-DNDEBUG`, `-DFLASHINFER_ENABLE_BF16`, etc.
   - Include paths: `include/` (CUTLASS), `csrc/` (tvm_ffi_utils.h), `wrapper/` (kernel_traits.cuh)
3. `JitSpec.build_and_load()` acquires a file lock, then:
   - Generates `build.ninja` in `~/.cache/flashinfer/sm100_grouped_gemm/100a/cached_ops/mxfp8_rc_grouped_gemm/`
   - Runs `ninja -v -C <build_dir> -f build.ninja`
   - ninja invokes nvcc to compile each `.cu` → `.cuda.o`, then links → `.so`
   - Loads the `.so` via `tvm_ffi.load_module()`, which returns a module object
4. The module exposes `module.mxfp8_rc_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt, output, workspace, mma_sm)`
5. That calls `mxfp8_rc_grouped_gemm_run()` in `launcher.cu`, exported via `TVM_FFI_DLL_EXPORT_TYPED_FUNC` in `binding.cu`
6. On subsequent calls (same process): `@functools.cache` returns the loaded module instantly
7. On subsequent runs (new process): ninja sees `.so` is up-to-date → skips compilation → just loads

## What This Computes

Mixture-of-Experts (MoE) layers dispatch tokens to different experts, each performing a GEMM
with its own weight matrix. The kernel is block-scaled: A and B use `mx_float8_t` (MX FP8 E4M3)
with `float_ue8m0_t` scale factors per block (MXFP8 microscaling format).

## Ragged Contiguous (RC) Design

### The Core Idea

In a standard grouped GEMM, every operand (A, B, C, D) is supplied as an array of pointers —
one independent allocation per group. "Ragged Contiguous" is an optimized variant for MoE
workloads where **M and K are the same across all groups**, but **N varies per group**.

The name captures both halves:

- **Contiguous** — Matrix A is a single contiguous buffer spanning all groups.
  Groups are stacked along a batch dimension with stride = M * K, so group `i` lives at
  offset `i * M * K` from the base pointer.
- **Ragged** — Matrix B, C, and D are separate per-group allocations accessed
  through pointer arrays. Their N dimension is "ragged" — it differs per group.

### Dimension Mapping: CUTLASS A/B vs MoE Semantics

**This is the most confusing part.** In a typical MoE framework you think of:
- Activations (tokens): varying token count per expert, same hidden dimension
- Weights: identical shape across all experts (hidden_in × hidden_out)

So you would expect the *activation* matrix to be ragged and the *weight* matrix to be
uniform. But in this CUTLASS example, A is contiguous (fixed shape) and B is ragged. This
seems backwards — until you look at what A and B actually represent.

The GEMM computes `D = A × B^T` (A is RowMajor M×K, B is ColumnMajor N×K):

| CUTLASS name | Shape | Layout | MoE role |
|---|---|---|---|
| **A** | M × K × G (contiguous) | RowMajor, 3D stride | **Expert weights** — one M×K slice per expert, all stacked |
| **B** | N_i × K per group (ragged) | ColumnMajor, pointer array | **Activations/tokens** — N_i tokens routed to expert i |
| **D** | M × N_i per group | RowMajor, pointer array | Output activations per expert |

The key: **CUTLASS's "A" is the weight matrix, "B" is the activation matrix.** This is the
opposite of the neural-network convention where "activations × weights = output".

Why does this work?
- All experts have the same weight shape (M × K) → A is uniform → **contiguous** storage
- Each expert gets a different number of tokens (N_i varies) → B is varying → **ragged** storage
- `tokens_per_expert[i]` = N_i = the number of tokens routed to expert `i`

```
MoE dimension mapping:

CUTLASS A (expert weights, contiguous):    CUTLASS B (tokens, ragged):
┌──────────────┐                           ptr_B[0] ──► ┌──────────┐ N₀ tokens × K
│  Expert 0    │ M × K                    ptr_B[1] ──► ┌────┐ N₁ tokens × K
│  Expert 1    │ M × K                    ptr_B[2] ──► ┌───────────────┐ N₂ tokens × K
│  Expert 2    │ M × K                          ...
│  ...         │
│  Expert G-1  │ M × K                    Each expert receives a different
└──────────────┘                           number of tokens (ragged N dim).
  All experts have the same
  M×K shape → contiguous.

GEMM per expert i:  D_i = A_i × B_i^T    →   (M × K) × (K × N_i) = M × N_i
```

You can verify this in `randomize_problems()` (line 373-380):
```cpp
for (int i = groups; i > 0; i--) {
    int n = cmd_line_n;
    if (n < 0) {
        n = alignment * ((rand() % 64) + 1);  // N randomized per expert
    }
    problem_sizes_host.push_back({m, n, k});   // M, K fixed; N varies
    tokens_per_expert_host.push_back(n);        // N = tokens for this expert
}
```

And in `MoEProblemShape::get_problem_shape()` (`group_array_problem_shape.hpp` line 107):
```cpp
expert_problem_dims = {max_m, tokens_per_expert[group_idx], max_k};
//                      ^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^
//                      fixed   varies per group (ragged)    fixed
```

### Where This Appears in Code

**1. Type-level distinction — the `.cu` file (lines 105-111, 208-211):**

```cpp
// A: stride-based (no pointer, contiguous)
using LayoutA  = cutlass::layout::RowMajor;          // plain layout tag
using StrideA  = cutlass::detail::TagToStrideA_t<LayoutA>;

// B: pointer-array-based (ragged)
using LayoutB  = cutlass::layout::ColumnMajor;       // note: LayoutB * (pointer)
using StrideB  = typename Gemm::GemmKernel::InternalStrideB;
```

The `*` in `LayoutB *` (used at line 166 in the collective builder) is the signal to CUTLASS
that B comes as a pointer array rather than a single strided buffer.

**2. Host-side allocation — `allocate()` and `initialize()` (lines 524-632):**

```cpp
// A: one allocation for ALL groups
stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, options.groups});
auto layout_A = make_layout(make_shape(options.m, options.k, options.groups), stride_A);
block_A.reset(cutlass::make_Coord(size(layout_A)));   // single buffer

// B: separate allocation PER group, stored as pointer array
for (int32_t i = 0; i < options.groups; ++i) {
    ptr_B_host.at(i) = block_B.at(i).device_data();   // each group's own buffer
}
ptr_B.reset(options.groups);
ptr_B.copy_from_host(ptr_B_host.data());               // array of device pointers
```

**3. Kernel arguments — `args_from_options()` (lines 688-695):**

```cpp
arguments = typename Gemm::Arguments {
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {options.m, options.n, options.k, options.groups, tokens_per_expert.get()},
    {block_A.device_data(),   // single pointer   (contiguous)
     ptr_B.get(),             // pointer-to-array  (ragged)
     block_SFA.device_data(), // single pointer    (contiguous, same as A)
     ptr_SFB.get()},          // pointer-to-array  (ragged, same as B)
    ...
};
```

**4. MoEProblemShape — `group_array_problem_shape.hpp` (line 95):**

```cpp
template <class ProblemShape_>
struct MoEProblemShape {
    int32_t max_m, max_n, max_k, num_groups;
    int32_t* tokens_per_expert;   // per-group N on device

    UnderlyingProblemShape get_problem_shape(int32_t group_idx) const {
        return {max_m, tokens_per_expert[group_idx], max_k};
    }
};
```

Unlike the generic `GroupProblemShape` (which allows M, N, K to all differ per group),
`MoEProblemShape` fixes M and K and only varies N via `tokens_per_expert[]`.

**5. Stride computation for A — `moe_stride_utils.hpp`:**

```cpp
// RowMajor A: stride = (K, 1, M*K)
//   dim 0 (M stride) = K
//   dim 1 (K stride) = 1
//   dim 2 (group stride) = M * K   ← groups are contiguous
get<2>(s_copy) = M * K;  // when num_groups > 1
```

## TMA Descriptors: Single vs Per-Group Updates

Both A and B are loaded via TMA (Tensor Memory Access), but the way the hardware knows
*where* and *how much* to load differs fundamentally between them. Understanding this
requires knowing what a TMA descriptor actually is.

The mainloop collective lives in:
`include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized_rcggemm.hpp`

### What Is a TMA Descriptor?

A TMA descriptor is a small (~128-byte) opaque structure that tells the TMA hardware unit
how to perform a multi-dimensional memory copy. It encodes:

```
TMA Descriptor contents:
┌──────────────────────────────────────────┐
│  base_address    — where the data starts │
│  dimensions[]    — size of each dim      │
│  strides[]       — byte stride per dim   │
│  element_type    — data type / size      │
│  swizzle_mode    — shared memory layout  │
│  ...other fields                         │
└──────────────────────────────────────────┘
```

When you issue a TMA load, you give it a descriptor + a set of coordinates. The hardware
uses the descriptor's strides and dimensions to compute the source address and transfer
the tile. You do NOT compute pointers yourself — the descriptor does it.

### Matrix A — One Descriptor Covers All Groups

Because A is contiguous with shape `(M, K, G)` and stride `(K, 1, M*K)`, all expert
weights sit in a single buffer. The TMA descriptor is created **once at init** with
a 3D layout that includes the group dimension:

```cpp
// sm100_blockscaled_mma_array_warpspecialized_rcggemm.hpp, lines 440-474

// 3D shape: (M, K, num_groups)
auto shape_a = make_shape(init_M, init_K_A, problem_shapes.groups());
InternalStrideA stride_a = make_internal_packed_stride(InternalStrideA{}, shape_a);

// Tensor A: single pointer, 3D layout covering ALL groups
Tensor tensor_a = make_tensor(ptr_A_first_batch,
    make_layout(make_shape(init_M, init_K_A, problem_shapes.groups()), stride_a));

// One descriptor is created from this 3D tensor
typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<...>(
    GmemTiledCopyA{}, tensor_a, ...);
```

At load time, selecting a different group is just an index into the 3rd dimension:

```cpp
// Line 895 — curr_batch selects which group to load from
Tensor tAgA = tAgA_mkl(_, cta_coord_m, _, curr_batch);
//                                        ^^^^^^^^^^
//                     batch coordinate indexes into dim 2 of the descriptor

// Line 918 — TMA load: the hardware computes
//   address = base_address + curr_batch * stride[2] + k_tile * stride[1] + ...
copy(tma_load_a->with(*tma_barrier, mcast_mask_a),
     tAgA(_, *k_tile_iter),
     tAsA(_, write_stage));
```

**No descriptor modification needed.** Switching from expert 3 to expert 7 is just
changing the batch coordinate from 3 to 7. The hardware applies `7 * M * K * sizeof(elem)`
as the offset — that's already encoded in the descriptor's stride[2].

A's descriptor is **not stored in shared memory** — it never changes, so it stays in
kernel `Params` (read-only):

```cpp
// Lines 292-295: shared memory only has B and SFB descriptors
struct TensorMapStorage : cute::aligned_struct<128, _0> {
    cute::TmaDescriptor smem_tensormap_B;     // B only
    cute::TmaDescriptor smem_tensormap_SFB;   // SFB only
    // A is NOT here — it never needs updating
} tensormaps;
```

### Matrix B — Descriptor Rewritten Per Group

B is a pointer array — each expert's tokens live in a **separate allocation** at a
**different address**, and each has a **different N dimension**. No single 3D descriptor
can cover this because:
1. The buffers are not contiguous (no single base_address + stride formula works)
2. The N dimension changes per group (descriptor dimensions must change)

So B's descriptor is created initially with placeholder values:

```cpp
// Lines 442-449: B has shape (N, K, 1) — NO group dimension
InternalStrideB stride_b = InternalStrideB{};   // 2D stride, no batch dim

Tensor tensor_b = make_tensor(ptr_B_first_batch,
    make_layout(make_shape(init_N, init_K_B, init_L), stride_b));
//                                           ^^^^^^
//                         L = 1, not num_groups — descriptor is per-group only
```

B's descriptor is copied into **shared memory** so it can be modified on the fly by
the producer warp:

```cpp
// Shared memory holds a mutable copy of B's descriptor
struct TensorMapStorage {
    cute::TmaDescriptor smem_tensormap_B;    // will be rewritten per group
    cute::TmaDescriptor smem_tensormap_SFB;
};
```

When the kernel finishes one group and moves to the next, **three things are patched**
in the shared-memory descriptor:

```cpp
// Step 1: Replace base address — point to the next expert's buffer
// (lines 1157-1162)
cute::tma_descriptor_replace_addr_in_shared_mem(
    shared_tensormaps.smem_tensormap_B,
    mainloop_params.ptr_B[next_batch]);      // ptr_B[3] → ptr_B[4]

// Step 2: Replace dimensions and strides — N changes per expert
// (lines 1174-1212)
auto [N_new, K_new, L_new] = new_problem_shape;
cute::tma_descriptor_replace_dims_strides_in_shared_mem(
    shared_tensormaps.smem_tensormap_B,
    prob_shape_B, prob_stride_B);            // N=640 → N=112

// Step 3: Fence to make the updated descriptor visible to TMA loads
// (line 1260)
cute::tma_descriptor_cp_fence_release(
    shared_tensormaps.smem_tensormap_B, ...);
```

Then the TMA load for B reads from the **shared-memory descriptor** (not from Params):

```cpp
// Line 919 — note: get<0>(input_tensormaps) is the smem descriptor pointer
copy(tma_load_b->with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_b),
//                     ^^^^^^^^^^^^^^^^^^^^^^^^
//                     shared-memory descriptor (mutable, updated per group)
     tBgB(_, *k_tile_iter),
     tBsB(_, write_stage));
```

### Visual Comparison

```
Matrix A (single descriptor):              Matrix B (per-group descriptor updates):

Descriptor created once:                   Descriptor patched every group switch:
  base = &A[0]                               Group 0: base = ptr_B[0], dims = {N₀, K}
  dims = {M, K, G}                           Group 1: base = ptr_B[1], dims = {N₁, K}  ← rewrite
  strides = {K, 1, M*K}                      Group 2: base = ptr_B[2], dims = {N₂, K}  ← rewrite
                                              ...
Load group 0: coord = (m, k, 0)
Load group 1: coord = (m, k, 1)            Each switch requires:
Load group 2: coord = (m, k, 2)              1. replace base address
  ↑ just change the coordinate                2. replace dims (N changes)
  ↑ descriptor stays the same                 3. fence + release
```

### How Many Descriptor Copies Exist? Per-CTA, Not Per-Group

**Short answer: there is exactly ONE mutable TMA descriptor per CTA, rewritten in-place
on each group switch.** Groups do NOT get their own descriptor. The same shared-memory
slot is overwritten with new address + dimensions when the CTA moves to the next group.

Here is the full lifecycle:

```
Descriptor lifecycle for matrix B (one CTA's perspective):
═══════════════════════════════════════════════════════════

     ┌─────────────────────────────────────────────────────────────┐
     │                    Kernel Params (gmem, read-only)          │
     │  tma_load_b descriptor: initial template with placeholder  │
     │  values for base_addr, dims, strides                       │
     └────────────────────────┬────────────────────────────────────┘
                              │
                    ① INIT (once per CTA launch)
                    copy 128 bytes: Params → smem
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │      CTA's shared memory (private, mutable)                │
     │  ┌─────────────────────────────────────────────────┐       │
     │  │         smem_tensormap_B  (128 bytes)           │       │
     │  │  This is the ONLY mutable copy. It gets         │       │
     │  │  rewritten in-place for each group switch.      │       │
     │  └────────────────────┬────────────────────────────┘       │
     └───────────────────────┼────────────────────────────────────┘
                             │
              ② UPDATE (on each group switch)
              One elected thread patches smem:
                a) replace base address  → ptr_B[next_group]
                b) replace dims/strides  → new N, same K
              Then entire warp issues fence:
                c) tensormap.cp_fenceproxy smem → gmem
                             │
                             ▼
     ┌─────────────────────────────────────────────────────────────┐
     │     Global memory workspace (per-SM descriptor slots)      │
     │  gmem_tensormap[sm_idx * NumTmaDescriptorsPerSm + offset]  │
     │  The fence atomically publishes the updated descriptor     │
     │  here, making it visible to the TMA hardware unit.         │
     └────────────────────────┬────────────────────────────────────┘
                              │
              ③ TMA LOAD (hardware reads gmem descriptor)
              TMA unit uses the published gmem descriptor to
              compute source address and transfer a tile.
                              │
                              ▼
                    ┌───────────────────┐
                    │  Tile data lands  │
                    │  in CTA's smem    │
                    └───────────────────┘
```

Key point: the smem descriptor is a **staging area**, not the final source for TMA.
The `tensormap.cp_fenceproxy` instruction copies from smem → gmem and makes the gmem
copy visible to the TMA hardware. The TMA load references the gmem slot:

```cpp
// Line 919 — get<0>(input_tensormaps) is a gmem descriptor pointer
copy(tma_load_b->with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_b),
//                     ^^^^^^^^^^^^^^^^^^^^^^^^
//                     gmem descriptor (published from smem via fence)
     tBgB(_, *k_tile_iter),
     tBsB(_, write_stage));
```

#### Per-CTA shared memory copy

Shared memory is private to each CTA. The `TensorMapStorage` struct lives inside the
CTA's `SharedStorage`, so every CTA that runs a tile of any group has its own
`smem_tensormap_B`:

```cpp
// Lines 292-295: this is inside SharedStorage — per CTA
struct TensorMapStorage : cute::aligned_struct<128, _0> {
    cute::TmaDescriptor smem_tensormap_B;
    cute::TmaDescriptor smem_tensormap_SFB;
} tensormaps;
```

#### Initialization: each CTA copies from Params to its own smem

When a CTA starts, one elected thread copies the descriptor from read-only kernel Params
into that CTA's shared memory (`tensormaps_init`, lines 1112-1122):

```cpp
if (cute::elect_one_sync()) {
    // Copy from Params (global, read-only) → this CTA's smem (mutable)
    Tensor pB = make_tensor(observed_tma_load_b_->get_tma_descriptor(), ...);
    Tensor sB = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), ...);
    copy(recast<uint128_t>(pB), recast<uint128_t>(sB));
}
```

#### Multiple CTAs on the same group each have their own descriptor

Consider a group whose output tile grid is 4×8 = 32 tiles. 32 CTAs are launched for
this group. Each of these 32 CTAs has its own smem copy of the B descriptor, all
initialized to the same values (same `ptr_B[i]`, same N_i). They operate independently:

```
Group i: N_i = 640, ptr_B[i] = 0x7f00...

   CTA 0 (tile 0,0)         CTA 1 (tile 0,1)        CTA 31 (tile 3,7)
  ┌──────────────────┐      ┌──────────────────┐     ┌──────────────────┐
  │ smem_tensormap_B │      │ smem_tensormap_B │     │ smem_tensormap_B │
  │  base=ptr_B[i]   │      │  base=ptr_B[i]   │     │  base=ptr_B[i]   │
  │  dims={640, K}   │      │  dims={640, K}   │     │  dims={640, K}   │
  └──────────────────┘      └──────────────────┘     └──────────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
   TMA loads tile (0,0)     TMA loads tile (0,1)     TMA loads tile (3,7)
   from the SAME gmem       from the SAME gmem       from the SAME gmem
   buffer at ptr_B[i]       buffer at ptr_B[i]       buffer at ptr_B[i]
```

Each CTA uses its own descriptor to TMA-load a different tile coordinate from the same
underlying buffer. The descriptors are identical in content but physically separate in
each CTA's shared memory.

**Why not share a single descriptor across CTAs in the same group?** Because shared
memory is architecturally private to each CTA — there is no mechanism for one CTA to
read another CTA's shared memory. The duplication is a hardware constraint, not a design
choice.

#### When a CTA switches to a different group

In grouped GEMM, a single CTA may process tiles from multiple groups sequentially (the
tile scheduler assigns work). When a CTA finishes all its tiles for group `i` and moves
to group `j`, it rewrites **its own** smem descriptor in-place via
`tensormaps_perform_update` (lines 1215-1242):

```cpp
// Called when did_batch_change is true (kernel loop, line ~950)
if (did_batch_change) {
    collective_mainloop.tensormaps_perform_update(
        shared_storage.tensormaps[0].mainloop,  // this CTA's smem
        params.mainloop,
        get_tensormap(input_tensormaps, tma_desc_offset),
        problem_shape,
        curr_batch);                            // next group index
}
```

Inside `tensormaps_perform_update`, one elected thread patches address and dims, then
the entire warp issues the fence:

```cpp
if (cute::elect_one_sync()) {
    // Step 1: point to next expert's buffer
    tma_descriptor_replace_addr_in_shared_mem(
        shared_tensormaps.smem_tensormap_B,
        mainloop_params.ptr_B[next_batch]);

    // Step 2: update N dimension (K stays the same)
    tma_descriptor_replace_dims_strides_in_shared_mem(
        shared_tensormaps.smem_tensormap_B,
        prob_shape_B, prob_stride_B);
}
__syncwarp();
// Step 3: entire warp publishes smem → gmem and fences
tma_descriptor_cp_fence_release(
    get<0>(input_tensormaps),                    // gmem slot
    shared_tensormaps.smem_tensormap_B);          // smem source
```

Visually, the single smem slot is overwritten:

```
CTA 5 processing group 2, then group 7:

  Time T₁ (group 2):              Time T₂ (group 7):
  ┌──────────────────┐             ┌──────────────────┐
  │ smem_tensormap_B │   ──────►   │ smem_tensormap_B │
  │  base=ptr_B[2]   │  rewrite    │  base=ptr_B[7]   │
  │  dims={640, K}   │  in-place   │  dims={224, K}   │
  └──────────────────┘             └──────────────────┘
  Same smem address, new content.
  Old descriptor is gone — no per-group copies are kept.
```

This rewrite is entirely local to CTA 5. Other CTAs still working on group 2 are
unaffected — their smem descriptors still point to `ptr_B[2]`.

#### Global memory workspace: descriptor versioning per SM

In addition to the smem copy, there is a **global memory workspace** that stores
multiple versions of descriptors per SM (lines 559-561):

```cpp
return sm_count * sizeof(TensorMaps) * NumTmaDescriptorsPerSm;
//                                     ^^^^^^^^^^^^^^^^^^^^^^^^^
//     NumTmaDescriptorsPerSm = SchedulerPipelineStageCount + Stages + 2
```

The TMA hardware unit does **not** read descriptors directly from shared memory. Instead,
the `tensormap.cp_fenceproxy` instruction atomically copies the updated descriptor from
smem to a gmem slot and makes it visible to TMA. Multiple gmem slots per SM allow
**pipelining**: a CTA can prepare the next group's descriptor in one gmem slot while the
TMA unit is still reading the current group's descriptor from another. The gmem slot
index rotates via modular arithmetic:

```cpp
// Line 1139-1140
idx = idx % NumTmaDescriptorsPerSm;
```

#### Fence scope is CTA-level, not cluster-level

The fence instruction that publishes the updated descriptor operates at CTA scope:

```asm
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
//                             ^^^^^
//                             CTA scope — not cluster-wide
```

Within a cluster, TMA data can **multicast** (one TMA load delivers data to multiple
CTAs' shared memory simultaneously). But the descriptor itself is NOT multicast — each
CTA in the cluster maintains and updates its own copy independently.

### Why This Matters for Performance

The RC design exists specifically to **minimize TMA descriptor updates**. In a fully
pointer-array grouped GEMM (where A is also ragged), the kernel would need to rewrite
descriptors for *both* A and B on every group switch. By making A contiguous, only B's
descriptor needs updating — cutting the descriptor management overhead roughly in half.

This is also why only B (not A) needs:
- Shared memory for a mutable descriptor copy (per CTA)
- Global memory workspace for descriptor versioning (per SM)
- Fence/release instructions after each update

### Scale Factors Follow the Same Pattern

- **SFA** (scale factors for A): contiguous, single TMA descriptor — same as A.
- **SFB** (scale factors for B): pointer array, per-group TMA updates — same as B.

### Summary

| Aspect | Matrix A (+ SFA) | Matrix B (+ SFB) |
|--------|-------------------|-------------------|
| MoE role | Expert weights | Activation tokens |
| Memory | Single contiguous buffer | Separate allocation per group |
| Pointer | `ptr_A` (one pointer) | `ptr_B[]` (pointer array) |
| TMA descriptor shape | 3D: `(M, K, G)` | 2D: `(N_i, K)` per group |
| Descriptor storage | Kernel Params (read-only) | Per-CTA shared memory (mutable) |
| Descriptor copies | 1 (in Params, shared by all CTAs) | 1 per CTA (each updates independently) |
| Group switch cost | Change coordinate only | Rewrite address + dims + fence |
| Cluster behavior | Coordinate indexes batch dim | Each CTA in cluster updates its own copy |

## Directory Layout

```
.
├── CMakeLists.txt
├── README.md
├── 92_blackwell_moe_gemm_blockscaled_rcgrouped.cu   # Main example (entry point)
└── include/
    ├── helper.h                          # GPU timer + error-check macros
    ├── cute/                             # CuTe tensor algebra library
    │   ├── tensor.hpp                    # Core tensor type
    │   ├── layout.hpp                    # Layout algebra
    │   ├── algorithm/                    # copy, gemm, cooperative_copy, ...
    │   ├── arch/                         # SM-specific copy/mma atoms
    │   ├── atom/                         # TiledMMA, TiledCopy atom definitions
    │   └── ...
    └── cutlass/
        ├── cutlass.h                     # Top-level CUTLASS header
        ├── gemm/
        │   ├── collective/               # Collective mainloop builders
        │   │   └── collective_builder.hpp
        │   ├── kernel/
        │   │   └── gemm_universal.hpp    # GemmUniversal kernel template
        │   ├── device/
        │   │   └── gemm_universal_adapter.h  # Host-side launch adapter
        │   ├── dispatch_policy.hpp       # Kernel schedule tags
        │   └── group_array_problem_shape.hpp  # MoEProblemShape
        ├── epilogue/
        │   ├── collective/               # Collective epilogue builders
        │   └── fusion/                   # LinCombEltActBlockScaleFactor, etc.
        └── util/
            └── reference/                # Host-side reference GEMM for verification
```

## Key Source Files to Read

Start with the main `.cu` file, then explore headers in this order:

| Order | File | What It Defines |
|-------|------|-----------------|
| 1 | `92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` | Type aliases, options, allocate/initialize/run/verify |
| 2 | `include/cutlass/gemm/group_array_problem_shape.hpp` | `MoEProblemShape` — how groups + token routing is described |
| 3 | `include/cutlass/gemm/dispatch_policy.hpp` | `KernelPtrArrayTmaWarpSpecialized{1,2}SmMxf8f6f4Sm100` schedule tags |
| 4 | `include/cutlass/gemm/collective/collective_builder.hpp` | Builder that selects the right mainloop collective |
| 5 | `include/cutlass/epilogue/collective/collective_builder.hpp` | Builder that selects the right epilogue collective |
| 6 | `include/cutlass/gemm/kernel/gemm_universal.hpp` | `GemmUniversal` — the top-level kernel template |
| 7 | `include/cutlass/gemm/device/gemm_universal_adapter.h` | Host-side adapter: `get_workspace_size`, `initialize`, `run` |

## Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│ GemmUniversalAdapter (device/gemm_universal_adapter.h)    │
│   Host-side: workspace alloc, argument packing, launch    │
│                                                           │
│ ┌───────────────────────────────────────────────────────┐ │
│ │ GemmUniversal (kernel/gemm_universal.hpp)             │ │
│ │   Kernel entry: tile scheduler + mainloop + epilogue  │ │
│ │                                                       │ │
│ │  ┌────────────────────┐  ┌────────────────────────┐   │ │
│ │  │ CollectiveMainloop │  │ CollectiveEpilogue     │   │ │
│ │  │  TMA loads A,B,SFs │  │  Accumulator → Output  │   │ │
│ │  │  Block-scaled MMA  │  │  α·AB + β·C + SiLU     │   │ │
│ │  │  Warp-specialized  │  │  (optional SF output)  │   │ │
│ │  └────────────────────┘  └────────────────────────┘   │ │
│ └───────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

### Two Kernel Configs

This example instantiates exactly **two** kernel configurations. They share everything
(element types, layouts, epilogue, problem shape) and differ **only** in the MMA SM mode:

| Config | Type alias | Schedule Tag | MmaTileShape | Constraint |
|--------|-----------|--------------|--------------|------------|
| **1SM** | `Gemm1SM` | `KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100` | 128×256×128 | None |
| **2SM** | `Gemm2SM` | `KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100` | 256×256×128 | `cluster_shape.x >= 2` |

Both are run sequentially in `main()` (lines 882-884):

```cpp
std::cout << "Running kernel with 1SM MMA config:" << std::endl;
run<Gemm1SM>(options);
std::cout << "Running kernel with 2SM MMA config:" << std::endl;
run<Gemm2SM>(options);
```

The two configs are defined at lines 141-151:

```cpp
struct MMA1SMConfig {
  using MmaTileShape     = Shape<_128,_256,_128>;
  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

struct MMA2SMConfig {
  using MmaTileShape     = Shape<_256,_256,_128>;
  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};
```

**1SM vs 2SM**: In 1SM mode, each SM independently computes an MMA tile (M=128). In
2SM mode, two SMs cooperate on a larger tile (M=256), which can improve utilization on
large problems but requires the cluster's X dimension to be at least 2.

Everything else is fixed across both configs:
- **Element types**: `mx_float8_t` (MXFP8 E4M3) for A and B, `bfloat16_t` for C/D, `float` accumulator
- **Layouts**: A is RowMajor (contiguous), B is ColumnMajor pointer-array (ragged)
- **Scale factors**: `float_ue8m0_t` (E8M0) for both SFA and SFB
- **Epilogue**: `LinCombEltActBlockScaleFactor` with SiLU (defined but commented out at line 161)
- **Cluster shape**: Runtime-configurable via `--cluster_m` / `--cluster_n`

### Block Scaling (MXFP8)

Operands A and B use the MX microscaling format:
- Data stored as `float_e4m3_t` (FP8 E4M3)
- Each block of elements shares a `float_ue8m0_t` (E8M0 unsigned) scale factor
- Scale factor layouts (`LayoutSFA`, `LayoutSFB`) are computed by `Sm1xxBlkScaledConfig`

### MoE Problem Shape

`MoEProblemShape` describes the grouped GEMM:
- Shared `M`, `K` across all groups; per-group `N` (= tokens per expert)
- `tokens_per_expert` device array tells the kernel the N dimension per group
- A is contiguous (all groups stacked), B/C/D use pointer arrays

## Using as a Library: Input/Output Contract and BF16 Conversion

If you want to call `Gemm1SM` or `Gemm2SM` from another `.cu` file (treating this
example as a library), you need to prepare inputs that match the kernel's exact type
contract. This section documents what goes in, what comes out, and how to bridge
from BF16 data.

### Input/Output Summary

```
Inputs                                          Outputs
══════                                          ═══════
A    : float_e4m3_t [M × K × G]  (contiguous)  D     : bfloat16_t [M × N_i] per group (ptr array)
SFA  : float_ue8m0_t              (contiguous)  (SFD) : float_ue4m3_t (optional, currently disabled)
B    : float_e4m3_t [N_i × K]    (ptr array)
SFB  : float_ue8m0_t             (ptr array)
C    : bfloat16_t   [M × N_i]    (ptr array)   ← optional (can be void-C)
alpha: float (scalar or per-group ptr array)
beta : float (scalar or per-group ptr array)

GEMM per expert i:  D_i = alpha * A_i × B_i^T + beta * C_i
                    (M×K) × (K×N_i) = (M×N_i)
```

### Detailed Type Table

Every input/output has a specific CUTLASS type. These are **not** interchangeable
with raw `float` or `__nv_bfloat16` — they are CUTLASS wrapper types defined in
`include/cutlass/float8.h` and `include/cutlass/float_subbyte.h`.

| Operand | CUTLASS Type | Underlying Bits | Layout | Storage | Defined At |
|---------|-------------|-----------------|--------|---------|-----------|
| **A** (weights) | `mx_float8_t<float_e4m3_t>` | 8-bit FP (E4M3) | RowMajor `(M, K, G)` | Single contiguous buffer | `.cu` line 105 |
| **SFA** (scale for A) | `float_ue8m0_t` | 8-bit unsigned E8M0 | Computed by `Sm1xxBlkScaledConfig` | Single contiguous buffer | `.cu` line 97 |
| **B** (tokens) | `mx_float8_t<float_e4m3_t>` | 8-bit FP (E4M3) | ColumnMajor `(N_i, K)` | Pointer array (`G` pointers) | `.cu` line 110 |
| **SFB** (scale for B) | `float_ue8m0_t` | 8-bit unsigned E8M0 | Computed by `Sm1xxBlkScaledConfig` | Pointer array (`G` pointers) | `.cu` line 97 |
| **C** (bias) | `bfloat16_t` | 16-bit BF16 | RowMajor `(M, N_i)` | Pointer array (`G` pointers) | `.cu` line 98 |
| **D** (output) | `bfloat16_t` | 16-bit BF16 | RowMajor `(M, N_i)` | Pointer array (`G` pointers) | `.cu` line 115 |
| **alpha, beta** | `float` | 32-bit FP32 | Scalar or per-group | Scalar or pointer array | `.cu` line 119 |

Note: `mx_float8_t<float_e4m3_t>` is a **compile-time tag** (see `float8.h` line 1306).
It tells the CUTLASS builder "this operand is FP8 E4M3 with MXFP8 block scaling." The
actual device memory contains raw `float_e4m3_t` bytes — `mx_float8_t` has no runtime
storage of its own. Scale factors are passed separately as `SFA`/`SFB`.

### Step-by-Step: Calling the Kernel from Your Own Code

```cpp
#include "92_blackwell_moe_gemm_blockscaled_rcgrouped.cu"
// Or: copy the type aliases and #includes into your own file.

// --- 1. Define problem dimensions ---
int M = 256;                          // hidden dim (same for all experts)
int K = 512;                          // input dim  (same for all experts)
int G = 8;                            // number of experts
std::vector<int32_t> tokens_per_expert = {128, 64, 256, 32, 128, 96, 64, 192};
//                                        N_0  N_1  N_2  ...  per expert

// --- 2. Allocate A (contiguous: all experts stacked) ---
// Shape: (M, K, G), RowMajor → stride = (K, 1, M*K)
// Total elements: M * K * G
auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, G});
cutlass::DeviceAllocation<float_e4m3_t> dev_A(M * K * G);

// --- 3. Allocate SFA (scale factors for A, contiguous) ---
auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    cute::make_shape(M, /*N_max=*/256, K, G));
cutlass::DeviceAllocation<float_ue8m0_t> dev_SFA(cute::size(cute::filter_zeros(layout_SFA)));

// --- 4. Allocate B, SFB, C, D per group (pointer arrays) ---
std::vector<cutlass::DeviceAllocation<float_e4m3_t>> dev_B(G);
std::vector<cutlass::DeviceAllocation<float_ue8m0_t>> dev_SFB(G);
std::vector<cutlass::DeviceAllocation<bfloat16_t>> dev_C(G), dev_D(G);

std::vector<const float_e4m3_t*> ptr_B_host(G);
std::vector<const float_ue8m0_t*> ptr_SFB_host(G);
std::vector<const bfloat16_t*> ptr_C_host(G);
std::vector<bfloat16_t*> ptr_D_host(G);

for (int i = 0; i < G; ++i) {
    int N_i = tokens_per_expert[i];
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N_i, K, 1});
    dev_B[i].reset(N_i * K);
    dev_SFB[i].reset(cute::size(cute::filter_zeros(
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N_i, K, 1)))));
    dev_C[i].reset(M * N_i);
    dev_D[i].reset(M * N_i);

    ptr_B_host[i]   = dev_B[i].get();
    ptr_SFB_host[i] = dev_SFB[i].get();
    ptr_C_host[i]   = dev_C[i].get();
    ptr_D_host[i]   = dev_D[i].get();
}

// Copy pointer arrays to device
cutlass::DeviceAllocation<const float_e4m3_t*> dev_ptr_B(G);
cutlass::DeviceAllocation<const float_ue8m0_t*> dev_ptr_SFB(G);
cutlass::DeviceAllocation<const bfloat16_t*> dev_ptr_C(G);
cutlass::DeviceAllocation<bfloat16_t*> dev_ptr_D(G);
dev_ptr_B.copy_from_host(ptr_B_host.data());
dev_ptr_SFB.copy_from_host(ptr_SFB_host.data());
dev_ptr_C.copy_from_host(ptr_C_host.data());
dev_ptr_D.copy_from_host(ptr_D_host.data());

// Copy tokens_per_expert to device
cutlass::DeviceAllocation<int32_t> dev_tokens(G);
dev_tokens.copy_from_host(tokens_per_expert.data());

// --- 5. Build arguments ---
cutlass::KernelHardwareInfo hw_info;
hw_info.device_id = 0;
hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
hw_info.cluster_shape = dim3(2, 1, 1);
hw_info.cluster_shape_fallback = dim3(2, 1, 1);

typename Gemm1SM::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {M, /*max_N=*/256, K, G, dev_tokens.get()},         // problem shape
    {dev_A.get(), dev_ptr_B.get(),                       // mainloop: A, B
     dev_SFA.get(), dev_ptr_SFB.get()},                  // mainloop: SFA, SFB
    {{/*alpha=*/1.0f, /*beta=*/0.0f},                    // epilogue: fusion args
     dev_ptr_C.get(), /*stride_C=*/nullptr,
     dev_ptr_D.get(), /*stride_D=*/nullptr},
    hw_info
};

// --- 6. Run ---
Gemm1SM gemm;
size_t workspace_size = Gemm1SM::get_workspace_size(arguments);
cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
CUTLASS_CHECK(gemm.can_implement(arguments));
CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
CUTLASS_CHECK(gemm.run());
cudaDeviceSynchronize();

// --- 7. Read output ---
// D[i] on device: bfloat16_t buffer of shape (M, N_i), RowMajor
```

### What If My Data Is BF16? Quantization to MXFP8

The kernel **requires** FP8 E4M3 data with E8M0 block scale factors. You cannot pass
BF16 tensors directly. You must quantize BF16 → MXFP8 before calling the kernel.

**MXFP8 block-scaling format recap:**

```
Original BF16 data:   [v₀, v₁, v₂, ..., v₃₁]     ← 32 contiguous elements (one block)

Quantization:
  1. shared_exp = max exponent across the 32 elements
  2. scale_factor = float_ue8m0_t encoding of shared_exp   ← goes into SFA or SFB
  3. each vᵢ is divided by 2^shared_exp and rounded to E4M3 ← goes into A or B
```

Each block of 32 elements shares one `float_ue8m0_t` scale factor (the
`SfVectorSize = 32` that appears throughout the builder). The scale factor array
layout is non-trivial (it follows the tiled MMA layout) and must be computed via
`Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`.

**You need to write (or use) a quantization kernel.** Here is the conceptual logic:

```cpp
// Pseudocode: BF16 → MXFP8 quantization for one block of 32 elements
//
// Input:  bf16_data[32]    (bfloat16_t)
// Output: fp8_data[32]     (float_e4m3_t)
//         scale_factor     (float_ue8m0_t)

// Step 1: Find the maximum absolute value in the block
float amax = 0.0f;
for (int i = 0; i < 32; i++) {
    amax = max(amax, abs(float(bf16_data[i])));
}

// Step 2: Compute shared exponent (E8M0 has no mantissa — it's a pure power of 2)
// E8M0 encodes: value = 2^(stored_bits - 127)   (like IEEE float exponent with bias 127)
int shared_exp = floor(log2(amax));             // biased exponent
float scale = exp2f(shared_exp);                // 2^shared_exp
scale_factor = float_ue8m0_t(scale);            // store as E8M0

// Step 3: Scale each element and convert to E4M3
for (int i = 0; i < 32; i++) {
    float scaled = float(bf16_data[i]) / scale;
    fp8_data[i] = float_e4m3_t(scaled);         // round to E4M3 range
}
```

**In practice** you would implement this as a CUDA kernel that processes the full
matrix. The tricky part is getting the scale factor **layout** right — it must match
the tiled layout that `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB` produces,
which is not a simple row-major array. You can use these helpers to compute the layout
and then fill the scale factors in the expected order.

**Quantization for A (weights, contiguous):**
```cpp
// A is RowMajor (M, K, G) — blocks of 32 along K dimension
// SFA layout: computed by tile_atom_to_shape_SFA(make_shape(M, N, K, G))
auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    cute::make_shape(M, N, K, G));
// Total scale factors: size(filter_zeros(layout_SFA))

// For each expert g, for each row m, for each block of 32 along K:
//   block = A[g][m][k_block*32 .. k_block*32+31]
//   → quantize to 32 × float_e4m3_t + 1 × float_ue8m0_t
```

**Quantization for B (tokens, per-group):**
```cpp
// B[i] is ColumnMajor (N_i, K) — blocks of 32 along K dimension
// SFB layout: computed per group
auto layout_SFB_i = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
    cute::make_shape(M, N_i, K, 1));

// For each token n, for each block of 32 along K:
//   block = B[i][n][k_block*32 .. k_block*32+31]
//   → quantize to 32 × float_e4m3_t + 1 × float_ue8m0_t
```

### End-to-End Data Flow (BF16 → Kernel → BF16)

```
Your BF16 data                        Kernel boundary                    Your BF16 output
═══════════════                        ═══════════════                    ════════════════

weights_bf16 [M,K,G]                                                     D[i] [M,N_i]
    │                                                                       ▲
    ├── quantize ──► A  [float_e4m3_t, M×K×G contiguous]                    │
    └── quantize ──► SFA [float_ue8m0_t, tiled layout]                      │
                              │                                              │
                              ├──► Gemm1SM or Gemm2SM ──► D (bfloat16_t) ───┘
                              │    D_i = α·A_i·B_i^T + β·C_i
tokens_bf16 [N_i,K] per group│                                              │
    │                         │                                              │
    ├── quantize ──► B[i]  [float_e4m3_t, N_i×K per group, ptr array] ──────┘
    └── quantize ──► SFB[i] [float_ue8m0_t, tiled layout, ptr array]

C[i] [bfloat16_t, M×N_i] ──► passed directly (no conversion needed)
alpha, beta [float]  ──────► passed directly
```

**Key takeaway:** The kernel takes MXFP8 in and produces BF16 out. The quantization
BF16 → MXFP8 is your responsibility and happens **before** calling the kernel. The
output D is already BF16 — no dequantization needed on the output side.

### Bridging from Triton `Mgemm_mxfp8` Inputs

#### Triton `Mgemm_mxfp8` Parameter Layouts

| Parameter | Shape | Dtype | Strides | Layout |
|-----------|-------|-------|---------|--------|
| `x_fp8` | `[M, 2560]` | `float8_e4m3fn` | `(2560, 1)` | Row-major. Each row = one token's hidden representation, contiguous. |
| `x_scale` | `[M, 80]` | `uint8` (E8M0) | `(80, 1)` | Row-major. 80 = 2560/32. Each row = one token's per-group exponent scales. |
| `w_fp8` | `[32, 1536, 2560]` | `float8_e4m3fn` | `(3932160, 2560, 1)` | Row-major. Dim 0 = expert, dim 1 = N (2×ffn_dim), dim 2 = K (hidden_dim). |
| `w_scale` | `[32, 1536, 80]` | `uint8` (E8M0) | `(122880, 80, 1)` | Row-major. Same logical layout as `w_fp8`, last dim divided by 32. |
| `cnt` (slice_offs) | `[33]` | `int32` | `(1,)` | 1D. Cumulative prefix sum with 0 at front. |
| **output `y`** | `[M, 1536]` | `bfloat16` | `(1536, 1)` | Row-major. Allocated inside `Mgemm_mxfp8`. |

*(Concrete values shown for the test config: hidden_dim=2560, ffn_dim=768, E=32. In general: N=2×ffn_dim for W13 GEMM, K=hidden_dim, scale last dim = K//32.)*

If you already have PyTorch tensors in the format expected by the Triton `Mgemm_mxfp8`
function (from `triton-moe/llm/kernel/gemm.py` lines 515-522):

```python
def Mgemm_mxfp8(
    x_fp8: torch.Tensor,    # [M_total, K] float8_e4m3fn  — activations (all tokens concatenated)
    x_scale: torch.Tensor,  # [M_total, K//32] uint8      — activation E8M0 scales
    w_fp8: torch.Tensor,    # [E, N, K] float8_e4m3fn     — expert weights
    w_scale: torch.Tensor,  # [E, N, K//32] uint8         — weight E8M0 scales
    cnt: torch.Tensor,      # [E+1] cumulative prefix sum  — expert boundaries
    max_M_per_E: int,
    out_dtype: torch.dtype = torch.bfloat16,
):
```

the Triton kernel computes: `output[tokens_of_e, :] = x_fp8[tokens_of_e, :] @ w_fp8[e].T`
per expert, producing `output [M_total, N]` in bf16.

The CUTLASS kernel computes: `D_i = A_i × B_i^T` per expert, where A = weights
(contiguous), B = activations (ragged pointer array).

There are **four incompatibilities** between the Triton and CUTLASS interfaces:

#### Incompatibility 1: Role naming is swapped

The most confusing part: **CUTLASS "A" = weights, "B" = activations** — opposite of
the neural-network convention used by Triton.

```
Triton                          CUTLASS
──────                          ──────
x_fp8    (activations)    ───►  B   (ragged, pointer array)
w_fp8    (weights)        ───►  A   (contiguous, single buffer)
x_scale  (act scales)     ───►  SFB (ragged, pointer array)
w_scale  (weight scales)  ───►  SFA (contiguous, single buffer)
output   [M_total, N]     ◄──  D   (pointer array, BUT transposed — see below)
cnt      [E+1] prefix sum ───►  tokens_per_expert[G]  (differenced)
```

#### Incompatibility 2: B (activations) must be ColumnMajor — requires transpose

Triton's `x_fp8 [M_total, K]` is RowMajor (each token is a contiguous row of K elements).
CUTLASS B is `[N_i, K]` **ColumnMajor** per group, meaning the token index (N_i) varies
fastest — physically stored as K columns of N_i elements each.

```
Triton x_fp8 (RowMajor):            CUTLASS B[i] (ColumnMajor):
token 0: [k₀, k₁, k₂, ..., k_{K-1}]    k=0: [t₀, t₁, t₂, ..., t_{N_i-1}]
token 1: [k₀, k₁, k₂, ..., k_{K-1}]    k=1: [t₀, t₁, t₂, ..., t_{N_i-1}]
...  contiguous rows                      ...  contiguous columns
```

These are transposed in memory. You must transpose each expert's token block.

#### Incompatibility 3: Scale factor layouts are completely different

Triton uses simple RowMajor `uint8` arrays for scales:
- `x_scale [M_total, K//32]` — one uint8 per group of 32 elements along K
- `w_scale [E, N, K//32]` — same, per expert

CUTLASS uses a **hardware-specific tiled layout** for scale factors, computed by
`Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`. This is NOT RowMajor — the scales
are arranged to match the MMA tile decomposition for efficient hardware access.

There is no simple reshape or transpose that converts between them. You need a
reshuffling kernel that reads scales in RowMajor order and writes them in the
tiled layout order.

#### Incompatibility 4: Output D is transposed relative to Triton's output

Triton output: `[M_total, N]` RowMajor — each token's output is a contiguous row of N
elements, all experts concatenated.

CUTLASS D_i: `[M, N_i]` RowMajor per group — where M = N_hidden (weight dim),
N_i = tokens for expert i. So each expert's output is `[N_hidden, num_tokens]` RowMajor.

```
Triton output[tokens_of_e]:  [num_tokens, N_hidden] RowMajor
CUTLASS D_i:                 [N_hidden, num_tokens] RowMajor   ← transposed!
```

After CUTLASS runs, each D_i must be transposed and gathered into a single contiguous
`[M_total, N]` buffer to match the Triton output format.

#### Complete Preprocessing Pipeline

```python
import torch

def triton_to_cutlass_inputs(x_fp8, x_scale, w_fp8, w_scale, cnt):
    """
    Convert Triton Mgemm_mxfp8 inputs to CUTLASS kernel inputs.

    WARNING: SFA/SFB tiled layout conversion is a placeholder.
    You must implement the actual tiled reshuffling to match
    Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB.
    """
    M_total, K = x_fp8.shape
    E, N, _ = w_fp8.shape

    # --- tokens_per_expert: difference the cumulative prefix sum ---
    # Triton: cnt [E+1], cnt[0]=0, cnt[i+1] = sum of tokens for experts 0..i
    # CUTLASS: tokens_per_expert [G], tokens_per_expert[i] = N_i
    tokens_per_expert = (cnt[1:] - cnt[:-1]).to(torch.int32)  # [E]

    # --- A (weights): w_fp8 [E, N, K] → CUTLASS A [M, K, G] RowMajor ---
    # CUTLASS A[m, k, g] at offset g*M*K + m*K + k  (stride: K, 1, M*K)
    # Triton  w_fp8[e, n, k] at offset e*N*K + n*K + k
    # These are identical when M=N, G=E. No data movement needed!
    A_ptr = w_fp8  # pass w_fp8.data_ptr() directly as CUTLASS A

    # --- B (activations): x_fp8 [M_total, K] RowMajor
    #     → per-group [N_i, K] ColumnMajor (pointer array) ---
    # ColumnMajor [N_i, K] = physically [K, N_i] RowMajor
    # So: transpose each expert's chunk from [N_i, K] to [K, N_i]
    B_buffers = []
    for i in range(E):
        start = cnt[i].item()
        end = cnt[i + 1].item()
        N_i = end - start
        if N_i > 0:
            chunk = x_fp8[start:end, :]               # [N_i, K] RowMajor
            chunk_colmajor = chunk.T.contiguous()      # [K, N_i] RowMajor
                                                       # = [N_i, K] ColumnMajor
            B_buffers.append(chunk_colmajor)
        else:
            B_buffers.append(torch.empty(0, dtype=x_fp8.dtype, device=x_fp8.device))

    # Build device pointer array for B
    B_ptrs = [buf.data_ptr() for buf in B_buffers]

    # --- SFA (weight scales): w_scale [E, N, K//32] RowMajor uint8
    #     → CUTLASS tiled layout via Sm1xxBlkScaledConfig ---
    # >>> THIS REQUIRES A CUSTOM RESHUFFLING KERNEL <<<
    # The tiled layout is hardware-specific and cannot be expressed as a
    # simple transpose or reshape. See note below.
    SFA = reshuffle_to_tiled_layout_SFA(w_scale, N, K, E)  # placeholder

    # --- SFB (activation scales): x_scale [M_total, K//32] RowMajor uint8
    #     → per-group tiled layout (pointer array) ---
    SFB_buffers = []
    for i in range(E):
        start = cnt[i].item()
        end = cnt[i + 1].item()
        N_i = end - start
        if N_i > 0:
            chunk_scale = x_scale[start:end, :]       # [N_i, K//32]
            # >>> RESHUFFLE to tiled layout for this group <<<
            SFB_buffers.append(
                reshuffle_to_tiled_layout_SFB(chunk_scale, N, N_i, K))
        else:
            SFB_buffers.append(
                torch.empty(0, dtype=torch.uint8, device=x_scale.device))

    return A_ptr, B_buffers, SFA, SFB_buffers, tokens_per_expert


def cutlass_output_to_triton_format(D_buffers, cnt, N, M_total):
    """
    Convert CUTLASS per-group output back to Triton's [M_total, N] format.

    CUTLASS D_i: [N_hidden, num_tokens_i] RowMajor (pointer array)
    Triton output: [M_total, N_hidden] RowMajor (single contiguous buffer)
    """
    output = torch.empty(M_total, N, dtype=torch.bfloat16, device=D_buffers[0].device)
    E = len(D_buffers)
    for i in range(E):
        start = cnt[i].item()
        end = cnt[i + 1].item()
        if end > start:
            # D_i is [N_hidden, num_tokens] RowMajor → transpose to [num_tokens, N_hidden]
            output[start:end, :] = D_buffers[i].T
    return output
```

#### Summary of All Preprocessing Steps

```
Triton inputs                                  CUTLASS inputs
═════════════                                  ══════════════

w_fp8 [E, N, K]  ──── no copy needed ────────► A [M=N, K, G=E] contiguous
  (RowMajor)           (layouts match)           (RowMajor, same memory)

w_scale [E, N, K//32] ── RESHUFFLE ──────────► SFA [tiled layout, contiguous]
  (RowMajor uint8)       (tiled layout)          (hardware-specific)

                         ┌─ split by cnt ─┐
x_fp8 [M_total, K] ─────┤                ├──► B[i] [N_i, K] ColumnMajor (ptr array)
  (RowMajor)             │ TRANSPOSE each │      (= [K, N_i] RowMajor physically)
                         └────────────────┘

                         ┌─ split by cnt ──┐
x_scale [M_total, K//32]┤                 ├─► SFB[i] [tiled layout] (ptr array)
  (RowMajor uint8)       │ RESHUFFLE each  │     (hardware-specific)
                         └─────────────────┘

cnt [E+1]  ──── difference ──────────────────► tokens_per_expert [E] int32
  (prefix sum)    cnt[i+1] - cnt[i]

(no equivalent) ─────────────────────────────► alpha=1.0, beta=0.0

(no equivalent) ─────────────────────────────► C = nullptr (void-C mode)

CUTLASS D[i] [N,N_i] ── TRANSPOSE + GATHER ─► output [M_total, N] (Triton format)
  (RowMajor, ptr array)   per expert             (RowMajor, contiguous)
```

#### The Scale Factor Layout Problem

The hardest part is converting between Triton's simple RowMajor scale factors and
CUTLASS's hardware-tiled scale factor layout. The CUTLASS tiled layout is computed by:

```cpp
auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    cute::make_shape(M, N, K, G));
```

This produces a CuTe layout that interleaves scale factors to match the MMA tile
decomposition. The exact tiling depends on the MMA atom shape and is not documented
as a simple formula — it's derived from the hardware's scale factor access pattern.

**Practical options:**
1. **Write a C++/CUDA conversion kernel** that uses the CuTe layout algebra to map
   from RowMajor indices to tiled indices
2. **Quantize directly into tiled layout** — modify your quantization pipeline to
   produce scale factors in the CUTLASS-expected layout from the start
3. **Use the Triton `Mgemm_mxfp8` kernel instead** — if you want to stay in PyTorch
   land with simple RowMajor scales, the Triton kernel handles the scale factor
   access pattern in software (via pointer arithmetic at lines 470-481)

#### Why Not Just Change CUTLASS B to RowMajor?

You might wonder: instead of transposing B, can you change `LayoutB` from `ColumnMajor`
to `RowMajor` in the CUTLASS kernel? The MXF8F6F4 instruction does support non-K-major
inputs (`sm100_blockscaled_umma_builder.inl` line 158):

```cpp
static_assert(UseMxf8f6f4 || (is_k_major_A<...>() && is_k_major_B<...>()),
    "Only MMA.MXF8F6F4 supports non-K major inputs");
```

So in principle, yes — you could modify the `.cu` file to use `LayoutB = RowMajor`.
This would eliminate the B transpose. However:
- The SFB tiled layout would change (it depends on UmmaMajorB)
- You'd need to rebuild the kernel
- Performance may differ (K-major is typically more efficient for the TMA unit)

This is a kernel modification, not a preprocessing step.

## Prerequisites

- NVIDIA Blackwell GPU (SM100/SM101/SM103)
- CUDA Toolkit 12.8 or newer
- CMake 3.18+
- C++17 compiler

## Build

Verified on NVIDIA B200 (SM100a) with CUDA 13.0 and GCC 13.3.

```bash
cd /home/xule/sm100-grouped-gemm
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="100a"
make -j$(nproc)
```

If you need a different compute capability (e.g., SM101), adjust:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="101a"
```

## Run

```bash
# Basic run: 10 groups, all with M=128, K=128, N randomized
./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=128 --k=128 --groups=10

# Fixed dimensions for all groups
./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=256 --n=512 --k=256 --groups=5

# Custom alpha/beta, skip verification for faster profiling
./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=1024 --n=512 --k=1024 --groups=10 \
    --alpha=2 --beta=0.707 --no_verif

# Benchmark from file
./92_blackwell_moe_gemm_blockscaled_rcgrouped --benchmark=test_benchmark.txt

# Cluster shape tuning
./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=256 --k=256 --groups=8 \
    --cluster_m=2 --cluster_n=1

# Enable Programmatic Dependent Launch
./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=128 --k=128 --groups=10 --use_pdl
```

### Benchmark File Format

```
0 256x512x256
1 256x128x256
2 256x256x256
```

Each line: `<group_index> <M>x<N>x<K>`. M and K must be consistent across groups.

## Example Output

Tested on B200 (SM100a), CUDA 13.0, 10 groups with M=128, K=128, N randomized:

```
Running kernel with 1SM MMA config:
  Problem Sizes, Alpha, Beta
    (128,640,128), 4, 4
    (128,112,128), 2, 1
    ...
  Groups      : 10
  Disposition: Passed
  Avg runtime : 0.0981 ms
  TFLOPS      : 1.79

Running kernel with 2SM MMA config:
  ...
  Disposition: Passed
  Avg runtime : 0.0948 ms
  TFLOPS      : 1.85
```

The program runs both 1SM and 2SM kernel configs, performs host-side verification
(unless `--no_verif`), and reports average runtime and TFLOPS.
# cutlass_grouped_gemm
