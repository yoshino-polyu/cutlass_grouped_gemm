# MX FP8 RC Grouped GEMM — CUTLASS Wrapper

## Purpose

Wraps the CUTLASS SM100 MX FP8 block-scaled ragged-contiguous grouped GEMM kernels
so they can be called from Python with the **same interface** as Triton's `Mgemm_mxfp8`.

## Input/Output Contract

**Inputs** (identical to Triton `Mgemm_mxfp8`):

| Tensor    | Shape         | Dtype            | Description                        |
|-----------|---------------|------------------|------------------------------------|
| `x_fp8`   | `[M, K]`      | float8_e4m3fn    | Activations                        |
| `x_scale` | `[M, K//32]`  | uint8 (E8M0)     | Activation block scales            |
| `w_fp8`   | `[E, N, K]`   | float8_e4m3fn    | Expert weights                     |
| `w_scale` | `[E, N, K//32]`| uint8 (E8M0)    | Weight block scales                |
| `cnt`     | `[E+1]`       | int32            | Cumulative token prefix sum        |

**Output**: `y [M, N]` bfloat16

`cnt[i]` to `cnt[i+1]` marks the row range in `x_fp8` assigned to expert `i`.
For expert `i`: `y[cnt[i]:cnt[i+1], :] = x_fp8[cnt[i]:cnt[i+1], :] @ w_fp8[i].T`.

## Architecture

```
Python                    C++ / CUDA
──────                    ──────────
api.py (Layer 1)     ──→  binding.cu (Layer 3)  ──→  launcher.cu (Layer 4)
  │                                                       │
jit_module.py              preprocess_kernels.cuh ◄───────┤
(Layer 2, JIT build)       kernel_traits.cuh ◄────────────┘
                           (Layer 5, CUTLASS types)
```

## File Descriptions

### `kernel_traits.cuh` — CUTLASS type definitions

Defines all CUTLASS template types for the grouped GEMM:
- `ElementA/B = mx_float8_t<float_e4m3_t>`, `ElementC/D = bfloat16_t`
- `LayoutA = RowMajor`, `LayoutB = ColumnMajor`, `LayoutC = RowMajor`
- `Gemm1SM`: tile `128×256×128`, schedule `KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100`
- `Gemm2SM`: tile `256×256×128`, schedule `KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100`
- Uses `MoEProblemShape` (fixed M/K, variable N per group)
- Default epilogue (no FusionOperation): `D = accumulator` cast to bfloat16

### `preprocess_kernels.cuh` — GPU preprocessing kernels

Five CUDA kernels that bridge Triton↔CUTLASS data format differences:

1. **`compute_tokens_per_expert`** — `cnt[E+1]` → `tokens[E]` via differencing
2. **`transpose_b_batched`** — Per-expert: activations RowMajor → ColumnMajor
3. **`reshuffle_scale_factors`** — RowMajor `[rows, K_sf]` → SfAtom-tiled layout (for weight SFA)
4. **`reshuffle_sfb_per_expert`** — Same tiling but per-expert (for activation SFB)
5. **`transpose_output_gather`** — CUTLASS output `D_i[N, tokens_i]` → `output[tokens_i, N]` with scatter
6. **`build_pointer_arrays`** — Populates `ptr_B[E]`, `ptr_SFB[E]`, `ptr_D[E]` from base+offsets

### `launcher.cu` — C++ orchestrator

Main entry point `mxfp8_rc_grouped_gemm_run(...)`. Executes a 7-step GPU pipeline:

```
Step 1: compute_tokens_per_expert    (cnt → tokens_per_expert)
Step 2: transpose_b_batched          (x_fp8 → B_transposed, col-major per expert)
Step 3: reshuffle_scale_factors      (w_scale → SFA_tiled, contiguous)
Step 4: reshuffle_sfb_per_expert     (x_scale → SFB_tiled, per expert)
Step 5: build_pointer_arrays         (fill ptr_B, ptr_SFB, ptr_D)
Step 6: CUTLASS grouped GEMM        (A=weights, B=activations → D per expert)
Step 7: transpose_output_gather      (D per expert → output [M, N])
```

All scratch memory comes from a single pre-allocated workspace buffer, partitioned
into sub-allocations at fixed offsets (computed on host).

Dispatches between `Gemm1SM` and `Gemm2SM` via the `mma_sm` parameter.

### `binding.cu` — TVM-FFI export

Minimal file: forward-declares `mxfp8_rc_grouped_gemm_run` and exports it via
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`.

### `jit_module.py` — JIT compilation

Uses FlashInfer's `gen_jit_spec()` to compile `launcher.cu` + `binding.cu` with:
- `-gencode=arch=compute_100a,code=sm_100a`
- `--expt-relaxed-constexpr`
- Include paths: CUTLASS headers, FlashInfer csrc, wrapper dir

### `api.py` — Python API

```python
def mxfp8_rc_grouped_gemm(
    x_fp8, x_scale, w_fp8, w_scale, cnt, mma_sm=1
) -> torch.Tensor:
```
- Pre-allocates `[M, N]` bfloat16 output
- Manages a cached workspace buffer (auto-grows if too small)
- Loads the JIT-compiled module via `get_module()`

### `test_wrapper.py` — End-to-end test

- Generates random FP8 data + E8M0 scales
- Computes reference via dequantize → FP32 matmul → BF16 cast
- Compares CUTLASS output to reference (tolerance ~0.1 abs for FP8)
- Tests both `mma_sm=1` and `mma_sm=2`

## Key Design Decisions

### 1. Role Swap (Triton → CUTLASS)

CUTLASS MoE grouped GEMM has:
- **A** = shared/contiguous matrix, RowMajor `[M_cutlass, K]` with `L` groups
- **B** = per-group pointer array, ColumnMajor `[N_i, K]`

Our mapping:
- **A** ← `w_fp8 [E, N, K]` (weights, contiguous). CUTLASS `M_cutlass = N_hidden`.
- **B** ← `x_fp8` slices (activations, per expert). CUTLASS `N_i = tokens_per_expert[i]`.

So CUTLASS computes `D_i[N_hidden, tokens_i] = W_i[N_hidden, K] @ X_i[tokens_i, K]^T`.

### 2. Why Transpose B?

`x_fp8` is RowMajor `[M, K]`. CUTLASS needs B in ColumnMajor.
ColumnMajor `[tokens_i, K]` stores data as `(k, n)` → offset `k * tokens_i + n`.
The `transpose_b_batched` kernel does this conversion per expert.

### 3. Scale Factor Reshuffling (SfAtom Layout)

CUTLASS SM100 expects scale factors in a hardware-specific tiled layout defined by `SfAtom`.
For MX FP8 (SFVecSize=32, K-major):

```
Atom shape: (128 MN, 4 K_sf) = 512 unique values
Indexing:
  tile_mn = mn / 128,  tile_k = k_sf / 4
  local_mn = mn % 128, local_k = k_sf % 4
  mn_0 = local_mn % 32, mn_1 = local_mn / 32
  within_atom = mn_0 * 16 + mn_1 * 4 + local_k
  atom_idx = tile_mn * num_k_tiles + tile_k  (K innermost)
  dest = atom_idx * 512 + within_atom
```

### 4. Output Transpose + Gather

CUTLASS outputs `D_i[N_hidden, tokens_i]` RowMajor per expert.
The final output needs to be `[M, N_hidden]` with expert results at `cnt[i]:cnt[i+1]`.
The `transpose_output_gather` kernel does the transpose and scatter in one pass.

### 5. Single Workspace Buffer

All GPU scratch memory is carved from one large buffer to avoid multiple allocations.
Layout (in order):
```
tokens_per_expert [E×4B] | expert_B_offsets [E×8B] | expert_sfb_offsets [E×8B] |
expert_D_offsets [E×8B] | B_transposed [M×K] | SFA_tiled [E×tiled_size] |
SFB_tiled [E×tiled_size] | D_scratch [E×N×M×2B] | ptr_arrays [3×E×8B] |
CUTLASS_workspace [remainder]
```

## Usage

```python
from wrapper import mxfp8_rc_grouped_gemm

y = mxfp8_rc_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt, mma_sm=1)
```

## Testing

```bash
python -m wrapper.test_wrapper
```
