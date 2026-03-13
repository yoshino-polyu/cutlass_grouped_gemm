# Bugs Found and Files Modified

Summary of all bugs discovered and fixes applied while working toward running
`wrapper/test_wrapper.py` (the end-to-end correctness test for the CUTLASS
MX FP8 RC grouped GEMM wrapper).

---

## Files Modified

| File | Kind of change |
|---|---|
| `csrc/tvm_ffi_utils.h` | Removed `using tvm::ffi::Tensor;` to fix namespace collision |
| `wrapper/kernel_traits.cuh` | Removed `using namespace cute;`; changed `ElementC` to `void`; set `AlignmentC = 0` |
| `wrapper/launcher.cu` | Fixed element-type casts; fixed epilogue args; fixed B_transposed allocation; fixed max_tokens computation |
| `wrapper/api.py` | Fixed `_estimate_workspace_size` to match corrected B_transposed allocation |

---

## Bug 1 — `Tensor` name is ambiguous (compilation error)

**Symptom:** Compilation fails inside CUTLASS epilogue headers with
`error: "Tensor" is ambiguous`.

**Root cause:** Two unrelated `Tensor` types are pulled into global scope:
- `tvm::ffi::Tensor` via `using tvm::ffi::Tensor;` in `csrc/tvm_ffi_utils.h`
- `cute::Tensor` via `using namespace cute;` in `wrapper/kernel_traits.cuh`

Inside CUTLASS headers that already do `using namespace cute;`, the compiler
finds both and cannot resolve which one to use.

**Fix:**
- `csrc/tvm_ffi_utils.h` — removed the line `using tvm::ffi::Tensor;`
  (only `using tvm::ffi::TensorView;` is kept, which has no collision).
- `wrapper/kernel_traits.cuh` — removed `using namespace cute;` and
  fully qualified all CuTe types (e.g. `cute::Shape`, `cute::_128`).

---

## Bug 2 — Element type mismatch in mainloop arguments (compilation error)

**Symptom:** Compilation fails with cannot-convert errors when constructing
the CUTLASS mainloop `Arguments` struct:
`const mx_float8_t<float_e4m3_t>*` cannot initialize `const float_e4m3_t*`.

**Root cause:** `ElementA = cutlass::mx_float8_t<float_e4m3_t>` is a **tag
type** used by the CUTLASS builder to select block-scaled tensor-op
pipelines. The actual storage element type used in the mainloop `Arguments`
struct is the unwrapped `float_e4m3_t`, exposed as
`CollectiveMainloop::ArrayElementA`.

**Fix (`wrapper/launcher.cu`):**
```cpp
using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
// ... then use reinterpret_cast<const ArrayElementA*>(A_ptr) instead of ElementA*
```

---

## Bug 3 — NaN output from non-void-C epilogue (runtime, all outputs NaN)

**Symptom:** Every element of every expert's output is NaN. CUTLASS reports
`can_implement`, `initialize`, and `run` all succeed.

**Root cause:** `ElementC` was set to `cutlass::bfloat16_t`. This tells CUTLASS
the epilogue has a C input matrix (`D = alpha * Acc + beta * C`). CUTLASS
unconditionally issues TMA loads for C. Because we pass `nullptr` for the C
pointer array, the TMA reads from address 0 → garbage/NaN bit patterns.

**Fix (`wrapper/kernel_traits.cuh`):**
```cpp
using ElementC = void;    // was cutlass::bfloat16_t
constexpr int AlignmentC = 0;   // was 128/sizeof_bits<ElementD>
```
The void-C epilogue computes `D = alpha * Acc` with no C read.

---

## Bug 4 — Wrong epilogue argument field count (compilation error)

**Symptom:** Compilation error when constructing the epilogue `Arguments`
with 3 fields `{fusion_args, ptr_D, nullptr}`.

**Root cause:** Even with `ElementC = void`, the CUTLASS epilogue `Arguments`
struct still has 5 fields. The `ptr_C` field exists as `const void**` (it's
just never dereferenced at runtime).

**Fix (`wrapper/launcher.cu`):**
```cpp
{fusion_args,
 nullptr,   // ptr_C array (void-C, not read)
 nullptr,   // stride_C
 reinterpret_cast<ElementD**>(ptr_D_dev),
 nullptr}   // stride_D
```

---

## Bug 5 — B_transposed buffer overflow (runtime, Expert 1 always NaN)

**Symptom:** Expert index 1 always produces NaN. Exactly 4096 = 128 x 32
NaN elements regardless of token count. All other experts produce correct
results.

**Root cause:** The B_transposed workspace region was allocated as
`M_total * K` bytes (the **total** across all experts). But per-expert byte
offsets were computed as `i * max_tokens_ub * K` where
`max_tokens_ub = M_total`. This means expert 0 gets `[0, M_total*K)`, and
expert 1 starts at byte `M_total * K` — **past the end of the buffer**.

Expert 1's B_transposed pointer landed in the SFA_tiled region. The
transpose kernel wrote activation data there (corrupting SFA), and CUTLASS
read SFA scale-factor bytes (value 0x7F = 127) as FP8 data — 0x7F is
**NaN in float8_e4m3fn format**.

**Diagnostic evidence:**
```
B_transposed expert 1 (first 16 bytes):
  0x7f 0x7f 0x7f 0x7f ...   ← these are SFA bytes (scale=127), not FP8 data
```

**Fix (`wrapper/launcher.cu` line 181, `wrapper/api.py` line 36):**
```cpp
// Before (WRONG): only M_total * K total
offset = B_transposed_off + (int64_t)M_total * K;

// After (CORRECT): E * max_tokens_ub * K (each expert gets its own region)
offset = B_transposed_off + (int64_t)E * max_tokens_ub * K;
```
Matching change in `_estimate_workspace_size` in `api.py`.

---

## Bug 6 — max_tokens passed as M_total to CUTLASS (runtime, minor)

**Symptom:** Not directly observed to cause errors on its own, but incorrect
per the CUTLASS API contract.

**Root cause:** `MoEProblemShape.max_n` should be the **maximum** number of
tokens assigned to any single expert, not the total token count.  We were
passing `M_total` (sum of all experts' tokens).

**Fix (`wrapper/launcher.cu`):**
Copy `cnt` to host, compute `max_tokens = max(cnt[i+1] - cnt[i])`, and pass
that to `run_gemm` instead of `max_tokens_ub = M_total`.

---

## Open Bug 7 — Scale factor tiling dimension mismatch (runtime, wrong output with non-uniform scales)

**Status: NOT YET FIXED**

**Symptom:** With random data and random E8M0 scale factors, the CUTLASS
output has large numerical errors vs. the FP32 reference
(`max_abs_diff ~ 450000`). With uniform data (all ones) and uniform
scales (all 127), the output is exactly correct.

**Root cause (suspected):** Our `tiled_sf_size()` and reshuffle kernels
compute the number of K-tiles as `ceil(K_sf / 4)` where `K_sf = K / 32`.
But CUTLASS's `tile_atom_to_shape_SFA` receives the **full K** (not K_sf)
and passes it to `tile_to_shape(SfAtom, make_shape(M, K, L), ...)`.

The SfAtom has K-mode size 4, but each K position in the atom represents
**one scale-factor value** (covering SFVecSize=32 FP8 elements). Internally,
CUTLASS's `tile_to_shape` divides K by the atom's effective K coverage
(SFVecSize × 4 = 128) rather than by the raw atom K-mode size (4). The
exact division rule is encoded in the atom's stride structure where the K
stride is `Stride<_0, _1>` — the `_0` stride on the SFVecSize sub-mode
means those positions are **degenerate** (broadcast), so the effective K
tile count is `ceil(K / 128)`, not `ceil(K / 4)`.

For K=256: `ceil(256/128) = 2`, which matches our `ceil(K_sf/4) = ceil(8/4) = 2`.
This is why the all-ones test passes — the tile count happens to be the same.

However, the tiling formula within each atom may still be wrong. The
`within_atom` index mapping and the atom-to-tile ordering need to be
verified against what CUTLASS actually reads at runtime. This is the most
likely cause of the remaining numerical errors with non-uniform data.

**Additionally**, with non-uniform FP8 data and uniform scale=127, there
are still large errors (`max_abs_diff ~ 93`), suggesting there may also be
an issue with the B transpose stride or the output transpose logic that is
independent of scale factor tiling.

---

## Summary Table

| # | Bug | Severity | Status | Root cause |
|---|---|---|---|---|
| 1 | `Tensor` ambiguous | Build error | Fixed | Namespace collision between `cute::Tensor` and `tvm::ffi::Tensor` |
| 2 | Element type mismatch | Build error | Fixed | `mx_float8_t` is a tag type; mainloop uses unwrapped `float_e4m3_t` |
| 3 | All-NaN output | Runtime | Fixed | Non-void `ElementC` triggers TMA load from nullptr |
| 4 | Wrong epilogue field count | Build error | Fixed | Void-C epilogue still has 5-field `Arguments` struct |
| 5 | Expert 1 NaN | Runtime | Fixed | B_transposed buffer allocated `M_total*K` instead of `E*M_total*K` |
| 6 | Wrong max_tokens | Runtime | Fixed | Passed sum-of-tokens instead of max-per-expert to `MoEProblemShape` |
| 7 | Wrong output values | Runtime | **Open** | Scale factor tiling and/or B transpose layout mismatch with CUTLASS expectations |
