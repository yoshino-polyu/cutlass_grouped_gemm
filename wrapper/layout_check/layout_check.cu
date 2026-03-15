#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/coord.h"

#include "tvm_ffi_utils.h"

void check_block_a_layout_run(TensorView w_fp8, int64_t print_coords) {
  // w_fp8 is [E, N, K] in PyTorch terms — the weight tensor for grouped GEMM.
  // In CUTLASS, block_A has shape [M, K, G] where M=N, G=E, with RowMajor layout.
  // Physically the E experts are stacked as a single (E*N, K) RowMajor matrix.
  // This test verifies that the PyTorch tensor's memory is correctly interpretable
  // by both CUTLASS 2.x HostTensor and CUTLASS 3.x CuTe layout primitives.
  TVM_FFI_ICHECK_EQ(w_fp8.ndim(), 3) << "w_fp8 must be a 3D tensor [E, N, K]";

  const int32_t E = static_cast<int32_t>(w_fp8.size(0));
  const int32_t N = static_cast<int32_t>(w_fp8.size(1));
  const int32_t K = static_cast<int32_t>(w_fp8.size(2));
  const int64_t total = (int64_t)E * N * K;

  printf("=== block_A Layout Parity Check ===\n");
  printf("Tensor shape: [E=%d, N=%d, K=%d]\n", E, N, K);
  printf("PyTorch strides: [%ld, %ld, %ld]\n",
         (long)w_fp8.stride(0), (long)w_fp8.stride(1), (long)w_fp8.stride(2));

  // ================================================================
  // Step 1: Build HostTensor<float_e4m3_t, RowMajor> — CUTLASS 2.x view
  // ================================================================
  // In CUTLASS grouped GEMM, block_A for all E experts is a single
  // contiguous (E*N) x K row-major matrix. We construct a HostTensor
  // with that extent and copy the GPU data into it.
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using HostTensorA = cutlass::HostTensor<ElementA, LayoutA>;

  auto extent_A = cutlass::MatrixCoord(E * N, K);
  HostTensorA host_A(extent_A, LayoutA::packed(extent_A), /*device_backed=*/false);
  host_A.copy_in_device_to_host(
      reinterpret_cast<const ElementA*>(w_fp8.data_ptr()), total);

  printf("HostTensor extent: (%d, %d), stride: %ld\n",
         extent_A.row(), extent_A.column(), (long)host_A.stride(0));

  // ================================================================
  // Step 2: Build CuTe layout — CUTLASS 3.x ground truth
  // ================================================================
  // TagToStrideA_t<RowMajor> = Stride<int64_t, Int<1>, int64_t> for [M, K, L].
  // make_cute_packed_stride fills in stride_M=K, stride_K=1, stride_G=M*K.
  //
  // Why PyTorch shape (E, N, K) and CuTe shape (N, K, E) produce the same
  // offsets despite listing dimensions in different order:
  //
  //   The two APIs use different axis-ordering conventions, but bind the
  //   same strides to the same physical dimensions:
  //
  //   Memory position       | PyTorch          | CuTe layout_A
  //   (inner -> outer)      |                  |
  //   ----------------------+------------------+-------------------
  //   Innermost (stride 1)  | K  (dim 2)       | K  (dim 1)
  //   Middle    (stride K)  | N  (dim 1)       | M=N (dim 0)
  //   Outermost (stride NK) | E  (dim 0)       | G=E (dim 2)
  //
  //   PyTorch orders dims as (outermost, ..., innermost): dim 0 has
  //   the largest stride.  CuTe orders dims by GEMM semantics (M, K, L);
  //   the shape tuple (N, K, E) does NOT imply memory nesting — the
  //   strides (K, 1, N*K) do.
  //
  //   So the offset formulas are identical (addition is commutative):
  //     PyTorch:  e * (N*K) + n * K + k * 1
  //     CuTe:    layout_A(m=n, k=k, g=e) = n * K + k * 1 + e * (N*K)
  //
  using StrideA = cutlass::detail::TagToStrideA_t<cutlass::layout::RowMajor>;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {N, K, E});
  auto layout_A = cute::make_layout(cute::make_shape(N, K, E), stride_A);

  printf("CuTe layout_A: ");
  cute::print(layout_A);
  printf("\n");

  // ================================================================
  // Step 3: Parity check — HostTensor.at() vs CuTe layout offset
  // ================================================================
  // For every (m, k, g) in block_A's logical space:
  //   - CuTe computes an offset via layout_A(m, k, g)
  //   - HostTensor computes an offset via RowMajor(MatrixCoord(g*N+m, k))
  // Both use CUTLASS primitives. If they produce the same byte value
  // from the same raw data, the layout is semantically correct.
  printf("\n--- HostTensor vs CuTe Parity Check ---\n");
  bool parity_pass = true;
  int64_t checked = 0;
  int64_t printed = 0;

  // Get the raw host pointer for CuTe-indexed access
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(host_A.host_data());

  for (int g = 0; g < E; ++g) {
    for (int m = 0; m < N; ++m) {
      for (int k = 0; k < K; ++k) {
        // CuTe offset from the 3.x layout
        int64_t cute_offset = layout_A(m, k, g);

        // HostTensor access via 2.x RowMajor coordinate
        // block_A stacks G groups of M rows: row = g*M + m, col = k
        auto coord = cutlass::MatrixCoord(g * N + m, k);
        int64_t host_offset = host_A.offset(coord);

        uint8_t cute_val = raw[cute_offset];
        uint8_t host_val = *reinterpret_cast<const uint8_t*>(
            &host_A.at(coord));

        if (cute_offset != host_offset || cute_val != host_val) {
          if (printed < print_coords) {
            printf("  MISMATCH (m=%d, k=%d, g=%d): cute_off=%ld host_off=%ld "
                   "cute_val=0x%02x host_val=0x%02x\n",
                   m, k, g, (long)cute_offset, (long)host_offset,
                   cute_val, host_val);
            printed++;
          }
          parity_pass = false;
        } else if (printed < print_coords) {
          printf("  OK (m=%d, k=%d, g=%d): offset=%ld val=0x%02x\n",
                 m, k, g, (long)cute_offset, cute_val);
          printed++;
        }
        checked++;
      }
    }
  }
  printf("Coordinates checked: %ld\n", (long)checked);
  printf("HostTensor vs CuTe parity: %s\n", parity_pass ? "PASS" : "FAIL");

  // ================================================================
  // Step 4: Stride consistency — HostTensor stride vs CuTe stride
  // ================================================================
  // HostTensor's RowMajor leading dimension should equal CuTe's stride
  // for the M mode (dim 0). Both are computed by CUTLASS from the shape.
  printf("\n--- Stride Consistency ---\n");
  auto cute_stride_M = cute::get<0>(stride_A);
  int64_t host_stride = host_A.stride(0);  // RowMajor leading dim = K
  bool stride_pass = (host_stride == cute_stride_M);
  printf("HostTensor stride(0) = %ld, CuTe stride_M = %ld: %s\n",
         (long)host_stride, (long)cute_stride_M,
         stride_pass ? "PASS" : "FAIL");

  // Also verify total size agrees
  bool size_pass = ((int64_t)cute::size(layout_A) == total &&
                    (int64_t)host_A.size() == total);
  printf("size(layout_A)=%ld, host_A.size()=%ld, E*N*K=%ld: %s\n",
         (long)cute::size(layout_A), (long)host_A.size(), (long)total,
         size_pass ? "PASS" : "FAIL");

  // ================================================================
  // Step 5: PyTorch stride sanity — is the tensor even contiguous?
  // ================================================================
  // For the PyTorch tensor to be directly usable as block_A, it must be
  // row-major contiguous: stride = [N*K, K, 1]. We check this without
  // manually mapping dims — just compare against what a packed (E,N,K)
  // tensor should have.
  printf("\n--- PyTorch Contiguity ---\n");
  bool contiguous = (w_fp8.stride(2) == 1 &&
                     w_fp8.stride(1) == K &&
                     w_fp8.stride(0) == (int64_t)N * K);
  printf("Expected strides [%ld, %d, 1], got [%ld, %ld, %ld]: %s\n",
         (long)N * K, K,
         (long)w_fp8.stride(0), (long)w_fp8.stride(1), (long)w_fp8.stride(2),
         contiguous ? "PASS" : "FAIL");

  // ================================================================
  // Summary
  // ================================================================
  printf("\n=== Summary ===\n");
  printf("HostTensor vs CuTe parity: %s\n", parity_pass ? "PASS" : "FAIL");
  printf("Stride consistency:        %s\n", stride_pass ? "PASS" : "FAIL");
  printf("Size consistency:          %s\n", size_pass ? "PASS" : "FAIL");
  printf("PyTorch contiguity:        %s\n", contiguous ? "PASS" : "FAIL");

  bool all_pass = parity_pass && stride_pass && size_pass && contiguous;
  printf("Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILED");
}
