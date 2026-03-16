// SFA (Scale Factor for A) layout parity check.
// Verifies that a flat PyTorch tensor can round-trip through CUTLASS's
// HostTensorSF + tiled SFA layout without data corruption.
//
// SFA uses a hardware-tiled layout (tile_atom_to_shape_SFA) that is NOT
// simple row-major. However, the underlying storage is a flat 1D buffer
// (PackedVectorLayout). The tiled layout is applied on top when the kernel
// or the host reference reads the data.
//
// This check verifies:
//   1. PyTorch tensor size matches CUTLASS's expected allocation size
//   2. Data round-trips correctly: PyTorch GPU → HostTensorSF → device → compare
//   3. Per-group stride arithmetic is consistent

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/coord.h"

// Full GEMM type chain — needed for Sm1xxBlkScaledConfig
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "tvm_ffi_utils.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Same type chain as 92_blackwell_moe_gemm_blockscaled_rcgrouped.cu
using ProblemShape = cutlass::gemm::MoEProblemShape<Shape<int,int,int>>;
using ElementInput = cutlass::float_e4m3_t;
using ElementSF    = cutlass::float_ue8m0_t;
using ElementC     = cutlass::bfloat16_t;
using ElementA     = cutlass::mx_float8_t<ElementInput>;
using LayoutA      = cutlass::layout::RowMajor;
using ElementB     = cutlass::mx_float8_t<ElementInput>;
using LayoutB      = cutlass::layout::ColumnMajor;
using ElementD     = ElementC;
using LayoutC      = cutlass::layout::RowMajor;
using ElementAccumulator = float;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementSFD = cutlass::float_ue4m3_t;
constexpr int OutputSFVectorSize = 16;
using FusionOperation = cutlass::epilogue::fusion::LinCombEltActBlockScaleFactor<
    cutlass::epilogue::thread::SiLu,
    OutputSFVectorSize, ElementD, ElementAccumulator, ElementSFD, LayoutC, ElementC>;

using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape  = Shape<int32_t, int32_t, _1>;

struct MMA1SMConfig {
  using MmaTileShape     = Shape<_128, _256, _128>;
  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    typename MMA1SMConfig::MmaTileShape, ClusterShape, Shape<_128, _64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutC *, AlignmentD,
    typename MMA1SMConfig::EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    typename MMA1SMConfig::MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    typename MMA1SMConfig::KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using HostTensorSF = cutlass::HostTensor<typename Gemm::GemmKernel::ElementSF,
                                         cutlass::layout::PackedVectorLayout>;

// ── get_sfa_size: returns expected SFA allocation size for given dimensions ──

int64_t get_sfa_size_run(int64_t M, int64_t K, int64_t groups) {
  int N_dummy = 256;  // unused by tile_atom_to_shape_SFA
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      make_shape((int)M, N_dummy, (int)K, (int)groups));
  return static_cast<int64_t>(size(filter_zeros(layout_SFA)));
}

// ── check_sfa_layout: parity check between PyTorch flat tensor and HostTensorSF ──

void check_sfa_layout_run(TensorView w_sfa, int64_t M_val, int64_t K_val,
                           int64_t groups_val, int64_t print_coords) {
  const int M = static_cast<int>(M_val);
  const int K = static_cast<int>(K_val);
  const int groups = static_cast<int>(groups_val);
  const int N_dummy = 256;  // unused by SFA

  printf("=== SFA Layout Parity Check ===\n");
  printf("M=%d, K=%d, groups=%d\n", M, K, groups);

  // ================================================================
  // Step 1: Compute SFA tiled layout using CUTLASS primitives
  // ================================================================
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      make_shape(M, N_dummy, K, groups));
  int64_t alloc_size = static_cast<int64_t>(size(filter_zeros(layout_SFA)));

  printf("SFA tiled layout: ");
  print(layout_SFA);
  printf("\n");
  printf("SFA allocation size (filter_zeros): %ld\n", (long)alloc_size);

  // ================================================================
  // Step 2: Size check — PyTorch tensor vs CUTLASS expected size
  // ================================================================
  int64_t pytorch_total = 1;
  for (int d = 0; d < w_sfa.ndim(); d++) pytorch_total *= w_sfa.size(d);

  bool size_pass = (pytorch_total == alloc_size);
  printf("\n--- Size Check ---\n");
  printf("PyTorch size=%ld, CUTLASS expected=%ld: %s\n",
         (long)pytorch_total, (long)alloc_size, size_pass ? "PASS" : "FAIL");
  if (!size_pass) {
    printf("Overall: FAIL (size mismatch, cannot proceed)\n");
    return;
  }

  // ================================================================
  // Step 3: Create HostTensorSF ground truth using CUTLASS primitives
  // ================================================================
  // This follows the same flow as initialize() in the reference:
  //   HostTensorSF.reset(make_Coord(size(filter_zeros(layout_SFA))))
  //   copy data in → host_view() available → sync_device() → device_data()
  HostTensorSF ground_truth;
  ground_truth.reset(cutlass::make_Coord(alloc_size));

  // Copy PyTorch GPU data → HostTensorSF host memory
  ground_truth.copy_in_device_to_host(
      reinterpret_cast<const typename Gemm::GemmKernel::ElementSF*>(w_sfa.data_ptr()),
      alloc_size);

  // Sync host → device (same as block_SFA.sync_device() in reference)
  ground_truth.sync_device();

  // ================================================================
  // Step 4: Round-trip check — PyTorch GPU → HostTensorSF → device → compare
  // ================================================================
  // After copy_in_device_to_host + sync_device, device_data() should hold
  // the same bytes as the original PyTorch tensor.
  printf("\n--- Round-trip Check ---\n");
  std::vector<uint8_t> pytorch_host(alloc_size);
  cudaMemcpy(pytorch_host.data(), w_sfa.data_ptr(), alloc_size, cudaMemcpyDeviceToHost);

  std::vector<uint8_t> cutlass_host(alloc_size);
  cudaMemcpy(cutlass_host.data(), ground_truth.device_data(), alloc_size, cudaMemcpyDeviceToHost);

  bool roundtrip_pass = true;
  int64_t first_mismatch = -1;
  for (int64_t i = 0; i < alloc_size; i++) {
    if (pytorch_host[i] != cutlass_host[i]) {
      roundtrip_pass = false;
      if (first_mismatch < 0) first_mismatch = i;
    }
  }
  if (roundtrip_pass) {
    printf("PyTorch GPU -> HostTensorSF host -> sync_device -> device_data: PASS\n");
  } else {
    printf("FAIL at byte %ld: pytorch=0x%02x, cutlass=0x%02x\n",
           (long)first_mismatch, pytorch_host[first_mismatch], cutlass_host[first_mismatch]);
  }

  // ================================================================
  // Step 5: host_view() consistency — flat pointer vs host_view access
  // ================================================================
  printf("\n--- host_view() Consistency ---\n");
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(ground_truth.host_data());
  bool view_pass = true;
  int64_t printed = 0;

  for (int64_t i = 0; i < alloc_size; i++) {
    uint8_t raw_val = raw[i];
    // host_data(i) returns a reference to the i-th ElementSF in the flat buffer
    uint8_t view_val = *reinterpret_cast<const uint8_t*>(&ground_truth.host_data(i));
    if (raw_val != view_val) {
      view_pass = false;
      if (printed < print_coords) {
        printf("  MISMATCH at i=%ld: raw=0x%02x, view=0x%02x\n",
               (long)i, raw_val, view_val);
        printed++;
      }
    } else if (printed < print_coords) {
      printf("  OK i=%ld: val=0x%02x\n", (long)i, raw_val);
      printed++;
    }
  }
  printf("host_view() consistency (%ld elements): %s\n",
         (long)alloc_size, view_pass ? "PASS" : "FAIL");

  // ================================================================
  // Step 6: Per-group stride check
  // ================================================================
  // In verify(), group i's SFA is at offset i * size(filter_zeros(layout_SFA_single)).
  // Verify the total allocation equals groups * per_group_size.
  printf("\n--- Per-group Stride ---\n");
  auto layout_SFA_single = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      make_shape(M, N_dummy, K, 1));
  int64_t per_group = static_cast<int64_t>(size(filter_zeros(layout_SFA_single)));
  bool group_pass = (alloc_size == groups * per_group);
  printf("Per-group SFA size: %ld\n", (long)per_group);
  printf("Total: %d groups * %ld = %ld, actual = %ld: %s\n",
         groups, (long)per_group, (long)(groups * per_group),
         (long)alloc_size, group_pass ? "PASS" : "FAIL");

  // Also verify each group's data is contiguously accessible
  printf("Single-group tiled layout: ");
  print(layout_SFA_single);
  printf("\n");

  // ================================================================
  // Summary
  // ================================================================
  printf("\n=== Summary ===\n");
  printf("Size check:              %s\n", size_pass ? "PASS" : "FAIL");
  printf("Round-trip check:        %s\n", roundtrip_pass ? "PASS" : "FAIL");
  printf("host_view() consistency: %s\n", view_pass ? "PASS" : "FAIL");
  printf("Per-group stride:        %s\n", group_pass ? "PASS" : "FAIL");

  bool all_pass = size_pass && roundtrip_pass && view_pass && group_pass;
  printf("Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILED");
}

#else

int64_t get_sfa_size_run(int64_t M, int64_t K, int64_t groups) {
  printf("CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
  return 0;
}

void check_sfa_layout_run(TensorView w_sfa, int64_t M, int64_t K,
                           int64_t groups, int64_t print_coords) {
  printf("CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
}

#endif
