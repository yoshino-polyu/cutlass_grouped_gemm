// Condensed RC grouped GEMM test where block_A comes from a PyTorch TensorView.
// Everything else (B, SFB, SFA, C, D) is allocated and initialized locally.
// See 92_blackwell_moe_gemm_blockscaled_rcgrouped.cu for full documentation.

#include <iostream>
#include <vector>
#include <float.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "helper.h"
#include "tvm_ffi_utils.h"

using namespace cute;

// ── Type definitions (same as reference) ──

using ProblemShape = cutlass::gemm::MoEProblemShape<Shape<int,int,int>>;
using ElementInput = cutlass::float_e4m3_t;
using ElementSF    = cutlass::float_ue8m0_t;
using ElementC     = cutlass::bfloat16_t;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::mx_float8_t<ElementInput>;
using LayoutA  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB = cutlass::mx_float8_t<ElementInput>;
using LayoutB  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;

using ElementD = ElementC;
using LayoutC  = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;

using ElementSFD = cutlass::float_ue4m3_t;
constexpr int OutputSFVectorSize = 16;
using FusionOperation = cutlass::epilogue::fusion::LinCombEltActBlockScaleFactor<
    cutlass::epilogue::thread::SiLu,
    OutputSFVectorSize, ElementD, ElementAccumulator, ElementSFD, LayoutC, ElementC>;

using ArchTag        = cutlass::arch::Sm100;
using OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape   = Shape<int32_t, int32_t, _1>;

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

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using Sm1xxBlockScaledOutputConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<
    OutputSFVectorSize,
    cute::is_same_v<typename FusionOperation::GmemLayoutTagScalefactor,
        cutlass::layout::RowMajor> ? cute::UMMA::Major::K : cute::UMMA::Major::MN>;
using LayoutSFD = typename Sm1xxBlockScaledOutputConfig::LayoutSF;

using HostTensorA  = cutlass::HostTensor<typename Gemm::ElementA, cutlass::layout::PackedVectorLayout>;
using HostTensorB  = cutlass::HostTensor<typename Gemm::ElementB, cutlass::layout::PackedVectorLayout>;
using HostTensorSF = cutlass::HostTensor<typename Gemm::GemmKernel::ElementSF, cutlass::layout::PackedVectorLayout>;
using HostTensorC  = cutlass::HostTensor<typename Gemm::ElementC, cutlass::layout::PackedVectorLayout>;
using HostTensorD  = cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, cutlass::layout::PackedVectorLayout>;

template <typename T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }

template <typename Element, typename Layout>
bool initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  double scope_max, scope_min;
  constexpr int bits = cutlass::sizeof_bits<Element>::value;
  if constexpr (bits <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4; scope_min = 1;
    } else {
      scope_max = 1; scope_min = -1;
    }
  } else {
    scope_max = 4; scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
  return true;
}

// ── The test function ──

void gemm_test_run(
    TensorView w_fp8,       // [E, M, K] float8_e4m3fn on GPU — block_A data from PyTorch
    int64_t n_tokens,       // N per expert (fixed for all groups)
    int64_t num_groups,     // number of experts (= E)
    int64_t alpha_val,      // epilogue alpha (cast to float)
    int64_t beta_val)       // epilogue beta (cast to float)
{
  TVM_FFI_ICHECK_EQ(w_fp8.ndim(), 3) << "w_fp8 must be [E, M, K]";

  const int M = static_cast<int>(w_fp8.size(1));
  const int K = static_cast<int>(w_fp8.size(2));
  const int E = static_cast<int>(w_fp8.size(0));
  const int N = static_cast<int>(n_tokens);
  const int groups = static_cast<int>(num_groups);
  const float alpha = static_cast<float>(alpha_val);
  const float beta  = static_cast<float>(beta_val);

  TVM_FFI_ICHECK_EQ(E, groups) << "w_fp8 dim0 must equal groups";

  printf("gemm_test: M=%d, N=%d, K=%d, groups=%d, alpha=%.1f, beta=%.1f\n",
         M, N, K, groups, alpha, beta);

  // ── Problem sizes (same N for all groups) ──
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<int32_t> tokens_per_expert_host;
  for (int i = 0; i < groups; i++) {
    problem_sizes_host.push_back({M, N, K});
    tokens_per_expert_host.push_back(N);
  }

  // ── Allocate per-group ragged buffers (B, SFB, C, D, ref_D) ──
  std::vector<HostTensorB> block_B;
  std::vector<HostTensorSF> block_SFB;
  std::vector<HostTensorC> block_C;
  std::vector<HostTensorD> block_D;
  std::vector<HostTensorSF> block_SFD;
  std::vector<HostTensorD> block_ref_D;

  for (int i = 0; i < groups; i++) {
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
    auto layout_C = make_layout(make_shape(M, N, 1), stride_C);
    auto layout_D = make_layout(make_shape(M, N, 1), stride_D);
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    auto layout_SFD_i = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(make_shape(M, N, K, 1));

    block_B.push_back(HostTensorB(cutlass::make_Coord(size(layout_B))));
    block_SFB.push_back(HostTensorSF(cutlass::make_Coord(size(filter_zeros(layout_SFB)))));
    block_C.push_back(HostTensorC(cutlass::make_Coord(size(layout_C))));
    block_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D))));
    block_SFD.push_back(HostTensorSF(cutlass::make_Coord(size(filter_zeros(layout_SFD_i)))));
    block_ref_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D))));
  }

  // ── SFA: allocated and initialized locally (tiled layout) ──
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, groups));
  HostTensorSF block_SFA;
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  initialize_block(block_SFA.host_view(), 2024);
  block_SFA.sync_device();

  // ── block_A: from PyTorch. Make a host copy for verify(). ──
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, groups});
  auto layout_A = make_layout(make_shape(M, K, groups), stride_A);
  int64_t total_A = size(layout_A);

  // Host copy for CPU reference GEMM
  HostTensorA host_A;
  host_A.reset(cutlass::make_Coord(total_A));
  host_A.copy_in_device_to_host(
      reinterpret_cast<const typename Gemm::ElementA*>(w_fp8.data_ptr()), total_A);

  // The device pointer for the kernel comes directly from PyTorch
  const auto* A_device_ptr = reinterpret_cast<const typename Gemm::ElementA*>(w_fp8.data_ptr());

  // ── Initialize B, SFB, C and build pointer arrays ──
  uint64_t seed = 2020;
  cutlass::DeviceAllocation<int32_t> tokens_per_expert_dev;
  tokens_per_expert_dev.reset(tokens_per_expert_host.size());
  tokens_per_expert_dev.copy_from_host(tokens_per_expert_host.data());

  std::vector<typename Gemm::ElementB *> ptr_B_host(groups);
  std::vector<typename Gemm::GemmKernel::ElementSF *> ptr_SFB_host(groups);
  std::vector<typename Gemm::ElementC *> ptr_C_host(groups);
  std::vector<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D_host(groups);
  std::vector<typename Gemm::GemmKernel::ElementSF *> ptr_SFD_host(groups);
  std::vector<ElementAccumulator> alpha_host(groups, alpha);
  std::vector<ElementAccumulator> beta_host(groups, beta);

  for (int i = 0; i < groups; i++) {
    initialize_block(block_B.at(i).host_view(), seed + 2022);
    initialize_block(block_C.at(i).host_view(), seed + 2023);
    initialize_block(block_SFB.at(i).host_view(), seed + 2025);
    block_B.at(i).sync_device();
    block_C.at(i).sync_device();
    block_SFB.at(i).sync_device();

    ptr_B_host.at(i) = block_B.at(i).device_data();
    ptr_SFB_host.at(i) = block_SFB.at(i).device_data();
    ptr_C_host.at(i) = block_C.at(i).device_data();
    ptr_D_host.at(i) = block_D.at(i).device_data();
    ptr_SFD_host.at(i) = block_SFD.at(i).device_data();
  }

  cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B(groups);
  ptr_B.copy_from_host(ptr_B_host.data());
  cutlass::DeviceAllocation<const typename Gemm::GemmKernel::ElementSF *> ptr_SFB(groups);
  ptr_SFB.copy_from_host(ptr_SFB_host.data());
  cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C(groups);
  ptr_C.copy_from_host(ptr_C_host.data());
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D(groups);
  ptr_D.copy_from_host(ptr_D_host.data());
  cutlass::DeviceAllocation<typename Gemm::GemmKernel::ElementSF *> ptr_SFD(groups);
  ptr_SFD.copy_from_host(ptr_SFD_host.data());

  // ── Build kernel arguments ──
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  hw_info.cluster_shape = dim3(2, 1, 1);
  hw_info.cluster_shape_fallback = dim3(2, 1, 1);

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = alpha;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.beta = beta;
  fusion_args.beta_ptr = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dBeta = {_0{}, _0{}, 0};

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;

  arguments = typename Gemm::Arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {M, N, K, groups, tokens_per_expert_dev.get()},
    // block_A from PyTorch (device pointer), SFA from local HostTensor
    {A_device_ptr, ptr_B.get(), block_SFA.device_data(), ptr_SFB.get()},
    {fusion_args, ptr_C.get(), nullptr, ptr_D.get(), nullptr},
    hw_info, scheduler
  };

  // ── Run kernel ──
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
  cudaDeviceSynchronize();

  // ── Verify ──
  bool passed = true;
  for (int i = 0; i < groups; i++) {
    auto stride_A_i = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B_i = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C_i = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D_i = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_A_i = make_layout(make_shape(M, K, 1), stride_A_i);
    auto layout_B_i = make_layout(make_shape(N, K, 1), stride_B_i);
    auto layout_C_i = make_layout(make_shape(M, N, 1), stride_C_i);
    auto layout_D_i = make_layout(make_shape(M, N, 1), stride_D_i);
    auto layout_SFA_i = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB_i = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    // host_A has the PyTorch data copied to host; index into group i
    Tensor tensor_A = make_tensor(
        make_iterator(host_A.host_data()) + size_t(1) * i * size(layout_A_i), layout_A_i);
    Tensor tensor_SFA = make_tensor(
        block_SFA.host_data() + size_t(1) * i * size(filter_zeros(layout_SFA_i)), layout_SFA_i);
    Tensor tensor_B = make_tensor(make_iterator(block_B.at(i).host_data()), layout_B_i);
    Tensor tensor_SFB = make_tensor(block_SFB.at(i).host_data(), layout_SFB_i);

    cutlass::reference::host::GettBlockScalingMainloopParams<ElementAccumulator,
        decltype(tensor_A), decltype(tensor_SFA),
        decltype(tensor_B), decltype(tensor_SFB)>
      mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

    auto tensor_C = make_tensor(make_iterator(block_C.at(i).host_data()), layout_C_i);
    auto tensor_ref_D = make_tensor(make_iterator(block_ref_D.at(i).host_data()), layout_D_i);

    cutlass::reference::host::GettEpilogueParams<
        float, float, ElementAccumulator, ElementAccumulator,
        decltype(tensor_C), decltype(tensor_ref_D)> epilogue_params{};
    epilogue_params.C = tensor_C;
    epilogue_params.D = tensor_ref_D;
    epilogue_params.alpha = alpha_host.at(i);
    epilogue_params.beta = beta_host.at(i);

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    block_D.at(i).sync_host();
    bool group_passed = cutlass::reference::host::TensorEquals(
        block_ref_D.at(i).host_view(), block_D.at(i).host_view());
    if (!group_passed) {
      printf("  Group %d: FAILED\n", i);
    }
    passed &= group_passed;
  }

  printf("Disposition: %s\n", passed ? "Passed" : "Failed");
}

#else

void gemm_test_run(TensorView w_fp8, int64_t n, int64_t groups,
                   int64_t alpha_val, int64_t beta_val) {
  printf("CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined. Skipping.\n");
}

#endif
