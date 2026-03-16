// Full RC grouped GEMM check with all 4 operands from PyTorch.
// A [E,M,K] and SFA [flat] as contiguous single pointers.
// B [total_tokens, K] with offsets_B [E+1] for per-expert slicing.
// SFB [total_sfb] with offsets_SFB [E+1] for per-expert slicing.
// Runs the GEMM kernel and verifies against CPU reference.
//
// This mirrors how a real MoE layer works: tokens are routed to experts
// with variable counts, and the activation buffer is a single packed
// tensor where offsets_B[i] marks where expert i's tokens begin.

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

#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "helper.h"
#include "tvm_ffi_utils.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// ── GEMM type chain (same as sfa_check.cu / gemm_test.cu) ──

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

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using Sm1xxBlockScaledOutputConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<
    OutputSFVectorSize,
    cute::is_same_v<typename FusionOperation::GmemLayoutTagScalefactor,
        cutlass::layout::RowMajor> ? cute::UMMA::Major::K : cute::UMMA::Major::MN>;

using HostTensorA  = cutlass::HostTensor<typename Gemm::ElementA, cutlass::layout::PackedVectorLayout>;
using HostTensorB  = cutlass::HostTensor<typename Gemm::ElementB, cutlass::layout::PackedVectorLayout>;
using HostTensorSF = cutlass::HostTensor<typename Gemm::GemmKernel::ElementSF, cutlass::layout::PackedVectorLayout>;
using HostTensorC  = cutlass::HostTensor<typename Gemm::ElementC, cutlass::layout::PackedVectorLayout>;
using HostTensorD  = cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, cutlass::layout::PackedVectorLayout>;

template <typename T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }

template <typename Element, typename Layout>
void initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
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
}

// ── Size helpers ──

int64_t get_b_size_run(int64_t N, int64_t K) {
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {(int)N, (int)K, 1});
  auto layout_B = make_layout(make_shape((int)N, (int)K, 1), stride_B);
  return static_cast<int64_t>(size(layout_B));
}

int64_t get_sfb_size_run(int64_t M, int64_t N, int64_t K) {
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      make_shape((int)M, (int)N, (int)K, 1));
  return static_cast<int64_t>(size(filter_zeros(layout_SFB)));
}

// ── Main GEMM check ──
//
// w_act is a packed buffer [total_tokens, K] where tokens for all experts
// are concatenated. offsets_B [E+1] gives the token boundaries:
//   expert i owns tokens [offsets_B[i], offsets_B[i+1]),
//   so N_i = offsets_B[i+1] - offsets_B[i].
//   ptr_B[i] = w_act.data_ptr() + offsets_B[i] * K
//
// Same pattern for SFB: w_sfb is flat [total_sfb], offsets_SFB [E+1] gives
//   ptr_SFB[i] = w_sfb.data_ptr() + offsets_SFB[i]

void gemm_check_run(
    TensorView w_fp8,        // [E, M, K] float8_e4m3fn — block_A
    TensorView w_sfa,        // [sfa_total] uint8 — SFA flat
    TensorView w_act,        // [total_tokens, K] float8_e4m3fn — packed B for all experts
    TensorView w_sfb,        // [total_sfb] uint8 — packed SFB for all experts
    TensorView offsets_B_tv, // [E+1] int32 — token offsets per expert
    TensorView offsets_SFB_tv, // [E+1] int32 — SFB offsets per expert
    int64_t alpha_val,
    int64_t beta_val)
{
  TVM_FFI_ICHECK_EQ(w_fp8.ndim(), 3) << "w_fp8 must be [E, M, K]";
  TVM_FFI_ICHECK_EQ(w_act.ndim(), 2) << "w_act must be [total_tokens, K]";

  const int M = static_cast<int>(w_fp8.size(1));
  const int K = static_cast<int>(w_fp8.size(2));
  const int E = static_cast<int>(w_fp8.size(0));
  const int groups = E;
  const float alpha = static_cast<float>(alpha_val);
  const float beta  = static_cast<float>(beta_val);

  TVM_FFI_ICHECK_EQ(w_act.size(1), K) << "w_act dim1 must equal K";
  TVM_FFI_ICHECK_EQ(offsets_B_tv.size(0), E + 1) << "offsets_B must be [E+1]";
  TVM_FFI_ICHECK_EQ(offsets_SFB_tv.size(0), E + 1) << "offsets_SFB must be [E+1]";

  // Copy offset arrays to host
  std::vector<int32_t> offsets_B(E + 1);
  std::vector<int32_t> offsets_SFB(E + 1);
  cudaMemcpy(offsets_B.data(), offsets_B_tv.data_ptr(),
             (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(offsets_SFB.data(), offsets_SFB_tv.data_ptr(),
             (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);

  // Derive per-expert N_i from offsets
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<int32_t> tokens_per_expert_host;
  int max_N = 0;
  printf("gemm_check: M=%d, K=%d, groups=%d, alpha=%.1f, beta=%.1f\n",
         M, K, groups, alpha, beta);
  printf("  tokens per expert: [");
  for (int i = 0; i < groups; i++) {
    int N_i = offsets_B[i + 1] - offsets_B[i];
    problem_sizes_host.push_back({M, N_i, K});
    tokens_per_expert_host.push_back(N_i);
    if (N_i > max_N) max_N = N_i;
    if (i > 0) printf(", ");
    printf("%d", N_i);
  }
  printf("]\n");

  // ── A: contiguous from PyTorch, host copy for verify ──
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, groups});
  auto layout_A = make_layout(make_shape(M, K, groups), stride_A);
  int64_t total_A = size(layout_A);

  HostTensorA host_A;
  host_A.reset(cutlass::make_Coord(total_A));
  host_A.copy_in_device_to_host(
      reinterpret_cast<const typename Gemm::ElementA*>(w_fp8.data_ptr()), total_A);
  const auto* A_device_ptr = reinterpret_cast<const typename Gemm::ElementA*>(w_fp8.data_ptr());

  // ── SFA: contiguous from PyTorch, host copy for verify ──
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, max_N, K, groups));
  int64_t sfa_total = size(filter_zeros(layout_SFA));

  HostTensorSF host_SFA;
  host_SFA.reset(cutlass::make_Coord(sfa_total));
  host_SFA.copy_in_device_to_host(
      reinterpret_cast<const typename Gemm::GemmKernel::ElementSF*>(w_sfa.data_ptr()), sfa_total);
  const auto* SFA_device_ptr = reinterpret_cast<const typename Gemm::GemmKernel::ElementSF*>(w_sfa.data_ptr());

  // ── B: packed [total_tokens, K] from PyTorch, use offsets for per-expert slicing ──
  const auto* B_base = reinterpret_cast<const typename Gemm::ElementB*>(w_act.data_ptr());

  std::vector<HostTensorB> block_B;  block_B.reserve(groups);
  std::vector<typename Gemm::ElementB*> ptr_B_host(groups);

  for (int i = 0; i < groups; i++) {
    int N_i = tokens_per_expert_host[i];
    int64_t b_offset = (int64_t)offsets_B[i] * K;  // element offset into packed buffer
    int64_t b_count = (int64_t)N_i * K;

    // Host copy for CPU reference verify
    block_B.push_back(HostTensorB(cutlass::make_Coord(b_count)));
    block_B.at(i).copy_in_device_to_host(B_base + b_offset, b_count);

    // Device pointer for kernel: points into the packed PyTorch buffer
    ptr_B_host.at(i) = const_cast<typename Gemm::ElementB*>(B_base + b_offset);
  }

  cutlass::DeviceAllocation<const typename Gemm::ElementB*> ptr_B(groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  // ── SFB: packed [total_sfb] from PyTorch, use offsets for per-expert slicing ──
  const auto* SFB_base = reinterpret_cast<const typename Gemm::GemmKernel::ElementSF*>(w_sfb.data_ptr());

  std::vector<HostTensorSF> block_SFB;  block_SFB.reserve(groups);
  std::vector<typename Gemm::GemmKernel::ElementSF*> ptr_SFB_host(groups);

  for (int i = 0; i < groups; i++) {
    int64_t sfb_offset = offsets_SFB[i];
    int64_t sfb_count = offsets_SFB[i + 1] - offsets_SFB[i];

    block_SFB.push_back(HostTensorSF(cutlass::make_Coord(sfb_count)));
    block_SFB.at(i).copy_in_device_to_host(SFB_base + sfb_offset, sfb_count);

    ptr_SFB_host.at(i) = const_cast<typename Gemm::GemmKernel::ElementSF*>(SFB_base + sfb_offset);
  }

  cutlass::DeviceAllocation<const typename Gemm::GemmKernel::ElementSF*> ptr_SFB(groups);
  ptr_SFB.copy_from_host(ptr_SFB_host.data());

  // ── C, D, SFD, ref_D: allocated locally ──
  std::vector<HostTensorC> block_C;    block_C.reserve(groups);
  std::vector<HostTensorD> block_D;    block_D.reserve(groups);
  std::vector<HostTensorSF> block_SFD; block_SFD.reserve(groups);
  std::vector<HostTensorD> block_ref_D; block_ref_D.reserve(groups);

  std::vector<typename Gemm::ElementC*> ptr_C_host(groups);
  std::vector<typename Gemm::EpilogueOutputOp::ElementOutput*> ptr_D_host(groups);
  std::vector<typename Gemm::GemmKernel::ElementSF*> ptr_SFD_host(groups);
  std::vector<ElementAccumulator> alpha_host(groups, alpha);
  std::vector<ElementAccumulator> beta_host(groups, beta);

  for (int i = 0; i < groups; i++) {
    int N_i = tokens_per_expert_host[i];
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N_i, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N_i, 1});
    auto layout_C_i = make_layout(make_shape(M, N_i, 1), stride_C);
    auto layout_D_i = make_layout(make_shape(M, N_i, 1), stride_D);
    auto layout_SFD_i = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(make_shape(M, N_i, K, 1));

    block_C.push_back(HostTensorC(cutlass::make_Coord(size(layout_C_i))));
    block_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D_i))));
    block_SFD.push_back(HostTensorSF(cutlass::make_Coord(size(filter_zeros(layout_SFD_i)))));
    block_ref_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D_i))));

    initialize_block(block_C.at(i).host_view(), 2023 + i);
    block_C.at(i).sync_device();

    ptr_C_host.at(i) = block_C.at(i).device_data();
    ptr_D_host.at(i) = block_D.at(i).device_data();
    ptr_SFD_host.at(i) = block_SFD.at(i).device_data();
  }

  cutlass::DeviceAllocation<const typename Gemm::ElementC*> ptr_C(groups);
  ptr_C.copy_from_host(ptr_C_host.data());
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput*> ptr_D(groups);
  ptr_D.copy_from_host(ptr_D_host.data());
  cutlass::DeviceAllocation<typename Gemm::GemmKernel::ElementSF*> ptr_SFD(groups);
  ptr_SFD.copy_from_host(ptr_SFD_host.data());

  cutlass::DeviceAllocation<int32_t> tokens_per_expert_dev;
  tokens_per_expert_dev.reset(tokens_per_expert_host.size());
  tokens_per_expert_dev.copy_from_host(tokens_per_expert_host.data());

  // ── Kernel arguments ──
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
    {M, max_N, K, groups, tokens_per_expert_dev.get()},
    {A_device_ptr, ptr_B.get(), SFA_device_ptr, ptr_SFB.get()},
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

  // ── Verify against CPU reference ──
  bool passed = true;
  auto layout_SFA_single = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, max_N, K, 1));
  int64_t sfa_per_group = size(filter_zeros(layout_SFA_single));

  for (int i = 0; i < groups; i++) {
    int N_i = tokens_per_expert_host[i];

    auto stride_A_i = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B_i = cutlass::make_cute_packed_stride(StrideB{}, {N_i, K, 1});
    auto stride_C_i = cutlass::make_cute_packed_stride(StrideC{}, {M, N_i, 1});
    auto stride_D_i = cutlass::make_cute_packed_stride(StrideD{}, {M, N_i, 1});
    auto layout_A_i = make_layout(make_shape(M, K, 1), stride_A_i);
    auto layout_B_i = make_layout(make_shape(N_i, K, 1), stride_B_i);
    auto layout_C_i = make_layout(make_shape(M, N_i, 1), stride_C_i);
    auto layout_D_i = make_layout(make_shape(M, N_i, 1), stride_D_i);
    auto layout_SFA_i = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N_i, K, 1));
    auto layout_SFB_i = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N_i, K, 1));

    Tensor tensor_A = make_tensor(
        make_iterator(host_A.host_data()) + size_t(1) * i * size(layout_A_i), layout_A_i);
    Tensor tensor_SFA = make_tensor(
        host_SFA.host_data() + size_t(1) * i * sfa_per_group, layout_SFA_i);
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
      printf("  Group %d (N=%d): FAILED\n", i, N_i);
    }
    passed &= group_passed;
  }

  printf("Disposition: %s\n", passed ? "Passed" : "Failed");
}

#else

int64_t get_b_size_run(int64_t N, int64_t K) {
  printf("CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
  return 0;
}
int64_t get_sfb_size_run(int64_t M, int64_t N, int64_t K) {
  printf("CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
  return 0;
}
void gemm_check_run(TensorView w_fp8, TensorView w_sfa, TensorView w_act,
                     TensorView w_sfb, TensorView offsets_B, TensorView offsets_SFB,
                     int64_t alpha, int64_t beta) {
  printf("CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
}

#endif
