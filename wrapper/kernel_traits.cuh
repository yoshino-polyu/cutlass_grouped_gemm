#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

// ---------- Element types ----------
using ProblemShape = cutlass::gemm::MoEProblemShape<cute::Shape<int,int,int>>;
using ElementInput  = cutlass::float_e4m3_t;
using ElementSF     = cutlass::float_ue8m0_t;
using ElementC      = void;                    // void-C: no C input, pure D = alpha * accumulator
using ElementD      = cutlass::bfloat16_t;
using ElementA      = cutlass::mx_float8_t<ElementInput>;
using ElementB      = cutlass::mx_float8_t<ElementInput>;

// ---------- Layouts ----------
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// ---------- Alignments ----------
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 0;   // unused for void-C
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;

// ---------- Architecture ----------
using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape  = cute::Shape<int32_t, int32_t, cute::_1>;

// ---------- 1SM config ----------
struct MMA1SMConfig {
  using MmaTileShape   = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

// ---------- 2SM config ----------
struct MMA2SMConfig {
  using MmaTileShape   = cute::Shape<cute::_256, cute::_256, cute::_128>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

// ========== 1SM Gemm ==========
using CollectiveEpilogue1SM = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MMA1SMConfig::MmaTileShape, ClusterShape,
    cute::Shape<cute::_128, cute::_64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutC*, AlignmentD,
    MMA1SMConfig::EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop1SM = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB*, AlignmentB,
    ElementAccumulator,
    MMA1SMConfig::MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue1SM::SharedStorage))>,
    MMA1SMConfig::KernelSchedule
>::CollectiveOp;

using GemmKernel1SM = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop1SM, CollectiveEpilogue1SM>;
using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel1SM>;

// ========== 2SM Gemm ==========
using CollectiveEpilogue2SM = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MMA2SMConfig::MmaTileShape, ClusterShape,
    cute::Shape<cute::_128, cute::_64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutC*, AlignmentD,
    MMA2SMConfig::EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop2SM = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB*, AlignmentB,
    ElementAccumulator,
    MMA2SMConfig::MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue2SM::SharedStorage))>,
    MMA2SMConfig::KernelSchedule
>::CollectiveOp;

using GemmKernel2SM = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop2SM, CollectiveEpilogue2SM>;
using Gemm2SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

// ========== Derived type aliases (from 1SM, shared structure) ==========
using StrideA  = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB  = typename Gemm1SM::GemmKernel::InternalStrideB;
using StrideD  = typename Gemm1SM::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm1SM::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm1SM::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm1SM::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
