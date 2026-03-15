/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/



/*! \file
    \brief Ragged Contiguous Blockscaled Grouped GEMM example using CUTLASS 3 APIs for the NVIDIA Blackwell SM100 architecture.

    This example demonstrates an implementation of Ragged Contiguous Grouped GEMM using a TMA + Blackwell SM100 TensorOp-based warp-specialized kernel for narrow precisions (FP4) with Scale Factors (In and Out).
    For this example all scheduling work is performed on the device.

    To run this example:

      $ ./examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped --m=128 --k=128 --groups=10

      The above example command makes all 10 groups to be sized at the given m, n, k sizes.
      Note that m and k remain consistent across groups and only n is randomized if it's not provided through the args.
      Alpha and beta values are randomized across the different groups.

    To run this example for a set of problems using the benchmark option:

      $ ./examples/92_blackwell_grouped_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped --benchmark=./test_benchmark.txt

      Where the test_benchmark.txt may look as such:
        0 256x512x256
        1 256x128x256
        2 256x256x256 and so on
      Note that one must keep m and k consistent across groups in the benchmark file.

    ==============================================================================================
    CALL STACK TRACE for:
      ./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=128 --k=128 --groups=10 --alpha=1 --beta=0
    ==============================================================================================

    With --alpha=1 --beta=0, the GEMM simplifies to:
      D_i = 1.0 * A_i × B_i^T + 0.0 * C_i  =  A_i × B_i^T    (C is effectively ignored)

    Where:
      A (CUTLASS) = expert weights, [128, 128] per expert, FP8 E4M3 with block scales
      B (CUTLASS) = activation tokens, [N_i, 128] per expert (N_i random), FP8 E4M3
      D = output, [128, N_i] per expert, BF16

    "Ragged Contiguous": A is one stacked buffer [128,128,10]; B is separate alloc per expert.

    main() [line 1437]  — Program entry point
    │
    ├─ CUDA version check: __CUDACC_VER_MAJOR__ >= 12.8              [line 1440]  — Reject if toolkit < 12.8 (Blackwell needs 12.8+)
    │
    ├─ GPU capability check: props.major == 10                       [line 1454]  — Reject if GPU is not Blackwell SM 10.x
    │
    ├─ options.parse(argc, args)                                     [line 1470 → line 527]  — Parse all CLI flags into the Options struct
    │  │
    │  ├─ CommandLine cmd(argc, args)                                [line 528]  — Tokenize argv into searchable key=value pairs
    │  ├─ cmd.get_cmd_line_argument("m", m)       → m = 128         [line 541]  — Expert weight matrix row count
    │  ├─ cmd.get_cmd_line_argument("n", n)       → n unchanged     [line 542]  — Token count per expert (not given, stays default)
    │  ├─ cmd.get_cmd_line_argument("k", k)       → k = 128         [line 543]  — Reduction (hidden) dimension
    │  ├─ cmd.get_cmd_line_argument("groups", ..) → groups = 10     [line 544]  — Number of MoE experts
    │  ├─ cmd.get_cmd_line_argument("alpha", ..)  → alpha = 1.0     [line 545]  — Epilogue scaling factor for A×B^T product
    │  ├─ cmd.get_cmd_line_argument("beta", ..)   → beta = 0.0      [line 546]  — Epilogue scaling factor for bias C (0 = ignore C)
    │  │
    │  └─ randomize_problems(cmd)  (--benchmark not given)           [line 565 → line 646]  — Build per-group {M,N,K} problem sizes
    │     ├─ cmd_line_m=128, cmd_line_n=-1, cmd_line_k=128           [line 647-650]  — Re-read M/N/K into locals for randomization logic
    │     ├─ m=128, k=128 (fixed for all groups)                     [line 654-655]  — Set globals; skip randomization since both > 0
    │     └─ for i in 0..9:                                          [line 663]     — One iteration per expert group
    │        └─ n = 16 * rand_in_[1,64]  (random per group)         [line 665-666]  — Random N since --n not given (simulates MoE routing)
    │           push {128, N_i, 128} into problem_sizes_host         [line 668]     — Store this expert's GEMM dimensions
    │           push N_i into tokens_per_expert_host                 [line 669]     — Store token count (used by kernel's tile scheduler)
    │
    ├─ allocate(options)                                             [line 1479 → line 838]  — Allocate host+device memory for all per-group operands
    │  │
    │  ├─ for i in 0..9:                                             [line 839]  — One allocation set per expert
    │  │  ├─ {M=128, N=N_i, K=128} = problem_sizes_host[i]          [line 840-843]  — Unpack this expert's dimensions
    │  │  ├─ stride_B = make_cute_packed_stride(StrideB{},{N,K,1})   [line 920]  — Fill dynamic stride slot from shape (see proof at line 848)
    │  │  │    → stride_B = {K=128, 1, 0}   (K-contiguous)
    │  │  ├─ stride_C = make_cute_packed_stride(StrideC{},{M,N,1})   [line 965]  — Same overload, but modes are [M,N,L] not [N,K,L]
    │  │  │    → stride_C = {N_i, 1, 0}     (N-contiguous, RowMajor)
    │  │  ├─ stride_D = make_cute_packed_stride(StrideD{},{M,N,1})   [line 966]  — Output stride, identical layout to C
    │  │  │    → stride_D = {N_i, 1, 0}     (same as C)
    │  │  ├─ layout_B  = make_layout({N_i, K, 1}, stride_B)         [line 971]  — Pair shape+stride into CuTe layout for size()
    │  │  ├─ layout_C  = make_layout({M, N_i, 1}, stride_C)         [line 972]  — Bias matrix layout
    │  │  ├─ layout_D  = make_layout({M, N_i, 1}, stride_D)         [line 973]  — Output matrix layout
    │  │  ├─ layout_SFB = tile_atom_to_shape_SFB({M,N,K,1})         [line 1003]  — HW-tiled scale factor layout for B (uses N,K; ignores M)
    │  │  ├─ layout_SFD = tile_atom_to_shape_SFD({M,N,K,1})         [line 1004]  — HW-tiled scale factor layout for D output
    │  │  ├─ block_B[i]     = HostTensor(size(layout_B))            [line 1022]  — Alloc N_i×K FP8 elements (host + device pair)
    │  │  ├─ block_SFB[i]   = HostTensor(size(filter_zeros(SFB)))   [line 1023]  — Alloc E8M0 scale factors; filter_zeros removes broadcast modes
    │  │  ├─ block_C[i]     = HostTensor(size(layout_C))            [line 1024]  — Alloc M×N_i BF16 bias (unused when beta=0, still allocated)
    │  │  ├─ block_D[i]     = HostTensor(size(layout_D))            [line 1025]  — Alloc M×N_i BF16 output (written by GPU kernel)
    │  │  ├─ block_SFD[i]   = HostTensor(size(filter_zeros(SFD)))   [line 1026]  — Alloc output scale factors (unused, SF output disabled)
    │  │  └─ block_ref_D[i] = HostTensor(size(layout_D))            [line 1027]  — Alloc M×N_i BF16 reference output (for host verification)
    │  │
    │  ├─ block_alpha.reset(10)                                      [line 1030]  — Alloc device array for 10 per-group alpha scalars
    │  └─ block_beta.reset(10)                                       [line 1031]  — Alloc device array for 10 per-group beta scalars
    │
    ├─ initialize(options)                                           [line 1481 → line 1042]  — Fill all operands with data and copy to GPU
    │  │
    │  ├─ tokens_per_expert → copy to device                         [line 1047-1048]  — Kernel reads N_i per expert from this device array
    │  ├─ print "Tokens per expert (N_i): [...]"                     [line 1050-1055]  — Debug output showing each expert's token count
    │  ├─ layout_SFA = tile_atom_to_shape_SFA({m,n,k,groups})        [line 1078]  — HW-tiled scale factor layout for A (uses M,K; ignores N)
    │  ├─ stride_A = make_cute_packed_stride(StrideA{},{128,128,10}) [line 1083]  — RowMajor stride for A: K-contiguous, group stride = M*K
    │  │    → stride_A = {K=128, 1, M*K=16384}
    │  ├─ block_A.reset(size(layout_A))  → 128*128*10 = 163840      [line 1085]  — Alloc contiguous A for all 10 experts (single buffer)
    │  ├─ block_SFA.reset(size(filter_zeros(layout_SFA)))            [line 1087]  — Alloc contiguous SFA for all 10 experts
    │  ├─ initialize_block(block_A, seed)  → random FP8 [-1,1]      [line 1089]  — Fill A with pseudo-random FP8 E4M3 values
    │  ├─ initialize_block(block_SFA, seed) → random E8M0 [1,4]     [line 1090]  — Fill SFA with pseudo-random E8M0 scale factors
    │  ├─ block_A.sync_device()    (cudaMemcpy H→D)                 [line 1093]  — Transfer A from host to device memory
    │  ├─ block_SFA.sync_device()  (cudaMemcpy H→D)                 [line 1094]  — Transfer SFA from host to device memory
    │  │
    │  ├─ for i in 0..9:                                             [line 1098]  — Fill and transfer each expert's ragged operands
    │  │  ├─ initialize_block(B[i], seed)   → random FP8 [-1,1]     [line 1101]  — Fill B[i] activation tokens with random FP8
    │  │  ├─ initialize_block(C[i], seed)   → random BF16 [-4,4]    [line 1102]  — Fill C[i] bias with random BF16 (ignored when beta=0)
    │  │  ├─ initialize_block(SFB[i], seed) → random E8M0 [1,4]     [line 1103]  — Fill B[i]'s block scale factors with random E8M0
    │  │  ├─ B[i].sync_device(), C[i].sync_device(), SFB[i].sync()  [line 1105-1107]  — cudaMemcpy H→D for this expert's 3 buffers
    │  │  ├─ collect device ptrs: ptr_B[i], ptr_SFB[i], etc.        [line 1111-1115]  — Build pointer arrays for kernel indirection
    │  │  ├─ alpha_host[i] = 1.0  (from --alpha=1, not FLT_MAX)     [line 1120]  — Uniform scalar alpha (not per-group random)
    │  │  └─ beta_host[i]  = 0.0  (from --beta=0, not FLT_MAX)      [line 1121]  — Uniform scalar beta (C is zeroed out)
    │  │
    │  ├─ ptr_B → copy pointer array to device                      [line 1131-1132]  — GPU can now dereference ptr_B[i] to find expert i's B
    │  ├─ ptr_SFB, ptr_C, ptr_D, ptr_SFD → same                    [line 1134-1144]  — Same pointer-array pattern for all ragged operands
    │  ├─ alpha_device[i] = &block_alpha[i], beta_device[i] = ...   [line 1148-1151]  — Per-group scalar pointers (unused since alpha/beta are uniform)
    │  ├─ block_alpha → copy [1,1,1,...,1] to device                [line 1153]  — All experts use alpha=1.0
    │  ├─ block_beta  → copy [0,0,0,...,0] to device                [line 1154]  — All experts use beta=0.0
    │  └─ norm_constant_device = 1.0                                [line 1157-1158]  — Output normalization (unused, SF output disabled)
    │
    ├─ print "Running kernel with 1SM MMA config:"                   [line 1491]
    ├─ run<Gemm1SM>(options)                                         [line 1492 → line 1323]  — Launch, verify, and benchmark the 1-SM MMA kernel
    │  │
    │  ├─ print problem sizes + alpha + beta for each group          [line 1326-1331]  — Log each expert's {M,N_i,K}, alpha, beta
    │  │    e.g. "(128,640,128), 1, 0"
    │  │
    │  ├─ args_from_options<Gemm1SM>(options)                        [line 1337 → line 1165]  — Pack all inputs into CUTLASS kernel Arguments struct
    │  │  ├─ hw_info.sm_count = query_device_multiprocessor_count()  [line 1170]  — Query GPU SM count (e.g. 160 for B200)
    │  │  ├─ fusion_args.alpha = 1.0 (scalar, same for all groups)   [line 1191]  — Scalar broadcast: one alpha for all groups
    │  │  │    fusion_args.alpha_ptr_array = nullptr                  [line 1192]  — No per-group pointer array needed
    │  │  │    dAlpha = {0, 0, 0}  (no per-group indirection)        [line 1193]  — Zero stride = same alpha everywhere
    │  │  ├─ fusion_args.beta = 0.0 (scalar, same for all groups)    [line 1203]  — Scalar broadcast: one beta for all groups
    │  │  │    fusion_args.beta_ptr_array = nullptr                   [line 1204]  — No per-group pointer array needed
    │  │  │    dBeta = {0, 0, 0}                                     [line 1205]  — Zero stride = same beta everywhere
    │  │  └─ Gemm::Arguments{kGrouped, problem_shape,                [line 1228-1235]  — Final struct passed to kernel init/launch
    │  │       {A_ptr, B_ptr_array, SFA_ptr, SFB_ptr_array},
    │  │       {fusion_args, C_ptr_array, nullptr, D_ptr_array, nullptr},
    │  │       hw_info, scheduler{raster=AlongN}}
    │  │
    │  ├─ workspace_size = Gemm::get_workspace_size(arguments)       [line 1341]  — Query bytes for tile scheduler state + TMA descriptor slots
    │  ├─ workspace = device_memory::allocation(workspace_size)      [line 1344]  — Allocate workspace on device
    │  ├─ gemm.can_implement(arguments)  → check alignment, etc.    [line 1348]  — Validate K alignment, SM arch, and problem constraints
    │  ├─ gemm.initialize(arguments, workspace)                      [line 1352]  — Copy args to device, build TMA descriptors, init scheduler
    │  │    → copy args to device, set up TMA descriptors, init tile scheduler
    │  ├─ gemm.run()  ← GPU KERNEL LAUNCH (all 10 groups in 1 launch) [line 1356]  — Single kernel launch processes all 10 expert GEMMs
    │  ├─ cudaDeviceSynchronize()                                    [line 1358]  — Block host until kernel completes
    │  │
    │  ├─ verify(options)                                            [line 1365 → line 1245]  — Compare GPU output against CPU reference for correctness
    │  │  └─ for i in 0..9:                                          [line 1248]  — Verify each expert independently
    │  │     ├─ build stride/layout for A[i], B[i], C[i], D[i]      [line 1254-1264]  — Reconstruct CuTe layouts for host-side reference
    │  │     ├─ reference::host::Gemm3x(mainloop, epilogue)          [line 1293]  — CPU reference GEMM with block scaling (FP32 accumulation)
    │  │     │    → CPU reference: ref_D[i] = 1.0*(A[i]*B[i]^T) + 0.0*C[i]
    │  │     ├─ block_D[i].sync_host()  (copy GPU result back)      [line 1295]  — cudaMemcpy D→H to bring kernel output to host for comparison
    │  │     └─ TensorEquals(ref_D[i], D[i])  → compare             [line 1297]  — Element-wise BF16 equality check (with tolerance)
    │  │
    │  ├─ print "Disposition: Passed"                                [line 1366]  — Report verification result
    │  │
    │  ├─ Warmup: 1000 × {initialize + run}                         [line 1380-1383]  — Stabilize GPU clocks, fill caches before timing
    │  ├─ timer.start()                                              [line 1385]  — Begin GPU-side timing via CUDA events
    │  ├─ Benchmark: 1000 × {initialize + run}                      [line 1386-1389]  — Timed iterations; init needed each time (scheduler consumed)
    │  ├─ timer.stop()                                               [line 1390]  — End GPU-side timing
    │  └─ print avg_runtime_ms, TFLOPS                               [line 1399-1400]  — Report average latency and throughput
    │
    ├─ print "Running kernel with 2SM MMA config:"                   [line 1495]
    ├─ run<Gemm2SM>(options)   (same as above, tile 256×256×128)     [line 1496]  — Launch, verify, and benchmark the 2-SM cooperative kernel
    │  └─ ... (identical flow to run<Gemm1SM> above)
    │
    └─ return 0                                                      [line 1499]

*/

#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
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
using namespace cute;

// ═══════════════════════════════════════════════════════════════════════════════
// VARIABLE DEPENDENCY DIAGRAM
// ═══════════════════════════════════════════════════════════════════════════════
//
// This diagram shows all major global variables in the MoE grouped GEMM example,
// their data-flow dependencies (→ means "feeds into"), and a one-line motivation.
//
// ┌─────────────────────────── CLI / Options ──────────────────────────────────┐
// │                                                                           │
// │  options.m = 128       options.k = 128       options.groups = 10          │
// │  options.alpha = 1.0   options.beta = 0.0    options.n = max(N_i)         │
// │                                                                           │
// └──────┬───────────────────────┬──────────────────────┬─────────────────────┘
//        │                       │                      │
//        ▼                       ▼                      ▼
// ┌── Contiguous (A side) ──┐  ┌── Per-expert dims ─┐  ┌── Epilogue scalars ────┐
// │                         │  │                     │  │                        │
// │  stride_A ──────────┐   │  │  tokens_per_expert  │  │  alpha_host[G]         │
// │    Stride for the   │   │  │    DeviceAlloc<i32>  │  │    Per-group alpha     │
// │    single contiguous│   │  │    [N_0..N_9] on GPU │  │    (1.0 for all here)  │
// │    A buffer         │   │  │    Kernel reads this │  │         │              │
// │         │           │   │  │    to get each       │  │         ▼              │
// │         ▼           │   │  │    group's N_i       │  │  block_alpha[G]        │
// │  block_A ───────────┤   │  │                     │  │    DeviceAlloc on GPU  │
// │    HostTensor       │   │  └──────────┬──────────┘  │    contiguous array    │
// │    [M*K*G] FP8      │   │             │             │         │              │
// │    All experts'     │   │             │             │         ▼              │
// │    weights stacked  │   │             │             │  alpha_device[G]       │
// │         │           │   │             │             │    DeviceAlloc of ptrs │
// │         │           │   │             │             │    alpha_device[i] →   │
// │         ▼           │   │             │             │      &block_alpha[i]   │
// │  layout_SFA         │   │             │             │                        │
// │    Tiled SF layout  │   │             │             │  beta_host[G]          │
// │    for A, computed  │   │             │             │    Per-group beta      │
// │    from (M,K,G)     │   │             │             │    (0.0 for all here)  │
// │         │           │   │             │             │         │              │
// │         ▼           │   │             │             │         ▼              │
// │  block_SFA          │   │             │             │  block_beta[G]         │
// │    HostTensor       │   │             │             │    DeviceAlloc on GPU  │
// │    E8M0 scale       │   │             │             │         │              │
// │    factors for A    │   │             │             │         ▼              │
// │    (contiguous,     │   │             │             │  beta_device[G]        │
// │     all groups)     │   │             │             │    DeviceAlloc of ptrs │
// │                     │   │             │             │                        │
// └─────────────────────┘   │             │             └────────────────────────┘
//                           │             │
//        ┌──────────────────┘             │
//        │                                │
//        ▼                                ▼
// ┌── Ragged (B side, per-expert) ────────────────────────────────────────────┐
// │                                                                           │
// │  For each expert i = 0..9, with N_i tokens:                               │
// │                                                                           │
// │  block_B[i]          block_SFB[i]       block_C[i]        block_D[i]      │
// │    HostTensor          HostTensor         HostTensor        HostTensor     │
// │    [N_i * K] FP8       E8M0 scales        [M * N_i] BF16   [M * N_i] BF16│
// │    Expert i's          for B[i],          Bias matrix       KERNEL OUTPUT  │
// │    activation          one per 32         (for beta*C       = alpha*(A*B^T)│
// │    tokens              K-elements         term)             + beta*C       │
// │       │                   │                  │                  │          │
// │       ▼                   ▼                  ▼                  ▼          │
// │  ptr_B[i]            ptr_SFB[i]          ptr_C[i]           ptr_D[i]      │
// │    DeviceAlloc         DeviceAlloc         DeviceAlloc       DeviceAlloc   │
// │    of pointers         of pointers         of pointers       of pointers   │
// │    → B[i].device       → SFB[i].device     → C[i].device    → D[i].device│
// │                                                                           │
// │  block_SFD[i]        block_ref_D[i]                                       │
// │    HostTensor          HostTensor                                          │
// │    E8M0 scales         [M * N_i] BF16                                     │
// │    for D[i]            CPU REFERENCE                                      │
// │    (unused here,       output for                                         │
// │     SF output          VERIFICATION                                       │
// │     disabled)          against D[i]                                       │
// │       │                   │                                               │
// │       ▼                   ▼                                               │
// │  ptr_SFD[i]          ptr_ref_D[i]                                         │
// │    DeviceAlloc         DeviceAlloc                                        │
// │    of pointers         of pointers                                        │
// │                                                                           │
// └───────────────────────────────────────────────────────────────────────────┘
//
// ┌── Misc ───────────────────────────────────────────────────────────────────┐
// │  norm_constant_device   DeviceAlloc<float>, scalar = 1.0                  │
// │    Scales the output matrix to avoid tiny FP4 values. Not per-group.      │
// │    Unused in this run (SF output scaling is disabled).                     │
// └───────────────────────────────────────────────────────────────────────────┘
//
// DATA-FLOW SUMMARY (what feeds the kernel launch at gemm.run()):
//
//   Mainloop args:  block_A.device  ──→  single ptr   (contiguous A for all experts)
//                   ptr_B.get()     ──→  ptr array    (ragged B, one ptr per expert)
//                   block_SFA.device──→  single ptr   (contiguous SFA for all experts)
//                   ptr_SFB.get()   ──→  ptr array    (ragged SFB, one ptr per expert)
//
//   Epilogue args:  alpha (scalar=1.0) or alpha_device (ptr array, if per-group)
//                   beta  (scalar=0.0) or beta_device  (ptr array, if per-group)
//                   ptr_C.get()     ──→  ptr array    (ragged C bias, one per expert)
//                   ptr_D.get()     ──→  ptr array    (ragged D output, one per expert)
//
//   Problem shape:  tokens_per_expert.get() ──→  device array of N_i values
//                   m=128, k=128, groups=10
//
// KEY DESIGN PATTERN:
//   A is CONTIGUOUS (RC = Ragged Contiguous): all experts' weights are stacked in
//   one buffer [M, K, G]. The kernel uses stride_A's group stride to index expert i.
//   B is RAGGED: each expert has a separate buffer because N_i varies. The kernel
//   uses ptr_B[i] to find expert i's activation tokens.
//   This RC pattern is efficient for MoE because weights (A) are static and uniform
//   in size, while activations (B) have variable token counts per expert.
//
// VERIFICATION FLOW:
//   block_D[i] (GPU output) vs block_ref_D[i] (CPU reference) → verify() compares them.
//   The CPU reference uses the same alpha/beta/A/B/C/SFA/SFB to compute ref_D independently.
//
// ═══════════════════════════════════════════════════════════════════════════════

// MoEProblemShape: a specialized grouped GEMM problem shape where M and K are shared
// across all groups, and only N (tokens per expert) varies. It stores a device-side
// array tokens_per_expert[G] and computes per-group shapes as {M, tokens_per_expert[i], K}.
using ProblemShape = cutlass::gemm::MoEProblemShape<Shape<int,int,int>>; // <M,N,K> per group

// For --m=128 --k=128 --groups=10:
//   ElementInput = FP8 E4M3 (8-bit float: 4-bit exponent, 3-bit mantissa)
//   ElementSF    = E8M0 (8-bit unsigned exponent-only: encodes a power of 2 as block scale)
//   ElementC     = BF16 (16-bit bfloat: same exponent range as FP32, 8-bit mantissa)
using ElementInput = cutlass::float_e4m3_t;                                // Element type for Input matrix operands
using ElementSF    = cutlass::float_ue8m0_t;                               // Element type for SF matrix operands
using ElementC     = cutlass::bfloat16_t;                                      // Element type for C matrix operands

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
// A matrix configuration
// In MoE terms, A = expert WEIGHTS (contiguous across all groups).
// mx_float8_t<float_e4m3_t> is a compile-time tag telling CUTLASS "this is MXFP8 block-scaled".
// The actual memory holds raw float_e4m3_t bytes; scale factors are passed separately as SFA.
// RowMajor A with shape [M=128, K=128, G=10]: stride = (K=128, 1, M*K=16384).
// All 10 experts' weights are stacked contiguously in one buffer of 128*128*10 = 163,840 elements.
using ElementA = cutlass::mx_float8_t<ElementInput>;                        // Element type for A matrix operand
using LayoutA  = cutlass::layout::RowMajor;                                 // Layout type for A matrix operand
constexpr int AlignmentA  = 16;                                             // Alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
// In MoE terms, B = activation TOKENS (ragged -- separate allocation per group).
// ColumnMajor B with shape [N_i, K=128] per group: the token index varies fastest in memory.
// Physically stored as K=128 columns of N_i elements each.
// Note: LayoutB * (pointer) at line 166 signals CUTLASS that B is a pointer array, not a single buffer.
using ElementB = cutlass::mx_float8_t<ElementInput>;                        // Element type for A matrix operand
using LayoutB = cutlass::layout::ColumnMajor;                               // Layout type for B matrix operand
constexpr int AlignmentB  = 16;                                             // Alignment of A matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
// C = optional bias (BF16), D = output (BF16). Both are per-group pointer arrays.
// D_i has shape [M=128, N_i] RowMajor per expert.
using ElementD = ElementC;                                                  // Element type for D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Alignment of D matrix in units of elements (up to 16 bytes)
using ElementAccumulator  = float;                                          // Element type for internal accumulation

using ElementSFD  = cutlass::float_ue4m3_t;                                 // Element type for SF Output operands
constexpr int OutputSFVectorSize = 16;
using FusionOperation = cutlass::epilogue::fusion::LinCombEltActBlockScaleFactor<
    cutlass::epilogue::thread::SiLu,
    OutputSFVectorSize,
    ElementD,
    ElementAccumulator,
    ElementSFD,
    LayoutC,
    ElementC>;

// Core kernel configurations
using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;            // Operator class tag
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size

// Runtime Cluster Shape
using ClusterShape = Shape<int32_t,int32_t,_1>;

// Two kernel configurations are instantiated and both are run sequentially.
// They differ only in how many SMs cooperate on one MMA tile:
//
// 1SM config: each SM independently computes a 128x256x128 tile.
//   - For --m=128: one tile covers all M=128 rows, 256 columns of N, and full K=128 reduction.
//   - Works with any cluster shape.
//
// 2SM config: two SMs cooperate on a larger 256x256x128 tile (M dimension doubled).
//   - For --m=128: M=128 < tile_M=256, so some tile rows are unused (padding).
//   - Requires cluster_shape.x >= 2 (default is 2, so this works).
//   - Can improve utilization on large problems where M >= 256.
struct MMA1SMConfig {
  using MmaTileShape     = Shape<_128,_256,_128>;
  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;   // Kernel to launch
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;              // Epilogue to launch
};

struct MMA2SMConfig {
  using MmaTileShape     = Shape<_256,_256,_128>;
  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;   // Kernel to launch
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;              // Epilogue to launch
};

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    typename MMA1SMConfig::MmaTileShape, ClusterShape,
    Shape<_128,_64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC, // Set ElementC as void here to run kernel as void-C case
    ElementD, LayoutC *, AlignmentD,
    typename MMA1SMConfig::EpilogueSchedule
    // , FusionOperation  // Enable for SF Output
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
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;
using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Gemm = Gemm1SM;

using CollectiveEpilogue2SM = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    typename MMA2SMConfig::MmaTileShape, ClusterShape,
    Shape<_128,_64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC, // Set ElementC as void here to run kernel as void-C case
    ElementD, LayoutC *, AlignmentD,
    typename MMA2SMConfig::EpilogueSchedule
    // , FusionOperation  // Enable for SF Output
>::CollectiveOp;
using CollectiveMainloop2SM = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  ElementA, LayoutA, AlignmentA,
  ElementB, LayoutB *, AlignmentB,
  ElementAccumulator,
    typename MMA2SMConfig::MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue2SM::SharedStorage))>,
    typename MMA2SMConfig::KernelSchedule
>::CollectiveOp;
using GemmKernel2SM = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop2SM,
    CollectiveEpilogue2SM
>;
using Gemm2SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

// StrideA: RowMajor A [M, K, G] → Stride<int64_t, Int<1>, int64_t> = {K, 1, M*K}
//   M stride = K (skip one row of K elements), K stride = 1 (contiguous), G stride = M*K.
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;

// StrideB: the type chain is:
//   LayoutB = ColumnMajor, used as ColumnMajor* in the collective builder (line 220)
//   → TagToStrideB<ColumnMajor*>::type = Stride<int64_t, Int<1>, Int<0>>*
//   → InternalStrideB = remove_pointer_t<...> = Stride<int64_t, Int<1>, Int<0>>
//
// This is a rank-3 stride for B's shape [N, K, L]:
//   - N stride = int64_t (dynamic, filled in at runtime as K by make_cute_packed_stride)
//   - K stride = Int<1>  (static 1: K is the contiguous/fastest dimension)
//   - L stride = Int<0>  (static 0: no batch offset; each group is a separate pointer)
//
// Why "ColumnMajor" produces K-contiguous storage:
//   In BLAS convention, the B matrix in C = A*B has shape [K, N]. "ColumnMajor" for a [K, N]
//   matrix means columns (length K) are contiguous: stride-along-K = 1, stride-along-N = K.
//   CUTLASS stores B as [N, K] (transposed from BLAS), so the same physical layout gives
//   stride = {K, 1} — K is still the fast dimension.
//
//   Memory layout for B[i] with shape [N_i, K=128]:
//     B[0][0], B[0][1], ..., B[0][127],   ← token 0's 128 elements (contiguous)
//     B[1][0], B[1][1], ..., B[1][127],   ← token 1's 128 elements
//     ...
//     B[N_i-1][0], ..., B[N_i-1][127]     ← last token
//   Address of B[n][k] = base + n*128 + k
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using Sm1xxBlockScaledOutputConfig= cutlass::detail::Sm1xxBlockScaledOutputConfig<
                                        OutputSFVectorSize,
                                        cute::is_same_v<typename FusionOperation::GmemLayoutTagScalefactor,
                                            cutlass::layout::RowMajor> ? cute::UMMA::Major::K : cute::UMMA::Major::MN
                                     >;
using OutputSFAtom = typename Sm1xxBlockScaledOutputConfig::SfAtom;
using LayoutSFD = typename Sm1xxBlockScaledOutputConfig::LayoutSF;

// Host-side allocations
//
// alpha and beta are the standard BLAS GEMM epilogue scalars. Each expert i computes:
//
//   D_i = alpha_i * (A_i × B_i^T) + beta_i * C_i
//
//   - alpha scales the matrix multiplication result (the "new" computation from A × B^T)
//   - beta  scales the bias/residual matrix C (the "old" value being accumulated into)
//
// Common MoE use case: alpha=1.0, beta=0.0 (no bias), which simplifies to D_i = A_i × B_i^T.
// To set this on the command line, pass:
//   ./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=128 --k=128 --groups=10 --alpha=1 --beta=0
// This makes alpha and beta scalar (same for all experts), bypassing the pointer array path.
//
// In this example, since --alpha and --beta are not provided on the command line, they
// default to FLT_MAX, triggering per-expert random generation:
//   alpha_host[i] = random in {1, 2, 3, 4, 5}
//   beta_host[i]  = random in {0, 1, 2, 3, 4}
// This exercises the per-group alpha/beta pointer array path in the kernel epilogue.
std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

// HostTensor = paired host + device buffers with sync_device()/sync_host() for transfers.
// PackedVectorLayout = 1D flat layout (element count only, no multi-dim strides).
using HostTensorA = cutlass::HostTensor<typename Gemm::ElementA, cutlass::layout::PackedVectorLayout>;
using HostTensorB = cutlass::HostTensor<typename Gemm::ElementB, cutlass::layout::PackedVectorLayout>;
using HostTensorSF = cutlass::HostTensor<typename Gemm::GemmKernel::ElementSF, cutlass::layout::PackedVectorLayout>;
using HostTensorC = cutlass::HostTensor<typename Gemm::ElementC, cutlass::layout::PackedVectorLayout>;
using HostTensorD = cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, cutlass::layout::PackedVectorLayout>;

// block_A: single contiguous buffer for all experts' weights [M=128, K=128, G=10]
// block_SFA: single contiguous buffer for A's block scale factors (tiled layout)
HostTensorA block_A;
HostTensorSF block_SFA;
// block_B[i], block_SFB[i], block_C[i], block_D[i]: separate per-expert allocations (ragged)
// block_ref_D[i]: host-side reference output for verification
std::vector<HostTensorB> block_B;
std::vector<HostTensorSF> block_SFB;
std::vector<HostTensorC> block_C;
std::vector<HostTensorD> block_D;
std::vector<HostTensorSF> block_SFD;
std::vector<HostTensorD> block_ref_D;

// Device-side allocations
// tokens_per_expert: device array of 10 ints, each = N_i (randomized token count for expert i).
// This is what MoEProblemShape reads on-device to determine per-group N dimensions.
cutlass::DeviceAllocation<int32_t> tokens_per_expert;

cutlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
cutlass::DeviceAllocation<const typename Gemm::GemmKernel::ElementSF *> ptr_SFA;
cutlass::DeviceAllocation<const typename Gemm::GemmKernel::ElementSF *> ptr_SFB;
cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D;
cutlass::DeviceAllocation<typename Gemm::GemmKernel::ElementSF *> ptr_SFD;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_ref_D;

StrideA stride_A;
LayoutSFA layout_SFA;

// Note, this is an array of pointers to alpha and beta scaling values per group
cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;
// A matrix wide constant value to scale the output matrix
// Avoids generating small FP4 values.
// NormConst is a single device-side constant value, its not per-batch or per-group
cutlass::DeviceAllocation<ElementAccumulator> norm_constant_device;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
// Command line options parsing
struct Options {

  bool help = false;
  bool verification = true;
  bool use_pdl = false;

  float alpha = FLT_MAX;
  float beta  = FLT_MAX;
  float norm_constant = 1.0;
  int warmup = 1000;
  int iterations = 1000;
  int m = 1024, n = 2048, k = 512, groups = 10;
  dim3 cluster_shape = dim3(2,1,1);
  dim3 cluster_shape_fallback = dim3(2,1,1);
  RasterOrderOptions raster_order = RasterOrderOptions::AlongN;
  int max_sm_count = INT_MAX;
  std::string benchmark_path;
  std::vector<int32_t> tokens_per_expert_host;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  int const tma_alignment_bits = 128;
  int const alignment = tma_alignment_bits / cutlass::sizeof_bits<ElementInput>::value;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    if (cmd.check_cmd_line_flag("no_verif")) {
      verification = false;
    }
    if (cmd.check_cmd_line_flag("use_pdl")) {
      use_pdl = true;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("groups", groups);
    cmd.get_cmd_line_argument("alpha", alpha, FLT_MAX);
    cmd.get_cmd_line_argument("beta",  beta,  FLT_MAX);
    cmd.get_cmd_line_argument("norm_constant",  norm_constant,  float(1.0));
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);
    cmd.get_cmd_line_argument("cluster_m", cluster_shape.x);
    cmd.get_cmd_line_argument("cluster_n", cluster_shape.y);
    cmd.get_cmd_line_argument("cluster_fallback_m", cluster_shape_fallback.x);
    cmd.get_cmd_line_argument("cluster_fallback_n", cluster_shape_fallback.y);
    cmd.get_cmd_line_argument("max_sm_count", max_sm_count, INT_MAX);

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        problem_sizes_host.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster_order = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster_order = RasterOrderOptions::AlongM;
    }
  }

  // ── randomize_problems() ────────────────────────────────────────────────
  //
  // Called from parse() [line 565] when --benchmark is NOT provided (our case).
  //
  // Purpose: populate problem_sizes_host and tokens_per_expert_host — the two
  //          vectors that define each expert's GEMM dimensions.
  //
  // For: --m=128 --k=128 --groups=10 --alpha=1 --beta=0  (no --n)
  //
  //   1. Re-parse M, N, K from the command line into local variables:
  //        cmd_line_m = 128,  cmd_line_n = -1 (not given),  cmd_line_k = 128
  //                                                                  [lines 612-615]
  //
  //   2. Set the global M and K (shared by ALL groups):
  //        m = 128,  k = 128
  //      If either had been < 1 (not given), it would be randomized
  //      the same way as N below. But both are provided, so no randomization.
  //                                                                  [lines 619-626]
  //
  //   3. Loop 10 times (groups = 10), counting down i = 10..1:       [line 628]
  //      For each iteration:
  //        a. n = cmd_line_n = -1  (copy the CLI value)              [line 629]
  //        b. Since n < 0 (--n was NOT given):                       [line 630]
  //             n = alignment * ((rand() % 64) + 1)
  //               = 16 * random_int_in_[1, 64]
  //               → N_i is a random multiple of 16 in [16, 1024]    [line 631]
  //           (If --n HAD been given, e.g. --n=256, n would stay 256 for all groups.)
  //        c. problem_sizes_host.push_back({128, N_i, 128})          [line 633]
  //        d. tokens_per_expert_host.push_back(N_i)                  [line 634]
  //
  //   4. groups = problem_sizes_host.size() = 10 (unchanged)         [line 636]
  //
  // After this function returns:
  //   problem_sizes_host    = [{128,N_0,128}, {128,N_1,128}, ..., {128,N_9,128}]
  //   tokens_per_expert_host = [N_0, N_1, ..., N_9]
  //
  // Example (one possible run):
  //   problem_sizes_host    = [{128,640,128}, {128,112,128}, {128,480,128}, ...]
  //   tokens_per_expert_host = [640, 112, 480, ...]
  //
  // Note: --alpha=1 and --beta=0 do NOT affect this function. They are parsed
  // separately in parse() [line 429-430] and used later in initialize() and
  // args_from_options().
  // ─────────────────────────────────────────────────────────────────────────
  //
  // ── How to control N ──────────────────────────────────────────────────
  //
  // Option 1: --n=<int>  → Use the SAME N for every group.
  //   Example: ./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=128 --n=256 --k=128 --groups=10
  //   Result:  All 10 groups get problem size {128, 256, 128}.
  //
  // Option 2: Omit --n  → Each group gets a RANDOM N (multiple of 16, range [16..1024]).
  //   Example: ./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=128 --k=128 --groups=10
  //   Result:  10 groups with randomly varying N_i, simulating real MoE token routing.
  //
  // Option 3: --benchmark=<file>  → Specify DIFFERENT M×N×K per group from a file.
  //   File format (space-separated: group_index MxNxK):
  //     0 128x256x128
  //     1 128x512x128
  //     2 128x64x128
  //     ...
  //   Example: ./92_blackwell_moe_gemm_blockscaled_rcgrouped --benchmark=problems.txt
  //   Result:  Each group gets its own M, N, K from the file. The number of groups
  //            is determined by the number of lines in the file (--groups is ignored).
  //
  // There is NO option to specify a "total N" that gets split across groups.
  // ─────────────────────────────────────────────────────────────────────
  void randomize_problems(cutlass::CommandLine &cmd) {
    int cmd_line_m = -1, cmd_line_n = -1, cmd_line_k = -1;
    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes_host.reserve(groups);

    m = cmd_line_m;      // m = 128
    k = cmd_line_k;      // k = 128
    if (m < 1) {
      m = alignment * ((rand() % 64) + 1);
    }
    if (k < 1) {
      k = alignment * ((rand() % 64) + 1);
    }

    for (int i = groups; i > 0; i--) {
      int n = cmd_line_n;    // n = -1 (not provided via CLI)
      if (n < 0) {
        n = alignment * ((rand() % 64) + 1);  // Random N_i per group, multiple of 16
      }
      problem_sizes_host.push_back({m, n, k});      // {128, N_i, 128}
      tokens_per_expert_host.push_back(n);           // N_i = tokens for this expert
    }
    groups = static_cast<int>(problem_sizes_host.size());  // Still 10
  }

  /// Load a benchmark
  bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return false;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        extent.at(i) = std::atoi(tokens.at(i).c_str());
      }
      problem_sizes_host.push_back({extent.m(), extent.n(), extent.k()});
      tokens_per_expert_host.push_back(extent.n());
    }

    groups = static_cast<int>(problem_sizes_host.size());
    m = get<0>(problem_sizes_host.at(0));
    k = get<2>(problem_sizes_host.at(0));

    return true;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "92_blackwell_moe_gemm_blockscaled_rcgrouped\n\n"
      << "  Blackwell Block Scaled Narrow Precision Ragged Contiguous Grouped GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                                                       If specified, displays this usage statement\n\n"
      << "  --m=<int>                                                    Sets the M extent of the GEMM for all groups\n"
      << "  --n=<int>                                                    Sets the N extent of the GEMM for all groups (omit to randomize per group)\n"
      << "  --k=<int>                                                    Sets the K extent of the GEMM for all groups\n"
      << "  --groups=<int>                                               Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --benchmark=<file>                                           Load per-group MxNxK from file (overrides --m/--n/--k/--groups)\n"
      << "  --alpha=<f32>                                                Epilogue scalar alpha\n"
      << "  --beta=<f32>                                                 Epilogue scalar beta\n"
      << "  --norm_constant=<f32>                                        Epilogue scalar normalization constant for the output matrix\n\n"
      << "  --cluster_m=<int>          and --cluster_n=<int>             Sets the X,Y dims of the preferred cluster shape\n"
      << "  --cluster_fallback_m=<int> and --cluster_fallback_n=<int>    Sets the X,Y dims of the fallback cluster shape\n\n"
      << "  --raster=<char>                                              CTA Rasterization direction (N for along N, M for along M)\n\n"
      << "  --iterations=<int>                                           Number of profiling iterations to perform\n\n"
      << "  --benchmark=<str>                                            Executes a benchmark problem size\n"
      << "  --max_sm_count=<int>                                         Run kernels using only these number of SMs\n"
      << "  --no_verif                                                   Do not run (host-side) verification kernels\n"
      << "  --use_pdl                                                    Launch kernel with PDL (Programmatic Dependent Launch) enabled\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "92_blackwell_moe_gemm_blockscaled_rcgrouped" << " --m=128 --k=128 --groups=10\n"
      << "    # N is randomized per group (multiples of 16, range [16..1024])\n\n"
      << "$ " << "92_blackwell_moe_gemm_blockscaled_rcgrouped" << " --m=128 --n=256 --k=128 --groups=10\n"
      << "    # All 10 groups use N=256\n\n"
      << "$ " << "92_blackwell_moe_gemm_blockscaled_rcgrouped" << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=2 --beta=0.707\n"
      << "    # Fixed N=512, custom alpha/beta scalars\n\n"
      << "$ " << "92_blackwell_moe_gemm_blockscaled_rcgrouped" << " --benchmark=problems.txt\n"
      << "    # Per-group sizes from file (format: \"idx MxNxK\" per line, e.g. \"0 128x256x128\")\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  cudaError_t error = cudaSuccess;
  bool passed = false;
};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data with random values.
// The range depends on the element type's bit width:
//   - float_e4m3_t (8-bit, not E8M0): random in [-1, 1]  ← used for A and B data
//   - float_ue8m0_t (8-bit, E8M0):    random in [1, 4]   ← used for SFA, SFB scale factors
//   - bfloat16_t (16-bit):             random in [-4, 4]  ← used for C (bias)
// Values are generated uniformly on the host, then sync_device() copies them to GPU.
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;       // E8M0 scale factors: powers of 2 in [2^1, 2^4] = [2, 16]
      scope_min = 1;
    }
    else {
      scope_max = 1;       // FP8 E4M3 data: small values to avoid overflow
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;         // BF16 (bias C): wider range is fine
    scope_min = -4;
  }

  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Allocates device-side data
// For --m=128 --k=128 --groups=10:
// This allocates per-group (ragged) buffers for B, SFB, C, D, SFD, and ref_D.
// Note: A and SFA are NOT allocated here -- they are allocated in initialize() as single
// contiguous buffers spanning all 10 groups.
//
// For each group i with problem {M=128, N_i, K=128}:
//   block_B[i]     : N_i * K = N_i * 128 elements of FP8 E4M3 (ColumnMajor)
//   block_SFB[i]   : scale factors for B[i], tiled layout computed by hardware config
//   block_C[i]     : M * N_i = 128 * N_i elements of BF16 (bias matrix)
//   block_D[i]     : M * N_i = 128 * N_i elements of BF16 (CUTLASS kernel output)
//   block_ref_D[i] : M * N_i elements of BF16 (host reference output for verification)
void allocate(const Options &options) {
  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);    // 128
    auto N = get<1>(problem);    // N_i (random, varies per group)
    auto K = get<2>(problem);    // 128

    // make_cute_packed_stride fills in the dynamic element of a stride tuple.
    //
    // StrideB{} starts as Stride<int64_t, Int<1>, Int<0>> = {???, 1, 0}.
    //
    // Proof chain for why StrideB = Stride<int64_t, Int<1>, Int<0>>:
    //
    //   (a) Line 153 of this file: using LayoutB = cutlass::layout::ColumnMajor;
    //       Line 220 of this file: passes LayoutB * (i.e., ColumnMajor *) to the collective builder.
    //
    //   (b) include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl, line 230:
    //         using StrideB = cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>;
    //       where GmemLayoutBTag = ColumnMajor *.
    //
    //   (c) include/cutlass/detail/layout.hpp, lines 117-123 (THE KEY DEFINITION):
    //         template <>
    //         struct TagToStrideB<layout::ColumnMajor *> {
    //           using UnderlyingType = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
    //           using type = UnderlyingType*;   // pointer because ColumnMajor * was passed
    //           using tag = layout::ColumnMajor;
    //         };
    //       So TagToStrideB_t<ColumnMajor *> = Stride<int64_t, Int<1>, Int<0>> *.
    //
    //   (d) include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized_rcggemm.hpp,
    //       line 171:
    //         using InternalStrideB = cute::remove_pointer_t<StrideB>;
    //       Strips the pointer → Stride<int64_t, Int<1>, Int<0>>.
    //
    //   (e) include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp, line 99:
    //         using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
    //       Forwarded unchanged to the kernel level.
    //
    //   (f) Line 288 of this file:
    //         using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    //       Final result: Stride<int64_t, Int<1>, Int<0>>.
    //
    // The shape argument {N, K, 1} tells it the matrix dimensions.
    // The function sees: position 0 is dynamic (int64_t), position 1 is static Int<1>.
    // It fills position 0 = get<1>(shape) = K.  (The "other" dimension becomes the leading dim.)
    //   Result: stride_B = {K, 1, 0}
    //
    // Proof that position 0 is filled with K (and N is ignored):
    //   include/cutlass/util/packed_stride.hpp, lines 115-124:
    //
    //     template <class StrideIntT>                                      // StrideIntT = int64_t
    //     cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>>
    //     make_cute_packed_stride(
    //         cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> s,     // matches our StrideB
    //         cute::Shape<int,int,int> shape_MKL) {                       // shape_MKL = {N, K, 1}
    //       auto s_copy = s;
    //       cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
    //       //   ^^^^^^^^^^^^^^^^^                          ^^^^^^^^^^^^^^^^^^^^^^^^
    //       //   sets position 0 (N stride)          =      get<1>({N, K, 1}) = K
    //       return s_copy;
    //     }
    //
    //   C++ template overload resolution selects this overload because our StrideB type
    //   Stride<int64_t, Int<1>, Int<0>> matches the pattern Stride<StrideIntT, Int<1>, Int<0>>
    //   with StrideIntT = int64_t.
    //
    //   The function ONLY sets position 0 (the dynamic int64_t slot). Positions 1 and 2
    //   are compile-time constants Int<1> and Int<0> — they cannot be assigned to, so the
    //   function doesn't touch them. N (= get<0>(shape_MKL)) is never read.
    //
    //   The logic: for a packed (no-gap) layout, if position 1 has stride 1 (K is contiguous),
    //   then position 0's stride must equal the size of dimension 1 (= K). N doesn't affect
    //   the stride — it only affects the total allocation size (N * K), computed later by
    //   size(layout_B).
    //
    // Concrete example for group i with N_i=640, K=128:
    //   stride_B = {128, 1, 0}
    //   B[n][k] = base + n*128 + k*1    → K=128 elements per token are contiguous
    //   Total elements = N_i * K = 640 * 128 = 81,920
    //
    // This is ColumnMajor in CUTLASS convention: K (reduction dim) is the fast axis.
    // Equivalently, it's how you'd store a [640, 128] array in C (row-major in C terms).
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    // StrideC and StrideD are RowMajor [M, N]: stride = {N, 1, 0}.
    // C[m][n] = base + m*N + n   → N elements per row are contiguous.
    //
    // Proof that StrideC = Stride<int64_t, Int<1>, Int<0>> (RowMajor, N-contiguous):
    //
    //   (a) Line 160 of this file: using LayoutC = cutlass::layout::RowMajor;
    //       Line 212 of this file: passes LayoutC * (= RowMajor *) to the epilogue CollectiveBuilder.
    //
    //   (b) include/cutlass/epilogue/collective/builders/sm100_builder.inl, line 1478:
    //         using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
    //       where GmemLayoutTagC = RowMajor *.
    //
    //   (c) include/cutlass/detail/layout.hpp, line 127 (TagToStrideC delegates to TagToStrideA):
    //         template <class LayoutTag>
    //         struct TagToStrideC : TagToStrideA<LayoutTag> { };
    //
    //   (d) include/cutlass/detail/layout.hpp, lines 94-99 (TagToStrideA<RowMajor *>):
    //         template <>
    //         struct TagToStrideA<layout::RowMajor *> {
    //           using UnderlyingType = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
    //           using type = UnderlyingType*;
    //           using tag = layout::RowMajor;
    //         };
    //
    //   So TagToStrideC_t<RowMajor *> = Stride<int64_t, Int<1>, Int<0>> *.
    //
    //   (e) include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp, line 110:
    //         using InternalStrideC = cute::remove_pointer_t<StrideC>;
    //       Strips the pointer → Stride<int64_t, Int<1>, Int<0>>.
    //
    //   (f) include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp, line 117:
    //         using InternalStrideC = typename CollectiveEpilogue::InternalStrideC;
    //       → this file line 289: using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    //
    //   The modes are [M, N, L]. Note: C's mode labels differ from B's [N, K, L]!
    //   For C: position 0 = M stride (dynamic int64_t), position 1 = N stride (Int<1>), L = Int<0>.
    //   "RowMajor" for C means N (columns) is contiguous — stride 1 along N.
    //
    //   make_cute_packed_stride fills position 0 = get<1>({M, N, 1}) = N.
    //   Result: stride_C = {N, 1, 0}.  C[m][n] = base + m*N + n.
    //
    //   (Same packed_stride.hpp overload as StrideB — lines 115-124 — because the type
    //   signature Stride<int64_t, Int<1>, Int<0>> is identical. But the MEANING differs:
    //   for B modes [N,K,L], position 1 = K; for C modes [M,N,L], position 1 = N.)
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // Combine shape + stride into a CuTe layout. size(layout) gives total element count
    // for allocation. E.g., layout_B with shape (640, 128, 1) and stride (128, 1, 0):
    //   size = 640 * 128 * 1 = 81,920 elements.
    auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
    auto layout_C = make_layout(make_shape(M, N, 1), stride_C);
    auto layout_D = make_layout(make_shape(M, N, 1), stride_D);
    // tile_atom_to_shape_SFB computes the hardware-specific tiled layout for B's scale factors.
    // This is NOT a simple row-major array -- it's arranged to match the MMA tile decomposition.
    // Block size = 32 elements along K: each group of 32 consecutive K elements shares one E8M0 scale.
    //
    // Why make_shape(M, N, K, 1) — all four dimensions when SFB only covers (N, K)?
    //
    //   The (M, N, K, L) 4-tuple is CUTLASS's standard GEMM problem shape convention:
    //     M = rows of A/C/D,  N = cols of B^T/C/D,  K = reduction dim,  L = batch/group count.
    //
    //   tile_atom_to_shape_SFB receives (M, N, K, L) but only uses N, K, and L:
    //
    //     include/cutlass/detail/sm100_blockscaled_layout.hpp, lines 110-112:
    //       auto [M, N, K, L] = problem_shape;                          // M is destructured...
    //       return tile_to_shape(SfAtom{}, make_shape(N, K, L), ...);   // ...but never used.
    //
    //   SFB's scale factors index over (N, K) because B has shape [N, K] and block scaling
    //   operates along K (every 32 elements of K share one E8M0 scale factor).
    //   M is A's row dimension — irrelevant for B's scale factors.
    //
    //   Compare with tile_atom_to_shape_SFA (lines 95-97), which does the opposite:
    //       auto [M, N, K, L] = problem_shape;                          // N is destructured...
    //       return tile_to_shape(SfAtom{}, make_shape(M, K, L), ...);   // ...but never used.
    //   SFA covers (M, K) because A has shape [M, K] — N is irrelevant for A's scale factors.
    //
    //   Both functions accept the same (M, N, K, L) tuple for API uniformity — the caller
    //   doesn't need to know which dimensions each function actually uses.
    //
    //   The L=1 here means: this is a single group's SFB (not batched). Each group gets its
    //   own SFB allocation because B is ragged (pointer array).
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1)); // [N, K]
    auto layout_SFD = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(cute::make_shape(M, N, K, 1)); // []

    // Allocate host+device paired buffers for this group's operands.
    //
    // Each line does: HostTensor(make_Coord(num_elements))
    //   - make_Coord(n) creates a 1D coordinate used as the allocation size
    //   - size(layout) computes total elements from shape × stride (e.g., N_i * K for layout_B)
    //   - HostTensor constructor allocates both a host array and a device array of that size
    //
    // filter_zeros is needed for SFB and SFD because their tiled layouts contain
    // stride-0 (broadcast) modes. size() on a layout with stride-0 modes would
    // overcount elements (it multiplies all shape dimensions, including broadcast ones).
    // filter_zeros removes modes with stride 0, so size() returns the actual number
    // of distinct elements that need storage.
    //
    // For example with K=128, SFVecSize=32: layout_SFB's SfAtom has a stride-0 mode
    // in the K-scaling dimension (the 32 elements within a block share one scale factor,
    // encoded as stride 0). filter_zeros collapses that, giving the true allocation count.
    block_B.push_back(HostTensorB(cutlass::make_Coord(size(layout_B))));          // N_i * K FP8 elements
    block_SFB.push_back(HostTensorSF(cutlass::make_Coord(size(filter_zeros(layout_SFB)))));  // scale factors for B[i]
    block_C.push_back(HostTensorC(cutlass::make_Coord(size(layout_C))));          // M * N_i BF16 elements (bias)
    block_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D))));          // M * N_i BF16 elements (kernel output)
    block_SFD.push_back(HostTensorSF(cutlass::make_Coord(size(filter_zeros(layout_SFD)))));  // scale factors for D[i] (unused, SF output disabled)
    block_ref_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D))));      // M * N_i BF16 elements (host reference output for verification)
  }
  // Allocate space for 10 alpha and 10 beta scalars on device
  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

/// Initialize operands to be used in the GEMM and reference GEMM
// For --m=128 --k=128 --groups=10:
// This function does three things:
//   1. Allocates and fills A [128, 128, 10] and SFA (contiguous across all experts) with random data
//   2. Fills each B[i], C[i], SFB[i] with random data and copies to device
//   3. Builds device pointer arrays (ptr_B, ptr_SFB, ptr_C, ptr_D, ptr_SFD) so the kernel
//      can access each expert's ragged buffers via indirection
//   4. Sets up per-group alpha/beta scalars (random since --alpha/--beta not provided)
void initialize(const Options &options) {
  uint64_t seed = 2020;

  // Copy tokens_per_expert (e.g., [640, 112, 480, ...]) from host to device.
  // The kernel reads this on-device to know each expert's N dimension.
  tokens_per_expert.reset(options.tokens_per_expert_host.size());
  tokens_per_expert.copy_from_host(options.tokens_per_expert_host.data());

  std::cout << "  Tokens per expert (N_i): [";
  for (size_t i = 0; i < options.tokens_per_expert_host.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << options.tokens_per_expert_host[i];
  }
  std::cout << "]" << std::endl;

  //
  // Assign pointers
  //

  std::vector<typename Gemm::ElementB *> ptr_B_host(options.groups);
  std::vector<typename Gemm::GemmKernel::ElementSF *> ptr_SFB_host(options.groups);
  std::vector<typename Gemm::ElementC *> ptr_C_host(options.groups);
  std::vector<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D_host(options.groups);
  std::vector<typename Gemm::GemmKernel::ElementSF *> ptr_SFD_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  // Compute the hardware-specific tiled layout for A's scale factors (contiguous across all groups).
  // For K=128 with block size 32: each row of A has 128/32 = 4 scale factors.
  //
  // The call passes make_shape(m, n, k, groups) — all 4 dimensions — but N is ignored inside.
  // Proof: include/cutlass/detail/sm100_blockscaled_layout.hpp, lines 95-98:
  //     auto [M, N, K, L] = problem_shape;                          // N is destructured...
  //     return tile_to_shape(SfAtom{}, make_shape(M, K, L), ...);   // ...but never used.
  // SFA covers (M, K, L) because A has shape [M, K, G] — N is B's dimension, irrelevant to A.
  // The function accepts (M, N, K, L) for API uniformity with tile_atom_to_shape_SFB.
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, options.groups));

  // A is a single contiguous buffer: shape [M=128, K=128, G=10], RowMajor.
  // stride_A = (K=128, 1, M*K=16384): group stride = 16384 elements between consecutive experts.
  // Total elements: 128 * 128 * 10 = 163,840 FP8 values.
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, options.groups});
  auto layout_A = make_layout(make_shape(options.m, options.k, options.groups), stride_A);
  block_A.reset(cutlass::make_Coord(size(layout_A)));

  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  // Fill A with random FP8 values in [-1, 1] and SFA with random E8M0 values in [1, 4]
  initialize_block(block_A.host_view(), seed + 2022);
  initialize_block(block_SFA.host_view(), seed + 2024);

  // Copy host data to device (cudaMemcpy)
  block_A.sync_device();
  block_SFA.sync_device();

  // For each of the 10 experts: fill B, C, SFB with random data, collect device pointers,
  // and generate random alpha/beta scalars.
  for (int32_t i = 0; i < options.groups; ++i) {

    // B[i]: FP8 values in [-1, 1], C[i]: BF16 values in [-4, 4], SFB[i]: E8M0 values in [1, 4]
    initialize_block(block_B.at(i).host_view(), seed + 2022);
    initialize_block(block_C.at(i).host_view(), seed + 2023);
    initialize_block(block_SFB.at(i).host_view(), seed + 2025);

    block_B.at(i).sync_device();
    block_C.at(i).sync_device();
    block_SFB.at(i).sync_device();

    // Collect device pointers for the pointer arrays.
    // These will be copied to device as arrays of pointers (one pointer per expert).
    ptr_B_host.at(i) = block_B.at(i).device_data();
    ptr_SFB_host.at(i) = block_SFB.at(i).device_data();
    ptr_C_host.at(i) = block_C.at(i).device_data();
    ptr_D_host.at(i) = block_D.at(i).device_data();
    ptr_SFD_host.at(i) = block_SFD.at(i).device_data();

    // alpha/beta not provided via CLI (== FLT_MAX), so randomize per expert:
    //   alpha_host[i] = random in {1, 2, 3, 4, 5}
    //   beta_host[i]  = random in {0, 1, 2, 3, 4}
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    // ptr_alpha_host[i] points to the i-th element of the device alpha array
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  // Copy the arrays of device pointers to the GPU.
  // After this, ptr_B is a device allocation holding 10 pointers, where ptr_B[i] points to
  // the device memory of expert i's B matrix. The kernel dereferences these on-device.
  // Same pattern for SFB, C, D, SFD.
  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_SFB.reset(options.groups);
  ptr_SFB.copy_from_host(ptr_SFB_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  ptr_SFD.reset(options.groups);
  ptr_SFD.copy_from_host(ptr_SFD_host.data());

  // alpha_device[i] = pointer to the i-th alpha scalar on device
  // block_alpha = contiguous array of 10 alpha values on device
  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

  // norm_constant = 1.0 by default (not used in this run since SF output is disabled)
  norm_constant_device.reset(1);
  norm_constant_device.copy_from_host(&options.norm_constant);
}

/// Populates a Gemm::Arguments structure from the given commandline options.
// This builds the argument struct that the CUTLASS kernel needs to launch.
// It packs together: problem shape, input pointers, epilogue scalars, and hardware config.
template <typename Gemm>
typename Gemm::Arguments args_from_options(Options &options) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  // Query how many SMs the GPU has (B200 has 160 SMs). The kernel uses all of them
  // unless --max_sm_count is provided.
  hw_info.sm_count = min(cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id), options.max_sm_count);

  if (!is_static_v<ClusterShape>) {
    if (size<0>(typename Gemm::GemmKernel::CollectiveMainloop::AtomThrShapeMNK{}) == 2 &&
        (options.cluster_shape.x < 2 || options.cluster_shape_fallback.x < 2)) {
      std::cout << "Error: MMA2SMConfig kernel config needs cluster_dim.x >= 2" << std::endl;
      exit(-1);
    }
    hw_info.cluster_shape = options.cluster_shape;
    hw_info.cluster_shape_fallback = options.cluster_shape_fallback;
  }

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;

  // If alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
  // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
  if (options.alpha != FLT_MAX){
    // Single alpha for all groups
    fusion_args.alpha = options.alpha;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 0};
  }
  else {
    fusion_args.alpha = 0;
    fusion_args.alpha_ptr_array = alpha_device.get();
    // Only one alpha per each group
    fusion_args.dAlpha = {_0{}, _0{}, 1};
  }
  if (options.beta != FLT_MAX) {
    // Single beta for all groups
    fusion_args.beta = options.beta;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dBeta = {_0{}, _0{}, 0};
  }
  else {
    fusion_args.beta = 0;
    fusion_args.beta_ptr_array = beta_device.get();
    // Only one beta per each group
    fusion_args.dBeta = {_0{}, _0{}, 1};
  }

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = options.raster_order;

  // Pack everything into the kernel arguments struct:
  //   Mode: kGrouped (multiple independent GEMMs scheduled by the device-side tile scheduler)
  //   Problem shape: {m=128, n=max_N, k=128, groups=10, tokens_per_expert device ptr}
  //     The kernel reads tokens_per_expert[i] on-device to get each group's actual N_i.
  //   Mainloop args: {A single ptr, B pointer array, SFA single ptr, SFB pointer array}
  //     A and SFA are contiguous (one buffer for all experts).
  //     B and SFB are pointer arrays (separate buffer per expert).
  //   Epilogue args: {alpha/beta config, C pointer array, C strides, D pointer array, D strides}
  //     C strides and D strides are nullptr because they're computed from the problem shape.
  //   Hardware info: {device_id, sm_count, cluster_shape}
  //   Scheduler args: {raster_order = AlongN}
  arguments = typename Gemm::Arguments {
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {options.m, options.n, options.k, options.groups, tokens_per_expert.get()},
    {block_A.device_data(), ptr_B.get(),
     block_SFA.device_data(), ptr_SFB.get()},
    {fusion_args, ptr_C.get(), nullptr, ptr_D.get(), nullptr},
    hw_info, scheduler
  };

  return arguments;
}

// Host-side verification: for each of the 10 experts, run a reference GEMM on the CPU
// using the same inputs and compare the output against what the CUTLASS kernel produced.
// This is slow (FP32 reference on CPU) but catches correctness bugs.
// The reference computes: ref_D_i = alpha_i * (A_i with SFA_i) * (B_i with SFB_i)^T + beta_i * C_i
// where the block scale factors are applied during the reference matmul.
bool verify(const Options &options) {
  using namespace cute;
  bool passed = true;
  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_A = make_layout(make_shape(M, K, 1), stride_A);
    auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
    auto layout_C = make_layout(make_shape(M, N, 1), stride_C);
    auto layout_D = make_layout(make_shape(M, N, 1), stride_D);
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    auto layout_SFD = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(cute::make_shape(M, N, K, 1));

    // Create the arguments for host reference implementation
    Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()) + size_t(1) * i * size(layout_A), layout_A);
    Tensor tensor_SFA = make_tensor(block_SFA.host_data() + size_t(1) * i * size(filter_zeros(layout_SFA)), layout_SFA);
    Tensor tensor_B = make_tensor(make_iterator(block_B.at(i).host_data()), layout_B);
    Tensor tensor_SFB = make_tensor(block_SFB.at(i).host_data(), layout_SFB);
    cutlass::reference::host::GettBlockScalingMainloopParams<ElementAccumulator,
        decltype(tensor_A),
        decltype(tensor_SFA),
        decltype(tensor_B),
        decltype(tensor_SFB)
      >
    mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

    auto tensor_C = cute::make_tensor(make_iterator(block_C.at(i).host_data()), layout_C);
    auto tensor_ref_D = cute::make_tensor(make_iterator(block_ref_D.at(i).host_data()), layout_D);

    cutlass::reference::host::GettEpilogueParams<
        float, float,
        ElementAccumulator, ElementAccumulator,
        decltype(tensor_C), decltype(tensor_ref_D)
      > epilogue_params{};

    epilogue_params.C = tensor_C;
    epilogue_params.D = tensor_ref_D;
    epilogue_params.alpha = alpha_host.at(i);
    epilogue_params.beta = beta_host.at(i);

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    block_D.at(i).sync_host();
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    passed &= cutlass::reference::host::TensorEquals(block_ref_D.at(i).host_view(), block_D.at(i).host_view());

  }
  return passed;
}

/// Execute a given example GEMM computation
// This is called twice from main(): once with Gemm=Gemm1SM, once with Gemm=Gemm2SM.
// For each call, it:
//   1. Prints the 10 problem sizes + alpha/beta values
//   2. Builds kernel arguments and allocates workspace (device memory for tile scheduler state,
//      TMA descriptors, etc.)
//   3. Runs one correctness iteration of the CUTLASS kernel
//   4. Runs host-side verification (comparing CUTLASS output vs CPU reference for all 10 experts)
//   5. Runs 1000 warmup iterations + 1000 timed iterations for benchmarking
//   6. Reports average runtime (ms) and TFLOPS
//
// The kernel itself (launched by gemm.run()) does:
//   - A device-side tile scheduler partitions the work: for 10 groups with varying N_i,
//     it computes how many M-tiles x N-tiles each group needs and assigns them to CTAs.
//   - Each CTA loads tiles of A and B via TMA (Tensor Memory Access hardware unit),
//     computes block-scaled FP8 matrix multiply using Blackwell's MMA instructions,
//     and writes the BF16 result to D.
//   - A's TMA descriptor covers all groups (3D: M,K,G); B's descriptor is rewritten
//     per-group (new base address + new N dimension) whenever a CTA switches experts.
//
// ── How does gemm.run() know this is a "grouped, block-scaled, ragged-contiguous, MXFP8" GEMM? ──
//
// The `Gemm` template parameter (= Gemm1SM or Gemm2SM) carries ALL of this information
// as compile-time types baked into the template chain. There is no runtime "mode switch" —
// the kernel binary is already specialized for exactly this configuration.
// Here is where each configuration knob is set:
//
// (1) BLOCK-SCALED (MXFP8)
//     WHERE: lines 261, 270 of this file:
//       using ElementA = cutlass::mx_float8_t<float_e4m3_t>;
//       using ElementB = cutlass::mx_float8_t<float_e4m3_t>;
//     HOW:  mx_float8_t<T> is a compile-time tag that wraps T. The collective builder
//           (sm100_blockscaled_umma_builder.inl, line 108) is a partial specialization of
//           CollectiveBuilder that matches when OperatorClass = OpClassBlockScaledTensorOp
//           (set at line 296 of this file) AND the schedule inherits from
//           KernelScheduleBlockScaledGemmSm100 (builder line 127). This selects the
//           block-scaled MMA path with separate scale factor (SFA/SFB) TMA loads,
//           instead of the normal dense GEMM path.
//
// (2) GROUPED (not batched, not single)
//     WHERE: line 1229 of this file (runtime):
//       cutlass::gemm::GemmUniversalMode::kGrouped
//     AND:   line 241 of this file (compile-time):
//       using ProblemShape = cutlass::gemm::MoEProblemShape<Shape<int,int,int>>;
//     HOW:  GemmUniversal (line 345) is instantiated with ProblemShape = MoEProblemShape.
//           The kernel's tile scheduler reads MoEProblemShape::groups to know how many
//           independent GEMMs to schedule, and MoEProblemShape::tokens_per_expert[i]
//           on-device to get each group's N_i. The mode kGrouped (vs kBatched or kGemm)
//           tells the runtime to use the grouped tile scheduler path.
//           A batched GEMM would use kBatched with a single {M,N,K} shape + batch count;
//           a single GEMM would use kGemm with just {M,N,K}. The grouped mode allows
//           per-group varying dimensions, which is what MoE needs.
//
// (3) RAGGED-CONTIGUOUS (RC)
//     WHERE: lines 262, 271 of this file:
//       using LayoutA = cutlass::layout::RowMajor;          // A is a single buffer (no pointer)
//       using LayoutB = cutlass::layout::ColumnMajor;       // B will become a pointer
//     AND:   lines 337-338 of this file (in CollectiveBuilder instantiation):
//       ElementA, LayoutA,   AlignmentA,     // LayoutA  = RowMajor   (no pointer → contiguous)
//       ElementB, LayoutB *, AlignmentB,     // LayoutB* = ColumnMajor * (pointer → ragged)
//     HOW:  The collective builder computes (sm100_blockscaled_umma_builder.inl, lines 244-245):
//       IsGroupGemm  = !(same(StrideA, InternalStrideA)) && !(same(StrideB, InternalStrideB))
//                    = !(same(RowMajor, RowMajor))   && ... = false  (A has no pointer)
//       IsRCGroupGemm = (same(StrideA, InternalStrideA)) && !(same(StrideB, InternalStrideB))
//                     = (same(RowMajor, RowMajor))   && !(same(ColMajor, ColMajor*)) = true
//           StrideA = TagToStrideA_t<RowMajor> = Stride<int64_t,Int<1>,int64_t>       (no pointer)
//           StrideB = TagToStrideB_t<ColumnMajor*> = Stride<int64_t,Int<1>,Int<0>> *  (pointer!)
//           InternalStrideB = remove_pointer_t<StrideB> = Stride<int64_t,Int<1>,Int<0>>
//           StrideB != InternalStrideB because StrideB is a pointer type.
//       This selects DispatchPolicy = MainloopSm100RCGroupGemmTmaUmmaWarpSpecializedBlockScaled
//       (builder lines 269-275), which is the Ragged-Contiguous grouped mainloop:
//         - A uses a single 3D TMA descriptor (M,K,G) — one buffer, descriptor shared across groups
//         - B uses a per-group pointer array — TMA descriptor is rewritten per group
//       For comparison:
//         - IsGroupGemm = true  → both A and B are pointer arrays (fully ragged)
//         - neither true        → normal single GEMM or batched GEMM (no pointer arrays)
//
// (4) 1SM vs 2SM MMA TILE SHAPE
//     WHERE: lines 313-323 of this file:
//       MMA1SMConfig: MmaTileShape = 128×256×128, KernelSchedule = ...1Sm...
//       MMA2SMConfig: MmaTileShape = 256×256×128, KernelSchedule = ...2Sm...
//     HOW:  The KernelSchedule tag (e.g. KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100)
//           inherits from both KernelSchedule1Sm and KernelSchedulePtrArrayMxf8f6f4Sm100
//           (dispatch_policy.hpp, line 847). The builder checks is_2sm via the tile shape
//           and cluster shape to determine 1SM vs 2SM cooperative MMA.
//
// (5) EPILOGUE: PtrArray (grouped output)
//     WHERE: lines 316, 322 of this file:
//       EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm (or 2Sm)
//     AND:   lines 330-331 of this file (CollectiveBuilder for epilogue):
//       ElementC, LayoutC *, AlignmentC,     // LayoutC* → C is a pointer array
//       ElementD, LayoutC *, AlignmentD,     // LayoutC* → D is a pointer array
//     HOW:  PtrArrayTmaWarpSpecialized1Sm inherits from TmaWarpSpecialized1Sm
//           (epilogue/dispatch_policy.hpp, line 79). The LayoutC* pointer type tells the
//           epilogue builder to generate a grouped epilogue that dereferences per-group
//           C and D pointers.
//
// (6) SM100 ARCHITECTURE
//     WHERE: line 295 of this file:
//       using ArchTag = cutlass::arch::Sm100;
//     HOW:  The collective builder's enable_if checks cute::is_same_v<ArchTag, arch::Sm100>
//           (builder line 123). This selects SM100-specific TMA, UMMA, and tile scheduler
//           implementations.
//
// Summary: gemm.run() at line 1356 launches a kernel whose entire behavior —
// block-scaled MXFP8 math, ragged-contiguous grouped scheduling, per-group pointer
// arrays for B/C/D, SM100 TMA/UMMA hardware — is determined at compile time by the
// Gemm template type, not by any runtime flag. The only runtime inputs are the
// dimensions, data pointers, and alpha/beta values packed in the Arguments struct.
// ─────────────────────────────────────────────────────────────────────────────────
template <typename Gemm>
int run(Options &options)
{
  // Print the 10 problems, e.g.: "(128,640,128), 4, 4" meaning M=128, N=640, K=128, alpha=4, beta=4
  std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
  for (int32_t i = 0; i < options.groups; ++i) {
    std::cout << "    " << options.problem_sizes_host.at(i);
    std::cout << ", " << alpha_host.at(i) << ", " << beta_host.at(i) << std::endl;
  }
  std::cout << "  Groups      : " << options.groups  << std::endl;

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<Gemm>(options);

  // Workspace includes: tile scheduler state, TMA descriptor slots (per-SM × pipeline stages),
  // and other internal buffers. Size depends on problem dimensions and SM count.
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Validate that the problem dimensions meet alignment and hardware constraints
  // (e.g., K must be divisible by 128 for MXFP8 block scaling -- K=128 is fine)
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize: copies arguments to device, sets up TMA descriptors, prepares tile scheduler.
  // This must be called before every run() because the tile scheduler state is consumed.
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Launch the CUTLASS kernel on the GPU (one kernel launch processes ALL 10 expert GEMMs).
  // The device-side tile scheduler distributes work across CTAs dynamically.
  CUTLASS_CHECK(gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ options.use_pdl));

  cudaDeviceSynchronize();

  // Check if output from CUTLASS kernel and reference kernel are equal or not.
  // Verification is ON by default (--no_verif was not passed).
  Result result;
  if (options.verification) {
    std::cout << "  Host-side verification is now running - may be very slow for large cases." << std::endl;
    result.passed = verify(options);
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
    if (!result.passed) {
      exit(-1);
    }
  }
  else {
    std::cout << "  Verification is turned off for this run." << std::endl;
  }

  // Profiling: 1000 warmup iterations (default) to stabilize GPU clocks and caches,
  // then 1000 timed iterations (default) to measure average kernel latency.
  // Note: initialize() is called every iteration because the tile scheduler workspace
  // is consumed by each run and must be reset.
  if (options.iterations > 0) {
    for (int iter = 0; iter < options.warmup; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ options.use_pdl));
    }
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ options.use_pdl));
    }
    timer.stop();

    // Compute total FLOPs across all 10 experts:
    //   FLOPs = sum over i of (2 * M * N_i * K) = 2 * 128 * sum(N_i) * 128
    // Then divide by runtime to get GFLOP/s, displayed as TFLOPS.
    float elapsed_ms       = timer.elapsed_millis();
    result.avg_runtime_ms  = double(elapsed_ms) / double(options.iterations);
    result.gflops          = options.gflops(result.avg_runtime_ms / 1000.0, options.problem_sizes_host);

    std::cout << "  Avg runtime : " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  TFLOPS      : " << result.gflops / 1000.0 << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

// ============================================================================
// ENTRY POINT
// For: ./92_blackwell_moe_gemm_blockscaled_rcgrouped --m=128 --k=128 --groups=10
//
// Execution order:
//   1. Check CUDA toolkit >= 12.8 (compile-time check via __CUDACC_VER_MAJOR__)
//   2. Check GPU is Blackwell (SM 10.0, 10.1, or 10.3)
//   3. Parse CLI: m=128, k=128, groups=10, N randomized per group
//   4. allocate(): create per-group device buffers for B, SFB, C, D
//   5. initialize(): fill all operands with random data, copy to device
//   6. run<Gemm1SM>(): run 1-SM kernel → verify → benchmark 1000 iterations
//   7. run<Gemm2SM>(): run 2-SM kernel → verify → benchmark 1000 iterations
//   8. Print results and exit
//
// Expected output (example):
//   Running kernel with 1SM MMA config:
//     Problem Sizes, Alpha, Beta
//       (128,640,128), 4, 4
//       (128,112,128), 2, 1
//       ...
//     Groups      : 10
//     Disposition: Passed
//     Avg runtime : 0.0981 ms
//     TFLOPS      : 1.79
//   Running kernel with 2SM MMA config:
//     ...
// ============================================================================
int main(int argc, char const **args) {

  // Compile-time check: CUDA toolkit must be 12.8+ for Blackwell SM100 support
  if (__CUDACC_VER_MAJOR__ < 12 ||
       ((__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)
       )
     ) {
    std::cerr << "This example requires CUDA 12.8 or newer.\n";
    return 0;
  }

  // Runtime check: GPU must be Blackwell (compute capability 10.x)
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 10 || (props.minor != 0 && props.minor != 1 && props.minor != 3
       )
     ) {
    std::cerr << "This example requires a GPU with compute capability 100a|f, 101a|f, or 103a|f)." << std::endl;
    return 0;
  }

  //
  // Parse options
  // For --m=128 --k=128 --groups=10: sets m=128, k=128, groups=10.
  // Since --n is not provided, randomize_problems() generates a random N_i per group.
  // Since --alpha/--beta are not provided, they stay FLT_MAX → per-group random scalars.
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  // Step 4: Allocate per-group device memory for B, SFB, C, D, SFD, ref_D
  allocate(options);
  // Step 5: Allocate contiguous A and SFA, fill everything with random data, copy to device
  initialize(options);

  //
  // Evaluate CUTLASS kernels
  // Both 1SM and 2SM configs run the same grouped GEMM on the same data.
  // They share the same output buffers, so the 2SM run overwrites the 1SM output.
  //

  // Step 6: Run the 1-SM MMA kernel (tile 128x256x128)
  // Each SM independently computes one tile. For M=128, one tile covers all M rows.
  std::cout << "Running kernel with 1SM MMA config:" << std::endl;
  run<Gemm1SM>(options);
  // Step 7: Run the 2-SM MMA kernel (tile 256x256x128)
  // Two SMs cooperate on each tile. For M=128 < 256, the tile is partially filled.
  std::cout << "Running kernel with 2SM MMA config:" << std::endl;
  run<Gemm2SM>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
