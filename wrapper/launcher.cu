#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

#include "kernel_traits.cuh"
#include "preprocess_kernels.cuh"

#include "tvm_ffi_utils.h"

// ---------- helpers ----------
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                        \
    }                                                                         \
  } while (0)

static inline int64_t align_up(int64_t x, int64_t a) {
  return (x + a - 1) / a * a;
}

// Compute tiled SF size for a block of [rows, K_sf] scale factors.
// num_mn_tiles = ceil(rows/128), num_k_tiles = ceil(K_sf/4), each atom = 512 bytes
static inline int64_t tiled_sf_size(int32_t rows, int32_t K_sf) {
  int32_t num_mn_tiles = (rows + 127) / 128;
  int32_t num_k_tiles  = (K_sf + 3) / 4;
  return (int64_t)num_mn_tiles * num_k_tiles * 512;
}

// ---------- Templated GEMM runner ----------
template <typename GemmType>
static void run_gemm(
    // Preprocessed device arrays
    const uint8_t* A_ptr,          // w_fp8 base [E*N, K] contiguous row-major
    const uint8_t** ptr_B_dev,     // [E] pointers into B_transposed (col-major)
    const uint8_t* SFA_tiled_ptr,  // tiled weight SFs
    const uint8_t** ptr_SFB_dev,   // [E] pointers into tiled activation SFs
    __nv_bfloat16** ptr_D_dev,     // [E] pointers into D scratch
    int32_t* tokens_per_expert_dev,
    int32_t N_hidden,              // = M in CUTLASS
    int32_t max_tokens,            // = max N_i
    int32_t K,
    int32_t E,
    void* cutlass_workspace,
    size_t cutlass_workspace_size,
    cudaStream_t stream)
{
  using Gemm = GemmType;
  using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  if constexpr (cute::is_same_v<Gemm, Gemm2SM>) {
    hw_info.cluster_shape = dim3(2, 1, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  } else {
    hw_info.cluster_shape = dim3(1, 1, 1);
    hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  }

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args{};
  fusion_args.alpha = 1.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;

  arguments = typename Gemm::Arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {N_hidden, max_tokens, K, E, tokens_per_expert_dev},
    {reinterpret_cast<const ArrayElementA*>(A_ptr),
     reinterpret_cast<const ArrayElementB**>(ptr_B_dev),
     reinterpret_cast<const ElementSF*>(SFA_tiled_ptr),
     reinterpret_cast<const ElementSF**>(ptr_SFB_dev)},
    {fusion_args,
     nullptr,   // ptr_C array (void-C)
     nullptr,   // stride_C
     reinterpret_cast<ElementD**>(ptr_D_dev),
     nullptr},  // stride_D
    hw_info,
    scheduler
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(arguments);
  if (ws_size > cutlass_workspace_size) {
    fprintf(stderr, "CUTLASS workspace too small: need %zu, have %zu\n",
            ws_size, cutlass_workspace_size);
    return;
  }

  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS can_implement failed: %d\n", (int)status);
    return;
  }

  status = gemm.initialize(arguments, cutlass_workspace);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS initialize failed: %d\n", (int)status);
    return;
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS run failed: %d\n", (int)status);
  }
}

// ---------- Main entry point ----------
void mxfp8_rc_grouped_gemm_run(
    TensorView x_fp8,       // [M_total, K] float8_e4m3fn   (activations -> becomes B)
    TensorView x_scale,     // [M_total, K_sf] uint8        (activation SFs -> becomes SFB)
    TensorView w_fp8,       // [E, N, K] float8_e4m3fn      (weights -> becomes A)
    TensorView w_scale,     // [E, N, K_sf] uint8           (weight SFs -> becomes SFA)
    TensorView cnt,         // [E+1] int32
    TensorView output,      // [M_total, N] bfloat16        (pre-allocated)
    TensorView workspace,   // scratch buffer
    int64_t mma_sm)
{
  // --- Extract dimensions ---
  int32_t M_total = static_cast<int32_t>(x_fp8.size(0));
  int32_t K       = static_cast<int32_t>(x_fp8.size(1));
  int32_t E       = static_cast<int32_t>(w_fp8.size(0));
  int32_t N_hidden= static_cast<int32_t>(w_fp8.size(1));
  int32_t K_sf    = K / 32;

  // --- Device pointers ---
  uint8_t* x_fp8_ptr   = reinterpret_cast<uint8_t*>(x_fp8.data_ptr());
  uint8_t* x_scale_ptr = reinterpret_cast<uint8_t*>(x_scale.data_ptr());
  uint8_t* w_fp8_ptr   = reinterpret_cast<uint8_t*>(w_fp8.data_ptr());
  uint8_t* w_scale_ptr = reinterpret_cast<uint8_t*>(w_scale.data_ptr());
  int32_t* cnt_ptr     = reinterpret_cast<int32_t*>(cnt.data_ptr());
  __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  uint8_t* ws_ptr      = reinterpret_cast<uint8_t*>(workspace.data_ptr());
  int64_t ws_size      = workspace.size(0);

  cudaStream_t stream = get_current_stream();

  // --- Compute actual max_tokens from cnt on host ---
  std::vector<int32_t> h_cnt(E + 1);
  CUDA_CHECK(cudaMemcpy(h_cnt.data(), cnt_ptr, (E + 1) * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));
  int32_t max_tokens = 0;
  for (int i = 0; i < E; i++) {
    max_tokens = std::max(max_tokens, h_cnt[i + 1] - h_cnt[i]);
  }

  // --- Workspace partitioning ---
  // Use M_total as upper bound for per-expert buffer allocation.
  int32_t max_tokens_ub = M_total;

  int64_t offset = 0;

  // 1. tokens_per_expert [E] int32
  int64_t tokens_per_expert_off = align_up(offset, 16);
  offset = tokens_per_expert_off + (int64_t)E * sizeof(int32_t);

  // 2. expert_B_offsets [E] int64
  int64_t expert_B_offsets_off = align_up(offset, 16);
  offset = expert_B_offsets_off + (int64_t)E * sizeof(int64_t);

  // 3. expert_sfb_offsets [E] int64
  int64_t expert_sfb_offsets_off = align_up(offset, 16);
  offset = expert_sfb_offsets_off + (int64_t)E * sizeof(int64_t);

  // 4. expert_D_offsets [E] int64
  int64_t expert_D_offsets_off = align_up(offset, 16);
  offset = expert_D_offsets_off + (int64_t)E * sizeof(int64_t);

  // 5. B_transposed: each expert gets max_tokens_ub * K bytes
  int64_t B_transposed_off = align_up(offset, 256);
  offset = B_transposed_off + (int64_t)E * max_tokens_ub * K;

  // 6. SFA_tiled (weight scales, contiguous for all E experts)
  int64_t sfa_tiled_size = tiled_sf_size(N_hidden, K_sf) * E;
  int64_t sfa_tiled_off = align_up(offset, 256);
  offset = sfa_tiled_off + sfa_tiled_size;

  // 7. SFB_tiled: per expert, each tiled_sf_size(max_tokens_ub, K_sf)
  int64_t sfb_per_expert_size = tiled_sf_size(max_tokens_ub, K_sf);
  int64_t sfb_tiled_off = align_up(offset, 256);
  offset = sfb_tiled_off + (int64_t)E * sfb_per_expert_size;

  // 8. D_scratch: per expert, [N_hidden, max_tokens_ub] * 2 bytes (bf16)
  int64_t D_per_expert_size = (int64_t)N_hidden * max_tokens_ub * 2;
  int64_t D_scratch_off = align_up(offset, 256);
  offset = D_scratch_off + (int64_t)E * D_per_expert_size;

  // 9. Pointer arrays: ptr_B[E], ptr_SFB[E], ptr_D[E]
  int64_t ptr_B_off = align_up(offset, 16);
  offset = ptr_B_off + (int64_t)E * sizeof(void*);
  int64_t ptr_SFB_off = align_up(offset, 16);
  offset = ptr_SFB_off + (int64_t)E * sizeof(void*);
  int64_t ptr_D_off = align_up(offset, 16);
  offset = ptr_D_off + (int64_t)E * sizeof(void*);

  // 10. CUTLASS workspace
  int64_t cutlass_ws_off = align_up(offset, 256);
  int64_t cutlass_ws_size = ws_size - cutlass_ws_off;
  if (cutlass_ws_size < 0) cutlass_ws_size = 0;

  // --- Workspace sub-pointers ---
  auto* tokens_per_expert_dev = reinterpret_cast<int32_t*>(ws_ptr + tokens_per_expert_off);
  auto* expert_B_offsets_dev  = reinterpret_cast<int64_t*>(ws_ptr + expert_B_offsets_off);
  auto* expert_sfb_offsets_dev= reinterpret_cast<int64_t*>(ws_ptr + expert_sfb_offsets_off);
  auto* expert_D_offsets_dev  = reinterpret_cast<int64_t*>(ws_ptr + expert_D_offsets_off);
  auto* B_transposed_base     = ws_ptr + B_transposed_off;
  auto* sfa_tiled_base        = ws_ptr + sfa_tiled_off;
  auto* sfb_tiled_base        = ws_ptr + sfb_tiled_off;
  auto* D_scratch_base        = reinterpret_cast<__nv_bfloat16*>(ws_ptr + D_scratch_off);
  auto* ptr_B_dev   = reinterpret_cast<const uint8_t**>(ws_ptr + ptr_B_off);
  auto* ptr_SFB_dev = reinterpret_cast<const uint8_t**>(ws_ptr + ptr_SFB_off);
  auto* ptr_D_dev   = reinterpret_cast<__nv_bfloat16**>(ws_ptr + ptr_D_off);
  auto* cutlass_ws  = ws_ptr + cutlass_ws_off;

  // --- Compute offsets on host and upload ---
  {
    std::vector<int64_t> h_B_offsets(E);
    std::vector<int64_t> h_sfb_offsets(E);
    std::vector<int64_t> h_D_offsets(E);
    for (int i = 0; i < E; i++) {
      h_B_offsets[i]   = (int64_t)i * max_tokens_ub * K;
      h_sfb_offsets[i] = (int64_t)i * sfb_per_expert_size;
      h_D_offsets[i]   = (int64_t)i * N_hidden * max_tokens_ub;  // element offset
    }
    CUDA_CHECK(cudaMemcpyAsync(expert_B_offsets_dev, h_B_offsets.data(),
               E * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(expert_sfb_offsets_dev, h_sfb_offsets.data(),
               E * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(expert_D_offsets_dev, h_D_offsets.data(),
               E * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
  }

  // --- Step 1: compute_tokens_per_expert ---
  {
    int threads = std::min(E, 256);
    int blocks  = (E + threads - 1) / threads;
    compute_tokens_per_expert<<<blocks, threads, 0, stream>>>(
        cnt_ptr, tokens_per_expert_dev, E);
  }

  // --- Step 2: transpose_b_batched ---
  {
    int64_t max_elems = (int64_t)max_tokens_ub * K;
    int threads = 256;
    int blocks_x = std::min((int)((max_elems + threads - 1) / threads), 65535);
    dim3 grid(blocks_x, E);
    transpose_b_batched<<<grid, threads, 0, stream>>>(
        x_fp8_ptr, B_transposed_base, cnt_ptr,
        expert_B_offsets_dev, K, E);
  }

  // --- Step 3: reshuffle SFA (weight scales) ---
  {
    int32_t total_rows = E * N_hidden;
    int32_t num_k_tiles = (K_sf + 3) / 4;
    int64_t total = (int64_t)total_rows * K_sf;
    int threads = 256;
    int blocks = std::min((int)((total + threads - 1) / threads), 65535);
    reshuffle_scale_factors<<<blocks, threads, 0, stream>>>(
        w_scale_ptr, sfa_tiled_base, total_rows, K_sf, num_k_tiles);
  }

  // --- Step 4: reshuffle SFB (activation scales, per expert) ---
  {
    int32_t num_k_tiles = (K_sf + 3) / 4;
    int64_t max_elems = (int64_t)max_tokens_ub * K_sf;
    int threads = 256;
    int blocks_x = std::min((int)((max_elems + threads - 1) / threads), 65535);
    dim3 grid(blocks_x, E);
    reshuffle_sfb_per_expert<<<grid, threads, 0, stream>>>(
        x_scale_ptr, sfb_tiled_base, cnt_ptr,
        expert_sfb_offsets_dev, K_sf, num_k_tiles, E, max_tokens_ub);
  }

  // --- Step 5: build pointer arrays ---
  {
    int threads = std::min(E, 256);
    int blocks  = (E + threads - 1) / threads;
    build_pointer_arrays<<<blocks, threads, 0, stream>>>(
        B_transposed_base, sfb_tiled_base,
        D_scratch_base,
        expert_B_offsets_dev, expert_sfb_offsets_dev, expert_D_offsets_dev,
        ptr_B_dev, ptr_SFB_dev, ptr_D_dev, E);
  }

  // --- Step 6: Run CUTLASS grouped GEMM ---
  if (mma_sm == 1) {
    run_gemm<Gemm1SM>(
        w_fp8_ptr, ptr_B_dev, sfa_tiled_base, ptr_SFB_dev,
        ptr_D_dev, tokens_per_expert_dev,
        N_hidden, max_tokens, K, E,
        cutlass_ws, cutlass_ws_size, stream);
  } else {
    run_gemm<Gemm2SM>(
        w_fp8_ptr, ptr_B_dev, sfa_tiled_base, ptr_SFB_dev,
        ptr_D_dev, tokens_per_expert_dev,
        N_hidden, max_tokens, K, E,
        cutlass_ws, cutlass_ws_size, stream);
  }

  // --- Step 7: Transpose + gather output ---
  {
    int64_t max_elems = (int64_t)max_tokens_ub * N_hidden;
    int threads = 256;
    int blocks_x = std::min((int)((max_elems + threads - 1) / threads), 65535);
    dim3 grid(blocks_x, E);
    transpose_output_gather<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16* const*>(ptr_D_dev),
        output_ptr, cnt_ptr, N_hidden, E);
  }
}
