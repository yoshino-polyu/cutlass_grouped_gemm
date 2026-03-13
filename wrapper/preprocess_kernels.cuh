#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================
// (a) compute_tokens_per_expert
//     cnt[E+1] cumulative prefix sum → tokens_per_expert[E]
// ============================================================
__global__ void compute_tokens_per_expert(
    const int32_t* __restrict__ cnt,   // [E+1]
    int32_t* __restrict__ tokens,      // [E]
    int32_t num_experts)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_experts) {
    tokens[i] = cnt[i + 1] - cnt[i];
  }
}

// ============================================================
// (b) transpose_b_batched
//     For each expert i, transpose x_fp8[cnt[i]:cnt[i+1], :]
//     from RowMajor [tokens_i, K] → ColumnMajor [tokens_i, K]
//     i.e. dest[k * tokens_i + n] = src[n * K + k]
//     Contiguous per-expert blocks written to B_transposed.
// ============================================================
__global__ void transpose_b_batched(
    const uint8_t* __restrict__ x_fp8,        // [M_total, K] row-major
    uint8_t* __restrict__ B_transposed,        // pre-allocated scratch
    const int32_t* __restrict__ cnt,           // [E+1]
    const int64_t* __restrict__ expert_B_offsets, // [E] byte offsets into B_transposed
    int32_t K,
    int32_t num_experts)
{
  // Grid: (ceil(max_elements / blockDim.x), num_experts)
  int expert = blockIdx.y;
  if (expert >= num_experts) return;

  int32_t start = cnt[expert];
  int32_t end   = cnt[expert + 1];
  int32_t tokens_i = end - start;
  if (tokens_i <= 0) return;

  int64_t total_elems = (int64_t)tokens_i * K;
  int64_t B_base = expert_B_offsets[expert];

  // Each thread handles one element in the transposed output
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total_elems;
       idx += (int64_t)gridDim.x * blockDim.x) {
    // idx in column-major order: idx = k * tokens_i + n
    int32_t k = idx / tokens_i;
    int32_t n = idx % tokens_i;
    // Source: row-major (start+n, k)
    uint8_t val = x_fp8[(int64_t)(start + n) * K + k];
    // Dest: column-major offset
    B_transposed[B_base + idx] = val;
  }
}

// ============================================================
// (c) reshuffle_scale_factors
//     Converts RowMajor [rows, K_sf] uint8 scales to SfAtom-tiled layout.
//
//     SfAtom (SFVecSize=32, K-major):
//       Atom covers (128 MN, 4 K_sf) = 512 unique SF values
//       local_mn = mn % 128,  mn_0 = local_mn % 32,  mn_1 = local_mn / 32
//       local_k  = k_sf % 4
//       within_atom = mn_0 * 16 + mn_1 * 4 + local_k
//       atom_idx = tile_mn * num_k_tiles + tile_k   (K innermost, Step<_2,_1>)
//       dest = atom_idx * 512 + within_atom
// ============================================================
__global__ void reshuffle_scale_factors(
    const uint8_t* __restrict__ sf_src,   // [rows, K_sf] row-major
    uint8_t* __restrict__ sf_dst,         // tiled output
    int32_t rows,
    int32_t K_sf,                         // = K / 32
    int32_t num_k_tiles)                  // = ceil(K_sf / 4)
{
  int64_t total = (int64_t)rows * K_sf;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    int32_t mn   = idx / K_sf;
    int32_t k_sf = idx % K_sf;

    // Tile indices
    int32_t tile_mn = mn / 128;
    int32_t tile_k  = k_sf / 4;

    // Local indices within tile
    int32_t local_mn = mn % 128;
    int32_t local_k  = k_sf % 4;

    int32_t mn_0 = local_mn % 32;
    int32_t mn_1 = local_mn / 32;

    int32_t within_atom = mn_0 * 16 + mn_1 * 4 + local_k;
    int64_t atom_idx = (int64_t)tile_mn * num_k_tiles + tile_k;
    int64_t dst_offset = atom_idx * 512 + within_atom;

    sf_dst[dst_offset] = sf_src[idx];
  }
}

// Variant: reshuffle per-expert SFB (activation scales)
// For each expert i, reshuffles x_scale[cnt[i]:cnt[i+1], K_sf]
// into a separate tiled buffer at sfb_dst + expert_sfb_offsets[i]
__global__ void reshuffle_sfb_per_expert(
    const uint8_t* __restrict__ x_scale,           // [M_total, K_sf] row-major
    uint8_t* __restrict__ sfb_dst,                 // scratch for all experts
    const int32_t* __restrict__ cnt,               // [E+1]
    const int64_t* __restrict__ expert_sfb_offsets, // [E] byte offsets
    int32_t K_sf,
    int32_t num_k_tiles,
    int32_t num_experts,
    int32_t max_tokens)
{
  int expert = blockIdx.y;
  if (expert >= num_experts) return;

  int32_t start = cnt[expert];
  int32_t tokens_i = cnt[expert + 1] - start;
  if (tokens_i <= 0) return;

  int64_t total = (int64_t)tokens_i * K_sf;
  int64_t dst_base = expert_sfb_offsets[expert];

  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    int32_t n    = idx / K_sf;   // local token index
    int32_t k_sf = idx % K_sf;

    // Source: row-major x_scale[(start+n), k_sf]
    uint8_t val = x_scale[(int64_t)(start + n) * K_sf + k_sf];

    // SfAtom tiling
    int32_t tile_mn = n / 128;
    int32_t tile_k  = k_sf / 4;
    int32_t local_mn = n % 128;
    int32_t local_k  = k_sf % 4;
    int32_t mn_0 = local_mn % 32;
    int32_t mn_1 = local_mn / 32;
    int32_t within_atom = mn_0 * 16 + mn_1 * 4 + local_k;
    int64_t atom_idx = (int64_t)tile_mn * num_k_tiles + tile_k;
    int64_t dst_offset = dst_base + atom_idx * 512 + within_atom;

    sfb_dst[dst_offset] = val;
  }
}

// ============================================================
// (d) transpose_output_gather
//     D_i is [N_hidden, tokens_i] RowMajor (CUTLASS output).
//     Transpose to [tokens_i, N_hidden] and write to output
//     at row offset cnt[i].
// ============================================================
__global__ void transpose_output_gather(
    const __nv_bfloat16* const* __restrict__ D_ptrs, // [E] pointers
    __nv_bfloat16* __restrict__ output,                     // [M_total, N_hidden]
    const int32_t* __restrict__ cnt,                        // [E+1]
    int32_t N_hidden,
    int32_t num_experts)
{
  int expert = blockIdx.y;
  if (expert >= num_experts) return;

  int32_t start = cnt[expert];
  int32_t tokens_i = cnt[expert + 1] - start;
  if (tokens_i <= 0) return;

  const __nv_bfloat16* D_i = D_ptrs[expert];
  int64_t total = (int64_t)tokens_i * N_hidden;

  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    // idx in output order: row n, col m  (output is [tokens_i, N_hidden])
    int32_t n = idx / N_hidden;   // token index
    int32_t m = idx % N_hidden;   // hidden dim

    // Source: D_i is [N_hidden, tokens_i] row-major → element (m, n)
    __nv_bfloat16 val = D_i[(int64_t)m * tokens_i + n];

    // Dest: output[(start+n), m]
    output[(int64_t)(start + n) * N_hidden + m] = val;
  }
}

// ============================================================
// (e) build_pointer_arrays
//     Populate device ptr_B, ptr_SFB, ptr_D arrays from base
//     pointers and per-expert offsets. Also build per-expert
//     LayoutSFB on device.
// ============================================================
__global__ void build_pointer_arrays(
    const uint8_t* __restrict__ B_transposed_base,
    const uint8_t* __restrict__ sfb_tiled_base,
    __nv_bfloat16* __restrict__ D_scratch_base,
    const int64_t* __restrict__ expert_B_offsets,
    const int64_t* __restrict__ expert_sfb_offsets,
    const int64_t* __restrict__ expert_D_offsets,
    const uint8_t** __restrict__ ptr_B,
    const uint8_t** __restrict__ ptr_SFB,
    __nv_bfloat16** __restrict__ ptr_D,
    int32_t num_experts)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_experts) {
    ptr_B[i]   = B_transposed_base + expert_B_offsets[i];
    ptr_SFB[i] = sfb_tiled_base + expert_sfb_offsets[i];
    ptr_D[i]   = D_scratch_base + expert_D_offsets[i];
  }
}
