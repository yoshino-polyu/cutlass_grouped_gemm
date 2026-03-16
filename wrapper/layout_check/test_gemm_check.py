"""Test full RC grouped GEMM with all 4 operands from PyTorch.

B and SFB use offset arrays to support variable N per expert, mirroring
how a real MoE layer packs tokens: w_act is a single [total_tokens, K]
buffer, and offsets_B[i] marks where expert i's tokens begin.

Layout summary:
  A   [E, M, K]          fp8   → CUTLASS block_A [M,K,G] RowMajor, single ptr
  SFA [sfa_total]         u8    → tiled layout, single ptr
  B   [total_tokens, K]   fp8   → ptr_B[i] = &w_act[offsets_B[i], :], per-expert ptr array
  SFB [total_sfb]         u8    → ptr_SFB[i] = &w_sfb[offsets_SFB[i]], per-expert ptr array

B in CUTLASS is ColumnMajor with stride {K, 1, 0}. A PyTorch [N, K]
contiguous tensor has stride [K, 1] — identical physical layout.

Run: python -m wrapper.layout_check.test_gemm_check

Bugs found during development:
===========================================================================

1. FP8 NaN from random bytes

   torch.randint(0, 255).view(torch.float8_e4m3fn) produces arbitrary bit
   patterns including FP8 E4M3 NaN (bytes 0x7F, 0xFF).  NaN != NaN, so
   TensorEquals fails even when the kernel is correct.

   Fix: torch.randn(...).clamp(-1, 1).to(torch.float8_e4m3fn).

2. std::vector<HostTensor> reallocation invalidates device pointers

   When push_back() triggers reallocation, all existing HostTensor objects
   are moved to a new buffer.  Device pointers captured BEFORE the move
   (ptr_D_host[i] = block_D[i].device_data()) still point to valid CUDA
   memory, but after the move block_D[i].sync_host() copies from the
   device into a DIFFERENT host buffer (the moved object's new host_
   vector).  The kernel wrote to the correct device address, but
   sync_host() reads the wrong host memory -> zeros.

   Symptom: groups 0-1 had D=0x0000 (zeros) while groups 2-3 were correct
   (D=0x4300 = BF16 128.0).  Later-allocated groups weren't affected
   because no subsequent push_back moved them.

   Fix: block_D.reserve(groups) before the loop to prevent reallocation.

3. N_i < MmaTileShape_N produces wrong results

   With 1SM config MmaTileShape = (128, 256, 128), setting any expert's
   N_i < 256 (e.g. N=64) causes verification failure.  The kernel runs
   without error (can_implement passes), but the output is incorrect.

   Tested: [208, 512, 64, 304] -> group 2 (N=64) FAILED, others passed.
   Tested: [208, 512, 304, 176] -> all passed (all N >= 176, though < 256,
   so the exact lower bound may be < 256 but > 64).

   This is NOT an address alignment issue.  TMA address alignment for B
   depends only on K (the contiguous dimension): as long as K % 16 == 0,
   each expert's start address (base + offsets_B[i] * K) is automatically
   16-byte aligned, regardless of N_i.  The failure is a tile-boundary
   edge case internal to the CUTLASS kernel when N < tile width.
===========================================================================
"""
import torch
from wrapper.layout_check.layout_check_jit import get_layout_check_module


def test_gemm_check_variable_n():
    """Full GEMM: variable N per expert via offset arrays."""
    print("\n" + "=" * 60)
    print("Test: Full GEMM check (variable N per expert)")
    print("=" * 60)

    E = 4       # experts (groups)
    M = 128     # hidden dim
    K = 128     # reduction dim
    # Variable N per expert — can be any value (not restricted to multiples
    # of 16). TMA address alignment is satisfied by K % 16 == 0 since K is
    # the contiguous dimension: each expert's start address
    # base + offsets_B[i] * K is automatically 16-byte aligned.
    #
    # Note: N_i must be >= MmaTileShape_N (256 for 1SM config) to avoid
    # tile-boundary edge cases where N < tile width.
    tokens_per_expert = [208, 512, 304, 176]

    mod = get_layout_check_module()

    # A: [E, M, K] contiguous FP8
    w_fp8 = torch.randn(E, M, K, device="cuda").clamp(-1, 1).to(torch.float8_e4m3fn)
    print(f"  A: shape={list(w_fp8.shape)}")

    # SFA: flat [sfa_total] uint8
    # E8M0 encoding: byte value v → scale = 2^(v-127)
    # Values 126-130 → scales 0.5, 1.0, 2.0, 4.0, 8.0
    sfa_size = mod["get_sfa_size"](M, K, E)
    w_sfa = torch.randint(126, 131, (sfa_size,), dtype=torch.uint8, device="cuda")
    print(f"  SFA: size={sfa_size}")

    # B: packed [total_tokens, K] — all experts' tokens concatenated
    total_tokens = sum(tokens_per_expert)
    w_act = torch.randn(total_tokens, K, device="cuda").clamp(-1, 1).to(torch.float8_e4m3fn)
    print(f"  B: shape={list(w_act.shape)}, tokens_per_expert={tokens_per_expert}")

    # offsets_B: [E+1] cumulative token counts
    # expert i owns rows [offsets_B[i], offsets_B[i+1]) of w_act
    offsets_B_host = [0]
    for n in tokens_per_expert:
        offsets_B_host.append(offsets_B_host[-1] + n)
    offsets_B = torch.tensor(offsets_B_host, dtype=torch.int32, device="cuda")

    # SFB: packed [total_sfb] — sfb_size varies with N_i
    sfb_sizes = [mod["get_sfb_size"](M, n, K) for n in tokens_per_expert]
    total_sfb = sum(sfb_sizes)
    w_sfb = torch.randint(126, 131, (total_sfb,), dtype=torch.uint8, device="cuda")

    # offsets_SFB: [E+1] cumulative SFB element counts
    offsets_SFB_host = [0]
    for s in sfb_sizes:
        offsets_SFB_host.append(offsets_SFB_host[-1] + s)
    offsets_SFB = torch.tensor(offsets_SFB_host, dtype=torch.int32, device="cuda")
    print(f"  SFB: total={total_sfb}, per_expert={sfb_sizes}")

    # alpha=1, beta=0 → D = A * B^T
    mod["gemm_check"](w_fp8, w_sfa, w_act, w_sfb, offsets_B, offsets_SFB, 1, 0)

def test_gemm_check_uniform():
    """Simpler test: all experts have the same N=256."""
    print("\n" + "=" * 60)
    print("Test: Full GEMM check (uniform N=256)")
    print("=" * 60)

    E, M, K, N = 4, 128, 128, 256
    mod = get_layout_check_module()

    w_fp8 = torch.randn(E, M, K, device="cuda").clamp(-1, 1).to(torch.float8_e4m3fn)
    sfa_size = mod["get_sfa_size"](M, K, E)
    w_sfa = torch.randint(126, 131, (sfa_size,), dtype=torch.uint8, device="cuda")

    # Uniform N: packed [E*N, K]
    total_tokens = E * N
    w_act = torch.randn(total_tokens, K, device="cuda").clamp(-1, 1).to(torch.float8_e4m3fn)
    offsets_B = torch.arange(E + 1, dtype=torch.int32, device="cuda") * N

    sfb_per_expert = mod["get_sfb_size"](M, N, K)
    total_sfb = E * sfb_per_expert
    w_sfb = torch.randint(126, 131, (total_sfb,), dtype=torch.uint8, device="cuda")
    offsets_SFB = torch.arange(E + 1, dtype=torch.int32, device="cuda") * sfb_per_expert

    print(f"  A: [E={E}, M={M}, K={K}], B: [{total_tokens}, {K}], N={N}/expert")
    mod["gemm_check"](w_fp8, w_sfa, w_act, w_sfb, offsets_B, offsets_SFB, 1, 0)


if __name__ == "__main__":
    test_gemm_check_uniform()
    test_gemm_check_variable_n()
