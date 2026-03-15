"""Test RC grouped GEMM with block_A sourced from PyTorch."""
import torch
from wrapper.gemm_test.gemm_test_jit import get_gemm_test_module


def test_pytorch_block_a():
    """Create w_fp8 [E, M, K] in PyTorch and run the full GEMM + verify."""
    E = 4       # number of experts
    M = 128     # hidden dim (= CUTLASS M)
    K = 128     # reduction dim
    N = 256     # tokens per expert (fixed for all groups)

    # Create a random FP8 weight tensor in PyTorch — this is what a real
    # MoE layer would produce (e.g., self.w13 quantized to FP8).
    # Use randn → clamp → cast to avoid NaN bit patterns in FP8 E4M3.
    w_fp8 = torch.randn(E, M, K, device="cuda").clamp(-1, 1).to(torch.float8_e4m3fn)

    print(f"w_fp8 shape: {w_fp8.shape}, strides: {w_fp8.stride()}, "
          f"contiguous: {w_fp8.is_contiguous()}")

    mod = get_gemm_test_module()
    # alpha=1, beta=0 → D = A × B^T (no bias)
    mod["gemm_test"](w_fp8, N, E, 1, 0)


if __name__ == "__main__":
    test_pytorch_block_a()
