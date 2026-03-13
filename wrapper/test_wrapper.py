"""End-to-end test for MX FP8 RC grouped GEMM wrapper.

Compares CUTLASS output against a dequantized FP32 reference matmul.
"""
import torch
import numpy as np


def dequantize_mxfp8(data_fp8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize MX FP8 (e4m3fn) + E8M0 block scales to FP32.

    Args:
        data_fp8:    [..., K] float8_e4m3fn
        scale_e8m0:  [..., K//32] uint8, each byte is an E8M0 exponent

    Returns:
        [..., K] float32
    """
    # Convert fp8 to float32
    data_f32 = data_fp8.to(torch.float32)

    # E8M0 scale: value = 2^(exponent - 127)
    exponents = scale_e8m0.to(torch.int32) - 127
    scale_f32 = torch.pow(2.0, exponents.to(torch.float32))  # [..., K//32]

    # Broadcast: each scale applies to 32 consecutive elements along K
    K = data_f32.shape[-1]
    K_sf = scale_f32.shape[-1]
    assert K == K_sf * 32, f"K={K}, K_sf={K_sf}"

    # Reshape scale to broadcast: [..., K_sf, 1] → [..., K_sf * 32]
    scale_expanded = scale_f32.unsqueeze(-1).expand(*scale_f32.shape, 32)
    scale_expanded = scale_expanded.reshape(*scale_f32.shape[:-1], K)

    return data_f32 * scale_expanded


def reference_grouped_gemm(
    x_fp8: torch.Tensor,     # [M, K] float8_e4m3fn
    x_scale: torch.Tensor,   # [M, K//32] uint8
    w_fp8: torch.Tensor,     # [E, N, K] float8_e4m3fn
    w_scale: torch.Tensor,   # [E, N, K//32] uint8
    cnt: torch.Tensor,       # [E+1] int32
) -> torch.Tensor:
    """FP32 reference: Y = X @ W^T per expert."""
    M = x_fp8.shape[0]
    N = w_fp8.shape[1]
    E = w_fp8.shape[0]

    # Dequantize weights: [E, N, K] → float32
    w_f32 = dequantize_mxfp8(w_fp8, w_scale)

    # Dequantize activations: [M, K] → float32
    x_f32 = dequantize_mxfp8(x_fp8, x_scale)

    output = torch.zeros(M, N, dtype=torch.float32, device=x_fp8.device)

    cnt_cpu = cnt.cpu().numpy()
    for i in range(E):
        start = int(cnt_cpu[i])
        end   = int(cnt_cpu[i + 1])
        if end <= start:
            continue
        # x_i: [tokens_i, K], w_i: [N, K]
        x_i = x_f32[start:end, :]  # [tokens_i, K]
        w_i = w_f32[i]             # [N, K]
        # Y_i = X_i @ W_i^T → [tokens_i, N]
        output[start:end, :] = x_i @ w_i.T

    return output.to(torch.bfloat16)


def test_mxfp8_rc_grouped_gemm(E=4, M=256, N=128, K=256, mma_sm=1, seed=42):
    """Test CUTLASS wrapper against FP32 reference."""
    from wrapper.api import mxfp8_rc_grouped_gemm

    torch.manual_seed(seed)
    device = torch.device("cuda:0")

    K_sf = K // 32

    # Generate random token counts per expert
    tokens_per_expert = torch.randint(16, M // E + 1, (E,), dtype=torch.int32)
    # Make sure total tokens = M
    tokens_per_expert[-1] = M - tokens_per_expert[:-1].sum()
    tokens_per_expert = tokens_per_expert.clamp(min=1)
    M_actual = int(tokens_per_expert.sum().item())

    # Cumulative prefix sum
    cnt = torch.zeros(E + 1, dtype=torch.int32, device=device)
    cnt[1:] = torch.cumsum(tokens_per_expert.to(device), dim=0)

    # Random FP8 activations
    x_f32 = torch.randn(M_actual, K, device=device)
    x_fp8 = x_f32.to(torch.float8_e4m3fn)

    # Random E8M0 scales for activations (exponents in reasonable range: 120-134 → ~0.008 to ~128)
    x_scale = torch.randint(120, 135, (M_actual, K_sf), dtype=torch.uint8, device=device)

    # Random FP8 weights
    w_f32 = torch.randn(E, N, K, device=device)
    w_fp8 = w_f32.to(torch.float8_e4m3fn)

    # Random E8M0 scales for weights
    w_scale = torch.randint(120, 135, (E, N, K_sf), dtype=torch.uint8, device=device)

    # --- Reference ---
    ref_output = reference_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt)

    # --- CUTLASS wrapper ---
    cutlass_output = mxfp8_rc_grouped_gemm(
        x_fp8, x_scale, w_fp8, w_scale, cnt, mma_sm=mma_sm
    )

    # --- Compare ---
    # FP8 matmul has limited precision; use generous tolerance
    ref_f32 = ref_output.float()
    out_f32 = cutlass_output.float()

    abs_diff = (ref_f32 - out_f32).abs()
    rel_diff = abs_diff / (ref_f32.abs().clamp(min=1e-6))

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()

    print(f"  mma_sm={mma_sm}, E={E}, M={M_actual}, N={N}, K={K}")
    print(f"  max_abs_diff={max_abs:.6f}, mean_abs_diff={mean_abs:.6f}, max_rel_diff={max_rel:.6f}")

    # Check: for FP8 with block scaling, ~1e-2 tolerance is reasonable
    atol = 0.1  # generous for FP8
    rtol = 0.05
    close = torch.allclose(ref_f32, out_f32, atol=atol, rtol=rtol)
    print(f"  allclose(atol={atol}, rtol={rtol}): {close}")

    if not close:
        # Find worst elements
        worst_idx = abs_diff.argmax()
        row = worst_idx // N
        col = worst_idx % N
        print(f"  Worst element at [{row}, {col}]: ref={ref_f32.flatten()[worst_idx]:.6f}, "
              f"got={out_f32.flatten()[worst_idx]:.6f}")

    return close


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MX FP8 RC Grouped GEMM Wrapper")
    print("=" * 60)

    passed = True

    for mma_sm in [1, 2]:
        print(f"\n--- Testing mma_sm={mma_sm} ---")
        ok = test_mxfp8_rc_grouped_gemm(E=4, M=256, N=128, K=256, mma_sm=mma_sm)
        passed = passed and ok

    print("\n" + "=" * 60)
    if passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)