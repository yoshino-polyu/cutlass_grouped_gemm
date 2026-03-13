"""Targeted diagnostic to identify which component causes wrong results."""
import torch
import numpy as np


def dequantize_mxfp8(data_fp8, scale_e8m0):
    data_f32 = data_fp8.to(torch.float32)
    exponents = scale_e8m0.to(torch.int32) - 127
    scale_f32 = torch.pow(2.0, exponents.to(torch.float32))
    K = data_f32.shape[-1]
    scale_expanded = scale_f32.unsqueeze(-1).expand(*scale_f32.shape, 32)
    scale_expanded = scale_expanded.reshape(*scale_f32.shape[:-1], K)
    return data_f32 * scale_expanded


def reference_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt):
    M = x_fp8.shape[0]
    N = w_fp8.shape[1]
    E = w_fp8.shape[0]
    w_f32 = dequantize_mxfp8(w_fp8, w_scale)
    x_f32 = dequantize_mxfp8(x_fp8, x_scale)
    output = torch.zeros(M, N, dtype=torch.float32, device=x_fp8.device)
    cnt_cpu = cnt.cpu().numpy()
    for i in range(E):
        start, end = int(cnt_cpu[i]), int(cnt_cpu[i + 1])
        if end <= start:
            continue
        output[start:end, :] = x_f32[start:end, :] @ w_f32[i].T
    return output.to(torch.bfloat16)


def run_cutlass(x_fp8, x_scale, w_fp8, w_scale, cnt, mma_sm=1):
    from wrapper.api import _get_workspace, _estimate_workspace_size
    from wrapper.jit_module import get_module
    M = x_fp8.shape[0]
    N = w_fp8.shape[1]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=x_fp8.device)
    E = w_fp8.shape[0]
    K = x_fp8.shape[1]
    ws_size = _estimate_workspace_size(M, K, N, E)
    workspace = _get_workspace(x_fp8.device, ws_size)
    workspace.zero_()
    module = get_module()
    module.mxfp8_rc_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt, output, workspace, mma_sm)
    torch.cuda.synchronize()
    return output


def test_case(name, x_fp8, x_scale, w_fp8, w_scale, cnt, mma_sm=1):
    ref = reference_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt)
    out = run_cutlass(x_fp8, x_scale, w_fp8, w_scale, cnt, mma_sm)
    ref_f32, out_f32 = ref.float(), out.float()

    abs_diff = (ref_f32 - out_f32).abs()
    nan_count = torch.isnan(out_f32).sum().item()
    max_abs = abs_diff[~torch.isnan(abs_diff)].max().item() if not torch.isnan(abs_diff).all() else float('nan')
    mean_abs = abs_diff[~torch.isnan(abs_diff)].mean().item() if not torch.isnan(abs_diff).all() else float('nan')

    passed = max_abs < 1.0 and nan_count == 0
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}, NaN={nan_count}")

    if not passed:
        # Show first few diffs per expert
        cnt_cpu = cnt.cpu().numpy()
        E = w_fp8.shape[0]
        for i in range(E):
            s, e = int(cnt_cpu[i]), int(cnt_cpu[i + 1])
            expert_diff = abs_diff[s:e, :]
            expert_max = expert_diff.max().item()
            # Show first element comparison
            if e > s:
                print(f"    Expert {i} [{s}:{e}]: max_diff={expert_max:.4f}, "
                      f"ref[0,:4]={ref_f32[s,:4].tolist()}, out[0,:4]={out_f32[s,:4].tolist()}")
    return passed


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    E, N, K = 4, 128, 256
    K_sf = K // 32
    tokens = [32, 32, 32, 32]
    M = sum(tokens)

    cnt = torch.zeros(E + 1, dtype=torch.int32, device=device)
    cnt[1:] = torch.cumsum(torch.tensor(tokens, dtype=torch.int32, device=device), dim=0)

    print("=" * 60)
    print("Diagnostic Tests")
    print("=" * 60)

    # Test 1: All ones data, uniform scale=127 (baseline, known good)
    print("\n--- Test 1: All ones, scale=127 ---")
    x_fp8 = torch.ones(M, K, dtype=torch.float8_e4m3fn, device=device)
    x_scale = torch.full((M, K_sf), 127, dtype=torch.uint8, device=device)
    w_fp8 = torch.ones(E, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.full((E, N, K_sf), 127, dtype=torch.uint8, device=device)
    test_case("ones/scale127", x_fp8, x_scale, w_fp8, w_scale, cnt)

    # Test 2: E=1, random FP8, uniform scale=127
    # Tests B transpose only (no scale factor contribution)
    print("\n--- Test 2: E=1, random data, scale=127 ---")
    cnt1 = torch.tensor([0, 64], dtype=torch.int32, device=device)
    x_fp8_r = torch.randn(64, K, device=device).to(torch.float8_e4m3fn)
    x_scale_u = torch.full((64, K_sf), 127, dtype=torch.uint8, device=device)
    w_fp8_r = torch.randn(1, N, K, device=device).to(torch.float8_e4m3fn)
    w_scale_u = torch.full((1, N, K_sf), 127, dtype=torch.uint8, device=device)
    test_case("E=1/random_data/scale127", x_fp8_r, x_scale_u, w_fp8_r, w_scale_u, cnt1)

    # Test 3: E=1, all ones, varying scales
    # Tests scale factor tiling for single expert
    print("\n--- Test 3: E=1, ones data, varying scales ---")
    x_fp8_1 = torch.ones(64, K, dtype=torch.float8_e4m3fn, device=device)
    x_scale_v = torch.randint(120, 135, (64, K_sf), dtype=torch.uint8, device=device)
    w_fp8_1 = torch.ones(1, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale_v = torch.randint(120, 135, (1, N, K_sf), dtype=torch.uint8, device=device)
    test_case("E=1/ones_data/varying_scales", x_fp8_1, x_scale_v, w_fp8_1, w_scale_v, cnt1)

    # Test 4: E=4, all ones, varying scales
    # Tests multi-expert scale factor tiling
    print("\n--- Test 4: E=4, ones data, varying scales ---")
    x_scale_v4 = torch.randint(120, 135, (M, K_sf), dtype=torch.uint8, device=device)
    w_scale_v4 = torch.randint(120, 135, (E, N, K_sf), dtype=torch.uint8, device=device)
    x_fp8_1m = torch.ones(M, K, dtype=torch.float8_e4m3fn, device=device)
    w_fp8_1m = torch.ones(E, N, K, dtype=torch.float8_e4m3fn, device=device)
    test_case("E=4/ones_data/varying_scales", x_fp8_1m, x_scale_v4, w_fp8_1m, w_scale_v4, cnt)

    # Test 5: E=4, random data, uniform scale=127
    # Tests B transpose across experts
    print("\n--- Test 5: E=4, random data, scale=127 ---")
    x_fp8_rm = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
    x_scale_um = torch.full((M, K_sf), 127, dtype=torch.uint8, device=device)
    w_fp8_rm = torch.randn(E, N, K, device=device).to(torch.float8_e4m3fn)
    w_scale_um = torch.full((E, N, K_sf), 127, dtype=torch.uint8, device=device)
    test_case("E=4/random_data/scale127", x_fp8_rm, x_scale_um, w_fp8_rm, w_scale_um, cnt)

    # Test 6: Full random (E=4, random data, random scales)
    print("\n--- Test 6: Full random ---")
    x_scale_full = torch.randint(120, 135, (M, K_sf), dtype=torch.uint8, device=device)
    w_scale_full = torch.randint(120, 135, (E, N, K_sf), dtype=torch.uint8, device=device)
    test_case("full_random", x_fp8_rm, x_scale_full, w_fp8_rm, w_scale_full, cnt)


if __name__ == "__main__":
    main()
