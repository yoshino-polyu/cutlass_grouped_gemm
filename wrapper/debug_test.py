"""Diagnostic test to identify where NaN comes from in the CUTLASS pipeline."""
import torch
import numpy as np


def debug_mxfp8_rc_grouped_gemm():
    from wrapper.api import _get_workspace, _estimate_workspace_size
    from wrapper.jit_module import get_module

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    E, N, K = 4, 128, 256
    K_sf = K // 32
    mma_sm = 1

    # Simple token distribution
    tokens_per_expert = torch.tensor([32, 32, 32, 32], dtype=torch.int32)
    M_actual = int(tokens_per_expert.sum().item())

    cnt = torch.zeros(E + 1, dtype=torch.int32, device=device)
    cnt[1:] = torch.cumsum(tokens_per_expert.to(device), dim=0)
    print(f"cnt = {cnt.cpu().tolist()}")

    # Simple data: small values to avoid overflow
    x_fp8 = torch.ones(M_actual, K, dtype=torch.float8_e4m3fn, device=device)
    # Scale = 127 means 2^(127-127) = 2^0 = 1.0
    x_scale = torch.full((M_actual, K_sf), 127, dtype=torch.uint8, device=device)

    w_fp8 = torch.ones(E, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.full((E, N, K_sf), 127, dtype=torch.uint8, device=device)

    # Expected: each output element = sum of K products of (1.0 * 1.0 * 1.0 * 1.0) = K = 256
    print(f"Expected output value (all-ones, scale=1): {K}")

    output = torch.empty(M_actual, N, dtype=torch.bfloat16, device=device)

    ws_size = _estimate_workspace_size(M_actual, K, N, E)
    workspace = _get_workspace(device, ws_size)

    # Zero workspace to detect uninitialized reads
    workspace.zero_()

    module = get_module()
    module.mxfp8_rc_grouped_gemm(
        x_fp8, x_scale, w_fp8, w_scale, cnt, output, workspace, mma_sm
    )

    torch.cuda.synchronize()

    out_f32 = output.float()
    print(f"\nOutput stats:")
    print(f"  shape: {output.shape}")
    print(f"  has NaN: {torch.isnan(out_f32).any().item()}")
    print(f"  has Inf: {torch.isinf(out_f32).any().item()}")
    nan_count = torch.isnan(out_f32).sum().item()
    print(f"  NaN count: {nan_count} / {out_f32.numel()}")
    if nan_count < out_f32.numel():
        valid = out_f32[~torch.isnan(out_f32)]
        print(f"  valid min: {valid.min().item():.4f}")
        print(f"  valid max: {valid.max().item():.4f}")
        print(f"  valid mean: {valid.mean().item():.4f}")

    # Check first few elements
    print(f"\nFirst 8 output elements [0,:8]: {out_f32[0, :8].tolist()}")
    print(f"First 8 output elements [1,:8]: {out_f32[1, :8].tolist()}")

    # Check each expert's output
    cnt_cpu = cnt.cpu().numpy()
    for i in range(E):
        start = int(cnt_cpu[i])
        end = int(cnt_cpu[i + 1])
        expert_out = out_f32[start:end, :]
        nan_pct = torch.isnan(expert_out).float().mean().item() * 100
        print(f"\n  Expert {i} [{start}:{end}]: NaN={nan_pct:.1f}%", end="")
        if nan_pct < 100:
            valid = expert_out[~torch.isnan(expert_out)]
            print(f", valid range=[{valid.min():.2f}, {valid.max():.2f}]", end="")
        print()


if __name__ == "__main__":
    debug_mxfp8_rc_grouped_gemm()
