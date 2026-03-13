"""Test to isolate B transpose correctness."""
import torch
import numpy as np


def run_cutlass(x_fp8, x_scale, w_fp8, w_scale, cnt, mma_sm=1):
    from wrapper.api import _get_workspace, _estimate_workspace_size
    from wrapper.jit_module import get_module
    M = x_fp8.shape[0]
    N = w_fp8.shape[1]
    E = w_fp8.shape[0]
    K = x_fp8.shape[1]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=x_fp8.device)
    ws_size = _estimate_workspace_size(M, K, N, E)
    workspace = _get_workspace(x_fp8.device, ws_size)
    workspace.zero_()
    module = get_module()
    module.mxfp8_rc_grouped_gemm(x_fp8, x_scale, w_fp8, w_scale, cnt, output, workspace, mma_sm)
    torch.cuda.synchronize()
    return output


def main():
    device = torch.device("cuda:0")
    E, N, K = 1, 128, 128
    K_sf = K // 32
    tokens = 128

    cnt = torch.tensor([0, tokens], dtype=torch.int32, device=device)

    # Test: only row 0 of x_fp8 has non-zero data.
    # x[0, :] = 1.0, x[n>0, :] = 0.0
    # w[0] = all 1.0
    # Expected: output[0, m] = K (=128) for all m, output[n>0, m] = 0
    x_fp8 = torch.zeros(tokens, K, dtype=torch.float8_e4m3fn, device=device)
    # Set row 0 to 1.0 by creating in float32 first
    x_f32 = torch.zeros(tokens, K, device=device)
    x_f32[0, :] = 1.0
    x_fp8 = x_f32.to(torch.float8_e4m3fn)

    x_scale = torch.full((tokens, K_sf), 127, dtype=torch.uint8, device=device)
    w_fp8 = torch.ones(E, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.full((E, N, K_sf), 127, dtype=torch.uint8, device=device)

    out = run_cutlass(x_fp8, x_scale, w_fp8, w_scale, cnt)
    out_f32 = out.float()

    print("Test: x[0,:]=1, x[n>0,:]=0, w=1, scale=1")
    print(f"  Expected: out[0,:] = {K}, out[n>0,:] = 0")
    print(f"  out[0, :4] = {out_f32[0, :4].tolist()}")
    print(f"  out[1, :4] = {out_f32[1, :4].tolist()}")
    print(f"  out[2, :4] = {out_f32[2, :4].tolist()}")

    # Check which row has value K
    row_sums = out_f32.sum(dim=1)
    nonzero_rows = (row_sums.abs() > 0.1).nonzero().squeeze()
    print(f"  Non-zero rows: {nonzero_rows.tolist()}")
    if len(nonzero_rows.shape) == 0:
        nonzero_rows = nonzero_rows.unsqueeze(0)
    for r in nonzero_rows[:5]:
        r = r.item()
        print(f"    Row {r}: sum={row_sums[r].item():.1f}, first4={out_f32[r, :4].tolist()}")

    # Test 2: only column 0 of x has non-zero data
    print("\nTest 2: x[:,0]=1, x[:,k>0]=0, w=1, scale=1")
    x_f32_2 = torch.zeros(tokens, K, device=device)
    x_f32_2[:, 0] = 1.0
    x_fp8_2 = x_f32_2.to(torch.float8_e4m3fn)

    out2 = run_cutlass(x_fp8_2, x_scale, w_fp8, w_scale, cnt)
    out2_f32 = out2.float()

    print(f"  Expected: out[n, m] = w[m, 0] = 1.0 for all n, m")
    print(f"  out[0, :4] = {out2_f32[0, :4].tolist()}")
    print(f"  out[1, :4] = {out2_f32[1, :4].tolist()}")
    print(f"  out mean = {out2_f32.mean().item():.4f} (expected 1.0)")

    # Test 3: Check the actual matmul with a simple pattern
    print("\nTest 3: x = identity-like pattern")
    # x[n, k] = 1.0 if k == 0 else 0.0 for first 32 rows
    # This means each row only contributes w[:, 0] to the output
    x_f32_3 = torch.zeros(tokens, K, device=device)
    for n in range(min(32, tokens)):
        x_f32_3[n, n] = 1.0  # diagonal pattern
    x_fp8_3 = x_f32_3.to(torch.float8_e4m3fn)

    out3 = run_cutlass(x_fp8_3, x_scale, w_fp8, w_scale, cnt)
    out3_f32 = out3.float()

    # Expected: out[n, m] = w[m, n] = 1.0 for n < 32, = 0 for n >= 32
    print(f"  Expected: out[n, m] = w[m, n] = 1.0 for n < 32 (diagonal)")
    print(f"  out[0, :8] = {out3_f32[0, :8].tolist()}")
    print(f"  out[1, :8] = {out3_f32[1, :8].tolist()}")
    print(f"  out[0, 0] = {out3_f32[0, 0].item()} (expected 1.0)")
    print(f"  out[0, 1] = {out3_f32[0, 1].item()} (expected 1.0)")
    print(f"  out[1, 0] = {out3_f32[1, 0].item()} (expected 1.0)")
    print(f"  out[1, 1] = {out3_f32[1, 1].item()} (expected 1.0)")
    print(f"  out[1, 2] = {out3_f32[1, 2].item()} (expected 1.0)")

    # Where is the 1.0?
    ones_mask = (out3_f32 - 1.0).abs() < 0.1
    ones_positions = ones_mask.nonzero()
    print(f"  Positions with value ~1.0 (first 10): {ones_positions[:10].tolist()}")


if __name__ == "__main__":
    main()
