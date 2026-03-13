"""Python API for MX FP8 RC grouped GEMM — same interface as Triton Mgemm_mxfp8."""
import torch
import functools
from .jit_module import get_module

# Cached workspace buffer per device
_workspace_cache: dict = {}

DEFAULT_WORKSPACE_SIZE = 256 * 1024 * 1024  # 256 MB


def _get_workspace(device: torch.device, min_size: int) -> torch.Tensor:
    """Get or grow a cached workspace buffer on the given device."""
    key = device.index if device.index is not None else 0
    if key not in _workspace_cache or _workspace_cache[key].numel() < min_size:
        _workspace_cache[key] = torch.empty(
            max(min_size, DEFAULT_WORKSPACE_SIZE),
            dtype=torch.uint8,
            device=device,
        )
    return _workspace_cache[key]


def _estimate_workspace_size(
    M_total: int, K: int, N_hidden: int, E: int
) -> int:
    """Conservative upper bound for workspace bytes."""
    K_sf = K // 32

    size = 0
    # tokens_per_expert [E] int32
    size += E * 4 + 256
    # expert offsets: 3 arrays of [E] int64
    size += 3 * (E * 8 + 256)
    # B_transposed: E * M_total * K (each expert gets M_total * K upper bound)
    size += E * M_total * K + 256
    # SFA tiled: E * tiled_sf_size(N_hidden, K_sf)
    num_mn_tiles_a = (N_hidden + 127) // 128
    num_k_tiles = (K_sf + 3) // 4
    size += E * num_mn_tiles_a * num_k_tiles * 512 + 256
    # SFB tiled: E * tiled_sf_size(M_total, K_sf)
    num_mn_tiles_b = (M_total + 127) // 128
    size += E * num_mn_tiles_b * num_k_tiles * 512 + 256
    # D scratch: E * N_hidden * M_total * 2 (bf16)
    size += E * N_hidden * M_total * 2 + 256
    # Pointer arrays: 3 * E * 8
    size += 3 * (E * 8 + 256)
    # CUTLASS workspace
    size += 32 * 1024 * 1024
    return size


def mxfp8_rc_grouped_gemm(
    x_fp8: torch.Tensor,     # [M, K] float8_e4m3fn
    x_scale: torch.Tensor,   # [M, K//32] uint8 (E8M0)
    w_fp8: torch.Tensor,     # [E, N, K] float8_e4m3fn
    w_scale: torch.Tensor,   # [E, N, K//32] uint8 (E8M0)
    cnt: torch.Tensor,       # [E+1] int32
    mma_sm: int = 1,         # 1 or 2
) -> torch.Tensor:
    """
    MX FP8 block-scaled ragged-contiguous grouped GEMM.

    Computes Y[M, N] = X @ W^T per expert, using CUTLASS SM100 kernels.
    Same interface as Triton Mgemm_mxfp8.

    Args:
        x_fp8:   Activations [M_total, K] float8_e4m3fn
        x_scale: Activation scales [M_total, K//32] uint8 (E8M0)
        w_fp8:   Expert weights [E, N, K] float8_e4m3fn
        w_scale: Weight scales [E, N, K//32] uint8 (E8M0)
        cnt:     Cumulative token counts [E+1] int32
        mma_sm:  1 or 2 (1SM or 2SM CUTLASS kernel)

    Returns:
        output: [M_total, N] bfloat16
    """
    assert x_fp8.ndim == 2, f"x_fp8 must be 2D, got {x_fp8.ndim}D"
    assert w_fp8.ndim == 3, f"w_fp8 must be 3D, got {w_fp8.ndim}D"
    assert cnt.ndim == 1, f"cnt must be 1D, got {cnt.ndim}D"

    M_total = x_fp8.shape[0]
    K = x_fp8.shape[1]
    E = w_fp8.shape[0]
    N_hidden = w_fp8.shape[1]

    # Pre-allocate output
    output = torch.empty(M_total, N_hidden, dtype=torch.bfloat16, device=x_fp8.device)

    # Get workspace
    ws_size = _estimate_workspace_size(M_total, K, N_hidden, E)
    workspace = _get_workspace(x_fp8.device, ws_size)

    # Call the compiled kernel
    module = get_module()
    module.mxfp8_rc_grouped_gemm(
        x_fp8, x_scale, w_fp8, w_scale, cnt, output, workspace, mma_sm
    )

    return output
