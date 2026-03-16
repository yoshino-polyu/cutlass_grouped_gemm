"""Test block_A layout parity between PyTorch and CUTLASS."""
import torch
from wrapper.layout_check.layout_check_jit import get_layout_check_module
'''
  test_contiguous vs test_sequential:                                                                                                                                 
  - test_contiguous uses random bytes. It can verify that PyTorch and CUTLASS compute the same offset for every (m, k, g) coordinate, but since values are random, two
  different offsets could coincidentally hold the same byte.
  - test_sequential fills the tensor with sequential bytes (0x00, 0x01, 0x02, ...) so every element has a unique value. This lets the C++ side verify not just that
  offsets match, but that the data values at CUTLASS coordinates match PyTorch's — confirming the traversal order is truly identical, not just the stride arithmetic.
'''


def test_contiguous():
    """Smoke test: random contiguous [E, N, K] tensor — expect ALL PASS.

    Uses random data to verify that stride values and coordinate offsets
    agree between PyTorch and CUTLASS for a standard contiguous tensor.
    Random bytes mean every offset is reachable but the *values* at each
    coordinate are not predictable, so this test can only confirm that
    the two sides compute the same offset — not that the offsets follow
    a particular order.
    """
    print("\n" + "=" * 60)
    print("Test 1: Contiguous tensor")
    print("=" * 60)
    E, N, K = 4, 8, 16
    w_fp8 = torch.randint(0, 255, (E, N, K), dtype=torch.uint8).cuda()
    w_fp8 = w_fp8.view(torch.float8_e4m3fn)

    mod = get_layout_check_module()
    # check_block_a_layout(w_fp8, print_coords):
    #   w_fp8       — the GPU tensor to check ([E, N, K] as float8_e4m3fn)
    #   print_coords — how many sample coordinates to print (for debugging);
    #                   the check still runs over ALL coordinates regardless.
    mod["check_block_a_layout"](w_fp8, 17)


def test_sequential():
    """Sequential byte pattern (0x00, 0x01, ...) to verify coordinate ordering.

    Unlike test_contiguous (random data), this fills the tensor with
    sequential bytes so that each element has a unique, predictable value.
    This lets the C++ side verify not just that offsets match, but that
    the *data* at CUTLASS coordinate (m, k, g) equals the data at the
    corresponding PyTorch coordinate — confirming the traversal order is
    identical, not just the stride arithmetic.
    """
    print("\n" + "=" * 60)
    print("Test 2: Sequential pattern")
    print("=" * 60)
    E, N, K = 2, 4, 8
    total = E * N * K
    data = torch.arange(total, dtype=torch.uint8).cuda()
    w_fp8 = data.view(E, N, K).view(torch.float8_e4m3fn)

    mod = get_layout_check_module()
    mod["check_block_a_layout"](w_fp8, 10)


def test_non_contiguous():
    """Non-contiguous (transposed) tensor — expect stride FAIL."""
    print("\n" + "=" * 60)
    print("Test 3: Non-contiguous tensor (permuted dims 1,2)")
    print("=" * 60)
    E, N, K = 4, 8, 16
    w_fp8 = torch.randint(0, 255, (E, N, K), dtype=torch.uint8).cuda()
    w_fp8 = w_fp8.view(torch.float8_e4m3fn)
    # Permute N and K dims -> shape [E, K, N], strides change
    w_transposed = w_fp8.permute(0, 2, 1)

    mod = get_layout_check_module()
    mod["check_block_a_layout"](w_transposed, 5)


def test_sfa_layout():
    """SFA layout check: flat PyTorch tensor round-trips through HostTensorSF.

    SFA uses a hardware-tiled layout, NOT simple row-major.  For M=128,
    K=128, groups=4, the CuTe layout printed is:

      Shape:  (((_32,_4),1),((_32,_4),1),(_1,4))
      Stride: (((_16,_4),512),((_0,_1),_512),(_0,512))

    Decomposition (CuTe Shape:Stride per mode):

      Mode 0 — M dimension (128 rows):
        _32 stride _16  : 32 rows within a tile, each 16 bytes apart
        _4  stride _4   : 4 tiles of 32 rows, interleaved at offset 4
        1   stride 512  : (placeholder)

      Mode 1 — K dimension (128 elements -> 4 scale-factor blocks):
        _32 stride _0   : 32 K elements within a block — stride 0 = broadcast
                          (all 32 share one SF, this IS block-scaling semantics)
        _4  stride _1   : 4 K-blocks (128/32), each at offset +1
        1   stride _512 : (placeholder)

      Mode 2 — Groups:
        _1  stride _0   : (placeholder from atom)
        4   stride 512  : 4 groups, each 512 bytes apart

    Offset formula: offset(m, k, group) =
        (m % 32) * 16 + (m / 32) * 4 + (k / 32) * 1 + group * 512

    The stride-0 in K's inner mode encodes the block-scaling rule: every 32
    consecutive K elements share one E8M0 scale factor.  filter_zeros removes
    these broadcast modes to give the allocation size: 512 per group
    (128 rows x 4 K-blocks).

    Why not simple row-major (offset = m * 4 + k_block)?

    The first 16 bytes in memory look like:

      byte 0:  SF(m=0,  k_blk=0)   byte 4:  SF(m=32, k_blk=0)
      byte 1:  SF(m=0,  k_blk=1)   byte 5:  SF(m=32, k_blk=1)
      byte 2:  SF(m=0,  k_blk=2)   byte 6:  SF(m=32, k_blk=2)
      byte 3:  SF(m=0,  k_blk=3)   byte 7:  SF(m=32, k_blk=3)
      byte 8:  SF(m=64, k_blk=0)   byte 12: SF(m=96, k_blk=0)
      byte 9:  SF(m=64, k_blk=1)   byte 13: SF(m=96, k_blk=1)
      byte 10: SF(m=64, k_blk=2)   byte 14: SF(m=96, k_blk=2)
      byte 11: SF(m=64, k_blk=3)   byte 15: SF(m=96, k_blk=3)

    Each 16-byte group packs scale factors for one row-within-tile across
    all 4 M-tiles and 4 K-blocks.  This interleaving is motivated by:

      1. 16-byte TMA alignment: 128 bits = one TMA load unit.  One load
         fetches all 16 SFs for a single row-position across all tiles
         and K-blocks.
      2. MMA tile access pattern: the UMMA processes 32 rows per tile,
         needing SFs at stride 16 (bytes 0, 16, 32, ..., 496) — a simple
         strided pattern TMA handles efficiently.
      3. K-block adjacency: SFs for consecutive K-blocks of the same row
         are adjacent (bytes 0,1,2,3), so K-dimension reduction reads
         them without extra loads.
    """
    print("\n" + "=" * 60)
    print("Test 4: SFA layout parity check")
    print("=" * 60)
    M, K, groups = 128, 128, 4

    mod = get_layout_check_module()

    # Query the expected SFA allocation size from CUTLASS.
    # SFA uses a hardware-tiled layout; the size is NOT simply M*K/block_size.
    sfa_size = mod["get_sfa_size"](M, K, groups)
    print(f"  SFA size for M={M}, K={K}, groups={groups}: {sfa_size}")

    # Create a flat tensor of E8M0 bytes and pass to the C++ check.
    # Any byte values work — we are checking layout parity, not arithmetic.
    w_sfa = torch.randint(1, 16, (sfa_size,), dtype=torch.uint8).cuda()
    mod["check_sfa_layout"](w_sfa, M, K, groups, 5)


def test_sfa_sequential():
    """SFA with sequential bytes to verify per-group offset arithmetic."""
    print("\n" + "=" * 60)
    print("Test 5: SFA sequential pattern")
    print("=" * 60)
    M, K, groups = 128, 128, 2

    mod = get_layout_check_module()
    sfa_size = mod["get_sfa_size"](M, K, groups)
    print(f"  SFA size for M={M}, K={K}, groups={groups}: {sfa_size}")

    # Sequential bytes so each position has a unique value
    w_sfa = torch.arange(sfa_size, dtype=torch.uint8).cuda()
    mod["check_sfa_layout"](w_sfa, M, K, groups, 10)


def test_sfa_wrong_size():
    """SFA with wrong size — expect size mismatch FAIL."""
    print("\n" + "=" * 60)
    print("Test 6: SFA wrong size (expect FAIL)")
    print("=" * 60)
    M, K, groups = 128, 128, 4

    mod = get_layout_check_module()
    sfa_size = mod["get_sfa_size"](M, K, groups)

    # Pass a tensor that is too small — should fail the size check
    w_sfa = torch.randint(0, 255, (sfa_size // 2,), dtype=torch.uint8).cuda()
    mod["check_sfa_layout"](w_sfa, M, K, groups, 5)


if __name__ == "__main__":
    # test_contiguous()
    # test_sequential()
    # test_non_contiguous()
    test_sfa_layout()
    test_sfa_sequential()
    test_sfa_wrong_size()
    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
