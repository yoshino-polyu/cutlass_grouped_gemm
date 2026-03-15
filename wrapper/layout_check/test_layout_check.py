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


if __name__ == "__main__":
    test_contiguous()
    test_sequential()
    test_non_contiguous()
    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
