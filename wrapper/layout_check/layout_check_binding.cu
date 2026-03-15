#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

using TensorView = tvm::ffi::TensorView;

void check_block_a_layout_run(TensorView w_fp8, int64_t print_coords);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(check_block_a_layout, check_block_a_layout_run);
