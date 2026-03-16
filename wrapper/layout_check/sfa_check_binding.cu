#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

using TensorView = tvm::ffi::TensorView;

int64_t get_sfa_size_run(int64_t M, int64_t K, int64_t groups);
void check_sfa_layout_run(TensorView w_sfa, int64_t M, int64_t K,
                           int64_t groups, int64_t print_coords);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_sfa_size, get_sfa_size_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(check_sfa_layout, check_sfa_layout_run);
