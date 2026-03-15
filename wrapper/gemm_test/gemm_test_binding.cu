#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

using TensorView = tvm::ffi::TensorView;

void gemm_test_run(TensorView w_fp8, int64_t n, int64_t groups,
                   int64_t alpha_val, int64_t beta_val);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_test, gemm_test_run);
