#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

using TensorView = tvm::ffi::TensorView;

void mxfp8_rc_grouped_gemm_run(
    TensorView x_fp8,
    TensorView x_scale,
    TensorView w_fp8,
    TensorView w_scale,
    TensorView cnt,
    TensorView output,
    TensorView workspace,
    int64_t mma_sm);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp8_rc_grouped_gemm, mxfp8_rc_grouped_gemm_run);
