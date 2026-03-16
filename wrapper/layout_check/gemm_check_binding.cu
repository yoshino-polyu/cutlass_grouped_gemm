#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

using TensorView = tvm::ffi::TensorView;

int64_t get_b_size_run(int64_t N, int64_t K);
int64_t get_sfb_size_run(int64_t M, int64_t N, int64_t K);
void gemm_check_run(TensorView w_fp8, TensorView w_sfa,
                     TensorView w_act, TensorView w_sfb,
                     TensorView offsets_B, TensorView offsets_SFB,
                     int64_t alpha_val, int64_t beta_val);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_b_size, get_b_size_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_sfb_size, get_sfb_size_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_check, gemm_check_run);
