cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(blackwell_moe_gemm_blockscaled_rcgrouped LANGUAGES CXX CUDA)

# Require CUDA 12.8+
find_package(CUDAToolkit 12.8 REQUIRED)

# SM100 (Blackwell) target architecture
set(CMAKE_CUDA_ARCHITECTURES "100a" CACHE STRING "CUDA architectures")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Include paths: our self-contained headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(92_blackwell_moe_gemm_blockscaled_rcgrouped
    92_blackwell_moe_gemm_blockscaled_rcgrouped.cu
)

target_compile_options(92_blackwell_moe_gemm_blockscaled_rcgrouped PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcudafe --diag_suppress=20208
    >
)

target_link_libraries(92_blackwell_moe_gemm_blockscaled_rcgrouped
    CUDA::cudart
)
