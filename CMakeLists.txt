cmake_minimum_required(VERSION 3.5.1)
set (CMAKE_CXX_STANDARD 11)
project(acocuda)

find_package(CUDA REQUIRED)
set(
CUDA_NVCC_FLAGS
${CUDA_NVCC_FLAGS};
-w -g -G -O0 -use_fast_math -prec-div=false -prec-sqrt=false
-gencode arch=compute_50,code=sm_50 -std=c++11 -rdc=true
)

set(CUDA_SEPARABLE_COMPILATION TRUE)

include_directories(.
)


set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(BUILD_SHARED_LIBS ON)

cuda_add_executable( acocuda acocuda.cu acocuda.h)

target_compile_options( acocuda PRIVATE -ftz=true -prec-div=false -prec-sqrt=false)

#####MAKE nvcc -arch=compute_50 -code=sm_50 -std=c++11 -rdc=true acocuda.cu -o acocuda -lcudadevrt -ftz=true -prec-div=false -prec-sqrt=false