cmake_minimum_required(VERSION 3.5.1)
project(acocuda)

find_package(CUDA REQUIRED)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
			-w 
-gencode arch=compute_50,code=sm_50
#    -O3 -gencode arch=compute_22,code=sm_22
		  
)

include_directories(

)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(BUILD_SHARED_LIBS ON)

cuda_add_executable( acocuda acocuda.cu )
