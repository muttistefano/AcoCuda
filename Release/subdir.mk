################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../acocuda.cu 

OBJS += \
./acocuda.o 

CU_DEPS += \
./acocuda.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -O3 -gencode arch=compute_50,code=sm_50 -m64 -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

