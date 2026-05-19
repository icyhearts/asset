cmake_minimum_required(VERSION 3.18)
project(tma_demo LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(tma_demo tma_demo.cu)
target_link_libraries(tma_demo PRIVATE CUDA::cuda_driver)

add_executable(fp16_matmul fp16_matmul.cu)
target_link_libraries(fp16_matmul PRIVATE CUDA::cuda_driver)
