#include <cstdio>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>  // for std::stoll
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while (0)

#define CHECK_CUBLAS(x) do { \
  cublasStatus_t err = (x); \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error %s:%d\n", __FILE__, __LINE__); \
    exit(1); \
  } \
} while (0)

int main(int argc, char **argv) {
  char const* memg_s = std::getenv("MEMG");
  char const* slms_s = std::getenv("SLMS");
  int mem = atoi(memg_s);
  long long slms = std::stoll(slms_s);
  printf("mem=%d, slms:%lld\n", mem, slms);
  // -------------------------------
  // 1. 占用 ~79GB 显存（纯占位）
  // -------------------------------
  size_t target_bytes = (size_t)(mem * 1000)* 1024 * 1024;
  void* big_buffer = nullptr;
  CHECK_CUDA(cudaMalloc(&big_buffer, target_bytes));
  CHECK_CUDA(cudaMemset(big_buffer, 0, target_bytes));

  printf("Allocated %.2f GiB dummy buffer\n",
         target_bytes / 1024.0 / 1024 / 1024);

  // -------------------------------
  // 2. GEMM 参数（真正跑算力）
  // -------------------------------
  // 16384^3 ≈ 8.8 TFLOPs / GEMM（FP16 TensorCore）
  const int M = 16384;
  const int N = 16384;
  const int K = 16384;

  size_t bytesA = (size_t)M * K * sizeof(__half);
  size_t bytesB = (size_t)K * N * sizeof(__half);
  size_t bytesC = (size_t)M * N * sizeof(__half);

  __half *A, *B, *C;
  CHECK_CUDA(cudaMalloc(&A, bytesA));
  CHECK_CUDA(cudaMalloc(&B, bytesB));
  CHECK_CUDA(cudaMalloc(&C, bytesC));

  CHECK_CUDA(cudaMemset(A, 1, bytesA));
  CHECK_CUDA(cudaMemset(B, 1, bytesB));
  CHECK_CUDA(cudaMemset(C, 0, bytesC));

  // -------------------------------
  // 3. cuBLAS setup
  // -------------------------------
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  float alpha = 1.0f;
  float beta  = 0.0f;

  printf("Entering infinite GEMM loop...\n");

  // -------------------------------
  // 4. 死循环 GEMM
  // -------------------------------
  while (true) {
    CHECK_CUBLAS(
      cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
      )
    );

    // 保证 kernel 真执行完，避免 host loop 空转
    CHECK_CUDA(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(slms));
  }

  return 0;
}

