/**
 * FP16 Matrix Multiplication with TMA + Software Pipeline on H100 (SM90)
 *
 * Demonstrates:
 *   - TMA (Tensor Memory Accelerator) for async global→shared memory loads
 *   - Software pipelining with mbarrier (NUM_STAGES deep)
 *   - WMMA tensor core computation (16×16×16, half→float accumulate)
 *
 * Layout: A (M×K), B (K×N), C (M×N), all row-major FP16.
 * Output C is FP16; accumulation is FP32.
 *
 * Tile: BM=128, BN=128, BK=64.  Thread block: 128 threads (4 warps).
 * Each warp handles 128 rows × 32 cols of the output tile.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

using namespace nvcuda;

// ── Error-check macros ──────────────────────────────────────────────────────

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult err = (call);                                                     \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *str;                                                         \
      cuGetErrorString(err, &str);                                             \
      fprintf(stderr, "CUDA Driver error %d: %s @ %s:%d\n", err, str,         \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Runtime error: %s @ %s:%d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ── Tile configuration ──────────────────────────────────────────────────────

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int NUM_STAGES = 3;

// WMMA tile dimensions
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// ── TMA load wrapper (inline PTX) ──────────────────────────────────────────

__device__ __forceinline__ void tma_load_2d(void *smem_ptr,
                                             const void *tensor_map,
                                             int coord_x, int coord_y,
                                             uint64_t *barrier_ptr) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(smem_addr), "l"(tensor_map), "r"(coord_x), "r"(coord_y),
        "r"(bar_addr)
      : "memory");
}

// ── Barrier helpers ─────────────────────────────────────────────────────────

__device__ __forceinline__ void barrier_init(uint64_t *barrier_ptr,
                                              uint32_t arrive_count) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"
               :
               : "r"(bar_addr), "r"(arrive_count)
               : "memory");
}

__device__ __forceinline__ void barrier_expect_tx(uint64_t *barrier_ptr,
                                                   uint32_t tx_bytes) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
               :
               : "r"(bar_addr), "r"(tx_bytes)
               : "memory");
}

__device__ __forceinline__ void barrier_wait(uint64_t *barrier_ptr,
                                              uint32_t phase) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "WAIT_%=:\n"
      "  mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
      "  @!P bra WAIT_%=;\n"
      "}\n"
      :
      : "r"(bar_addr), "r"(phase)
      : "memory");
}

// ── Kernel ──────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(128)
fp16_matmul_kernel(const __grid_constant__ CUtensorMap tma_A,
                   const __grid_constant__ CUtensorMap tma_B,
                   half *__restrict__ C,
                   int M, int N, int K,
                   int N_stride) {
  // Shared memory layout
  extern __shared__ char smem_raw[];
  // Barriers: NUM_STAGES × 8 bytes, aligned to 8
  uint64_t *barriers = reinterpret_cast<uint64_t *>(smem_raw);
  // A tiles: NUM_STAGES × BM × BK × sizeof(half), aligned to 128
  constexpr int BARRIER_BYTES = NUM_STAGES * sizeof(uint64_t);
  // Pad to 128-byte alignment
  constexpr int A_OFFSET = ((BARRIER_BYTES + 127) / 128) * 128;
  constexpr int A_STAGE_BYTES = BM * BK * sizeof(half);  // 16384 bytes
  constexpr int B_OFFSET = A_OFFSET + NUM_STAGES * A_STAGE_BYTES;
  // Pad B_OFFSET to 128-byte alignment
  constexpr int B_OFFSET_ALIGNED = ((B_OFFSET + 127) / 128) * 128;

  half *A_smem_base = reinterpret_cast<half *>(smem_raw + A_OFFSET);
  half *B_smem_base = reinterpret_cast<half *>(smem_raw + B_OFFSET_ALIGNED);

  auto A_smem = [&](int stage) -> half * {
    return A_smem_base + stage * BM * BK;
  };
  auto B_smem = [&](int stage) -> half * {
    return B_smem_base + stage * BK * BN;
  };

  const int bm = blockIdx.y;  // block row
  const int bn = blockIdx.x;  // block col
  const int warp_id = threadIdx.x / 32;
  const int num_k_tiles = (K + BK - 1) / BK;

  // Each warp handles a 128×32 slice of the output (warp_id selects 32 columns)
  const int warp_n_offset = warp_id * 32;  // 0, 32, 64, 96

  // Number of WMMA tiles along m and n dimensions for each warp
  constexpr int WARP_M_TILES = BM / WMMA_M;     // 8
  constexpr int WARP_N_TILES = 32 / WMMA_N;     // 2
  constexpr int K_TILES = BK / WMMA_K;          // 4

  // Declare accumulators
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      acc[WARP_M_TILES][WARP_N_TILES];
  #pragma unroll
  for (int mi = 0; mi < WARP_M_TILES; mi++) {
    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++) {
      wmma::fill_fragment(acc[mi][ni], 0.0f);
    }
  }

  // Phase tracking for barriers
  uint32_t phase[NUM_STAGES] = {0};

  // Expected bytes per TMA operation
  constexpr uint32_t A_TX_BYTES = BM * BK * sizeof(half);  // 16384
  constexpr uint32_t B_TX_BYTES = BK * BN * sizeof(half);  // 16384
  constexpr uint32_t TOTAL_TX_BYTES = A_TX_BYTES + B_TX_BYTES;

  // ── Prologue: fill pipeline stages ──
  #pragma unroll
  for (int s = 0; s < NUM_STAGES; s++) {
    if (threadIdx.x == 0) {
      barrier_init(&barriers[s], 1);
    }
  }
  __syncthreads();

  auto issue_tma_load = [&](int k_tile, int stage) {
    if (threadIdx.x == 0) {
      barrier_expect_tx(&barriers[stage], TOTAL_TX_BYTES);
      // A tile: rows [bm*BM .. bm*BM+BM), cols [k_tile*BK .. k_tile*BK+BK)
      // TMA coord: (col_offset, row_offset) = (k_tile*BK, bm*BM)
      tma_load_2d(A_smem(stage), &tma_A,
                  k_tile * BK, bm * BM, &barriers[stage]);
      // B tile: rows [k_tile*BK .. k_tile*BK+BK), cols [bn*BN .. bn*BN+BN)
      // TMA coord: (col_offset, row_offset) = (bn*BN, k_tile*BK)
      tma_load_2d(B_smem(stage), &tma_B,
                  bn * BN, k_tile * BK, &barriers[stage]);
    }
  };

  // Issue prologue loads
  for (int s = 0; s < NUM_STAGES && s < num_k_tiles; s++) {
    issue_tma_load(s, s);
  }

  // ── Main loop ──
  for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    int stage = k_tile % NUM_STAGES;

    // Wait for this stage's data
    barrier_wait(&barriers[stage], phase[stage]);

    // Compute: iterate over WMMA k-steps within BK
    half *a_base = A_smem(stage);
    half *b_base = B_smem(stage);

    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
      #pragma unroll
      for (int mi = 0; mi < WARP_M_TILES; mi++) {
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ni++) {
          wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                         wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                         wmma::row_major> b_frag;

          // A: row = mi*16, col = ki*16 within BM×BK tile (row-major, stride=BK)
          wmma::load_matrix_sync(a_frag,
              a_base + (mi * WMMA_M) * BK + ki * WMMA_K, BK);
          // B: row = ki*16, col = warp_n_offset + ni*16 within BK×BN tile (row-major, stride=BN)
          wmma::load_matrix_sync(b_frag,
              b_base + (ki * WMMA_K) * BN + warp_n_offset + ni * WMMA_N, BN);

          wmma::mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
        }
      }
    }

    // Flip phase for this stage
    phase[stage] ^= 1;

    // Ensure all threads finished reading from this stage's shared memory
    // before thread 0 issues a new TMA load that overwrites it.
    __syncthreads();

    // Issue next TMA load into this stage (data is consumed, safe to overwrite)
    int next_k = k_tile + NUM_STAGES;
    if (next_k < num_k_tiles) {
      issue_tma_load(next_k, stage);
    }
  }

  // ── Epilogue: store results to global memory ──
  int block_row = bm * BM;
  int block_col = bn * BN;

  #pragma unroll
  for (int mi = 0; mi < WARP_M_TILES; mi++) {
    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++) {
      // Convert FP32 accumulator to FP16
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
      for (int i = 0; i < acc[mi][ni].num_elements; i++) {
        c_frag.x[i] = __float2half(acc[mi][ni].x[i]);
      }

      int out_row = block_row + mi * WMMA_M;
      int out_col = block_col + warp_n_offset + ni * WMMA_N;

      // Bounds check: only store if the entire 16×16 tile is within bounds
      if (out_row + WMMA_M <= M && out_col + WMMA_N <= N) {
        wmma::store_matrix_sync(C + out_row * N_stride + out_col, c_frag,
                                N_stride, wmma::mem_row_major);
      } else if (out_row < M && out_col < N) {
        // Partial tile: store element by element via per-warp shared memory staging
        // 4 warps × 16×16 half = 2048 bytes
        __shared__ half staging[4][WMMA_M][WMMA_N];
        wmma::store_matrix_sync(&staging[warp_id][0][0], c_frag, WMMA_N,
                                wmma::mem_row_major);
        __syncwarp();
        for (int idx = threadIdx.x % 32; idx < WMMA_M * WMMA_N; idx += 32) {
          int r = idx / WMMA_N;
          int c = idx % WMMA_N;
          if (out_row + r < M && out_col + c < N) {
            C[(out_row + r) * N_stride + (out_col + c)] = staging[warp_id][r][c];
          }
        }
      }
    }
  }
}

// ── Host: Create TMA descriptors ────────────────────────────────────────────

// Create TMA descriptor for a row-major matrix.
// rows/cols are the padded dimensions (must be >= box dimensions).
// row_stride_bytes is the byte stride between rows (= cols * sizeof(half) for dense).
CUtensorMap create_tma_desc(half *d_ptr, int rows, int cols,
                            int box_rows, int box_cols,
                            int row_stride_bytes) {
  CUtensorMap tensor_map{};
  // 2D: dim0=cols (contiguous), dim1=rows
  cuuint64_t global_dim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t global_strides[1] = {(cuuint64_t)row_stride_bytes};
  cuuint32_t box_dim[2] = {(cuuint32_t)box_cols, (cuuint32_t)box_rows};
  cuuint32_t elem_strides[2] = {1, 1};

  CU_CHECK(cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      2,          // tensorRank
      d_ptr,      // globalAddress
      global_dim,
      global_strides,
      box_dim,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

  return tensor_map;
}

static inline int pad_up(int x, int align) { return ((x + align - 1) / align) * align; }

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  CU_CHECK(cuInit(0));

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 9) {
    fprintf(stderr, "This kernel requires SM90+ (Hopper). Current: SM%d%d\n",
            prop.major, prop.minor);
    return 1;
  }
  printf("Device: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);

  // Parse dimensions. Optional -b flag for benchmark-only mode.
  bool bench_only = false;
  int arg_start = 1;
  if (argc > 1 && strcmp(argv[1], "-b") == 0) {
    bench_only = true;
    arg_start = 2;
  }
  int M = (argc > arg_start) ? atoi(argv[arg_start]) : 512;
  int N = (argc > arg_start + 1) ? atoi(argv[arg_start + 1]) : 1024;
  int K = (argc > arg_start + 2) ? atoi(argv[arg_start + 2]) : 2048;
  // Auto-skip verification for very large matrices (CPU ref would be too slow)
  double total_ops = 2.0 * M * N * K;
  if (total_ops > 4e9) bench_only = true;
  printf("Matrix: M=%d, N=%d, K=%d%s\n", M, N, K,
         bench_only ? " (benchmark only)" : "");

  // Pad dimensions up to tile multiples for TMA (boxDim <= globalDim required)
  int Mp = pad_up(M, BM);
  int Np = pad_up(N, BN);
  int Kp = pad_up(K, BK);
  printf("Padded: Mp=%d, Np=%d, Kp=%d\n", Mp, Np, Kp);

  // Allocate host memory (original sizes)
  size_t size_A = (size_t)M * K;
  size_t size_B = (size_t)K * N;
  size_t size_C = (size_t)M * N;

  half *h_A = (half *)malloc(size_A * sizeof(half));
  half *h_B = (half *)malloc(size_B * sizeof(half));
  half *h_C = bench_only ? NULL : (half *)malloc(size_C * sizeof(half));
  float *h_ref = bench_only ? NULL : (float *)malloc(size_C * sizeof(float));

  // Initialize with small random values to avoid overflow
  srand(42);
  for (size_t i = 0; i < size_A; i++)
    h_A[i] = __float2half((float)(rand() % 100 - 50) / 50.0f);
  for (size_t i = 0; i < size_B; i++)
    h_B[i] = __float2half((float)(rand() % 100 - 50) / 50.0f);

  // CPU reference (FP32 accumulation)
  if (!bench_only) {
    printf("Computing CPU reference...\n");
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += __half2float(h_A[m * K + k]) * __half2float(h_B[k * N + n]);
        }
        h_ref[m * N + n] = sum;
      }
    }
  }

  // Allocate padded device memory (zero-filled, then copy real data row by row)
  size_t padded_A = (size_t)Mp * Kp;
  size_t padded_B = (size_t)Kp * Np;
  size_t padded_C = (size_t)Mp * Np;

  half *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, padded_A * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_B, padded_B * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_C, padded_C * sizeof(half)));
  CUDA_CHECK(cudaMemset(d_A, 0, padded_A * sizeof(half)));
  CUDA_CHECK(cudaMemset(d_B, 0, padded_B * sizeof(half)));
  CUDA_CHECK(cudaMemset(d_C, 0, padded_C * sizeof(half)));

  // Copy A into padded allocation using cudaMemcpy2D (M rows, K→Kp width)
  CUDA_CHECK(cudaMemcpy2D(d_A, (size_t)Kp * sizeof(half),
                           h_A, (size_t)K * sizeof(half),
                           (size_t)K * sizeof(half), M,
                           cudaMemcpyHostToDevice));
  // Copy B into padded allocation using cudaMemcpy2D (K rows, N→Np width)
  CUDA_CHECK(cudaMemcpy2D(d_B, (size_t)Np * sizeof(half),
                           h_B, (size_t)N * sizeof(half),
                           (size_t)N * sizeof(half), K,
                           cudaMemcpyHostToDevice));

  // Create TMA descriptors using padded dimensions
  // A: Mp×Kp, row stride = Kp * sizeof(half)
  CUtensorMap tma_A = create_tma_desc(d_A, Mp, Kp, BM, BK, Kp * sizeof(half));
  // B: Kp×Np, row stride = Np * sizeof(half)
  CUtensorMap tma_B = create_tma_desc(d_B, Kp, Np, BK, BN, Np * sizeof(half));

  // Compute shared memory size
  constexpr int BARRIER_BYTES = NUM_STAGES * sizeof(uint64_t);
  constexpr int A_OFFSET = ((BARRIER_BYTES + 127) / 128) * 128;
  constexpr int A_STAGE_BYTES = BM * BK * sizeof(half);
  constexpr int B_OFFSET = A_OFFSET + NUM_STAGES * A_STAGE_BYTES;
  constexpr int B_OFFSET_ALIGNED = ((B_OFFSET + 127) / 128) * 128;
  constexpr int B_STAGE_BYTES = BK * BN * sizeof(half);
  constexpr int SMEM_SIZE = B_OFFSET_ALIGNED + NUM_STAGES * B_STAGE_BYTES;

  printf("Shared memory: %d bytes (%.1f KB)\n", SMEM_SIZE, SMEM_SIZE / 1024.0f);

  // Set max dynamic shared memory
  CUDA_CHECK(cudaFuncSetAttribute(
      fp16_matmul_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      SMEM_SIZE));

  // Query and print occupancy
  int max_active_blocks;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, fp16_matmul_kernel, 128, SMEM_SIZE));
  printf("Max active blocks per SM: %d\n", max_active_blocks);

  // Launch config — grid covers padded dimensions (exact tile multiples)
  dim3 grid(Np / BN, Mp / BM);
  dim3 block(128);
  printf("Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);

  // Warmup
  fp16_matmul_kernel<<<grid, block, SMEM_SIZE>>>(tma_A, tma_B, d_C, M, N, Kp, Np);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verification
  int errors = 0;
  if (!bench_only) {
    // Copy back results (C is stored with stride Np, extract M×N region)
    CUDA_CHECK(cudaMemcpy2D(h_C, (size_t)N * sizeof(half),
                             d_C, (size_t)Np * sizeof(half),
                             (size_t)N * sizeof(half), M,
                             cudaMemcpyDeviceToHost));

    // Verify using combined absolute + relative tolerance.
    // FP16 accumulation over K elements introduces rounding proportional to sqrt(K).
    float atol = 5e-3f;   // absolute tolerance
    float rtol = 1e-2f;   // relative tolerance (1%)
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
      float got = __half2float(h_C[i]);
      float ref = h_ref[i];
      float abs_err = fabsf(got - ref);
      float rel_err = (fabsf(ref) > 1e-8f) ? abs_err / fabsf(ref) : 0.0f;
      if (abs_err > max_abs_err) max_abs_err = abs_err;
      if (rel_err > max_rel_err) max_rel_err = rel_err;
      float tol = fmaxf(atol, rtol * fabsf(ref));
      if (abs_err > tol) {
        if (errors < 10) {
          int m = i / N, n = i % N;
          printf("MISMATCH [%d,%d]: got %.6f, ref %.6f, abs_err=%.6f, rel_err=%.4f\n",
                 m, n, got, ref, abs_err, rel_err);
        }
        errors++;
      }
    }

    if (errors == 0) {
      printf("PASSED! Max abs_err: %.6f, max rel_err: %.6f\n", max_abs_err, max_rel_err);
    } else {
      printf("FAILED: %d / %d mismatches (max_abs_err=%.6f, max_rel_err=%.6f)\n",
             errors, M * N, max_abs_err, max_rel_err);
    }
  }

  // Benchmark
  constexpr int NUM_ITERS = 100;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < NUM_ITERS; i++) {
    fp16_matmul_kernel<<<grid, block, SMEM_SIZE>>>(tma_A, tma_B, d_C, M, N, Kp, Np);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / NUM_ITERS;
  double flops = 2.0 * M * N * K;
  double tflops = flops / (avg_ms * 1e-3) / 1e12;

  printf("Average: %.3f ms, %.2f TFLOPS\n", avg_ms, tflops);

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_ref);

  return errors ? 1 : 0;
}
