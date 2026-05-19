/**
 * TMA (Tensor Memory Accelerator) Demo for NVIDIA Hopper GPUs (SM90+)
 *
 * Demonstrates using TMA to bulk-copy a 2D tile from global memory to shared
 * memory, double each element in shared memory, then bulk-copy the result back
 * to global memory.
 *
 * Key APIs:
 *   Host:   cuTensorMapEncodeTiled  – create a TMA descriptor
 *   Device: cp.async.bulk.tensor    – TMA load  (global → shared)
 *           cp.async.bulk.tensor    – TMA store (shared → global)
 *           barrier                 – synchronize the async copy
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ── helpers ──────────────────────────────────────────────────────────────────

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

// ── tile dimensions ─────────────────────────────────────────────────────────
// We copy a TILE_ROWS × TILE_COLS tile per TMA operation.
// The full matrix is ROWS × COLS.

static constexpr int TILE_ROWS = 32;
static constexpr int TILE_COLS = 64;
static constexpr int ROWS = 128;
static constexpr int COLS = 128;

// ── TMA load / store wrappers (inline PTX) ──────────────────────────────────

// TMA load: global → shared, 2D tiled
__device__ void tma_load_2d(void *smem_ptr, const void *tensor_map,
                            int coord_x, int coord_y,
                            uint64_t *barrier_ptr) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint64_t bar_addr = static_cast<uint64_t>(__cvta_generic_to_shared(barrier_ptr));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(smem_addr), "l"(tensor_map), "r"(coord_x), "r"(coord_y),
        "r"((uint32_t)bar_addr)
      : "memory");
}

// TMA store: shared → global, 2D tiled
__device__ void tma_store_2d(const void *tensor_map, void *smem_ptr,
                             int coord_x, int coord_y) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
      " [%0, {%1, %2}], [%3];"
      :
      : "l"(tensor_map), "r"(coord_x), "r"(coord_y), "r"(smem_addr)
      : "memory");
}

// ── kernel ──────────────────────────────────────────────────────────────────

__global__ void tma_demo_kernel(const __grid_constant__ CUtensorMap tensor_map_load,
                                const __grid_constant__ CUtensorMap tensor_map_store) {
  // Each block processes one tile.
  // blockIdx.x → tile column index, blockIdx.y → tile row index.

  // ── shared memory: tile + mbarrier ──
  __shared__ alignas(128) float smem_tile[TILE_ROWS][TILE_COLS];
  __shared__ alignas(8) uint64_t mbarrier;

  // ── Step 1: Initialize mbarrier (thread 0 only) ──
  if (threadIdx.x == 0) {
    // Initialize barrier; expect `expected_bytes` bytes to arrive.
    uint32_t expected_bytes = TILE_ROWS * TILE_COLS * sizeof(float);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :
                 : "r"((uint32_t)__cvta_generic_to_shared(&mbarrier)),
                   "r"((uint32_t)1) // arrival count = 1 (one TMA op)
                 : "memory");
    // Arrive with expected tx bytes
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(&mbarrier)),
          "r"(expected_bytes)
        : "memory");
  }
  __syncthreads();

  // ── Step 2: TMA load (thread 0 issues the copy) ──
  int tile_col = blockIdx.x * TILE_COLS; // coordinate in elements
  int tile_row = blockIdx.y * TILE_ROWS;

  if (threadIdx.x == 0) {
    tma_load_2d(&smem_tile[0][0], &tensor_map_load,
                tile_col, tile_row, &mbarrier);
  }

  // ── Step 3: Wait for TMA load to complete ──
  // All threads wait on the mbarrier.
  uint32_t phase = 0;
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "WAIT_LOOP:\n"
      "  mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
      "  @!P bra WAIT_LOOP;\n"
      "}\n"
      :
      : "r"((uint32_t)__cvta_generic_to_shared(&mbarrier)), "r"(phase)
      : "memory");

  // ── Step 4: Compute – double each element ──
  for (int r = threadIdx.x; r < TILE_ROWS; r += blockDim.x) {
    for (int c = 0; c < TILE_COLS; c++) {
      smem_tile[r][c] *= 2.0f;
    }
  }
  __syncthreads();

  // ── Step 5: TMA store (thread 0 issues the copy) ──
  // Fence to make shared memory writes visible to TMA engine
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (threadIdx.x == 0) {
    tma_store_2d(&tensor_map_store, &smem_tile[0][0],
                 tile_col, tile_row);
    // Commit the async store
    asm volatile("cp.async.bulk.commit_group;");
    // Wait for the store to complete
    asm volatile("cp.async.bulk.wait_group 0;");
  }
  __syncthreads();
}

// ── host: create TMA descriptor ─────────────────────────────────────────────

CUtensorMap create_tma_descriptor(float *d_ptr, int rows, int cols,
                                  int tile_rows, int tile_cols) {
  CUtensorMap tensor_map{};

  // 2D tensor: dim[0] = cols (inner/contiguous), dim[1] = rows (outer)
  cuuint64_t global_dim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  // Stride of dim[1] in bytes (stride of dim[0] is implicit = element size)
  cuuint64_t global_strides[1] = {(cuuint64_t)(cols * sizeof(float))};
  // Box (tile) dimensions
  cuuint32_t box_dim[2] = {(cuuint32_t)tile_cols, (cuuint32_t)tile_rows};
  cuuint32_t elem_strides[2] = {1, 1};

  CU_CHECK(cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      2,                                    // tensorRank
      d_ptr,                                // globalAddress
      global_dim,
      global_strides,
      box_dim,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  return tensor_map;
}

// ── main ────────────────────────────────────────────────────────────────────

int main() {
  // Initialize CUDA driver API
  CU_CHECK(cuInit(0));

  // Check SM version
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 9) {
    fprintf(stderr, "TMA requires SM90+ (Hopper). Current: SM%d%d\n",
            prop.major, prop.minor);
    return 1;
  }
  printf("Device: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);

  // Allocate host data
  int N = ROWS * COLS;
  float *h_in = (float *)malloc(N * sizeof(float));
  float *h_out = (float *)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) h_in[i] = (float)i;

  // Allocate device data
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

  // Create TMA descriptors (host-side)
  CUtensorMap tma_load = create_tma_descriptor(d_in, ROWS, COLS, TILE_ROWS, TILE_COLS);
  CUtensorMap tma_store = create_tma_descriptor(d_out, ROWS, COLS, TILE_ROWS, TILE_COLS);

  // Launch kernel
  dim3 grid(COLS / TILE_COLS, ROWS / TILE_ROWS); // 2 × 4 = 8 blocks
  dim3 block(128);                                // 128 threads per block
  printf("Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);
  printf("Matrix: %d × %d, Tile: %d × %d\n", ROWS, COLS, TILE_ROWS, TILE_COLS);

  tma_demo_kernel<<<grid, block>>>(tma_load, tma_store);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify
  CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  int errors = 0;
  for (int i = 0; i < N; i++) {
    float expected = h_in[i] * 2.0f;
    if (h_out[i] != expected) {
      if (errors < 10)
        printf("MISMATCH [%d]: got %.1f, expected %.1f\n", i, h_out[i], expected);
      errors++;
    }
  }
  if (errors == 0)
    printf("PASSED! All %d elements correct (each doubled via TMA).\n", N);
  else
    printf("FAILED: %d / %d mismatches.\n", errors, N);

  // Cleanup
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  free(h_in);
  free(h_out);
  return errors ? 1 : 0;
}
