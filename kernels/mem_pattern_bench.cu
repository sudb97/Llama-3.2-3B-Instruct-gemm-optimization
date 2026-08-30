// mem_pattern_bench.cu
//
// Standalone microbenchmark isolating ONE variable: DRAM access pattern.
// Both kernels below touch the exact same 50,331,648 bytes (matching the
// up_proj/down_proj weight matrix size, 8192 x 3072 fp16) exactly once each
// -- same total unique-byte request, verified via ncu's L2 sector accounting
// to be at the theoretical minimum for the tiled GEMM kernel already.
//
// strided_read_kernel    : 16B chunk, stride = full row (6144B)
//                          -- mirrors the tiled GEMM's weight-tile access:
//                             fixed column offset, walk down rows.
// contiguous_read_kernel : 16B chunks, sequential across a full row (6144B)
//                          -- mirrors a GEMV weight-row read.
//
// Question this answers: does the ~1.17x DRAM amplification measured on the
// real kernel (dram__bytes_read.sum / lts__t_sectors_srcunit_tex_op_read.sum)
// come from the strided access pattern (recoverable by a GEMV rewrite), or
// is it invariant to access pattern (a fixed memory-system cost)?
//
// Build:
//   nvcc -O3 -arch=sm_75 mem_pattern_bench.cu -o mem_pattern_bench   (this box: T4)
//   nvcc -O3 -arch=sm_89 mem_pattern_bench.cu -o mem_pattern_bench   (project GPU: L4)
//
// Run:
//   ./mem_pattern_bench strided
//   ./mem_pattern_bench contiguous
//   ./mem_pattern_bench both        (default)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err__));                                    \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// Matrix shape verified directly from the ONNX initializer (protobuf-parsed,
// not assumed): weights are stored [K, N] row-major, NOT PyTorch's native
// [N, K] nn.Linear.weight orientation. The two MLP GEMMs have DIFFERENT row
// widths as a result -- set via -DMATMUL_SHAPE=UP_PROJ or DOWN_PROJ at build:
//   up_proj / gate_proj : [K=3072, N=8192] -> row = 8192 halves = 16384 B
//   down_proj            : [K=8192, N=3072] -> row = 3072 halves =  6144 B
#if !defined(MATMUL_SHAPE_UP_PROJ) && !defined(MATMUL_SHAPE_DOWN_PROJ)
#define MATMUL_SHAPE_DOWN_PROJ 1  // default: matches original benchmark
#endif

#if defined(MATMUL_SHAPE_UP_PROJ)
static constexpr int NROWS      = 3072;   // K
static constexpr int ROW_HALVES = 8192;   // N
#else
static constexpr int NROWS      = 8192;   // K
static constexpr int ROW_HALVES = 3072;   // N
#endif
static constexpr int    ROW_BYTES  = ROW_HALVES * 2;
static constexpr int    F4_PER_ROW = ROW_BYTES / 16;      // 16B = float4
static constexpr size_t TOTAL_BYTES =
    (size_t)NROWS * (size_t)ROW_BYTES;                    // 50,331,648 either way

static constexpr int THREADS = 256;
static constexpr int WARMUP  = 5;
static constexpr int ITERS   = 50;

// ---------------------------------------------------------------------------
// FRAGMENTED: reconstructed from measured evidence, not assumed. The real
// kernel's B-tile is 64(K) x 32(N); with weights confirmed [K,N] row-major
// via direct ONNX protobuf inspection, a 32-N-wide tile slice is only 64B
// contiguous per K-row. ncu's per-instruction sector count (~16 sectors =
// 512B, ZERO excess) is consistent with 8 independently-coalesced 64B
// sub-chunks per warp (4 threads x 16B each), each sub-chunk's K-row
// ROW_BYTES apart from the next -- NOT one 512B contiguous run (that
// would also show ~16 sectors with zero excess, so the aggregate sector
// count alone can't distinguish the two; this reconstruction is chosen
// because it's what the tile's actual 64B-per-row contiguity limit allows).
//
// group   : 4 threads, each 16B -> one coalesced 64B request (one K-row)
// warp    : 8 groups -> touches 8 DIFFERENT K-rows, ROW_BYTES apart each
// block   : owns one 64B-wide column slice, GROUPS_PER_BLOCK groups sweep
//           all NROWS rows -- grid size and iteration count matched to the
//           real kernel: 256 blocks, 48 iterations/group when NROWS=3072.
// ---------------------------------------------------------------------------
static constexpr int GROUP_SIZE        = 4;                    // threads/group, 4*16B = 64B
static constexpr int GROUPS_PER_BLOCK   = THREADS / GROUP_SIZE;  // 64
static constexpr int SLICE_F4           = GROUP_SIZE;            // 64B-wide column slice
static constexpr int NUM_SLICES         = F4_PER_ROW / SLICE_F4;

__global__ void fragmented_read_kernel(const float4* __restrict__ buf,
                                        float* __restrict__ out) {
  const int slice = blockIdx.x;
  const int tid    = threadIdx.x;
  const int group  = tid / GROUP_SIZE;
  const int lane   = tid % GROUP_SIZE;
  const int col    = slice * SLICE_F4 + lane;   // adjacent within the group -> coalesced 64B

  float acc = 0.f;
  for (int row = group; row < NROWS; row += GROUPS_PER_BLOCK) {
    float4 v = buf[(size_t)row * F4_PER_ROW + col];   // next iter for this group: +ROW_BYTES
    acc += v.x + v.y + v.z + v.w;
  }
  out[(size_t)slice * THREADS + tid] = acc;
}

// ---------------------------------------------------------------------------
// STRIDED: mirrors the real up_proj/down_proj kernel's weight-tile access --
// WITHIN one instruction, all 32 threads of a warp read ADJACENT float4
// lanes (fully coalesced, one contiguous 512B request, matching the sector
// accounting's 0%-excess finding on the real kernel). BETWEEN iterations,
// the warp advances to the next row, i.e. jumps by ROW_BYTES (6144B).
// This isolates the stride variable without also introducing an artificial
// within-warp coalescing penalty the real kernel does not have.
//
// grid.x : column-slice index (each slice is one warp wide = 32 float4 = 512B)
// grid.y : row-chunk index (splits the 8192-row sweep for occupancy)
// ---------------------------------------------------------------------------
static constexpr int STRIDE_ROWS_PER_CHUNK = 128;  // 8192 / 128 = 64 chunks

__global__ void strided_read_kernel(const float4* __restrict__ buf,
                                     float* __restrict__ out) {
  const int col       = blockIdx.x * 32 + threadIdx.x;   // adjacent within warp
  const int row_start  = blockIdx.y * STRIDE_ROWS_PER_CHUNK;
  const int row_end     = row_start + STRIDE_ROWS_PER_CHUNK;

  float acc = 0.f;
  for (int row = row_start; row < row_end; ++row) {
    float4 v = buf[(size_t)row * F4_PER_ROW + col];   // coalesced across the warp
    acc += v.x + v.y + v.z + v.w;                       // next iter: +ROW_BYTES (6144B)
  }
  out[(size_t)blockIdx.y * (F4_PER_ROW) + col] = acc;
}

// ---------------------------------------------------------------------------
// CONTIGUOUS: one block owns a fixed row, reads it end-to-end sequentially.
// Consecutive reads issued by consecutive threads are adjacent in memory --
// same shape as a GEMV row traversal (y[n] = dot(W[n,:], x)).
// ---------------------------------------------------------------------------
__global__ void contiguous_read_kernel(const float4* __restrict__ buf,
                                        float* __restrict__ out) {
  const int row      = blockIdx.x;
  const int tid       = threadIdx.x;
  const int nthreads  = blockDim.x;

  float acc = 0.f;
  for (int col = tid; col < F4_PER_ROW; col += nthreads) {
    float4 v = buf[(size_t)row * F4_PER_ROW + col];
    acc += v.x + v.y + v.z + v.w;
  }
  out[(size_t)row * nthreads + tid] = acc;
}

static double run_and_time(void (*launch)(const float4*, float*, dim3, int),
                            const float4* d_buf, float* d_out, dim3 grid,
                            int threads, const char* label) {
  for (int i = 0; i < WARMUP; ++i) launch(d_buf, d_out, grid, threads);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < ITERS; ++i) launch(d_buf, d_out, grid, threads);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  double avg_us = (ms * 1000.0) / ITERS;
  double gbps   = (TOTAL_BYTES / 1e9) / (avg_us / 1e6);
  printf("[%-10s] %9.3f us/iter   %8.2f GB/s (unique-bytes-requested basis, "
         "%.2f MB)\n",
         label, avg_us, gbps, TOTAL_BYTES / 1e6);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return avg_us;
}

static void launch_strided(const float4* buf, float* out, dim3 grid, int threads) {
  strided_read_kernel<<<grid, threads>>>(buf, out);
}
static void launch_contiguous(const float4* buf, float* out, dim3 grid, int threads) {
  contiguous_read_kernel<<<grid, threads>>>(buf, out);
}
static void launch_fragmented(const float4* buf, float* out, dim3 grid, int threads) {
  fragmented_read_kernel<<<grid, threads>>>(buf, out);
}

int main(int argc, char** argv) {
  std::string which = (argc > 1) ? argv[1] : "both";

  printf("Buffer size: %.2f MB (%d rows x %d bytes/row)\n",
         TOTAL_BYTES / 1e6, NROWS, ROW_BYTES);

  float4* d_buf = nullptr;
  CHECK_CUDA(cudaMalloc(&d_buf, TOTAL_BYTES));
  CHECK_CUDA(cudaMemset(d_buf, 0x3f, TOTAL_BYTES));  // arbitrary non-zero fill

  size_t out_elems = (size_t)std::max(F4_PER_ROW, NROWS) * THREADS;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_out, out_elems * sizeof(float)));

  if (which == "strided" || which == "both") {
    dim3 grid(F4_PER_ROW / 32, NROWS / STRIDE_ROWS_PER_CHUNK);  // 12 x 64 = 768 blocks, warp-wide
    run_and_time(launch_strided, d_buf, d_out, grid, 32, "strided");
  }
  if (which == "contiguous" || which == "both") {
    dim3 grid(NROWS);  // 8192 blocks, one per row
    run_and_time(launch_contiguous, d_buf, d_out, grid, THREADS, "contiguous");
  }
  if (which == "fragmented" || which == "both") {
    dim3 grid(NUM_SLICES);  // 256 blocks when NROWS=3072 (matches real up_proj grid)
    run_and_time(launch_fragmented, d_buf, d_out, grid, THREADS, "fragmented");
  }

  CHECK_CUDA(cudaFree(d_buf));
  CHECK_CUDA(cudaFree(d_out));
  return 0;
}
