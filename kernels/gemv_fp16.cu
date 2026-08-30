// gemv_fp16.cu
//
// Decode-time GEMV (M=1) for the Llama-3.2-3B MLP projections on L4 (sm_89).
// Replaces the TensorRT tactic
//   sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_...
// whose measured behaviour we bounded in PROJECT_PROGRESS.md:
//   - DRAM 89-93% of peak, compute ~15% of SM  -> memory bound
//   - L2 sector excess 0%                       -> coalescing inside its tile is optimal
//   - DRAM amplification 1.20-1.21x             -> but it overfetches at the DRAM level
// The mem_pattern_bench microbench showed the amplification tracks how much
// CONTIGUOUS run each row visit produces: 64 B chunks -> 1.41-1.45x, 512 B ->
// 1.14x, whole-row sequential -> 1.10-1.13x. So the only recoverable bytes come
// from restructuring who owns what, not from more occupancy.
//
// Weights are [K, N] row-major (verified by ONNX protobuf inspection), so a row
// W[k, :] is N contiguous halves and consecutive rows are adjacent in memory.
//
//   up_proj / gate_proj : K=3072, N=8192  -> row = 16384 B
//   down_proj            : K=8192, N=3072  -> row =  6144 B
//
// Partitioning: SPLIT-K, not split-N.
//
//   A block owns a contiguous CHUNK OF K ROWS and the FULL N width. It therefore
//   reads one unbroken slab W[k0:k1, :] -- hundreds of KB of purely sequential
//   DRAM. Splitting N instead would give each block a narrow column stripe and
//   reintroduce the row-stride jump that costs the TRT kernel its 1.20x.
//
//   Each thread owns a fixed set of output columns for the whole sweep, so its
//   partial sums live in REGISTERS -- no shared-memory accumulators, and no
//   staging of the weights at all. At M=1 every weight is consumed exactly once,
//   so a shared-memory B tile would buy no reuse while costing the barriers and
//   the 24 KiB that caps the TRT kernel at 2 blocks/SM.
//
//   Blocks then atomicAdd their partials into an fp32 accumulator. The output is
//   only N floats (32 KB at most), so that traffic stays L2-resident and costs
//   essentially no DRAM.
//
// Two row-assignment strategies are kept, because the difference between them
// is the whole point of the exercise:
//
//   CHUNK  : block b owns rows [b*rows_per_block, (b+1)*rows_per_block).
//            Each block alone is sequential, but the blocks RESIDENT AT THE SAME
//            TIME sit in widely separated regions of the matrix, so DRAM sees
//            ~grid concurrent scattered streams. Measured 74% of peak.
//   STRIDE : block b owns rows b, b+gridDim, b+2*gridDim, ...
//            Any instant has the resident blocks covering one contiguous window
//            of `grid` rows, so the ensemble walks the matrix front to back.
//            This is what the contiguous microbench did implicitly to reach 94%.
//
// For STRIDE the grid should be about the number of blocks that are actually
// co-resident; a larger grid breaks the window because the tail blocks cannot
// start until earlier ones retire.
//
// Build:
//   nvcc -O3 -arch=sm_89 -lineinfo gemv_fp16.cu -o gemv_fp16
//
// Run:
//   ./gemv_fp16 check                        # correctness vs CPU reference
//   ./gemv_fp16 bench                        # sweep variants x grid, both shapes
//   ./gemv_fp16 once up_proj <variant> <grid>   # one launch, for ncu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err__));                                    \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static constexpr int HALF_PER_VEC = 8;   // uint4 = 16 B = 8 halves

// ---------------------------------------------------------------------------
// Split-K GEMV.  y[n] = sum_k x[k] * W[k, n]
//
// grid.x  : K-chunk index. Block b sweeps rows [b*rows_per_block, ...).
// THREADS : block width.
// VEC_PER_THREAD : uint4 loads each thread issues per row. Chosen so that
//   THREADS * HALF_PER_VEC * VEC_PER_THREAD == N, i.e. the block covers the
//   FULL row width. That is what makes the slab sequential.
//
// Access pattern per row, for a fixed v: threads 0..THREADS-1 read consecutive
// uint4, i.e. one contiguous THREADS*16 B run. The VEC_PER_THREAD values of v
// tile the row end to end, and the next k is the adjacent row -- so the block's
// whole footprint is one sequential stream.
// ---------------------------------------------------------------------------
template <int THREADS, int VEC_PER_THREAD>
__device__ __forceinline__ void accumulate_row(
    const uint4* __restrict__ W, size_t row, float xk,
    float (&acc)[VEC_PER_THREAD * HALF_PER_VEC]) {
#pragma unroll
  for (int v = 0; v < VEC_PER_THREAD; ++v) {
    // ld.global.cs: the weights are streamed once and never revisited, so they
    // are marked evict-first rather than allowed to churn L2. Measured effect is
    // within noise (~1 us); kept because it states the intent correctly.
    const uint4 raw = __ldcs(&W[row + v * THREADS + threadIdx.x]);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __half22float2(h2[j]);
      acc[v * HALF_PER_VEC + 2 * j + 0] =
          fmaf(xk, f.x, acc[v * HALF_PER_VEC + 2 * j + 0]);
      acc[v * HALF_PER_VEC + 2 * j + 1] =
          fmaf(xk, f.y, acc[v * HALF_PER_VEC + 2 * j + 1]);
    }
  }
}

// Each thread owns the same columns for the whole sweep, so nothing is shared
// between threads and the reduction is a plain atomicAdd per owned column.
template <int THREADS, int VEC_PER_THREAD>
__device__ __forceinline__ void flush_acc(
    float* __restrict__ y_acc, const float (&acc)[VEC_PER_THREAD * HALF_PER_VEC]) {
#pragma unroll
  for (int v = 0; v < VEC_PER_THREAD; ++v) {
    const int col = (v * THREADS + threadIdx.x) * HALF_PER_VEC;
#pragma unroll
    for (int j = 0; j < HALF_PER_VEC; ++j) {
      atomicAdd(&y_acc[col + j], acc[v * HALF_PER_VEC + j]);
    }
  }
}

template <int THREADS, int VEC_PER_THREAD>
__global__ __launch_bounds__(THREADS) void gemv_chunk_kernel(
    const uint4* __restrict__ W,     // [K, N] row-major fp16, viewed as uint4
    const __half* __restrict__ x,    // [K]
    float* __restrict__ y_acc,       // [N] fp32, pre-zeroed
    int K, int vec_per_row, int rows_per_block) {
  constexpr int ACC = VEC_PER_THREAD * HALF_PER_VEC;

  const int k_begin = blockIdx.x * rows_per_block;
  int k_end = k_begin + rows_per_block;
  if (k_end > K) k_end = K;
  if (k_begin >= k_end) return;

  float acc[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) acc[i] = 0.f;

  for (int k = k_begin; k < k_end; ++k) {
    accumulate_row<THREADS, VEC_PER_THREAD>(W, (size_t)k * vec_per_row,
                                            __half2float(x[k]), acc);
  }
  flush_acc<THREADS, VEC_PER_THREAD>(y_acc, acc);
}

// Interleaved rows: the resident blocks together cover one contiguous window of
// the matrix at any instant, so DRAM sees a single front-to-back sweep rather
// than `grid` scattered streams.
template <int THREADS, int VEC_PER_THREAD>
__global__ __launch_bounds__(THREADS) void gemv_stride_kernel(
    const uint4* __restrict__ W, const __half* __restrict__ x,
    float* __restrict__ y_acc, int K, int vec_per_row) {
  constexpr int ACC = VEC_PER_THREAD * HALF_PER_VEC;

  float acc[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) acc[i] = 0.f;

  for (int k = blockIdx.x; k < K; k += gridDim.x) {
    accumulate_row<THREADS, VEC_PER_THREAD>(W, (size_t)k * vec_per_row,
                                            __half2float(x[k]), acc);
  }
  flush_acc<THREADS, VEC_PER_THREAD>(y_acc, acc);
}

__global__ void cast_f32_to_f16_kernel(const float* __restrict__ src,
                                       __half* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
struct Shape {
  const char* name;
  int K;
  int N;
};

static const Shape kShapes[] = {
    {"up_proj", 3072, 8192},    // also gate_proj
    {"down_proj", 8192, 3072},
};

// THREADS x 8 halves x VEC_PER_THREAD == N, so a block always spans exactly one
// full row. "narrow" keeps blocks small (more blocks, 32 accumulator registers);
// "wide" uses a whole row per instruction and only 8 accumulators, which lets
// far more warps be resident.
//   N=8192 : narrow 256 thr x 4 vec   wide 1024 thr x 1 vec
//   N=3072 : narrow 128 thr x 3 vec   wide  384 thr x 1 vec
enum Variant { kChunkNarrow = 0, kChunkWide, kStrideNarrow, kStrideWide };

static const char* variant_name(int v) {
  switch (v) {
    case kChunkNarrow: return "chunk/narrow";
    case kChunkWide:   return "chunk/wide";
    case kStrideNarrow:return "stride/narrow";
    default:           return "stride/wide";
  }
}

static int variant_threads(int variant, int N) {
  const bool wide = (variant == kChunkWide || variant == kStrideWide);
  if (N == 8192) return wide ? 1024 : 256;
  return wide ? 384 : 128;
}

// Best measured configuration per shape (ncu, cache-flushed, L4).
// Both want a SMALL grid: 14-29 blocks already saturate DRAM, and adding more
// only multiplies the number of concurrent streams the memory system has to
// interleave. Occupancy is a means here, not the goal -- 14 blocks of 1024
// threads reach 64% occupancy on a fraction of the SMs and still hit 90% DRAM.
static constexpr int kBestVariant = kChunkWide;
static int best_grid(int N) { return (N == 8192) ? 14 : 29; }

#define DISPATCH(THR, VEC)                                                     \
  do {                                                                         \
    if (variant == kChunkNarrow || variant == kChunkWide) {                    \
      gemv_chunk_kernel<THR, VEC>                                              \
          <<<grid, THR>>>(W, d_x, d_y_acc, K, vec_per_row, rows_per_block);    \
    } else {                                                                   \
      gemv_stride_kernel<THR, VEC><<<grid, THR>>>(W, d_x, d_y_acc, K,          \
                                                  vec_per_row);                \
    }                                                                          \
  } while (0)

static void launch_gemv(const void* d_W, const __half* d_x, float* d_y_acc,
                        int K, int N, int variant, int grid,
                        int rows_per_block) {
  const int vec_per_row = N / HALF_PER_VEC;
  const uint4* W = reinterpret_cast<const uint4*>(d_W);
  const bool wide = (variant == kChunkWide || variant == kStrideWide);

  if (N == 8192) {
    if (wide) DISPATCH(1024, 1);
    else      DISPATCH(256, 4);
  } else if (N == 3072) {
    if (wide) DISPATCH(384, 1);
    else      DISPATCH(128, 3);
  } else {
    fprintf(stderr, "no full-row tiling configured for N=%d\n", N);
    exit(1);
  }
}

// One GEMV = zero the accumulator, sweep, narrow back to fp16.
// For chunk variants `grid` is the split count; for stride variants it is the
// number of co-resident blocks that walk the matrix together.
static void run_gemv(const void* d_W, const __half* d_x, float* d_y_acc,
                     __half* d_y, int K, int N, int variant, int grid) {
  const int rows_per_block = (K + grid - 1) / grid;
  const int launch_grid = (variant == kChunkNarrow || variant == kChunkWide)
                              ? (K + rows_per_block - 1) / rows_per_block
                              : grid;
  CHECK_CUDA(cudaMemsetAsync(d_y_acc, 0, (size_t)N * sizeof(float)));
  launch_gemv(d_W, d_x, d_y_acc, K, N, variant, launch_grid, rows_per_block);
  cast_f32_to_f16_kernel<<<(N + 255) / 256, 256>>>(d_y_acc, d_y, N);
}

static void fill_host(std::vector<__half>& W, std::vector<__half>& x,
                      int K, int N) {
  // Small, well-conditioned values: fp16 accumulation error must not be
  // mistaken for a kernel bug when checking against the CPU reference.
  srand(1234);
  for (size_t i = 0; i < W.size(); ++i) {
    W[i] = __float2half(((rand() % 2001) - 1000) / 4000.0f);
  }
  for (int i = 0; i < K; ++i) {
    x[i] = __float2half(((rand() % 2001) - 1000) / 4000.0f);
  }
  (void)N;
}

static void cpu_reference(const std::vector<__half>& W,
                          const std::vector<__half>& x, std::vector<double>& y,
                          int K, int N) {
  y.assign(N, 0.0);
  for (int k = 0; k < K; ++k) {
    const double xk = __half2float(x[k]);
    if (xk == 0.0) continue;
    const __half* row = W.data() + (size_t)k * N;
    for (int n = 0; n < N; ++n) y[n] += xk * (double)__half2float(row[n]);
  }
}

static void check_shape(const Shape& s, int variant, int grid) {
  const size_t w_elems = (size_t)s.K * s.N;
  std::vector<__half> h_W(w_elems), h_x(s.K);
  fill_host(h_W, h_x, s.K, s.N);

  void* d_W = nullptr;
  __half* d_x = nullptr;
  __half* d_y = nullptr;
  float* d_y_acc = nullptr;
  CHECK_CUDA(cudaMalloc(&d_W, w_elems * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_x, (size_t)s.K * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y, (size_t)s.N * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y_acc, (size_t)s.N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), w_elems * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), (size_t)s.K * sizeof(__half),
                        cudaMemcpyHostToDevice));

  run_gemv(d_W, d_x, d_y_acc, d_y, s.K, s.N, variant, grid);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__half> h_y(s.N);
  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, (size_t)s.N * sizeof(__half),
                        cudaMemcpyDeviceToHost));

  std::vector<double> ref;
  cpu_reference(h_W, h_x, ref, s.K, s.N);

  double dot = 0, na = 0, nb = 0, max_abs = 0, max_rel = 0;
  for (int n = 0; n < s.N; ++n) {
    const double got = __half2float(h_y[n]);
    const double want = ref[n];
    dot += got * want;
    na += got * got;
    nb += want * want;
    const double abs_err = fabs(got - want);
    if (abs_err > max_abs) max_abs = abs_err;
    const double denom = fabs(want) > 1e-3 ? fabs(want) : 1e-3;
    if (abs_err / denom > max_rel) max_rel = abs_err / denom;
  }
  const double cos = dot / (sqrt(na) * sqrt(nb));
  printf("[check %-10s] K=%-5d N=%-5d %-14s grid=%-4d cosine=%.8f  max_abs=%.4g  max_rel=%.4g  %s\n",
         s.name, s.K, s.N, variant_name(variant), grid, cos, max_abs, max_rel,
         (cos > 0.9999 && max_rel < 0.05) ? "PASS" : "FAIL");

  CHECK_CUDA(cudaFree(d_W));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_y_acc));
}

// L4 has a 48 MB L2 and the weight matrix is 50.33 MB, so a naive repeat loop
// reports 800-1000 GB/s -- well above the ~300 GB/s DRAM peak -- because the
// matrix is almost L2-resident. In real decode the next visit to a given
// projection comes 6 GB of weights later, i.e. always cold. Rotating over
// several copies keeps every iteration cold and makes wall-clock agree with
// ncu's cache-flushed numbers.
static constexpr int WCOPIES = 4;

static void bench_shape(const Shape& s, const std::vector<int>& grids) {
  const size_t w_bytes = (size_t)s.K * s.N * sizeof(__half);

  void* d_W[WCOPIES] = {};
  __half* d_x = nullptr;
  __half* d_y = nullptr;
  float* d_y_acc = nullptr;
  for (int c = 0; c < WCOPIES; ++c) {
    CHECK_CUDA(cudaMalloc(&d_W[c], w_bytes));
    CHECK_CUDA(cudaMemset(d_W[c], 0x3c, w_bytes));
  }
  CHECK_CUDA(cudaMalloc(&d_x, (size_t)s.K * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y, (size_t)s.N * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y_acc, (size_t)s.N * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_x, 0x3c, (size_t)s.K * sizeof(__half)));

  printf("\n=== %s : K=%d N=%d, weights %.2f MB (%d rotating copies, L2-cold) ===\n",
         s.name, s.K, s.N, w_bytes / 1e6, WCOPIES);
  printf("    TensorRT sm80_xmma baseline: %.1f us\n", 223.9);

  const int WARMUP = 8, ITERS = 60;
  for (int variant = 0; variant < 4; ++variant) {
    for (int grid : grids) {
      for (int i = 0; i < WARMUP; ++i)
        run_gemv(d_W[i % WCOPIES], d_x, d_y_acc, d_y, s.K, s.N, variant, grid);
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int i = 0; i < ITERS; ++i)
        run_gemv(d_W[i % WCOPIES], d_x, d_y_acc, d_y, s.K, s.N, variant, grid);
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float ms = 0.f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
      const double us = (ms * 1000.0) / ITERS;
      printf("  %-14s thr=%-5d grid=%-4d  %8.3f us  %7.2f GB/s  %s\n",
             variant_name(variant), variant_threads(variant, s.N), grid, us,
             (w_bytes / 1e9) / (us / 1e6), us < 223.9 ? "<- beats TRT" : "");

      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
    }
  }

  for (int c = 0; c < WCOPIES; ++c) CHECK_CUDA(cudaFree(d_W[c]));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_y_acc));
}

// Single cold launch of the GEMV mainloop, so `ncu --launch-count 1` measures
// the weight sweep and nothing else.
static void once_shape(const Shape& s, int variant, int grid) {
  const size_t w_bytes = (size_t)s.K * s.N * sizeof(__half);
  void* d_W = nullptr;
  __half* d_x = nullptr;
  float* d_y_acc = nullptr;
  CHECK_CUDA(cudaMalloc(&d_W, w_bytes));
  CHECK_CUDA(cudaMalloc(&d_x, (size_t)s.K * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y_acc, (size_t)s.N * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_W, 0x3c, w_bytes));
  CHECK_CUDA(cudaMemset(d_x, 0x3c, (size_t)s.K * sizeof(__half)));
  CHECK_CUDA(cudaMemset(d_y_acc, 0, (size_t)s.N * sizeof(float)));

  const int rows_per_block = (s.K + grid - 1) / grid;
  const int launch_grid = (variant == kChunkNarrow || variant == kChunkWide)
                              ? (s.K + rows_per_block - 1) / rows_per_block
                              : grid;
  launch_gemv(d_W, d_x, d_y_acc, s.K, s.N, variant, launch_grid, rows_per_block);
  CHECK_CUDA(cudaDeviceSynchronize());
  printf("[once %s] %s thr=%d grid=%d weights=%.2f MB\n", s.name,
         variant_name(variant), variant_threads(variant, s.N), launch_grid,
         w_bytes / 1e6);

  CHECK_CUDA(cudaFree(d_W));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y_acc));
}

int main(int argc, char** argv) {
  const std::string mode = (argc > 1) ? argv[1] : "check";
  const std::string which = (argc > 2) ? argv[2] : "all";
  const int variant = (argc > 3) ? atoi(argv[3]) : kBestVariant;

  for (const Shape& s : kShapes) {
    if (which != "all" && which != s.name) continue;
    const int grid = (argc > 4) ? atoi(argv[4]) : best_grid(s.N);
    if (mode == "check") {
      for (int v = 0; v < 4; ++v)
        for (int g : {14, 29, 58, 232}) check_shape(s, v, g);
    } else if (mode == "bench") {
      bench_shape(s, {14, 29, 44, 58, 72, 87, 116});
    } else if (mode == "once") {
      once_shape(s, variant, grid);
    } else {
      fprintf(stderr,
              "usage: %s [check|bench|once] [up_proj|down_proj|all] "
              "[variant 0-3] [grid]\n",
              argv[0]);
      return 1;
    }
  }
  return 0;
}
