// gemv_fp8.cu
//
// Weight-only E4M3 decode GEMV for the Llama-3.2-3B MLP projections.
// Same split-K full-row ownership as gemv_fp16.cu; each weight is 1 byte so
// the unique stream is K*N = 25,165,824 B (~25.17 MB) instead of 50.33 MB.
//
// Scale scheme: native E4M3 (1 sign / 4 exp / 3 mantissa, SATFINITE). No extra
// scale tensor — unique bytes stay exactly 25165824. Dequant is register-only
// (__nv_cvt_fp8x2_to_halfraw2 -> fp32) then fmaf with the fp16 activation.
//
//   up_proj / gate_proj : K=3072, N=8192  -> row = 8192 B
//   down_proj            : K=8192, N=3072  -> row = 3072 B
//
// uint4 = 16 B = 16 E4M3, so THREADS * 16 * VEC_PER_THREAD == N:
//   N=8192 : narrow 256 thr x 2 vec   wide 512 thr x 1 vec
//   N=3072 : narrow  96 thr x 2 vec   wide 192 thr x 1 vec
//
// Build:
//   nvcc -O3 -arch=sm_89 -lineinfo gemv_fp8.cu -o gemv_fp8
//
// Run:
//   ./gemv_fp8 check
//   ./gemv_fp8 bench
//   ./gemv_fp8 once up_proj <variant> <grid>
//   ./gemv_fp8 checkreal <raw-fp16-KxN-blob> [K] [N] [x=<raw-fp16-K-blob>]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err__));                                    \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static constexpr int FP8_PER_VEC = 16;  // uint4 = 16 B = 16 e4m3
static constexpr size_t UNIQUE_WEIGHT_BYTES = 25165824ull;  // K*N for both shapes

// ---------------------------------------------------------------------------
// Split-K GEMV.  y[n] = sum_k x[k] * dequant(W_e4m3[k, n])
//
// grid.x  : K-chunk index. Block b sweeps rows [b*rows_per_block, ...).
// A block covers the FULL row width so the slab is one sequential stream.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float2 dequant_e4m3x2(__nv_fp8x2_storage_t packed) {
  const __half2_raw h2r = __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3);
  __half2 h2;
  *reinterpret_cast<__half2_raw*>(&h2) = h2r;
  return __half22float2(h2);
}

// One 32-bit word = 4 e4m3. Byte 0 sits in the low bits, matching the column
// order flush_acc() writes, so acc[0..3] follow the memory order of the word.
// Operating on the uint4 components keeps the staged loads in registers — a
// reinterpret_cast on a local array would force it to local memory.
__device__ __forceinline__ void fma_word(uint32_t w, float xk, float* acc) {
  const float2 lo =
      dequant_e4m3x2(static_cast<__nv_fp8x2_storage_t>(w & 0xFFFFu));
  const float2 hi =
      dequant_e4m3x2(static_cast<__nv_fp8x2_storage_t>(w >> 16));
  acc[0] = fmaf(xk, lo.x, acc[0]);
  acc[1] = fmaf(xk, lo.y, acc[1]);
  acc[2] = fmaf(xk, hi.x, acc[2]);
  acc[3] = fmaf(xk, hi.y, acc[3]);
}

__device__ __forceinline__ void fma_vec(const uint4& raw, float xk, float* acc) {
  fma_word(raw.x, xk, acc + 0);
  fma_word(raw.y, xk, acc + 4);
  fma_word(raw.z, xk, acc + 8);
  fma_word(raw.w, xk, acc + 12);
}

// TPR threads span one full row (TPR * VEC * 16 B == N). A block stacks R such
// row-groups, so it holds R full rows — R*N contiguous bytes — in flight while
// still owning a single contiguous K-chunk. Raising R instead of the grid buys
// bytes-in-flight without adding concurrent DRAM streams, which is what the
// grid sweep showed to be the binding constraint.
template <int TPR, int R, int VEC, int KU>
__global__ __launch_bounds__(TPR * R) void gemv_chunk_kernel(
    const uint4* __restrict__ W,     // [K, N] row-major e4m3, viewed as uint4
    const __half* __restrict__ x,    // [K]
    float* __restrict__ y_acc,       // [N] fp32, pre-zeroed
    int K, int vec_per_row, int rows_per_block) {
  constexpr int ACC = VEC * FP8_PER_VEC;
  const int lane = threadIdx.x % TPR;
  const int grp = threadIdx.x / TPR;

  const int k_begin = blockIdx.x * rows_per_block;
  int k_end = k_begin + rows_per_block;
  if (k_end > K) k_end = K;

  float acc[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) acc[i] = 0.f;

  // KU rows per group are loaded before any is dequantised, so each thread also
  // keeps KU*VEC loads outstanding on top of the R-way row parallelism.
  int k = k_begin + grp;
  for (; k + R * (KU - 1) < k_end; k += R * KU) {
    uint4 raw[KU][VEC];
    float xk[KU];
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      xk[u] = __half2float(x[k + u * R]);
      const size_t row = (size_t)(k + u * R) * vec_per_row;
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        raw[u][v] = __ldcs(&W[row + v * TPR + lane]);
      }
    }
#pragma unroll
    for (int u = 0; u < KU; ++u) {
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        fma_vec(raw[u][v], xk[u], acc + v * FP8_PER_VEC);
      }
    }
  }
  for (; k < k_end; k += R) {
    const float xk = __half2float(x[k]);
    const size_t row = (size_t)k * vec_per_row;
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
      fma_vec(__ldcs(&W[row + v * TPR + lane]), xk, acc + v * FP8_PER_VEC);
    }
  }

  // y_acc is already an atomic accumulator across K-chunks, so the R groups
  // sharing a column range need no extra reduction.
#pragma unroll
  for (int v = 0; v < VEC; ++v) {
    const int col = (v * TPR + lane) * FP8_PER_VEC;
#pragma unroll
    for (int j = 0; j < FP8_PER_VEC; ++j) {
      atomicAdd(&y_acc[col + j], acc[v * FP8_PER_VEC + j]);
    }
  }
}

__global__ void cast_f32_to_f16_kernel(const float* __restrict__ src,
                                       __half* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

// Optional dequant scale, used only by `checkreal <blob> tensor|channel`.
// A weight scale that is constant down a column folds entirely into the
// epilogue -- y[n] = s[n] * sum_k x[k] * q[k,n] -- so the GEMV kernel above is
// untouched and the scale tensor is read once per launch (N*2 B), not once per
// row. `s` is null for the per-tensor case, where `s0` carries the scalar.
__global__ void cast_scale_f32_to_f16_kernel(const float* __restrict__ src,
                                             const __half* __restrict__ s,
                                             float s0, __half* __restrict__ dst,
                                             int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float scale = (s != nullptr) ? __half2float(s[i]) : s0;
    dst[i] = __float2half(src[i] * scale);
  }
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

// Each variant is a (threads-per-row, row-groups, uint4-per-thread) triple with
// TPR * VEC * 16 == N, i.e. full-row ownership. Variants are ordered by the
// contiguous bytes a block keeps in flight (R * N).
struct Tiling {
  int tpr;
  int r;
  int vec;
};

static Tiling tiling_of(int variant, int N) {
  if (N == 8192) {
    switch (variant) {
      case 0:  return {512, 1, 1};   //  512 thr,  8 KB in flight
      case 1:  return {512, 2, 1};   // 1024 thr, 16 KB
      case 2:  return {256, 2, 2};   //  512 thr, 16 KB
      default: return {256, 4, 2};   // 1024 thr, 32 KB
    }
  }
  switch (variant) {
    case 0:  return {192, 1, 1};     //  192 thr,  3 KB
    case 1:  return {192, 2, 1};     //  384 thr,  6 KB
    case 2:  return {192, 4, 1};     //  768 thr, 12 KB
    default: return {96,  8, 2};     //  768 thr, 24 KB
  }
}

static int variant_threads(int variant, int N) {
  const Tiling t = tiling_of(variant, N);
  return t.tpr * t.r;
}

static const char* variant_name(int v) {
  switch (v) {
    case 0:  return "tpr/R1";
    case 1:  return "tpr/R2";
    case 2:  return "half/R2-4";
    default: return "wide/R4-8";
  }
}

// Tuned by cold-ncu sweep on an L4 (sm_89). Both shapes sit on a broad plateau;
// DRAM% falls off sharply above these grids because extra K-chunks add
// concurrent far-apart DRAM streams (grid 48 costs up_proj ~25 us).
static constexpr int kBestVariant = 0;
static int best_grid(int N) { return (N == 8192) ? 20 : 48; }
static int best_ku(int N) { return 4; }

#define DISPATCH_KU(TPR, R, VEC, KU)                                           \
  gemv_chunk_kernel<TPR, R, VEC, KU>                                           \
      <<<grid, (TPR) * (R)>>>(W, d_x, d_y_acc, K, vec_per_row, rows_per_block)

#define DISPATCH(TPR, R, VEC)                                                  \
  do {                                                                         \
    switch (ku) {                                                              \
      case 1:  DISPATCH_KU(TPR, R, VEC, 1); break;                             \
      case 2:  DISPATCH_KU(TPR, R, VEC, 2); break;                             \
      case 8:  DISPATCH_KU(TPR, R, VEC, 8); break;                             \
      default: DISPATCH_KU(TPR, R, VEC, 4); break;                             \
    }                                                                          \
  } while (0)

static void launch_gemv(const void* d_W, const __half* d_x, float* d_y_acc,
                        int K, int N, int variant, int grid,
                        int rows_per_block, int ku) {
  const int vec_per_row = N / FP8_PER_VEC;
  const uint4* W = reinterpret_cast<const uint4*>(d_W);

  if (N == 8192) {
    switch (variant) {
      case 0:  DISPATCH(512, 1, 1); break;
      case 1:  DISPATCH(512, 2, 1); break;
      case 2:  DISPATCH(256, 2, 2); break;
      default: DISPATCH(256, 4, 2); break;
    }
  } else if (N == 3072) {
    switch (variant) {
      case 0:  DISPATCH(192, 1, 1); break;
      case 1:  DISPATCH(192, 2, 1); break;
      case 2:  DISPATCH(192, 4, 1); break;
      default: DISPATCH(96, 8, 2); break;
    }
  } else {
    fprintf(stderr, "no full-row tiling configured for N=%d\n", N);
    exit(1);
  }
}

static void run_gemv(const void* d_W, const __half* d_x, float* d_y_acc,
                     __half* d_y, int K, int N, int variant, int grid, int ku) {
  const int rows_per_block = (K + grid - 1) / grid;
  const int launch_grid = (K + rows_per_block - 1) / rows_per_block;
  CHECK_CUDA(cudaMemsetAsync(d_y_acc, 0, (size_t)N * sizeof(float)));
  launch_gemv(d_W, d_x, d_y_acc, K, N, variant, launch_grid, rows_per_block, ku);
  cast_f32_to_f16_kernel<<<(N + 255) / 256, 256>>>(d_y_acc, d_y, N);
}

static void fill_host(std::vector<__half>& W, std::vector<__half>& x,
                      int K, int N) {
  srand(1234);
  for (size_t i = 0; i < W.size(); ++i) {
    W[i] = __float2half(((rand() % 2001) - 1000) / 4000.0f);
  }
  for (int i = 0; i < K; ++i) {
    x[i] = __float2half(((rand() % 2001) - 1000) / 4000.0f);
  }
  (void)N;
}

static void pack_e4m3(const std::vector<__half>& W_fp16,
                      std::vector<uint8_t>& W_fp8) {
  W_fp8.resize(W_fp16.size());
  for (size_t i = 0; i < W_fp16.size(); ++i) {
    const __nv_fp8_e4m3 q(W_fp16[i]);
    W_fp8[i] = q.__x;
  }
}

static double e4m3_to_double(uint8_t bits) {
  __nv_fp8_e4m3 q;
  q.__x = bits;
  return (double)float(q);
}

// fp64 reference from the *same* E4M3 bytes the kernel streams.
static void cpu_reference_e4m3(const std::vector<uint8_t>& W,
                               const std::vector<__half>& x,
                               std::vector<double>& y, int K, int N) {
  y.assign(N, 0.0);
  for (int k = 0; k < K; ++k) {
    const double xk = __half2float(x[k]);
    if (xk == 0.0) continue;
    const uint8_t* row = W.data() + (size_t)k * N;
    for (int n = 0; n < N; ++n) y[n] += xk * e4m3_to_double(row[n]);
  }
}

static void cpu_reference_fp16(const std::vector<__half>& W,
                               const std::vector<__half>& x,
                               std::vector<double>& y, int K, int N) {
  y.assign(N, 0.0);
  for (int k = 0; k < K; ++k) {
    const double xk = __half2float(x[k]);
    if (xk == 0.0) continue;
    const __half* row = W.data() + (size_t)k * N;
    for (int n = 0; n < N; ++n) y[n] += xk * (double)__half2float(row[n]);
  }
}

static void cosine_stats(const std::vector<__half>& got,
                         const std::vector<double>& want, int N,
                         double& cos, double& max_abs, double& max_rel) {
  double dot = 0, na = 0, nb = 0;
  max_abs = 0;
  max_rel = 0;
  for (int n = 0; n < N; ++n) {
    const double g = __half2float(got[n]);
    const double w = want[n];
    dot += g * w;
    na += g * g;
    nb += w * w;
    const double abs_err = fabs(g - w);
    if (abs_err > max_abs) max_abs = abs_err;
    const double denom = fabs(w) > 1e-3 ? fabs(w) : 1e-3;
    if (abs_err / denom > max_rel) max_rel = abs_err / denom;
  }
  cos = dot / (sqrt(na) * sqrt(nb));
}

static void check_shape(const Shape& s, int variant, int grid, int ku) {
  const size_t w_elems = (size_t)s.K * s.N;
  std::vector<__half> h_W_fp16(w_elems), h_x(s.K);
  fill_host(h_W_fp16, h_x, s.K, s.N);
  std::vector<uint8_t> h_W_fp8;
  pack_e4m3(h_W_fp16, h_W_fp8);

  void* d_W = nullptr;
  __half* d_x = nullptr;
  __half* d_y = nullptr;
  float* d_y_acc = nullptr;
  CHECK_CUDA(cudaMalloc(&d_W, w_elems));
  CHECK_CUDA(cudaMalloc(&d_x, (size_t)s.K * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y, (size_t)s.N * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y_acc, (size_t)s.N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_W, h_W_fp8.data(), w_elems, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), (size_t)s.K * sizeof(__half),
                        cudaMemcpyHostToDevice));

  run_gemv(d_W, d_x, d_y_acc, d_y, s.K, s.N, variant, grid, ku);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__half> h_y(s.N);
  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, (size_t)s.N * sizeof(__half),
                        cudaMemcpyDeviceToHost));

  std::vector<double> ref_e4, ref_f16;
  cpu_reference_e4m3(h_W_fp8, h_x, ref_e4, s.K, s.N);
  cpu_reference_fp16(h_W_fp16, h_x, ref_f16, s.K, s.N);

  double cos_e4 = 0, abs_e4 = 0, rel_e4 = 0;
  double cos_f16 = 0, abs_f16 = 0, rel_f16 = 0;
  cosine_stats(h_y, ref_e4, s.N, cos_e4, abs_e4, rel_e4);
  cosine_stats(h_y, ref_f16, s.N, cos_f16, abs_f16, rel_f16);

  const bool pass = (cos_e4 >= 0.999);
  printf("[check %-10s] K=%-5d N=%-5d %-14s grid=%-4d ku=%-3d "
         "cos_e4m3=%.8f  cos_fp16=%.8f  max_abs_e4=%.4g  %s\n",
         s.name, s.K, s.N, variant_name(variant), grid, ku, cos_e4, cos_f16,
         abs_e4, pass ? "PASS" : "FAIL");

  CHECK_CUDA(cudaFree(d_W));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_y_acc));
}

// ---------------------------------------------------------------------------
// checkreal: same correctness gate as `check`, but the weights come from a raw
// FP16 [K, N] row-major blob (a real Llama-3.2-3B MLP projection dumped out of
// the ONNX external-data file) instead of fill_host()'s synthetic uniform draw.
// The kernel, the packing and both fp64 references are unchanged, so cos_e4m3
// and cos_fp16 mean exactly what they mean in `check`.
// ---------------------------------------------------------------------------

// Activations only. Same distribution and same generator call as fill_host(),
// so `checkreal` and `check` draw x from an identical process.
static void fill_x(std::vector<__half>& x, int K, unsigned seed) {
  srand(seed);
  for (int i = 0; i < K; ++i) {
    x[i] = __float2half(((rand() % 2001) - 1000) / 4000.0f);
  }
}

// Real activation vector, as captured from a forward pass of the actual model
// by kernels/opt_loop/fp8_accuracy/capture_real_activations.py. The synthetic
// draw above is Uniform(-0.25, 0.25) with kurtosis ~1.8, whereas the true
// down_proj input is SiLU(gate)*up and reaches kurtosis ~1900, so the two are
// not interchangeable for an accuracy claim.
static void load_x(std::vector<__half>& x, int K, const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "checkreal: cannot open activation file %s\n", path.c_str());
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  const long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  if ((size_t)fsize != (size_t)K * sizeof(__half)) {
    fprintf(stderr, "checkreal: %s is %ld B, expected %zu B for K=%d\n",
            path.c_str(), fsize, (size_t)K * sizeof(__half), K);
    exit(1);
  }
  if (fread(x.data(), sizeof(__half), (size_t)K, f) != (size_t)K) {
    fprintf(stderr, "checkreal: short read on %s\n", path.c_str());
    exit(1);
  }
  fclose(f);
}

// Filenames written by kernels/opt_loop/fp8_accuracy/extract_real_weights.py
// look like L00.up_proj.K3072.N8192.fp16.bin.
static bool parse_kn_from_name(const std::string& path, int* K, int* N) {
  const size_t kp = path.rfind(".K");
  if (kp == std::string::npos) return false;
  const size_t np = path.find(".N", kp);
  if (np == std::string::npos) return false;
  *K = atoi(path.c_str() + kp + 2);
  *N = atoi(path.c_str() + np + 2);
  return *K > 0 && *N > 0;
}

enum ScaleMode { kScaleNone = 0, kScaleTensor = 1, kScaleChannel = 2 };

static const char* scale_name(ScaleMode m) {
  switch (m) {
    case kScaleTensor:  return "per-tensor scale";
    case kScaleChannel: return "per-channel scale";
    default:            return "no scale (shipped)";
  }
}

// fp64 reference over the dequantised weights s[n] * E4M3(w / s[n]).
static void cpu_reference_scaled(const std::vector<uint8_t>& W,
                                 const std::vector<__half>& x,
                                 const std::vector<__half>& s,
                                 std::vector<double>& y, int K, int N) {
  y.assign(N, 0.0);
  for (int k = 0; k < K; ++k) {
    const double xk = __half2float(x[k]);
    if (xk == 0.0) continue;
    const uint8_t* row = W.data() + (size_t)k * N;
    for (int n = 0; n < N; ++n) y[n] += xk * e4m3_to_double(row[n]);
  }
  for (int n = 0; n < N; ++n) y[n] *= __half2float(s[n]);
}

static void check_real(const std::string& path, int K, int N, unsigned seed,
                       ScaleMode smode, const std::string& xpath,
                       bool shipped_only) {
  const size_t w_elems = (size_t)K * N;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "checkreal: cannot open %s\n", path.c_str());
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  const long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  if ((size_t)fsize != w_elems * sizeof(__half)) {
    fprintf(stderr, "checkreal: %s is %ld B, expected %zu B for K=%d N=%d\n",
            path.c_str(), fsize, w_elems * sizeof(__half), K, N);
    exit(1);
  }
  std::vector<__half> h_W_fp16(w_elems);
  if (fread(h_W_fp16.data(), sizeof(__half), w_elems, f) != w_elems) {
    fprintf(stderr, "checkreal: short read on %s\n", path.c_str());
    exit(1);
  }
  fclose(f);

  std::vector<__half> h_x(K);
  if (xpath.empty()) {
    fill_x(h_x, K, seed);
  } else {
    load_x(h_x, K, xpath);
  }

  // Scales are the standard amax/448 choice, stored fp16 so the per-channel
  // variant costs exactly N*2 extra unique bytes.
  std::vector<__half> h_s(N, __float2half(1.0f));
  if (smode != kScaleNone) {
    std::vector<float> colmax(N, 0.f);
    float tmax = 0.f;
    for (int k = 0; k < K; ++k) {
      const __half* row = h_W_fp16.data() + (size_t)k * N;
      for (int n = 0; n < N; ++n) {
        const float a = fabsf(__half2float(row[n]));
        if (a > colmax[n]) colmax[n] = a;
        if (a > tmax) tmax = a;
      }
    }
    for (int n = 0; n < N; ++n) {
      const float amax = (smode == kScaleTensor) ? tmax : colmax[n];
      h_s[n] = __float2half(fmaxf(amax, 1e-12f) / 448.0f);
    }
    // Pre-divide the weights by their column scale before packing.
    for (int k = 0; k < K; ++k) {
      __half* row = h_W_fp16.data() + (size_t)k * N;
      for (int n = 0; n < N; ++n) {
        row[n] = __float2half(__half2float(row[n]) / __half2float(h_s[n]));
      }
    }
  }

  std::vector<uint8_t> h_W_fp8;
  pack_e4m3(h_W_fp16, h_W_fp8);

  // Reload the untouched FP16 weights: the fp16 reference must use the
  // original values, not the pre-divided ones.
  if (smode != kScaleNone) {
    f = fopen(path.c_str(), "rb");
    if (fread(h_W_fp16.data(), sizeof(__half), w_elems, f) != w_elems) {
      fprintf(stderr, "checkreal: short re-read on %s\n", path.c_str());
      exit(1);
    }
    fclose(f);
  }

  // Weight statistics, so the report can be cross-checked against the numpy
  // study without trusting that both read the same bytes.
  double wmin = 1e30, wmax = -1e30, wsum = 0, wsq = 0, amax = 0;
  size_t n_sub = 0, n_clip = 0;
  for (size_t i = 0; i < w_elems; ++i) {
    const double w = __half2float(h_W_fp16[i]);
    if (w < wmin) wmin = w;
    if (w > wmax) wmax = w;
    wsum += w;
    wsq += w * w;
    const double a = fabs(w);
    if (a > amax) amax = a;
    if (a < 0.015625 && a > 0.0) ++n_sub;  // below E4M3 smallest normal 2^-6
    if (a > 448.0) ++n_clip;               // would saturate under SATFINITE
  }
  const double mean = wsum / (double)w_elems;
  const double sd = sqrt(wsq / (double)w_elems - mean * mean);

  void* d_W = nullptr;
  __half* d_x = nullptr;
  __half* d_y = nullptr;
  __half* d_s = nullptr;
  float* d_y_acc = nullptr;
  CHECK_CUDA(cudaMalloc(&d_W, w_elems));
  CHECK_CUDA(cudaMalloc(&d_x, (size_t)K * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y, (size_t)N * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_s, (size_t)N * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y_acc, (size_t)N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_W, h_W_fp8.data(), w_elems, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), (size_t)K * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_s, h_s.data(), (size_t)N * sizeof(__half),
                        cudaMemcpyHostToDevice));

  // References depend only on the weights and x, so build them once.
  std::vector<double> ref_e4, ref_f16;
  if (smode == kScaleNone) {
    cpu_reference_e4m3(h_W_fp8, h_x, ref_e4, K, N);
  } else {
    cpu_reference_scaled(h_W_fp8, h_x, h_s, ref_e4, K, N);
  }
  cpu_reference_fp16(h_W_fp16, h_x, ref_f16, K, N);

  // Activation statistics, so the accuracy line can be read against the
  // distribution that produced it rather than assumed to be well behaved.
  double xsum = 0, xsq = 0, xamax = 0;
  for (int i = 0; i < K; ++i) {
    const double v = __half2float(h_x[i]);
    xsum += v;
    xsq += v * v;
    if (fabs(v) > xamax) xamax = fabs(v);
  }
  const double xmean = xsum / (double)K;
  const double xrms = sqrt(xsq / (double)K);
  double xk4 = 0;
  const double xvar = xsq / (double)K - xmean * xmean;
  for (int i = 0; i < K; ++i) {
    const double d = __half2float(h_x[i]) - xmean;
    xk4 += d * d * d * d;
  }
  const double xkurt = (xk4 / (double)K) / (xvar * xvar);

  printf("\n=== checkreal %s [%s] ===\n", path.c_str(), scale_name(smode));
  printf("    K=%d N=%d  min=%+.6f max=%+.6f mean=%+.3e std=%.6f amax/std=%.2f\n",
         K, N, wmin, wmax, mean, sd, amax / sd);
  printf("    x source: %s  rms=%.5f absmax=%.5f kurtosis=%.1f\n",
         xpath.empty() ? "SYNTHETIC Uniform(-0.25,0.25)" : xpath.c_str(),
         xrms, xamax, xkurt);
  printf("    |w| below E4M3 smallest normal (2^-6): %.2f%%   |w|>448 (would "
         "saturate): %zu\n",
         100.0 * (double)n_sub / (double)w_elems, n_clip);
  if (smode == kScaleTensor) {
    printf("    scale = %.6g, passed as a kernel immediate: 0 extra unique "
           "bytes\n", __half2float(h_s[0]));
  } else if (smode == kScaleChannel) {
    printf("    extra unique bytes for scales: %zu (%.3f%% of %zu)\n",
           (size_t)N * sizeof(__half),
           100.0 * (double)(N * sizeof(__half)) / (double)UNIQUE_WEIGHT_BYTES,
           UNIQUE_WEIGHT_BYTES);
  }

  double worst_e4 = 1.0, worst_f16 = 1.0;
  const int variants_lo = shipped_only ? kBestVariant : 0;
  const int variants_hi = shipped_only ? kBestVariant + 1 : 4;
  const std::vector<int> grids = shipped_only
      ? std::vector<int>{best_grid(N)}
      : std::vector<int>{14, 24, 58, 232, best_grid(N)};
  const std::vector<int> kus = shipped_only
      ? std::vector<int>{best_ku(N)}
      : std::vector<int>{1, 2, 4, 8};
  for (int v = variants_lo; v < variants_hi; ++v) {
    for (int g : grids) {
      for (int u : kus) {
        run_gemv(d_W, d_x, d_y_acc, d_y, K, N, v, g, u);
        if (smode != kScaleNone) {
          // Re-run the epilogue with the scale folded in.
          cast_scale_f32_to_f16_kernel<<<(N + 255) / 256, 256>>>(
              d_y_acc, (smode == kScaleChannel) ? d_s : nullptr,
              __half2float(h_s[0]), d_y, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<__half> h_y(N);
        CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, (size_t)N * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        double cos_e4 = 0, abs_e4 = 0, rel_e4 = 0;
        double cos_f16 = 0, abs_f16 = 0, rel_f16 = 0;
        cosine_stats(h_y, ref_e4, N, cos_e4, abs_e4, rel_e4);
        cosine_stats(h_y, ref_f16, N, cos_f16, abs_f16, rel_f16);
        if (cos_e4 < worst_e4) worst_e4 = cos_e4;
        if (cos_f16 < worst_f16) worst_f16 = cos_f16;
        const bool best = (v == kBestVariant && g == best_grid(N) &&
                           u == best_ku(N));
        printf("  %-14s grid=%-4d ku=%-3d cos_e4m3=%.8f  cos_fp16=%.8f  "
               "max_abs_e4=%.4g %s\n",
               variant_name(v), g, u, cos_e4, cos_f16, abs_e4,
               best ? "<- shipped config" : "");
      }
    }
  }
  printf("  WORST over %s: cos_e4m3=%.8f [%s]  cos_fp16=%.8f [%s]\n",
         shipped_only ? "shipped config" : "all 80 configs",
         worst_e4, worst_e4 >= 0.999 ? "PASS" : "FAIL", worst_f16,
         worst_f16 >= 0.999 ? "PASS" : "FAIL");

  CHECK_CUDA(cudaFree(d_W));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_s));
  CHECK_CUDA(cudaFree(d_y_acc));
}

static constexpr int WCOPIES = 4;

static void bench_shape(const Shape& s, const std::vector<int>& grids, int ku) {
  const size_t w_bytes = (size_t)s.K * s.N;  // e4m3

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

  printf("\n=== %s : K=%d N=%d, e4m3 weights %.2f MB (%d rotating copies) ===\n",
         s.name, s.K, s.N, w_bytes / 1e6, WCOPIES);
  printf("    unique bytes = %zu  TensorRT sm80_xmma baseline: %.1f us\n",
         UNIQUE_WEIGHT_BYTES, 223.9);

  const int WARMUP = 8, ITERS = 60;
  for (int variant = 0; variant < 4; ++variant) {
    for (int grid : grids) {
      for (int i = 0; i < WARMUP; ++i)
        run_gemv(d_W[i % WCOPIES], d_x, d_y_acc, d_y, s.K, s.N, variant, grid, ku);
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int i = 0; i < ITERS; ++i)
        run_gemv(d_W[i % WCOPIES], d_x, d_y_acc, d_y, s.K, s.N, variant, grid, ku);
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float ms = 0.f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
      const double us = (ms * 1000.0) / ITERS;
      printf("  %-14s thr=%-5d grid=%-4d ku=%-3d %8.3f us  %7.2f GB/s  %s\n",
             variant_name(variant), variant_threads(variant, s.N), grid, ku, us,
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

// One launch of the shipped config on the REAL weights and REAL activation, so
// the ncu capture profiles the production data path rather than a memset
// pattern. Duration is value independent for this kernel (no data-dependent
// control flow and no compression), but the sign-off should not have to rely on
// that argument.
static void once_real(const std::string& wpath, const std::string& xpath,
                      int K, int N, int grid, int ku) {
  const size_t w_elems = (size_t)K * N;
  FILE* f = fopen(wpath.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "oncereal: cannot open %s\n", wpath.c_str());
    exit(1);
  }
  std::vector<__half> h_W_fp16(w_elems);
  if (fread(h_W_fp16.data(), sizeof(__half), w_elems, f) != w_elems) {
    fprintf(stderr, "oncereal: short read on %s\n", wpath.c_str());
    exit(1);
  }
  fclose(f);

  std::vector<uint8_t> h_W_fp8;
  pack_e4m3(h_W_fp16, h_W_fp8);

  std::vector<__half> h_x(K);
  if (xpath.empty()) fill_x(h_x, K, 1234u);
  else load_x(h_x, K, xpath);

  void* d_W = nullptr;
  __half* d_x = nullptr;
  float* d_y_acc = nullptr;
  CHECK_CUDA(cudaMalloc(&d_W, w_elems));
  CHECK_CUDA(cudaMalloc(&d_x, (size_t)K * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_y_acc, (size_t)N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_W, h_W_fp8.data(), w_elems, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), (size_t)K * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_y_acc, 0, (size_t)N * sizeof(float)));

  const int rows_per_block = (K + grid - 1) / grid;
  const int launch_grid = (K + rows_per_block - 1) / rows_per_block;
  launch_gemv(d_W, d_x, d_y_acc, K, N, kBestVariant, launch_grid, rows_per_block, ku);
  CHECK_CUDA(cudaDeviceSynchronize());
  printf("[oncereal] %s K=%d N=%d %s grid=%d ku=%d e4m3_bytes=%zu unique=%zu\n",
         wpath.c_str(), K, N, variant_name(kBestVariant), launch_grid, ku,
         w_elems, UNIQUE_WEIGHT_BYTES);

  CHECK_CUDA(cudaFree(d_W));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y_acc));
}

static void once_shape(const Shape& s, int variant, int grid, int ku) {
  const size_t w_bytes = (size_t)s.K * s.N;
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
  const int launch_grid = (s.K + rows_per_block - 1) / rows_per_block;
  launch_gemv(d_W, d_x, d_y_acc, s.K, s.N, variant, launch_grid, rows_per_block, ku);
  CHECK_CUDA(cudaDeviceSynchronize());
  printf("[once %s] %s thr=%d grid=%d ku=%d weights=%.2f MB unique=%zu\n",
         s.name, variant_name(variant), variant_threads(variant, s.N),
         launch_grid, ku, w_bytes / 1e6, UNIQUE_WEIGHT_BYTES);

  CHECK_CUDA(cudaFree(d_W));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y_acc));
}

int main(int argc, char** argv) {
  const std::string mode = (argc > 1) ? argv[1] : "check";
  const std::string which = (argc > 2) ? argv[2] : "all";
  const int variant = (argc > 3) ? atoi(argv[3]) : kBestVariant;

  if (mode == "checkreal") {
    if (argc < 3) {
      fprintf(stderr,
              "usage: %s checkreal <raw-fp16-KxN-blob> [K N] "
              "[none|tensor|channel] [shipped] [x=<raw-fp16-K-blob>]\n",
              argv[0]);
      return 1;
    }
    // Remaining args in any order: two integers are K,N; a keyword is the
    // scale mode (default none = exactly what the kernel ships with);
    // x=<path> supplies a captured or synthetic activation vector instead of
    // the Uniform(-0.25,0.25) draw; shipped limits the sweep to the timed
    // launch config.
    ScaleMode smode = kScaleNone;
    std::string xpath;
    bool shipped_only = false;
    int K = 0, N = 0, nints = 0;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "none") smode = kScaleNone;
      else if (a == "tensor") smode = kScaleTensor;
      else if (a == "channel") smode = kScaleChannel;
      else if (a == "shipped") shipped_only = true;
      else if (a.rfind("x=", 0) == 0) xpath = a.substr(2);
      else if (nints == 0) { K = atoi(argv[i]); ++nints; }
      else if (nints == 1) { N = atoi(argv[i]); ++nints; }
    }
    if (nints < 2 && !parse_kn_from_name(which, &K, &N)) {
      fprintf(stderr,
              "checkreal: cannot infer K,N from '%s' -- pass them explicitly\n",
              which.c_str());
      return 1;
    }
    check_real(which, K, N, 1234u, smode, xpath, shipped_only);
    return 0;
  }

  if (mode == "oncereal") {
    if (argc < 3) {
      fprintf(stderr,
              "usage: %s oncereal <raw-fp16-KxN-blob> [K N] "
              "[x=<raw-fp16-K-blob>] [grid=<n>] [ku=<n>]\n",
              argv[0]);
      return 1;
    }
    std::string xpath;
    int K = 0, N = 0, nints = 0, grid = 0, ku = 0;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("x=", 0) == 0) xpath = a.substr(2);
      else if (a.rfind("grid=", 0) == 0) grid = atoi(a.c_str() + 5);
      else if (a.rfind("ku=", 0) == 0) ku = atoi(a.c_str() + 3);
      else if (nints == 0) { K = atoi(argv[i]); ++nints; }
      else if (nints == 1) { N = atoi(argv[i]); ++nints; }
    }
    if (nints < 2 && !parse_kn_from_name(which, &K, &N)) {
      fprintf(stderr,
              "oncereal: cannot infer K,N from '%s' -- pass them explicitly\n",
              which.c_str());
      return 1;
    }
    if (grid <= 0) grid = best_grid(N);
    if (ku <= 0) ku = best_ku(N);
    once_real(which, xpath, K, N, grid, ku);
    return 0;
  }

  for (const Shape& s : kShapes) {
    if (which != "all" && which != s.name) continue;
    const int grid = (argc > 4) ? atoi(argv[4]) : best_grid(s.N);
    const int ku = (argc > 5) ? atoi(argv[5]) : best_ku(s.N);
    if (mode == "check") {
      for (int v = 0; v < 4; ++v)
        for (int g : {14, 24, 58, 232, best_grid(s.N)})
          for (int u : {1, 2, 4, 8}) check_shape(s, v, g, u);
    } else if (mode == "bench") {
      bench_shape(s, {14, 24, 29, 44, 58, 72, 87, 116}, ku);
    } else if (mode == "once") {
      once_shape(s, variant, grid, ku);
    } else {
      fprintf(stderr,
              "usage: %s [check|bench|once] [up_proj|down_proj|all] "
              "[variant 0-3] [grid] [k-unroll]\n"
              "       %s checkreal <raw-fp16-KxN-blob> [K] [N] [x=<blob>]\n"
              "       %s oncereal  <raw-fp16-KxN-blob> [K] [N] [x=<blob>]\n",
              argv[0], argv[0], argv[0]);
      return 1;
    }
  }
  return 0;
}
