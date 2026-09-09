// Dump the CUDA host-side __nv_fp8_e4m3 rounding of every finite FP16 value,
// so the numpy/ml_dtypes emulation used in fp8_numpy_study.py can be checked
// against the exact conversion gemv_fp8.cu's pack_e4m3() performs.
//
// Build: nvcc -O3 -arch=sm_89 e4m3_selftest.cu -o e4m3_selftest
// Run:   ./e4m3_selftest > e4m3_table.csv     # fp16_bits,fp8_bits,fp8_value
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

int main() {
  printf("fp16_bits,fp8_bits,fp8_value\n");
  for (uint32_t b = 0; b < 65536u; ++b) {
    __half h;
    const uint16_t bits = (uint16_t)b;
    *reinterpret_cast<uint16_t*>(&h) = bits;
    const float f = __half2float(h);
    if (isnan(f) || isinf(f)) continue;
    const __nv_fp8_e4m3 q(h);
    printf("%u,%u,%.10g\n", b, (unsigned)q.__x, (double)float(q));
  }
  return 0;
}
