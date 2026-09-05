// Tiny host-only kernel so dry_run.sh can exercise nvcc + ncu + parse_ncu.py.
// Not a GEMV candidate. Do not log these numbers in EXPERIMENT_LOG.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void dummy_scale(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * 2.f;
}

int main() {
    const int n = 4096;
    float *a, *b;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    dummy_scale<<<1, 256>>>(a, b, n);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::printf("[dry_run] dummy_scale n=%d launched\n", n);
    cudaFree(a);
    cudaFree(b);
    return 0;
}
