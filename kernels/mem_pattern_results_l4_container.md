# Memory-pattern microbenchmark — verified inside project Docker container

Container: `489257d94e91` (image `cuda-profiler:latest`), GPU: NVIDIA L4 (sm_89), CUDA 12.6,
TensorRT 10.4.0.26. Built with `nvcc -O3 -arch=sm_89 -lineinfo`. `ncu` cold launch
(`--launch-skip 0 --launch-count 1`), no warmup, one launch per pattern.

Unique-bytes-requested basis for both weight shapes: 50.33 MB (8192 x 6144 B for
down_proj-shape buffer, 3072 x 16384 B for up_proj-shape buffer — same total bytes,
different row layout).

DRAM amplification = `dram__bytes_read.sum / 50331648`.

## down_proj shape (8192 rows x 6144 B/row)

| pattern    | dram__bytes_read.sum | amplification | L2 hit rate | DRAM % of peak | gpu time |
|------------|----------------------|----------------|-------------|-----------------|----------|
| contiguous | 55,542,912 B         | 1.1035x        | 14.38%      | 94.74%          | 205.5 us |
| strided    | 57,213,824 B         | 1.1367x        | 0.33%       | 94.75%          | 205.9 us |
| fragmented | 70,903,552 B         | 1.4087x        | 0.39%       | 92.58%          | 262.5 us |

## up_proj shape (3072 rows x 16384 B/row)

| pattern    | dram__bytes_read.sum | amplification | L2 hit rate | DRAM % of peak | gpu time |
|------------|----------------------|----------------|-------------|-----------------|----------|
| contiguous | 57,006,336 B         | 1.1326x        | 6.05%       | 94.86%          | 206.8 us |
| strided    | 57,345,280 B         | 1.1393x        | 0.31%       | 94.72%          | 206.4 us |
| fragmented | 72,984,320 B         | 1.4501x        | 0.73%       | 88.43%          | 282.8 us |

## Real TensorRT kernel (measured earlier, `down_projection_raw.csv`, same container/GPU class)

| kernel    | amplification |
|-----------|----------------|
| down_proj | 1.2104x        |
| up_proj   | 1.2044x        |

## Reading

The real kernel's amplification (1.20-1.21x) sits **between** the synthetic
`strided` (1.14x) and `fragmented` (1.41-1.45x) patterns for both weight
shapes — consistent with the real kernel's B-tile access being partially but
not fully coalesced across K-rows, as reconstructed from the ncu source-page
sector counts. `contiguous` (1.10-1.13x) is the practical floor reachable
without changing the tile shape entirely.

Numbers reproduce the earlier sandbox-GPU run to within ~0.1%, confirming
that run was already on equivalent L4 hardware and is safe to cite in the
writeup. This container run is the authoritative source going forward.
