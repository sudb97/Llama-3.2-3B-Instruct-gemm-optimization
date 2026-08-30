# FP16 decode GEMV vs TensorRT sm80_xmma tactic

Container `489257d94e91` (`cuda-profiler:latest`), NVIDIA L4 sm_89, CUDA 12.6.
Built `nvcc -O3 -arch=sm_89 -lineinfo gemv_fp16.cu`. All ncu numbers are cold
(`--launch-skip 0 --launch-count 1`, ncu flushes caches by default).

Source: `kernels/gemv_fp16.cu`. Baseline numbers from `bench/profile/down_projection_raw.csv`.

## Headline

| Shape | Kernel | Duration | DRAM read | Amplification | DRAM % peak | Occupancy |
|---|---|---|---|---|---|---|
| `up_proj` K=3072 N=8192 | TRT sm80_xmma | 223.9 us | 60.62 MB | 1.204x | ~90% | 16.1% |
| `up_proj` | **GEMV chunk/wide grid=14** | **215.6 us** | **56.96 MB** | **1.132x** | 90.1% | 64.7% |
| `down_proj` K=8192 N=3072 | TRT sm80_xmma | 223.6 us | 60.92 MB | 1.210x | ~91% | 13.4% |
| `down_proj` | **GEMV chunk/wide grid=29** | **208.9 us** | **56.28 MB** | **1.118x** | 91.9% | 24.6% |

Speedup **1.039x** (`up_proj`) and **1.070x** (`down_proj`) at equal DRAM saturation.

Correctness: cosine similarity **0.99999998** vs a float64 CPU reference for all
32 variant x grid combinations, max relative error 6.1e-4 (fp16 output rounding).

## Why this is the FP16 ceiling

The amplification landed at 1.132x / 1.118x against the microbenchmark's
contiguous floor of 1.133x / 1.104x (`mem_pattern_results_l4_container.md`).
The weight stream is now as sequential as a pure memory-read kernel, so there
are no more bytes to recover in FP16. The remaining gap to a 2x class win is
not addressable by access pattern -- it needs fewer bytes, i.e. FP8 or
weight-only quantization.

Predicted vs achieved: bytes fell 6.0-7.6%, time fell 3.7-6.6%. That matches
the 1.07-1.09x byte-level estimate recorded before the kernel was written.

## Design

Split-K, full-row blocks, register accumulators, no shared-memory weight staging.

- A block owns a contiguous chunk of K rows and the **full N width**, so it reads
  one unbroken slab. Splitting N instead would give each block a narrow column
  stripe and reintroduce the row-stride jump that costs the TRT kernel its 1.20x.
- `THREADS * 8 * VEC_PER_THREAD == N` so one instruction sweep covers exactly one
  row: 1024 thr x 1 vec for N=8192, 384 thr x 1 vec for N=3072.
- Each thread owns fixed output columns for the whole sweep, so partial sums stay
  in registers. At M=1 every weight is used exactly once, so a shared-memory B
  tile buys no reuse -- it would only cost barriers and the 24 KiB that caps the
  TRT kernel at 2 blocks/SM.
- Blocks `atomicAdd` into an fp32 accumulator. Output is at most 32 KB, so that
  traffic stays L2-resident.

## Two results that were not obvious

**Small grids win.** 14 blocks beat 58, 116 and 464 on `up_proj`. Every extra
block is another concurrent stream the memory system must interleave; past the
point where DRAM is saturated, more blocks only fragment the access. 14 blocks of
1024 threads use a fraction of the SMs and still reach 90% of DRAM peak.

**Occupancy was never the lever.** Going from 16.7% to 64.7% occupancy did not by
itself buy anything -- `chunk/narrow` at 16% occupancy and `chunk/wide` at 65%
land within 2 us of each other at the same grid. What bought the win was the byte
reduction. This confirms the pre-kernel analysis: the 2-blocks/SM limit was real
but not what was costing time.

## Rejected: far-apart K chunks

The first version gave each block a contiguous K-chunk but sized the grid to fill
the machine (`grid=116`). Amplification was fine (1.115x) but DRAM throughput
collapsed to 62-74% and it was **slower than the TRT baseline** (258-308 us),
because the co-resident blocks were streaming from widely separated regions. The
`stride` variant (block b takes rows b, b+grid, ...) was written to fix this by
keeping resident blocks in one contiguous window; it helped, but simply shrinking
the grid turned out to matter more, and `chunk` with a small grid is the best of
both. Both variants are kept in the source since the comparison is the evidence.

## Reproduce

```bash
./run_gemv_ncu.sh          # build, check, sweep, profile
./gemv_fp16 check          # correctness, all variants
./gemv_fp16 bench          # L2-cold wall-clock sweep
./gemv_fp16 once up_proj   # single launch for ncu
```

`bench` rotates over 4 weight copies. L4's L2 is 48 MB against a 50.33 MB matrix,
so a naive repeat loop reports 800-1000 GB/s -- above DRAM peak -- because the
matrix is nearly L2-resident. Real decode revisits a projection 6 GB later, i.e.
always cold.
