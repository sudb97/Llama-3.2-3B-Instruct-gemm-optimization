# Experiment log — L4 GEMV latency loop

Only append rows produced by `measure.sh` (record GPU + `CONTAINER`).

**Target:** ≥ 25% latency vs TRT XMMA on **both** shapes (up ≤ 167.9 µs, down ≤ 167.7 µs).

| tag | GPU | kernel | up µs | up amp | up DRAM% | up vs TRT | down µs | down amp | down DRAM% | down vs TRT | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_trt | L4 | sm80_xmma | 223.9 | 1.204 | ~90 | 0% | 223.6 | 1.210 | ~91 | 0% | TensorRT auto tactic |
| baseline_gemv_fp16 | L4 | chunk/wide | 215.6 | 1.132 | 90.1 | 3.7% | 208.9 | 1.118 | 91.9 | 7.0% | current best FP16 |

## Iteration notes

_(agent appends one subsection per iter)_
