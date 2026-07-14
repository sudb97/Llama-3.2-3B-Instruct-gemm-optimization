# Llama-3.2-3B-Instruct GEMM Optimization

Replace TensorRT's auto-selected decode GEMM tactic with a custom CUTLASS FP8 kernel (IPluginV3) on NVIDIA L4 (SM89).

## Layout

```
engine/          Download ONNX + build TRT engines (trtexec)
models/          HuggingFace ONNX weights (not committed)
engines/         Serialized .engine files (not committed)
bench/           trtexec benchmarks and Nsight scripts
kernels/         Custom CUDA/CUTLASS GEMM kernels
plugin/          IPluginV3 wrapper (C++ + CMake)
eval/            Correctness parity and downstream eval
```

## Quick start

```bash
# 1. Download FP16 ONNX model
./engine/download_model.sh

# 2. Inspect ONNX inputs (update shape flags if needed)
python3 engine/inspect_onnx.py

# 3. Build baseline engine
./engine/build_engine.sh decode

# 4. Benchmark
./bench/trtexec_baseline.sh decode
```

## Target GEMM (decode)

- MLP gate/up: `3072 -> 8192`, M = 1..8
- MLP down: `8192 -> 3072`, M = 1..8

See [tensorrt_custom_tactic_kernel_b07f17a4.plan.md](tensorrt_custom_tactic_kernel_b07f17a4.plan.md) for the full project plan.

**Progress log:** [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) — what was tried, what worked, learnings, and results (update as you go).

**Todo list:** [TODO.md](TODO.md) — current status of every project phase (persists across sessions).
