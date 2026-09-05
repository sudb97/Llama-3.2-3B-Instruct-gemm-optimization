You are the FP8-weight agent for the Llama-3.2-3B decode GEMV loop.

Workspace: this git worktree (branch `opt/fp8`). Do not edit files in the `dram` or `amp` worktrees.

Goal: stream E4M3 (or similar) weights with a register dequant so unique bytes drop from 50.33 MB toward ~25 MB. Same split-K full-row ownership as `gemv_fp16.cu`. This is the lever most likely to beat +25% vs TensorRT.

Add `kernels/gemv_fp8.cu` (do not break the FP16 binary). Measure:

```bash
CONTAINER=<id> TAG=fp8_iterNN SRC=gemv_fp8.cu BIN=gemv_fp8 ./kernels/opt_loop/measure.sh
```

`parse_ncu.py` still divides by 50.33 MB unless you pass `--unique-bytes 25165824` after FP8 packing is real. Record both the raw DRAM bytes and the FP8-unique amplification.

Read `kernels/opt_loop/WORKFLOW.md`. Measure in `CONTAINER` (any GPU; record it). Cosine ≥ 0.999 vs fp64. Stop and write `FINAL_REPORT.md` if both shapes are ≥25% vs TRT (223.9 / 223.6 µs).
