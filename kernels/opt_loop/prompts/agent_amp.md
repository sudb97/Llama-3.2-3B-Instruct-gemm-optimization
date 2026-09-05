You are the DRAM-amplification agent for the Llama-3.2-3B decode GEMV loop.

Workspace: this git worktree (branch `opt/amp`). Do not edit files in the `dram` or `fp8` worktrees.

Goal: cut `dram__bytes_read.sum / 50331648` from ~1.13× toward 1.00× (128 B alignment, no extra epilogue traffic). Do not change numeric precision.

Read `kernels/opt_loop/WORKFLOW.md`. Measure in `CONTAINER` (any GPU; record it):

```bash
CONTAINER=<id> TAG=amp_iterNN SRC=gemv_fp16.cu ./kernels/opt_loop/measure.sh
```

Append results to **this worktree's** `kernels/opt_loop/EXPERIMENT_LOG.md`. If both shapes are ≥25% vs TRT (223.9 / 223.6 µs), write `kernels/opt_loop/FINAL_REPORT.md` and stop.

One kernel change per iteration. Keep cosine ≥ 0.999.
