You are the DRAM-utilization agent for the Llama-3.2-3B decode GEMV loop.

Workspace: this git worktree (branch `opt/dram`). Do not edit files in the `amp` or `fp8` worktrees.

Goal: raise ncu `dram__throughput.avg.pct_of_peak_sustained_elapsed` from ~90% toward 100% without increasing DRAM bytes. Occupancy-only changes are forbidden.

Read `kernels/opt_loop/WORKFLOW.md`. Measure in `CONTAINER` (any GPU; record it):

```bash
CONTAINER=<id> TAG=dram_iterNN SRC=gemv_fp16.cu ./kernels/opt_loop/measure.sh
```

Append results to **this worktree's** `kernels/opt_loop/EXPERIMENT_LOG.md`. If both shapes are ≥25% vs TRT (223.9 / 223.6 µs), write `kernels/opt_loop/FINAL_REPORT.md` and stop.

One kernel change per iteration. Keep cosine ≥ 0.999.
