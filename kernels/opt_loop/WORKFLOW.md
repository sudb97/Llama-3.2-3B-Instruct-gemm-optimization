# GEMV optimization loop

Measure inside the docker container passed as `CONTAINER` (id or name). That
container must already have `nvcc` + `ncu` and the workspace mount. **Do not
abort if the GPU is not L4** — record GPU name / sm in the run summary.

Start / track / stop: `COMMANDS.md`.

```bash
CONTAINER=<id> ./kernels/opt_loop/start.sh
CONTAINER=<id> TAG=iterNN_lever ./kernels/opt_loop/measure.sh
./kernels/opt_loop/status.sh
./kernels/opt_loop/stop.sh
```

Plumbing smoke test (no container required):

```bash
./kernels/opt_loop/dry_run.sh
```

## Goal

Beat TensorRT `sm80_xmma` decode GEMM latency by **≥ 25%** on both MLP shapes
(`up_proj` 223.9 µs → ≤167.9 µs, `down_proj` 223.6 µs → ≤167.7 µs), at cosine
≥ 0.999 vs the fp64 CPU reference.

Current FP16 GEMV (L4 ncu): **3.7–7.0%**. Three levers, in this order:

| Phase | Lever | Why it can move time |
|---|---|---|
| A | DRAM % → ~100% | Kernel is at 90–92% of L4 peak. Closing that is ~8–11% if bytes stay fixed. |
| B | Amplification → ~1.00× | Now 1.12–1.13× vs 50.33 MB unique. Closing that is ~11–13% bytes. |
| C | FP8 (or weight-only) weights | Halves the 50.33 MB stream. Roofline floor ~84 µs (~2.7× vs TRT) if still DRAM-bound. |

FP16 A+B stacked at the unique-byte / 300 GB/s floor is **168 µs ≈ 25% vs TRT**.
If A or B saturates short of 25%, **switch to C immediately**. Occupancy-only
rewrites are out of scope — they were already falsified.

## One iteration (do exactly this)

1. Read `EXPERIMENT_LOG.md`, latest `runs/*/summary.json`, and `baselines.json`.
2. Pick **one** change. Name it `iterNN_<lever>` (`dram`, `amp`, or `fp8`).
3. Implement in `kernels/` (prefer extending `gemv_fp16.cu` or adding
   `gemv_fp8.cu` — do not fork silently).
4. Correctness: `./gemv_fp16 check` (or the FP8 equivalent) inside `CONTAINER`.
5. Measure:
   ```bash
   CONTAINER=<id> TAG=iterNN_lever ./kernels/opt_loop/measure.sh
   ```
6. Append a row + a short ncu-gap note to `EXPERIMENT_LOG.md`. Record GPU.
7. **Stop** if `hit_25pct_both` is true → write `FINAL_REPORT.md` and halt.
8. **Stop** if 3 consecutive iters move min-gain by < 0.5 pp and Phase C has
   been tried → write `FINAL_REPORT.md` with the best kernel and why 25% failed.
9. Else choose the next lever from the ncu gaps below.

## How to read the ncu gaps

From `runs/<tag>/{up,down}_proj.json`:

- `dram_pct` < 96 → Phase A (issue more independent 16 B loads, `cp.async`,
  fewer concurrent far-apart streams, keep a small grid).
- `amplification` > 1.05 → Phase B (128 B alignment, no extra epilogue
  traffic, check `l2_sectors` vs unique/32).
- `dram_pct` ≥ 96 **and** `amplification` ≤ 1.05 **and** gain < 25% → Phase C.
  FP16 has no more bytes to give.

## Parallel agents

Three tracks can run at once (separate git worktrees). Setup: `PARALLEL.md`.
`measure.sh` serializes ncu with a file lock.

```bash
CONTAINER=<id> bash kernels/opt_loop/start_parallel.sh
```

## Hard rules

- `CONTAINER` required. `measure.sh` uses that container; it does not create one.
- Cold ncu only (`--launch-skip 0 --launch-count 1`). Do not quote L2-warm
  `./gemv_fp16 bench` GB/s (48 MB L2 vs 50 MB weights).
- Compare time to **TRT XMMA 223.9 / 223.6 µs**, not dumpProfile 119 GB/s.
  Always write the capture GPU next to the number.
- Keep the split-K full-row ownership unless ncu shows it is the amp problem.
- Cosine must stay ≥ 0.999. A faster wrong kernel is a failed iter.
