# GEMV optimization loop

Measure inside the docker container passed as `CONTAINER` (id or name). That
container must already have `nvcc` + `ncu` and the workspace mount. **Do not
abort if the GPU is not L4** — record GPU name / sm in the run summary.

Start / track / stop: `COMMANDS.md`.

```bash
CONTAINER=local ./kernels/opt_loop/start.sh          # already inside the container
CONTAINER=local TAG=iterNN_lever ./kernels/opt_loop/measure.sh
./kernels/opt_loop/status.sh
./kernels/opt_loop/stop.sh
```

Plumbing smoke test (no container required):

```bash
./kernels/opt_loop/dry_run.sh
```

## Goal

Beat TensorRT `sm80_xmma` decode GEMM latency on both MLP shapes at cosine
≥ 0.999 vs the fp64 CPU reference. **Two bars** (user decision 2026-09-07):

**Reference corrected 2026-09-07** to `up_proj` **187.1 µs**, `down_proj`
**181.9 µs**. The old 223.9 / 223.6 pair was captured under GPU contention and
overstated by ~20%. See `TRT_RECHECK.md`.

| Bar | Rule | up_proj | down_proj | Reachable in FP16? |
|---|---|---|---|---|
| **PASS** | `gain = trt/us − 1 ≥ 25%` | ≤ **149.68 µs** | ≤ **145.52 µs** | **no** |
| **STRETCH** | `us ≤ 0.75 × trt` | ≤ **140.33 µs** | ≤ **136.43 µs** | **no** |

Both bars are now *below* the FP16 roofline floor (50.33 MB / 300 GB/s =
167.8 µs), so **neither is reachable in FP16 by any means.** FP8 is the only
route, and it meets both.

| Variant | up_proj | down_proj | vs corrected TRT |
|---|---|---|---|
| FP16 baseline | 192.4 µs | 187.9 µs | −2.7% / −3.2% (**slower than TRT**) |
| FP16 best (no-atomics) | 181.95 µs | 181.70 µs | +2.8% / +0.1% (**parity**) |
| **FP8 E4M3** | **110.40 µs** | **111.42 µs** | **+69.5% / +63.3% — both bars MET** |

| Phase | Lever | Status |
|---|---|---|
| A | DRAM % | **CLOSED.** 94% is the practical ceiling. Read-only probe (all arithmetic deleted) is *slower*, so nothing above DRAM throttles. |
| B | Amplification | **CLOSED.** Already 1.0009 unmodified — and TRT is 1.0006, so there were never excess bytes to recover. The original thesis was void. |
| C | FP8 weights | **DELIVERED.** Halves the stream to 25.17 MB. |

Occupancy-only rewrites are out of scope — already falsified (16% vs 65%
within 2 µs).

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
  fewer concurrent far-apart streams, keep a small grid, avoid serialized
  epilogue read-modify-write).
- `amplification` > 1.05 → would be Phase B, but **verify against a same-session
  control build of the unmodified kernel first**. Amplification is 1.000 on this
  machine; a high reading is an environment artifact, not a kernel property.
- `dram_pct` ≥ 96 **and** gain < 25% → Phase C. FP16 has no more bytes to give.

**Always sanity-check a surprising capture against a control build of
`git show HEAD:kernels/gemv_fp16.cu` in the same session.** That is how the
1.13× amplification error was caught.

## Parallel agents

Three tracks can run at once (separate git worktrees). Setup: `PARALLEL.md`.
`measure.sh` serializes ncu with a file lock.

```bash
CONTAINER=<id> bash kernels/opt_loop/start_parallel.sh
```

## Hard rules

- `CONTAINER` required. `measure.sh` uses that container; it does not create one.
- **Verify the GPU is idle before every official capture**
  (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` must be empty).
  `dram__bytes_read.sum` is **device-wide**: a concurrent unprofiled process
  inflates both bytes and duration. This single mistake invalidated the TRT
  reference for two months and produced a phantom 1.20× amplification.
- **Cross-check every capture:** `dram__bytes_read.sum` should equal
  `lts__t_sectors_aperture_device_lookup_miss.sum × 32` (that counter is
  per-kernel). Ratio 1.0000 = clean. Anything above = foreign traffic.
- Cold ncu only (`--launch-skip 0 --launch-count 1`). Do not quote L2-warm
  `./gemv_fp16 bench` GB/s (48 MB L2 vs 50 MB weights). Note tensors under
  48 MB (e.g. `o_proj`) *do* go L2-warm on repeat launches — split by DRAM
  bytes before averaging.
- Compare time to **corrected TRT XMMA 187.1 / 181.9 µs**, not the superseded
  223.9 / 223.6 and not dumpProfile. Always record the capture GPU.
- Keep the split-K full-row ownership unless ncu shows it is the amp problem.
- Cosine must stay ≥ 0.999. A faster wrong kernel is a failed iter.
