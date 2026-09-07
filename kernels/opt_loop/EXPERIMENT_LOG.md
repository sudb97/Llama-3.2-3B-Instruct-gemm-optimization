# Experiment log — L4 GEMV latency loop

Only append rows produced by `measure.sh` (record GPU + `CONTAINER`).

**Target:** ≥ 25% vs TRT XMMA on **both** shapes.

> **The TRT reference was corrected on 2026-09-07** from 223.9 / 223.6 µs to
> **187.1 / 181.9 µs**. The old figures were captured under GPU contention.
> The `up vs TRT` / `down vs TRT` columns below are **against the old
> reference** and are therefore overstated by ~20 pp. The corrected scoring
> table is at the bottom of this file. See `TRT_RECHECK.md`.

| tag | GPU | kernel | up µs | up amp | up DRAM% | up vs TRT | down µs | down amp | down DRAM% | down vs TRT | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ~~baseline_trt~~ | L4 | sm80_xmma | ~~223.9~~ | ~~1.204~~ | ~~90~~ | — | ~~223.6~~ | ~~1.210~~ | ~~91~~ | — | **SUPERSEDED — contended capture** |
| **baseline_trt_corrected** | L4 | sm80_xmma | **187.1** | **1.0006** | 90.0 | 0% | **181.9** | **1.0007** | 92.4 | 0% | idle GPU, measured twice independently |
| ~~baseline_gemv_fp16~~ | L4 | chunk/wide | ~~215.6~~ | ~~1.132~~ | ~~90.1~~ | ~~3.7%~~ | ~~208.9~~ | ~~1.118~~ | ~~91.9~~ | ~~7.0%~~ | **SUPERSEDED** — see correction |
| ~~tick1_baseline~~ | L4 | chunk/wide | ~~214.3~~ | ~~1.1314~~ | ~~90.51~~ | ~~4.5%~~ | ~~208.0~~ | ~~1.1182~~ | ~~92.27~~ | ~~7.5%~~ | **SUPERSEDED** — not reproducible |
| **baseline_corrected** | L4 | chunk/wide | **192.4** | **1.0009** | 88.98 | **16.4%** | **187.9** | **1.0007** | 91.1 | **19.0%** | mean of 3 parent re-measures, unmodified HEAD |
| amp_iter01 | L4 | chunk/wide, no atomics | 181.95 | 1.0003 | 94.04 | 23.05% | 183.50 | 1.0005 | 93.48 | 21.85% | agent, n=3 |
| amp_iter02 | L4 | chunk/wide, plain float4 store | 182.85 | 1.0003 | 93.60 | 22.45% | 182.33 | 1.0004 | 93.91 | 22.64% | agent, n=3 |
| **parent_verify_amp02** | L4 | amp worktree kernel | **181.95** | **1.0003** | **94.04** | **23.05%** | **181.70** | **1.0004** | **94.20** | **23.06%** | parent-verified, single cold capture |
| dram_iter01 | L4 | SW-pipelined K loop | 192.26 | 1.0009 | 89.06 | 16.46% | 187.84 | 1.0007 | 91.09 | 19.04% | agent; null (ptxas already pipelined) |
| dram_iter02 | L4 | ROW_BATCH 8/1 | 190.91 | 1.0010 | 89.63 | 17.28% | 187.71 | 1.0007 | 91.19 | 19.12% | agent; median of 3 |
| dram_probe_readonly | L4 | all arithmetic deleted | 195.5 | — | 88.30 | — | — | — | — | — | **slower than real kernel → memory path is the limit** |
| fp8_iter01 | L4 | E4M3 split-K | 121.4 | — | 63–69 | — | 133.2 | — | — | — | agent; already past target |
| **fp8_iter02** | L4 | E4M3, retuned grid | **110.98** | 1.0019* | 75.91 | **101.76%** | **111.78** | 1.0015* | 75.32 | **100.04%** | agent, n=2 |
| **parent_verify_fp8** | L4 | E4M3, retuned grid | **110.40** | **1.0019*** | **76.30** | **102.81%** | **111.42** | **1.0015*** | **75.54** | **100.67%** | **parent-verified. STRETCH BAR HIT** |

\* FP8 amplification is vs 25,165,824 unique FP8 bytes. Versus the original
50,331,648 FP16 bytes it is **0.501×** — the stream genuinely halved.

## Correction — 2026-09-07 (baseline was wrong by ~12 pp)

Both the Aug-2026 baseline and my own `tick1_baseline` recorded amplification
~1.12–1.13×. **No capture taken on 2026-09-07 reproduces this.** Three parent
re-measures of the *unmodified* `HEAD` kernel, plus independent controls from
two agents, all land at **amp ≈ 1.000** and **~192 / ~188 µs**.

Evidence the difference is environmental, not a kernel property:

- `lts__t_sectors_srcunit_tex_op_read` is **identical** across old and new
  captures (1,573,187 vs 1,573,191 on up_proj). The excess bytes never reached
  L2, so no kernel change could have caused or fixed them.
- Run-to-run byte noise is ~400 B. The discrepancy is ~6.6 MB — four orders larger.
- The byte ratio 56.94/50.38 = **1.130** is close to GDDR6 inline-ECC overhead
  (1/8 = 1.125). `nvidia-smi` now reports `ecc.mode.current = Disabled`.
- Cross-process counter contamination was tested and falsified by the AMP agent.

The prior ECC state cannot be proven retroactively, so the cause is inferred,
not established. What is established: **amplification is 1.000 on the current
machine, so Phase B (amp → 1.00×) is closed — there are no bytes to recover.**

The real headline: the FP16 kernel was already **16.4% / 19.0%** faster than
TensorRT, not 3.7% / 7.0%.

## Target definition ambiguity — unresolved

Two incompatible rules are live in this repo:

| Source | Rule | up_proj budget | Status |
|---|---|---|---|
| `parse_ncu.py` (`hit_25pct_vs_trt`) | `trt/us - 1 ≥ 0.25` | ≤ **179.12 µs** | reachable in FP16 (~1.6% away) |
| `WORKFLOW.md` | `us ≤ 0.75 × trt` | ≤ **167.9 µs** | **unreachable in FP16** |

The FP16 roofline floor is 50.33 MB / 300 GB/s = **167.8 µs**, so the
`WORKFLOW.md` budget demands 100% of theoretical peak bandwidth at
amplification 1.00. Under that rule FP8 is mandatory. Under the script rule,
one more DRAM% push (94.0 → ~95.5%) finishes the job in FP16.

## Iteration notes

**amp lever — CLOSED.** Amplification was already 1.000; the lever's premise
was void. The atomicAdd epilogue removal is a null on bytes (−0.07%) but a
genuine **DRAM%** win (89 → 94), worth ~5% latency. Reclassify it as Phase A.

Open caveat: the separate reduce kernel costs ~7.4 µs / 465 KB under ncu, but
ncu flushes L2 between launches so its production cost is unmeasured. Fusing
the reduction into the main kernel's tail would settle it.

**dram lever — CLOSED.** The decisive test was a read-only probe: keep the
addresses, grid and thread count identical but delete *all* arithmetic. It came
out **slower** (195.5 µs, 88.30%), proving nothing above DRAM was throttling.
Supporting negatives: MLP depth 2/4/8 within 1% (SASS shows ptxas already
issues four `LDG.E.EF.128` per unrolled step, so the software-pipelining
premise was false); grid 14/29 is a two-sided optimum; the contiguous-window
`stride` variant loses at every grid, so split-K full-row ownership is kept.
Ceiling at a literal 100% DRAM is 171 µs = +23.6%, short of PASS.

**fp8 lever — DELIVERED.** 2× TRT on both shapes, both bars met.
DRAM% *fell* to ~76%, so FP8 is no longer bandwidth-bound; it is latency-bound
with neither resource saturated (issue 38.2% of peak). Cause: sm_89 exposes FP8
only as a tensor-core operand, so `__nv_cvt_fp8x2_to_halfraw2` is **emulated**
(~2.4 non-FMA ALU instructions per weight byte in SASS). Closing the last 27 µs
to the 84 µs roofline needs a cheaper dequant, not more parallelism. Not
attempted — it risks the cosine gate for a target already met at 2×.
Negative results recorded: per-thread K-unrolling is null (ptxas already
pipelines); stacking row-groups per block is actively harmful.

## Corrected scoring (vs TRT 187.1 / 181.9 µs)

Bars: PASS ≤ 149.68 / 145.52 µs. STRETCH ≤ 140.33 / 136.43 µs.
Both are **below** the 167.8 µs FP16 roofline floor, so no FP16 kernel can pass.

| Variant | up µs | up gain | down µs | down gain | PASS | STRETCH |
|---|---|---|---|---|---|---|
| FP16 baseline | 192.40 | **−2.7%** | 187.90 | **−3.2%** | no | no |
| FP16 best (no-atomics) | 181.95 | **+2.8%** | 181.70 | **+0.1%** | no | no |
| **FP8 E4M3** | **110.40** | **+69.5%** | **111.42** | **+63.3%** | **YES** | **YES** |

The FP16 track never beat TensorRT: the baseline is slower than XMMA and the
best candidate is at parity. The +16–23% previously recorded was entirely an
artifact of the inflated reference.

## Root cause of all three measurement errors

One mistake: **profiling on a GPU that was not verified idle.**

`dram__bytes_read.sum` is **device-wide**; `lts__t_sectors_aperture_device_lookup_miss.sum`
is **per-kernel**. Foreign traffic inflates the former, and because bytes and
duration scale together the result still looks healthy at ~90% of DRAM peak.

The ECC hypothesis I recorded earlier is **refuted**: a fixed hardware overhead
must be a fixed percentage, but the excess was ~21% in the Jul-2026 TRT era and
~12–13% in the Sep-2026 GEMV era. Contention explains both (~47 and ~30 GB/s of
foreign read bandwidth).

Guard now mandated in `WORKFLOW.md`: verify idle before every official capture,
and confirm `dram__bytes_read.sum ÷ (L2_miss_sectors × 32) = 1.0000`.

## Open risks — must close before FINAL_REPORT

1. ~~TRT reference not re-validated.~~ **CLOSED 2026-09-07.** Re-measured twice
   on an idle GPU: 187.1 / 181.9 µs, amplification 1.0006 / 1.0007. The tick-3
   prediction held — FP8 stayed well past target (+69.5% / +63.3%) while the
   FP16 claim collapsed to parity, exactly as the sensitivity analysis said.
2. **FP8 accuracy validated only on synthetic uniform weights.** `fill_host`
   uses `Uniform(-0.25, 0.25)` — the best case for per-tensor E4M3, which this
   kernel applies with **no scale factor**. Real weights are heavy-tailed, so
   `cos_fp16 = 0.99972` is optimistic and is not a production accuracy number.
   **STILL OPEN** — gates `FINAL_REPORT.md`.
