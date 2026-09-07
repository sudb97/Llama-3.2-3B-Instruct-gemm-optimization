# NCU verdict (parent model) — tick 4

**The TensorRT reference was wrong.** Everything below is rescored against a
corrected, twice-independently-measured reference.

## The correction

`223.9 / 223.6 µs` does not reproduce on a verified-idle GPU.

| shape | superseded | agent re-measure | **parent re-measure** | amp now |
|---|---|---|---|---|
| up_proj | 223.9 µs, amp 1.204 | 187.27 µs | **186.88 µs** | **1.0006** |
| down_proj | 223.6 µs, amp 1.210 | 181.60 µs | **182.17 µs** | **1.0007** |

Adopted reference: **up_proj 187.1 µs, down_proj 181.9 µs.**

Method: `ncu` on the real `sm80_xmma_gemm_..._tilesize32x32x64_stage6_...`
kernels inside `engines/baseline_decode.engine` at the decode point
(batch 1, seq 1, past 128). Not `--dumpProfile`. I ran my own capture and
parsed it with my own script rather than reusing the agent's.

**Root cause: GPU contention, not ECC.** My earlier ECC hypothesis is refuted.
A fixed hardware ECC overhead must be a fixed *percentage* of the kernel's own
traffic, but the excess is ~21% in the Jul-2026 TRT captures and ~12–13% in the
Sep-2026 GEMV capture. Both are instead consistent with a concurrent unprofiled
process holding a roughly constant slice of DRAM bandwidth (~47 and ~30 GB/s).

The decisive counter: `dram__bytes_read.sum` is **device-wide**, while
`lts__t_sectors_aperture_device_lookup_miss.sum` is **per-kernel**. Today their
ratio is **1.0000** on every launch I captured. In the old captures ~21% of
DRAM bytes had no matching L2 miss — they belonged to another process.

The timing closes exactly: TRT's own bandwidth share was 50.33 MB / 223.92 µs =
224.8 GB/s then, versus 268.9 GB/s now. The ratio 1.196 is precisely the
duration ratio.

## Rescored results

Corrected bars: PASS ≤ 149.68 / 145.52 µs, STRETCH ≤ 140.33 / 136.43 µs.

| Variant | shape | µs | old gain | **corrected gain** | PASS | STRETCH |
|---|---|---|---|---|---|---|
| FP16 baseline | up | 192.40 | +16.4% | **−2.7%** | no | no |
| FP16 baseline | down | 187.90 | +19.0% | **−3.2%** | no | no |
| FP16 best (no-atomics) | up | 181.95 | +23.1% | **+2.8%** | no | no |
| FP16 best (no-atomics) | down | 181.70 | +23.1% | **+0.1%** | no | no |
| **FP8 E4M3** | up | **110.40** | +102.8% | **+69.5%** | **YES** | **YES** |
| **FP8 E4M3** | down | **111.42** | +100.7% | **+63.3%** | **YES** | **YES** |

## What this means

1. **The FP16 track never beat TensorRT.** The baseline is 2.7–3.2% *slower*
   than XMMA; the best candidate is +2.8% on up_proj and a statistical tie on
   down_proj (181.70 vs 181.9 ± 0.4). The +23% was entirely the inflated
   reference.
2. **The project's FP16 thesis was void from the start.** The premise was that
   TRT wasted 20% of its bytes and driving amplification 1.20 → 1.00 was worth
   ~7–9%. TRT's real amplification is **1.0006**. There were never excess bytes
   to recover — which is exactly why months of FP16 work landed at parity. Both
   kernels move the same 50.33 MB at ~90% of peak, so both take ~185 µs.
3. **FP8 survives and is the only real win.** +69.5% / +63.3%, both bars met
   with margin. Its win shrinks from ~2.0× to ~1.6–1.7× but does not depend on
   the disputed reference — that was predicted in tick 3 and has now held.
4. **Both bars are now below the FP16 roofline floor** (149.7 / 145.5 vs
   167.8 µs), so no FP16 kernel could ever have passed. The PASS/STRETCH
   distinction is moot.

## Process failure worth recording

Three separate measurement errors all traced to one root cause: **profiling on
a GPU that was not verified idle.** `dram__bytes_read.sum` is device-wide, so
foreign traffic silently inflates both bytes and duration, and the result still
*looks* healthy (~90% of DRAM peak) because bytes and time scale together.

`WORKFLOW.md` now mandates an idle check plus the per-kernel L2-miss
cross-check before any official capture. That check costs one command and would
have prevented all of this.

## Remaining gate

FP8 accuracy on real weights is still under validation. `cos_fp16 = 0.99972`
came from synthetic `Uniform(-0.25, 0.25)` weights — the best case for
per-tensor E4M3 with no scale factor. Until that returns, **do not publish the
FP8 speedup as production-ready.**

Do not write `FINAL_REPORT.md` until it lands.
