# TensorRT XMMA reference re-check — 2026-09-07

**Verdict: 223.9 / 223.6 µs does NOT reproduce on a verified-idle GPU.**
The corrected TensorRT XMMA decode reference is **187.27 µs (up_proj/gate_proj)** and
**181.60 µs (down_proj)** — the old figures were overstated by **19.6%** and **23.1%**.

The old capture's DRAM amplification of 1.204× / 1.210× also does not reproduce.
Measured amplification today is **1.0005× / 1.0007×**. TensorRT's XMMA tactic was
never over-fetching; it reads each weight byte exactly once.

---

## 1. Method

**Full engine, not a micronet.** This reproduces the original method as closely as
possible: `ncu` on the real `sm80_xmma_gemm_...` kernels inside the prebuilt
6.4 GB `engines/baseline_decode.engine`, driven by `trtexec --loadEngine`.

Provenance of the number being re-checked: `bench/profile/down_projection_raw.csv`,
an `ncu` raw export taken 2026-Jul-22 20:03 from `trtexec` PID 247, 10 kernel
launches (2 decoder layers). `PROJECT_PROGRESS.md` §"2026-08" and
`kernels/opt_loop/baselines.json` `trt_xmma` both derive 223.9 / 223.6 µs from it.
Those are the means of the grid-256 and 50 MB-grid-96 launches in that file.

Command actually run (per repeat, 4 repeats):

```
flock /tmp/gemv_opt_loop.ncu.lock \
ncu --csv \
    --kernel-name-base function \
    --kernel-name "regex:sm80_xmma_gemm" \
    --launch-skip 0 --launch-count 21 \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_aperture_device_lookup_miss.sum,\
gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,gpc__cycles_elapsed.avg.per_second \
    trtexec --loadEngine=engines/baseline_decode.engine \
            --shapes="$OPT_SHAPES" --warmUp=0 --duration=0 --iterations=4
```

- `--launch-skip 0` — cold, from the first launch of the process, matching how
  `kernels/opt_loop/measure.sh` and `kernels/run_gemv_ncu.sh` capture the GEMV kernels.
- Metric set is the GEMV set from `measure.sh`, plus `dram__bytes_write.sum`,
  `lts__t_sectors_aperture_device_lookup_miss.sum` (L2→DRAM read misses) and the
  GPC clock. The L2-miss counter is the cross-check that makes the old capture
  falsifiable: it is a **per-kernel** counter, whereas `dram__bytes_read.sum` is
  device-wide.
- `$OPT_SHAPES` from `engine/configs/baseline_decode.env`: batch 1, seq_len 1,
  past_sequence_length 128 (`attention_mask:1x129`) — the decode point the engine
  was tactic-profiled at.
- Serialized under `flock /tmp/gemv_opt_loop.ncu.lock` as required.
- TensorRT 10.4.0.26 (`trtexec` v100400), CUDA 12.6 `ncu`, driver 570.133.20.
  `trtexec` is not on `PATH` in this shell; it was run from the TensorRT 10.4.0.26
  tree with `LD_LIBRARY_PATH` set to its `lib`.

**`trtexec --dumpProfile` was deliberately not used as the reference**, per the known
119 vs 236 GB/s artifact documented in `PROJECT_PROGRESS.md`. No `--dumpProfile`
number appears anywhere in this document.

### Shape identification

Grid size fixes N (32-wide N-tile ⇒ `grid = N/32`); DRAM bytes fix K. Every one of
the four GEMM shapes in the decode step is identified unambiguously, and each
kernel's measured DRAM read matches its unique weight bytes to within 0.15%:

| Kernel | Grid | N = grid×32 | Unique FP16 weight bytes | Measured DRAM read |
|---|---|---|---|---|
| `qkv_proj` K=3072 | 160 | 5120 | 31,457,280 (31.457 MB) | 31.481 MB |
| `o_proj` K=3072 | 96 | 3072 | 18,874,368 (18.874 MB) | 18.897 MB |
| **`up_proj`/`gate_proj` K=3072** | **256** | **8192** | **50,331,648 (50.332 MB)** | **50.358 MB** |
| **`down_proj` K=8192** | **96** | **3072** | **50,331,648 (50.332 MB)** | **50.369 MB** |

Tactic (identical to the one named in `PROJECT_PROGRESS.md`, all four shapes):

```
sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt
```

---

## 2. GPU state

`ecc.mode.current` is **Disabled** before and after, and `ecc.mode.pending` is also
Disabled, so ECC did not change state during this session.

| | Before (01:55 UTC+5:30) | After (02:04 UTC+5:30) |
|---|---|---|
| `--query-compute-apps=pid,used_memory` | *(empty — no compute processes)* | *(empty — no compute processes)* |
| Name | NVIDIA L4 | NVIDIA L4 |
| `ecc.mode.current` / `pending` | Disabled / Disabled | Disabled / Disabled |
| `clocks.sm` | 2040 MHz (= max) | 2040 MHz (= max) |
| `clocks.mem` | 6251 MHz (= max) | 6251 MHz (= max) |
| `temperature.gpu` | 36 °C | 35 °C |
| `utilization.gpu` | 2% | 2% |
| `memory.used` | 0 MiB | 0 MiB |
| Persistence mode | Disabled | Disabled |

Driver 570.133.20, CUDA 12.8 runtime, compute capability 8.9.

**No contention appeared mid-run.** `nvidia-smi --query-compute-apps` was re-polled
between every repeat and returned empty every time. The in-band evidence is
stronger than the polling: across all 65 profiled launches,
`dram__bytes_read.sum` equalled `lts__t_sectors_aperture_device_lookup_miss.sum × 32`
to a ratio of **1.0000**, i.e. every DRAM byte counted is accounted for by an L2 miss
of the kernel under profile. There is no room for foreign traffic in these numbers.

During capture `ncu` locks the SM clock to base ≈ 795 MHz (`gpc__cycles_elapsed.avg.per_second`
790–819 MHz). The July capture shows 798–807 MHz, i.e. the same clock regime, so the
old and new durations are directly comparable. Memory clock is unaffected and stays
at 6251 MHz: measured 268.9 GB/s at 89.76% of peak implies a 299.6 GB/s peak,
matching L4's 300 GB/s spec.

---

## 3. Per-shape results

Unique FP16 weight bytes = 50,331,648 for both shapes.
Amplification = `dram__bytes_read.sum / 50,331,648`.

### up_proj / gate_proj — K=3072, N=8192, grid 256

| Repeat | n launches | duration_us | mean |
|---|---|---|---|
| rep1 | 3 | 187.55, 187.55, 187.33 | 187.48 |
| rep2 | 8 | 187.55, 186.14, 186.56, 187.36, 186.88, 188.51, 187.58, 187.36 | 187.24 |
| rep3 | 8 | 187.90, 187.52, 186.11, 186.24, 186.21, 186.78, 187.20, 186.37 | 186.79 |
| rep4 | 8 | 186.88, 188.26, 187.58, 187.01, 187.84, 188.70, 188.13, 187.30 | 187.71 |

- **mean = 187.27 µs**, min 186.11, max 188.70, spread 2.59 µs (1.38%), sd 0.71 µs, n = 27
- dram_bytes_read = **50.358 MB**, **amplification = 1.0005×**
- L2→DRAM miss bytes = 50.358 MB → `dram/L2miss = 1.0000`
- **dram_pct = 89.76%** of peak (268.9 GB/s)

### down_proj — K=8192, N=3072, grid 96

| Repeat | n launches | duration_us | mean |
|---|---|---|---|
| rep1 | 1 | 181.09 | 181.09 |
| rep2 | 3 | 181.38, 181.15, 181.89 | 181.47 |
| rep3 | 3 | 181.60, 182.30, 182.08 | 181.99 |
| rep4 | 3 | 181.28, 181.54, 181.66 | 181.49 |

- **mean = 181.60 µs**, min 181.09, max 182.30, spread 1.22 µs (0.67%), sd 0.40 µs, n = 10
- dram_bytes_read = **50.369 MB**, **amplification = 1.0007×**
- L2→DRAM miss bytes = 50.369 MB → `dram/L2miss = 1.0000`
- **dram_pct = 92.59%** of peak (277.4 GB/s)

### Other decode GEMMs (not part of the reference, recorded for completeness)

| Shape | n | mean_us | spread | dram_read | dram_pct |
|---|---|---|---|---|---|
| `qkv_proj` K=3072 N=5120 | 14 | 117.77 | 1.38 µs | 31.481 MB | 89.26% |
| `o_proj` K=3072 N=3072 | 14 | 72.82 | 1.22 µs | 18.897 MB | 86.72% |

---

## 4. What was wrong with the July capture

Every one of the four GEMM shapes in the July-22 file carries ~20–22% of DRAM read
bytes with no matching L2 miss, and is correspondingly ~18–23% slower:

| Shape | old_us | old MB | old amp | new_us | new MB | new amp | excess MB | excess as foreign BW | old/new time |
|---|---|---|---|---|---|---|---|---|---|
| `qkv_proj` | 138.74 | 38.15 | 1.213× | 117.77 | 31.481 | 1.0008× | 6.69 | 48.2 GB/s | 1.178× |
| `o_proj` | 86.43 | 22.96 | 1.217× | 72.82 | 18.897 | 1.0012× | 4.09 | 47.3 GB/s | 1.187× |
| `up_proj` | 223.92 | 60.62 | 1.204× | 187.27 | 50.358 | 1.0005× | 10.29 | 45.9 GB/s | 1.196× |
| `down_proj` | 223.63 | 60.92 | 1.210× | 181.60 | 50.369 | 1.0007× | 10.58 | 47.3 GB/s | 1.231× |

The timing story closes exactly. In the July capture the counted bus total was
~271 GB/s (reported as ~90% of peak, which is why it looked healthy), but the
kernel's *own* share was only 50.33 MB / 223.92 µs = **224.8 GB/s**. Today the
kernel gets **268.9 GB/s**. 268.9 / 224.8 = 1.196, which is precisely the observed
duration ratio. The old kernel was not doing extra work; it was being starved of
about a sixth of the bus.

### Mechanism: contention, not ECC

`baselines.json` currently attributes the superseded GEMV captures to inline ECC
having been enabled. That hypothesis does not survive being applied to both eras at
once. A fixed hardware ECC overhead is a fixed *percentage* of the kernel's own
traffic, so it must be the same percentage in every capture:

| Capture | excess | as % of unique bytes | as foreign bandwidth |
|---|---|---|---|
| TRT `up_proj` (Jul-22) | 10.29 MB | 20.4% | 45.9 GB/s |
| TRT `down_proj` (Jul-22) | 10.58 MB | 21.0% | 47.3 GB/s |
| GEMV `up_proj` (tick1, bogus) | 6.63 MB | 13.2% | 30.9 GB/s |
| GEMV `down_proj` (tick1, bogus) | 5.95 MB | 11.8% | 28.6 GB/s |

The percentage is *not* stable across eras (21% vs 12–13%), which rules out a fixed
ECC overhead. Both eras are, however, consistent with a concurrent unprofiled
process holding a roughly constant slice of DRAM read bandwidth — ~47 GB/s during
the TRT capture, ~30 GB/s during the GEMV capture. Combined with `dram__bytes_read.sum`
being device-wide while `lts__t_sectors_aperture_device_lookup_miss.sum` is
per-kernel, and with ECC reading Disabled/Disabled now, **GPU contention from
another process is the supported explanation.** The ECC note in `baselines.json`
should be corrected.

Within a single era the two hypotheses are nearly degenerate (all these kernels run
at ~90% of DRAM peak, so bytes and duration are proportional); it is only the
cross-era comparison that separates them.

---

## 5. Corrected reference and recomputed gains

Corrected reference: **up_proj 187.27 µs, down_proj 181.60 µs.**

Bars, with `gain = trt/us - 1`:

| Shape | corrected TRT | PASS bar (gain ≥ 25% ⇒ us ≤ trt/1.25) | STRETCH bar (us ≤ 0.75 × trt) |
|---|---|---|---|
| up_proj | 187.27 µs | ≤ 149.82 µs | ≤ 140.45 µs |
| down_proj | 181.60 µs | ≤ 145.28 µs | ≤ 136.20 µs |

| Variant | Shape | µs | gain vs OLD trt | **gain vs CORRECTED trt** | PASS (≥25%) | STRETCH (≤0.75×) |
|---|---|---|---|---|---|---|
| FP16 baseline | up | 192.40 | +16.37% | **−2.67%** | no | no |
| FP16 baseline | down | 187.90 | +19.00% | **−3.35%** | no | no |
| FP16 best (no-atomics) | up | 181.95 | +23.06% | **+2.92%** | no | no |
| FP16 best (no-atomics) | down | 181.70 | +23.06% | **−0.06%** | no | no |
| FP8 (E4M3) | up | 110.40 | +102.81% | **+69.63%** | **YES** | **YES** |
| FP8 (E4M3) | down | 111.42 | +100.68% | **+62.99%** | **YES** | **YES** |

### Consequences

1. **The FP16 track never beat TensorRT.** The FP16 "baseline" at 192.4 / 187.9 µs is
   2.7% / 3.4% *slower* than the XMMA tactic. The best FP16 candidate at
   181.95 / 181.70 µs is 2.9% faster on `up_proj` and statistically tied on
   `down_proj` (181.70 vs 181.60 ± 0.40 µs). The claimed +23% is entirely an
   artifact of the inflated reference.
2. **FP8 still clears both bars, with margin.** +69.6% / +63.0% against the corrected
   reference, comfortably inside both the PASS and STRETCH bars. The headline win
   survives; its size drops from ~2.0× to ~1.6–1.7×.
3. **The project's FP16 amplification thesis is void.** `PROJECT_PROGRESS.md`
   §"2026-08-19" argues that TRT's 1.20× amplification left 7–9% of bytes on the
   table and that driving amplification to 1.00 was the FP16 lever. TensorRT's real
   amplification is 1.0005×. There were never any excess bytes to recover, which is
   exactly why the FP16 work landed at parity: both kernels move the same 50.33 MB
   at ~90% of peak, so both take ~185 µs. The `mem_pattern_bench` "real TRT kernel
   1.204× / 1.210×" row in that table is contaminated too and should be struck.
4. **`target.definition_a_script` is not reachable in FP16.** Its 179.12 / 178.88 µs
   thresholds were derived from the inflated reference. The corrected PASS thresholds
   are 149.82 / 145.28 µs, which are *below* the 167.8 µs FP16 roofline floor. Under
   the corrected reference, definitions A and B are no longer meaningfully different:
   both are unreachable in FP16 and both are met by FP8.

### Caveat on the comparison numbers

The 192.4 / 187.9, 181.95 / 181.70 and 110.40 / 111.42 figures are taken as given
from `baselines.json` and `EXPERIMENT_LOG.md`; this task re-measured only the
TensorRT reference. The FP16 pair is documented as idle-GPU re-measured
(2026-09-07). The FP8 pair is recorded as "parent-verified" but I did not confirm
which GPU state it was captured under — if it predates the idle-GPU discipline it
should be re-checked the same way, since a contaminated FP8 capture would be
*over*stated in duration and so *under*state the FP8 gain.

Note also that comparing FP8 against an FP16 TensorRT reference is not
numerics-neutral; FP8 halves the weight bytes (25,165,824 unique), which is where
its win comes from.

---

## 6. Files

Raw `ncu` CSVs for the four repeats, plus the capture and parsing scripts, are at
`kernels/opt_loop/runs/trt_recheck_20260907/`:

- `engine_rep{1..4}.csv` — raw `ncu --csv` output
- `run_ncu_engine.sh`, `env.sh` — the capture as run
- `parse.py`, `agg.py` — parsing / aggregation used for the tables above

Note that `kernels/opt_loop/runs/` is gitignored (`.gitignore:42`), so these need a
`git add -f` if the evidence should be committed.

The original July capture being refuted is retained in-repo at
`bench/profile/down_projection_raw.csv`.

No kernels, engines or committed baselines were modified by this task. The only new
paths are this file and the `runs/trt_recheck_20260907/` directory. Nothing was
committed or pushed. `baselines.json`, `EXPERIMENT_LOG.md`, `VERDICT.md` and
`WORKFLOW.md` still carry the old 223.9 / 223.6 reference and need updating by the
parent.
