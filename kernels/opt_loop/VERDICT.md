# NCU verdict (parent model) — tick 0

**Source:** L4 cold ncu, `kernels/gemv_{up,down}_proj_ncu.csv` (sm_89, Aug 2026).
Not T4. `parse_ncu.py` numbers checked against the CSV by hand.

| | up_proj | down_proj |
|---|---|---|
| Kernel | `gemv_chunk_kernel<1024,1>` grid 14 | `gemv_chunk_kernel<384,1>` grid 29 |
| Duration | 215.6 µs | 208.9 µs |
| vs TRT 223.9 / 223.6 µs | **+3.9%** | **+7.0%** |
| DRAM read | 56.96 MB | 56.28 MB |
| Amplification vs 50.33 MB | **1.132×** | **1.118×** |
| DRAM % peak | **90.1%** | **91.9%** |
| Occupancy | 64.7% | 24.6% |

**Bound:** memory. ~90% of L4 DRAM peak, ~0.23 TFLOP/s. Occupancy is not the
time lever (already falsified: 16% vs 65% within 2 µs).

**Gap to +25% (≤168 µs):** need ~22% less time. FP16 unique-byte floor at
300 GB/s is 168 µs — only if amp ≈ 1.00 **and** DRAM ≈ 100%. Neither is there.

**Decisions this tick (all three levers still open — run three agents):**

1. **DRAM agent** — try to lift 90→96%+ without growing bytes (async copy / more
   independent loads / keep small grid). Expected: at most ~8–11% time if it works.
2. **AMP agent** — cut 1.13× toward 1.00× (alignment, no extra traffic).
   Expected: ~11% bytes if it works. Do not break full-row split-K without a
   reason in ncu sectors.
3. **FP8 agent** — pack weights to 8-bit + register dequant. This is the only
   lever that can **comfortably** clear +25% (floor ~84 µs if still DRAM-bound).

**Do not:** occupancy rewrites, L2-warm bench GB/s, official ncu on T4.

**Target miss:** min gain vs TRT = 3.9%. Continue.
