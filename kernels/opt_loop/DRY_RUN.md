# Dry-run verdict (T4, unofficial) — one tick

**Host:** Tesla T4 (sm_75). Docker not installed. `official: false`.
`dry_run.sh`: **7 passed, 0 failed**. Did **not** append `EXPERIMENT_LOG.md`.
Did **not** overwrite `VERDICT.md`.

## Plumbing

| check | result |
|---|---|
| `start_on_l4.sh` | exit 2 |
| `measure.sh` | exit 2 before Docker (`HOST_GPU=Tesla T4`) |
| `parse_ncu.py` on L4 `gemv_*_ncu.csv` | up **+3.87%**, down **+7.04%**, cc **8.9** |
| dummy `nvcc -arch=sm_75` + ncu + parse | CSV rows present; parser warned **CC 7.5 ≠ sm_89** |

## Parent vet of dummy ncu (do not rubber-stamp the parser)

Hand-checked `runs/dry_run_t4/dummy_ncu.csv`:

- Kernel is `dummy_scale` (4096 floats, 1×256), **not** GEMV.
- CC **7.5**, duration **3.808 µs**, DRAM read **3488 B**, DRAM% **0.30**, FFMA **0**.
- `parse_ncu.py` reports `hit_25pct_vs_trt: true` and `gain_vs_trt_pct: 5779`. **Reject.** That is a tiny memcpy-scale kernel compared to TRT 223.9 µs, not a 25% win.

**Decision:** workflow OK. Dummy timings must not enter `VERDICT.md` / `EXPERIMENT_LOG.md`. Optimization still uses L4 tick-0 ncu (min gain 3.9%). No DRAM/AMP/FP8 agents on this tick.

## Dummy Task

One smoke agent only (no kernel edit). Receipt: `runs/dry_run_t4/AGENT_REPORT.md`.
