#!/usr/bin/env python3
"""Parse an ncu --csv capture of one GEMV launch into loop metrics."""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

UNIQUE_FP16 = 50331648.0
# CORRECTED 2026-09-07. The old 223.9 / 223.6 pair was captured under GPU
# contention and is overstated by ~20%. Re-measured twice on a verified-idle
# L4 (agent 187.27 / 181.60, parent 186.88 / 182.17), amplification 1.0006 /
# 1.0007 -- not the 1.204 / 1.210 on record. See TRT_RECHECK.md.
TRT_US = {"up_proj": 187.1, "down_proj": 181.9}
TRT_US_SUPERSEDED = {"up_proj": 223.9, "down_proj": 223.6}
# Corrected 2026-09-07: mean of 3 re-measures of the unmodified kernel.
# The old 215.6/208.9 pair came from captures showing amp ~1.13x that no
# later run reproduces (see EXPERIMENT_LOG.md).
GEMV_US = {"up_proj": 192.4, "down_proj": 187.9}

# Two agreed bars (user decision 2026-09-07):
#   PASS    - gain = trt/us - 1 >= 25%  ->  us <= trt / 1.25
#   STRETCH - 25% less time             ->  us <= trt * 0.75
# Against the CORRECTED reference both bars fall BELOW the 167.8 us FP16
# roofline floor (PASS 149.7 / 145.5), so neither is reachable in FP16 at all.
# Only FP8 can meet them. The two definitions are no longer meaningfully apart.
PASS_US = {k: v / 1.25 for k, v in TRT_US.items()}
STRETCH_US = {k: v * 0.75 for k, v in TRT_US.items()}

METRIC_ALIASES = {
    "dram__bytes_read.sum": "dram_bytes",
    "gpu__time_duration.sum": "duration_ns",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "lts__t_sectors_srcunit_tex_op_read.sum": "l2_sectors",
    "lts__t_sectors_aperture_device_lookup_miss.sum": "l2_miss_sectors",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma",
}


def parse_csv_text(text: str) -> dict:
    # ncu prints banner lines before the CSV header
    lines = [ln for ln in text.splitlines() if ln.startswith('"ID"') or ln.startswith('"0"')]
    if not lines:
        raise SystemExit("no ncu CSV rows found")
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    out = {}
    kernel = None
    cc = None
    for row in reader:
        kernel = row.get("Kernel Name") or kernel
        cc = row.get("CC") or cc
        name = (row.get("Metric Name") or "").strip()
        val = (row.get("Metric Value") or "").strip().replace(",", "")
        if name in METRIC_ALIASES and val:
            out[METRIC_ALIASES[name]] = float(val)
    out["kernel"] = kernel
    out["cc"] = cc
    return out


def summarize(raw: dict, shape: str, unique_bytes: float) -> dict:
    dram = raw.get("dram_bytes")
    ns = raw.get("duration_ns")
    us = ns / 1000.0 if ns is not None else None
    amp = dram / unique_bytes if dram else None
    trt = TRT_US[shape]
    gemv = GEMV_US[shape]
    gain_trt = (trt / us - 1.0) * 100.0 if us else None
    gain_gemv = (gemv / us - 1.0) * 100.0 if us else None
    return {
        "shape": shape,
        "kernel": raw.get("kernel"),
        "cc": raw.get("cc"),
        "duration_us": round(us, 3) if us else None,
        "dram_bytes": int(dram) if dram else None,
        "dram_mb": round(dram / 1e6, 3) if dram else None,
        "amplification": round(amp, 4) if amp else None,
        "dram_pct": raw.get("dram_pct"),
        "occupancy_pct": raw.get("occupancy_pct"),
        "ffma": raw.get("ffma"),
        "l2_sectors": raw.get("l2_sectors"),
        "l2_miss_sectors": raw.get("l2_miss_sectors"),
        "dram_over_l2_miss": (
            round(dram / (raw["l2_miss_sectors"] * 32.0), 4)
            if dram and raw.get("l2_miss_sectors")
            else None
        ),
        "gain_vs_trt_pct": round(gain_trt, 2) if gain_trt is not None else None,
        "gain_vs_gemv_fp16_pct": round(gain_gemv, 2) if gain_gemv is not None else None,
        "hit_25pct_vs_trt": bool(gain_trt is not None and gain_trt >= 25.0),
        "pass_bar_us": round(PASS_US[shape], 2),
        "stretch_bar_us": round(STRETCH_US[shape], 2),
        "hit_pass_bar": bool(us is not None and us <= PASS_US[shape]),
        "hit_stretch_bar": bool(us is not None and us <= STRETCH_US[shape]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--shape", required=True, choices=["up_proj", "down_proj"])
    ap.add_argument("--unique-bytes", type=float, default=UNIQUE_FP16)
    ap.add_argument("--out", help="write JSON here")
    args = ap.parse_args()
    text = Path(args.csv_path).read_text()
    raw = parse_csv_text(text)
    summary = summarize(raw, args.shape, args.unique_bytes)
    blob = json.dumps(summary, indent=2)
    print(blob)
    if args.out:
        Path(args.out).write_text(blob + "\n")
    if summary.get("cc") and summary["cc"] != "8.9":
        print(f"NOTE: compute capability {summary['cc']} (L4 baseline is sm_89)", file=sys.stderr)


if __name__ == "__main__":
    main()
