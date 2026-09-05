#!/usr/bin/env python3
"""Parse an ncu --csv capture of one GEMV launch into loop metrics."""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

UNIQUE_FP16 = 50331648.0
TRT_US = {"up_proj": 223.9, "down_proj": 223.6}
GEMV_US = {"up_proj": 215.6, "down_proj": 208.9}

METRIC_ALIASES = {
    "dram__bytes_read.sum": "dram_bytes",
    "gpu__time_duration.sum": "duration_ns",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "lts__t_sectors_srcunit_tex_op_read.sum": "l2_sectors",
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
        "gain_vs_trt_pct": round(gain_trt, 2) if gain_trt is not None else None,
        "gain_vs_gemv_fp16_pct": round(gain_gemv, 2) if gain_gemv is not None else None,
        "hit_25pct_vs_trt": bool(gain_trt is not None and gain_trt >= 25.0),
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
