#!/usr/bin/env python3
"""Parse sign-off ncu captures and apply the measurement-discipline checks.

Beyond the numbers parse_ncu.py already reports, this enforces the guard that
WORKFLOW.md mandates for an official capture:

    dram__bytes_read.sum  ==  lts__t_sectors_aperture_device_lookup_miss.sum * 32

`dram__bytes_read.sum` is device-wide, the L2-miss counter is per-kernel, so a
ratio above 1.0 means DRAM traffic that this kernel did not cause. That check is
what invalidated the July-2026 TensorRT reference, so no number here is quoted
without it.

Multiple CSVs for the same configuration are aggregated so the report can state
a spread rather than a single sample.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path

SECTOR_BYTES = 32

ALIASES = {
    "dram__bytes_read.sum": "dram_read",
    "dram__bytes_write.sum": "dram_write",
    "lts__t_sectors_srcunit_tex_op_read.sum": "l2_tex_read_sectors",
    "lts__t_sectors_aperture_device_lookup_miss.sum": "l2_miss_sectors",
    "gpu__time_duration.sum": "duration_ns",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma",
    "launch__grid_size": "grid",
    "launch__block_size": "block",
    "gpc__cycles_elapsed.avg.per_second": "gpc_hz",
}


def parse_csv(path: Path) -> list[dict]:
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    if not lines:
        raise SystemExit(f"{path}: no ncu CSV rows")
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    per_launch: dict[str, dict] = {}
    for row in reader:
        name = (row.get("Metric Name") or "").strip()
        val = (row.get("Metric Value") or "").strip().replace(",", "")
        if name not in ALIASES or not val:
            continue
        key = f"{row.get('ID')}|{row.get('Kernel Name')}"
        rec = per_launch.setdefault(
            key, {"kernel": row.get("Kernel Name"), "cc": row.get("CC")}
        )
        try:
            rec[ALIASES[name]] = float(val)
        except ValueError:
            rec[ALIASES[name]] = val
    return list(per_launch.values())


def derive(rec: dict, unique_bytes: float, unique_fp16: float) -> dict:
    out = dict(rec)
    ns = rec.get("duration_ns")
    if ns:
        out["duration_us"] = ns / 1000.0
    dr = rec.get("dram_read")
    if dr:
        out["dram_read_mb"] = dr / 1e6
        out["amp_vs_unique"] = dr / unique_bytes
        out["amp_vs_fp16"] = dr / unique_fp16
        if ns:
            out["dram_read_gbs"] = dr / (ns * 1.0)  # bytes/ns == GB/s
    miss = rec.get("l2_miss_sectors")
    if miss:
        out["l2_miss_bytes"] = miss * SECTOR_BYTES
        if dr:
            out["dram_over_l2miss"] = dr / (miss * SECTOR_BYTES)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--unique-bytes", type=float, required=True)
    ap.add_argument("--unique-fp16", type=float, default=50331648.0)
    ap.add_argument("--kernel-filter", default="")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    launches = []
    for p in args.csvs:
        for rec in parse_csv(p):
            if args.kernel_filter and args.kernel_filter not in (rec.get("kernel") or ""):
                continue
            launches.append(derive(rec, args.unique_bytes, args.unique_fp16))
    if not launches:
        raise SystemExit("no launches matched")

    def vals(k):
        return [l[k] for l in launches if k in l and isinstance(l[k], float)]

    def agg(k):
        v = vals(k)
        if not v:
            return None
        return {
            "n": len(v),
            "mean": statistics.fmean(v),
            "min": min(v),
            "max": max(v),
            "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
        }

    summary = {
        "label": args.label,
        "kernel": launches[0].get("kernel"),
        "cc": launches[0].get("cc"),
        "n_launches": len(launches),
        "unique_bytes": args.unique_bytes,
    }
    for k in (
        "duration_us", "dram_read_mb", "dram_write", "amp_vs_unique",
        "amp_vs_fp16", "dram_pct", "occupancy_pct", "dram_read_gbs",
        "dram_over_l2miss", "ffma", "grid", "block", "gpc_hz",
    ):
        a = agg(k)
        if a:
            summary[k] = a

    print(f"=== {args.label} ===")
    print(f"  kernel        : {summary['kernel']}")
    print(f"  cc            : {summary['cc']}   launches: {summary['n_launches']}")
    for k, unit, prec in (
        ("duration_us", "us", 3),
        ("dram_read_mb", "MB", 4),
        ("dram_read_gbs", "GB/s", 2),
        ("amp_vs_unique", "x", 4),
        ("amp_vs_fp16", "x", 4),
        ("dram_pct", "%", 2),
        ("occupancy_pct", "%", 2),
        ("dram_over_l2miss", "ratio", 4),
        ("ffma", "", 0),
        ("grid", "", 0),
        ("block", "", 0),
    ):
        a = summary.get(k)
        if not a:
            continue
        print(
            f"  {k:<17}: mean={a['mean']:.{prec}f} min={a['min']:.{prec}f} "
            f"max={a['max']:.{prec}f} sd={a['sd']:.{prec}f} {unit}"
        )

    ratio = summary.get("dram_over_l2miss")
    if ratio:
        clean = abs(ratio["max"] - 1.0) <= 0.005
        print(
            f"  IDLE-GPU GUARD : dram/(L2miss*32) max={ratio['max']:.4f} -> "
            f"{'CLEAN' if clean else 'CONTAMINATED'}"
        )
    else:
        print("  IDLE-GPU GUARD : L2-miss counter absent -- cannot verify")

    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "launches": launches}, indent=2) + "\n")
        print(f"  [ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
