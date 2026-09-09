#!/usr/bin/env python3
"""Cosine of E4M3 GEMV vs FP16, real W + synthetic x (no GPU)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np

E4M3 = ml_dtypes.float8_e4m3fn
E4M3_MAX = 448.0


def quant(a):
    return np.clip(a, -E4M3_MAX, E4M3_MAX).astype(np.float32).astype(E4M3).astype(np.float64)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--synth-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    wman = { (r["layer"], r["proj"]): r for r in json.loads(args.weights.read_text()) }
    sm = json.loads(args.synth_manifest.read_text())
    rows = []
    for s in sm:
        key = (s["layer"], s["proj"])
        if key not in wman:
            continue
        wr = wman[key]
        K, N = wr["K"], wr["N"]
        w = np.fromfile(wr["path"], dtype=np.float16, count=K * N).reshape(K, N).astype(np.float64)
        x = np.fromfile(s["path"], dtype=np.float16, count=K).astype(np.float64)
        ref = x @ w
        y = x @ quant(w)
        rec = {
            "layer": s["layer"],
            "proj": s["proj"],
            "family": s["family"],
            "cos_none": cosine(y, ref),
            "x_rms": s["rms"],
            "x_kurtosis": s["kurtosis"],
            "w_absmax": float(np.abs(w).max()),
        }
        rows.append(rec)
        print(
            f"L{s['layer']:02d} {s['proj']:<10} {s['family']:<10} "
            f"cos_fp16={rec['cos_none']:.8f} rms={s['rms']:.4f} kurt={s['kurtosis']:.1f}"
        )
    c = np.array([r["cos_none"] for r in rows])
    print(
        f"\n=== synthetic-x / real-W  n={len(rows)}  "
        f"min={c.min():.8f} med={np.median(c):.8f} all>=0.999={bool((c>=0.999).all())}"
    )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
