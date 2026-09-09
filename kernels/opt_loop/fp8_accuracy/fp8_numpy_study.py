#!/usr/bin/env python3
"""Numpy model of the E4M3 quantisation error on REAL Llama-3.2-3B MLP weights.

For each raw FP16 [K, N] blob it reports
  1. weight statistics (min/max/mean/std, |w| tail percentiles, max|w|/std,
     and the spread of per-output-column amax),
  2. cosine similarity of y = x @ W between the original FP16 weights and
     three E4M3 schemes:
       (a) per-tensor, NO scale          -- exactly what gemv_fp8.cu does today
       (b) per-tensor, ONE optimal scale
       (c) per-channel scale (one fp16 scale per output column n)

E4M3 is emulated with ml_dtypes.float8_e4m3fn.  ml_dtypes maps overflow to NaN
whereas CUDA's __nv_fp8_e4m3 constructor uses SATFINITE, so inputs are clamped
to +-448 before the cast; that makes the emulation match the kernel's packing
bit-for-bit (verified separately by kernels/opt_loop/fp8_accuracy/e4m3_selftest).

Reference dot products accumulate in float64.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np

E4M3 = ml_dtypes.float8_e4m3fn
E4M3_MAX = 448.0


def quant_e4m3(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even cast to E4M3 with SATFINITE clamping, back to f64."""
    return np.clip(a, -E4M3_MAX, E4M3_MAX).astype(np.float32).astype(E4M3).astype(np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def best_scale(w: np.ndarray, axis=None, n_probe: int = 41) -> np.ndarray:
    """Scale s minimising ||w - s*E4M3(w/s)||^2, searched around the amax scale.

    axis=None -> single scalar for the whole tensor; axis=0 -> one per column.
    """
    amax = np.max(np.abs(w), axis=axis, keepdims=(axis is not None))
    base = np.maximum(amax, 1e-12) / E4M3_MAX
    mults = np.linspace(0.5, 1.5, n_probe)
    best_s = base.copy()
    best_err = np.full(base.shape, np.inf)
    for m in mults:
        s = base * m
        err = (w - s * quant_e4m3(w / s)) ** 2
        err = err.sum() if axis is None else err.sum(axis=axis, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_s = np.where(better, s, best_s)
    return best_s


def stats(w16: np.ndarray) -> dict:
    w = w16.astype(np.float64)
    aw = np.abs(w)
    colmax = np.max(aw, axis=0)  # amax per output column n
    return {
        "min": float(w.min()),
        "max": float(w.max()),
        "mean": float(w.mean()),
        "std": float(w.std()),
        "amax": float(aw.max()),
        "p50_abs": float(np.percentile(aw, 50)),
        "p99_abs": float(np.percentile(aw, 99)),
        "p999_abs": float(np.percentile(aw, 99.9)),
        "p9999_abs": float(np.percentile(aw, 99.99)),
        "amax_over_std": float(aw.max() / w.std()),
        "kurtosis": float(((w - w.mean()) ** 4).mean() / w.var() ** 2),
        "colmax_min": float(colmax.min()),
        "colmax_max": float(colmax.max()),
        "colmax_median": float(np.median(colmax)),
        "colmax_ratio_max_over_min": float(colmax.max() / colmax.min()),
        "frac_below_e4m3_smallest_normal": float((aw < 0.015625).mean()),
    }


def analyse(path: Path, K: int, N: int, n_x: int, seed: int) -> dict:
    w16 = np.fromfile(path, dtype=np.float16, count=K * N).reshape(K, N)
    st = stats(w16)
    w = w16.astype(np.float64)

    schemes = {}
    # (a) no scale at all -- the shipped kernel
    schemes["no_scale"] = quant_e4m3(w)
    # (b) one optimal scale for the whole tensor
    s_t = best_scale(w, axis=None)
    schemes["per_tensor_scale"] = s_t * quant_e4m3(w / s_t)
    # (c) one scale per output column
    s_c = best_scale(w, axis=0)
    schemes["per_channel_scale"] = s_c * quant_e4m3(w / s_c)

    st["per_tensor_scale"] = float(np.asarray(s_t).ravel()[0])
    st["amax_scale"] = float(st["amax"] / E4M3_MAX)

    rng = np.random.default_rng(seed)
    out = {name: [] for name in schemes}
    rel_fro = {}
    for name, wq in schemes.items():
        rel_fro[name] = float(np.linalg.norm(wq - w) / np.linalg.norm(w))
    for _ in range(n_x):
        # same distribution the C harness draws: Uniform(-0.25, 0.25), stored fp16
        x16 = np.float16(rng.uniform(-0.25, 0.25, size=K))
        x = x16.astype(np.float64)
        y_ref = x @ w
        for name, wq in schemes.items():
            out[name].append(cosine(x @ wq, y_ref))

    return {
        "path": str(path),
        "K": K,
        "N": N,
        "stats": st,
        "rel_fro_weight_err": rel_fro,
        "cos_mean": {k: float(np.mean(v)) for k, v in out.items()},
        "cos_min": {k: float(np.min(v)) for k, v in out.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n-x", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    recs = json.loads(args.manifest.read_text())
    results = []
    for r in recs:
        res = analyse(Path(r["path"]), r["K"], r["N"], args.n_x, args.seed)
        res["layer"], res["proj"] = r["layer"], r["proj"]
        results.append(res)
        s = res["stats"]
        print(f"\n=== L{r['layer']:02d} {r['proj']}  K={r['K']} N={r['N']} ===")
        print(f"  min={s['min']:+.6f} max={s['max']:+.6f} mean={s['mean']:+.3e} std={s['std']:.6f}")
        print(f"  |w|: p50={s['p50_abs']:.6f} p99={s['p99_abs']:.6f} "
              f"p99.9={s['p999_abs']:.6f} p99.99={s['p9999_abs']:.6f} amax={s['amax']:.6f}")
        print(f"  amax/std={s['amax_over_std']:.2f}  kurtosis={s['kurtosis']:.2f}  "
              f"frac |w|<2^-6={s['frac_below_e4m3_smallest_normal']*100:.2f}%")
        print(f"  per-column amax: min={s['colmax_min']:.5f} med={s['colmax_median']:.5f} "
              f"max={s['colmax_max']:.5f} (max/min={s['colmax_ratio_max_over_min']:.2f}x)")
        for k in ("no_scale", "per_tensor_scale", "per_channel_scale"):
            print(f"  {k:<18} cos_mean={res['cos_mean'][k]:.8f} "
                  f"cos_min={res['cos_min'][k]:.8f}  rel||dW||={res['rel_fro_weight_err'][k]:.5f}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
