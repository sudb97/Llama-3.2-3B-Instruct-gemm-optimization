#!/usr/bin/env python3
"""E4M3 accuracy over all 84 Llama-3.2-3B MLP projections, using REAL activations.

Supersedes fp8_all_layers.py, which drew x from generic synthetic distributions.
Here x is the tensor the model actually feeds each projection, captured from a
real forward pass by capture_real_activations.py. That matters: the down_proj
input is SiLU(gate)*up, whose kurtosis reaches ~1900, whereas the old harness
used Uniform(-0.25, 0.25) with kurtosis 1.8.

Weights are streamed from model.onnx_data at the offsets ONNX declares and are
never written to disk; each tensor is read, scored, and released.

For every tensor it reports cosine(y_fp8, y_fp16) with fp64 accumulation under
three scale schemes:
  none    - exactly what gemv_fp8.cu ships (no scale tensor at all)
  tensor  - one scalar for the whole matrix, folded into the epilogue
  channel - one fp16 scale per output column, folded into the epilogue
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx

E4M3 = ml_dtypes.float8_e4m3fn
E4M3_MAX = 448.0
PAT = re.compile(r"^/model/layers\.(\d+)/mlp/(gate_proj|up_proj|down_proj)/MatMul$")


def quant_e4m3(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even cast to E4M3 with SATFINITE clamping, back to f64."""
    return np.clip(a, -E4M3_MAX, E4M3_MAX).astype(np.float32).astype(E4M3).astype(np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def snr_db(got: np.ndarray, ref: np.ndarray) -> float:
    err = got - ref
    return float(10.0 * np.log10((ref @ ref) / (err @ err)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    model = onnx.load(str(args.onnx), load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    tensors = []
    for node in model.graph.node:
        m = PAT.match(node.name or "")
        if not m:
            continue
        init = next(inits[i] for i in node.input if i in inits)
        ext = {d.key: d.value for d in init.external_data}
        K, N = list(init.dims)
        tensors.append(
            (int(m.group(1)), m.group(2), args.onnx.parent / ext["location"],
             int(ext["offset"]), int(K), int(N))
        )

    rows = []
    for layer, proj, src, off, K, N in sorted(tensors):
        xp = args.acts / f"L{layer:02d}.{proj}.K{K}.x.fp16.bin"
        x = np.frombuffer(xp.read_bytes(), dtype=np.float16).astype(np.float64)
        assert x.shape == (K,), f"{xp}: got {x.shape}, want ({K},)"

        with open(src, "rb") as f:
            f.seek(off)
            w = np.frombuffer(f.read(K * N * 2), dtype=np.float16).reshape(K, N).astype(np.float64)

        ref = x @ w

        # (a) no scale -- the shipped kernel
        y_none = x @ quant_e4m3(w)

        # (b) per-tensor amax/448
        s_t = max(float(np.abs(w).max()), 1e-12) / E4M3_MAX
        y_tensor = (x @ quant_e4m3(w / s_t)) * s_t

        # (c) per-channel amax/448, stored fp16 (folds into the epilogue)
        s_c = np.maximum(np.abs(w).max(axis=0), 1e-12) / E4M3_MAX
        s_c = np.float16(s_c).astype(np.float64)
        y_chan = (x @ quant_e4m3(w / s_c)) * s_c

        aw = np.abs(w)
        rec = {
            "layer": layer,
            "proj": proj,
            "K": K,
            "N": N,
            "w_absmax": float(aw.max()),
            "w_std": float(w.std()),
            "w_frac_subnormal": float((aw < 0.015625).mean()),
            "w_n_saturating": int((aw > E4M3_MAX).sum()),
            "x_rms": float(np.sqrt((x**2).mean())),
            "x_kurtosis": float(((x - x.mean()) ** 4).mean() / x.var() ** 2),
            "cos_none": cosine(y_none, ref),
            "cos_tensor": cosine(y_tensor, ref),
            "cos_channel": cosine(y_chan, ref),
            "rel_l2_none": rel_l2(y_none, ref),
            "snr_db_none": snr_db(y_none, ref),
            "snr_db_channel": snr_db(y_chan, ref),
        }
        rows.append(rec)
        print(
            f"L{layer:02d} {proj:<10} K={K:<5} N={N:<5} "
            f"cos_none={rec['cos_none']:.8f} cos_chan={rec['cos_channel']:.8f} "
            f"relL2={rec['rel_l2_none']:.5f} SNR={rec['snr_db_none']:.1f}dB "
            f"x_kurt={rec['x_kurtosis']:.0f}"
            f"{'  <-- BELOW 0.999' if rec['cos_none'] < 0.999 else ''}",
            flush=True,
        )
        del w

    def col(name):
        return np.array([r[name] for r in rows])

    print(f"\n=== summary over {len(rows)} tensors (REAL weights, REAL activations) ===")
    for scheme in ("none", "tensor", "channel"):
        c = col(f"cos_{scheme}")
        print(
            f"  scale={scheme:<8} cos min={c.min():.8f} median={np.median(c):.8f} "
            f"max={c.max():.8f}  all>=0.999: {bool((c >= 0.999).all())}"
        )
    print(f"  rel L2 (no scale) max={col('rel_l2_none').max():.5f}")
    print(f"  SNR   (no scale) min={col('snr_db_none').min():.2f} dB")
    print(f"  SNR   (per-chan) min={col('snr_db_channel').min():.2f} dB")
    print(f"  weights that saturate E4M3 (|w|>448): {int(col('w_n_saturating').sum())}")

    worst = sorted(rows, key=lambda r: r["cos_none"])[:5]
    print("\n--- worst 5 by cos_none ---")
    for r in worst:
        print(
            f"  L{r['layer']:02d} {r['proj']:<10} cos_none={r['cos_none']:.8f} "
            f"cos_chan={r['cos_channel']:.8f} x_kurt={r['x_kurtosis']:.0f}"
        )

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\n[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
