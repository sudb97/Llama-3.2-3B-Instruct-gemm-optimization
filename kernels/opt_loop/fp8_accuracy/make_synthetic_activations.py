#!/usr/bin/env python3
"""Synthesize MLP input vectors matched to this model's captured decode stats.

gate/up_proj see post-RMSNorm residual (near-Gaussian; Transformer Circuits /
Elhage et al.). down_proj sees SiLU(gate)*up, which is heavy-tailed. Moments
are taken from capture_real_activations.py (real Llama-3.2-3B-Instruct ONNX
on a synthetic instruct prompt, last-token = decode-equivalent).

x is independent of W. Seeded so the GPU check is reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def student_t_nu(kurtosis: float) -> float:
    # kurtosis of t_nu is 3 + 6/(nu-4) for nu>4. Clamp to a valid df.
    k = max(float(kurtosis), 3.05)
    nu = 4.0 + 6.0 / (k - 3.0)
    return float(np.clip(nu, 4.05, 30.0))


def draw(kind: str, K: int, rms: float, kurtosis: float, rng: np.random.Generator) -> np.ndarray:
    if kind in ("gate_proj", "up_proj"):
        x = rng.normal(0.0, 1.0, size=K)
    else:
        x = rng.standard_t(student_t_nu(kurtosis), size=K)
    cur = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    if cur == 0.0:
        return np.zeros(K, dtype=np.float16)
    x = x * (rms / cur)
    return x.astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    recs = json.loads(args.manifest.read_text())
    rng = np.random.default_rng(args.seed)
    out = []
    for rec in recs:
        x = draw(rec["proj"], int(rec["K"]), float(rec["rms"]), float(rec["kurtosis"]), rng)
        dst = args.out / Path(rec["path"]).name.replace(".x.fp16.bin", ".synth.x.fp16.bin")
        dst.write_bytes(x.tobytes())
        f = x.astype(np.float64)
        a = np.abs(f)
        row = {
            "layer": rec["layer"],
            "proj": rec["proj"],
            "K": rec["K"],
            "path": str(dst),
            "matched_capture": rec["path"],
            "target_rms": rec["rms"],
            "target_kurtosis": rec["kurtosis"],
            "rms": float(np.sqrt((f**2).mean())),
            "absmax": float(a.max()),
            "kurtosis": float(((f - f.mean()) ** 4).mean() / f.var() ** 2),
            "family": "gaussian" if rec["proj"] != "down_proj" else "student_t",
        }
        out.append(row)
        print(
            f"L{rec['layer']:02d} {rec['proj']:<10} {row['family']:<10} "
            f"rms {rec['rms']:.4f}->{row['rms']:.4f} "
            f"kurt {rec['kurtosis']:.1f}->{row['kurtosis']:.1f}"
        )
    (args.out / "synth_x_manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
