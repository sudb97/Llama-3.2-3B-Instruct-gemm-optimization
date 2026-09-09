#!/usr/bin/env python3
"""Sweep the no-scale E4M3 cosine over EVERY MLP projection of Llama-3.2-3B.

The 12-tensor study in fp8_numpy_study.py could have missed a bad layer, so
this streams all 84 gate/up/down tensors straight out of model.onnx_data (seek
+ read at the offsets ONNX declares) and reports the no-scale cosine for each,
plus the worst layer.  It also re-runs the worst tensor under three activation
distributions, because the cosine is a property of (W, x) and the harness's
Uniform(-0.25, 0.25) x could in principle flatter the result.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx

E4M3 = ml_dtypes.float8_e4m3fn
E4M3_MAX = 448.0
PAT = re.compile(r"^/model/layers\.(\d+)/mlp/(gate_proj|up_proj|down_proj)/MatMul$")


def quant_e4m3(a):
    return np.clip(a, -E4M3_MAX, E4M3_MAX).astype(np.float32).astype(E4M3).astype(np.float64)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def draw_x(kind, K, rng):
    if kind == "uniform":  # what the C harness uses
        return np.float16(rng.uniform(-0.25, 0.25, size=K)).astype(np.float64)
    if kind == "gauss":
        return np.float16(rng.normal(0.0, 0.14, size=K)).astype(np.float64)
    if kind == "heavy":  # Student-t(3): a few activation outliers dominate
        return np.float16(rng.standard_t(3.0, size=K) * 0.08).astype(np.float64)
    if kind == "onehot":  # adversarial: y is a single weight row, no K-averaging
        v = np.zeros(K)
        v[rng.integers(K)] = 1.0
        return v
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
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
        tensors.append((int(m.group(1)), m.group(2), args.onnx.parent / ext["location"],
                        int(ext["offset"]), K, N))

    rng = np.random.default_rng(args.seed)
    xs = {}
    rows = []
    for layer, proj, src, off, K, N in sorted(tensors):
        with open(src, "rb") as f:
            f.seek(off)
            w = np.frombuffer(f.read(K * N * 2), dtype=np.float16).reshape(K, N).astype(np.float64)
        if K not in xs:
            xs[K] = draw_x("uniform", K, rng)
        x = xs[K]
        c = cosine(x @ quant_e4m3(w), x @ w)
        rows.append((layer, proj, K, N, c))
        print(f"L{layer:02d} {proj:<10} K={K:<5} N={N:<5} cos_fp16_noscale={c:.8f}"
              f"{'  <-- BELOW 0.999' if c < 0.999 else ''}", flush=True)

    rows.sort(key=lambda r: r[4])
    print("\n--- worst 5 of %d tensors ---" % len(rows))
    for layer, proj, K, N, c in rows[:5]:
        print(f"  L{layer:02d} {proj:<10} cos={c:.8f}")
    print(f"  min={min(r[4] for r in rows):.8f}  median={np.median([r[4] for r in rows]):.8f}"
          f"  max={max(r[4] for r in rows):.8f}")
    print(f"  ALL 84 >= 0.999 : {all(r[4] >= 0.999 for r in rows)}")

    # Activation-distribution sensitivity on the worst tensor.
    layer, proj, K, N, _ = rows[0]
    src, off = next((s, o) for (l, p, s, o, _K, _N) in tensors if l == layer and p == proj)
    with open(src, "rb") as f:
        f.seek(off)
        w = np.frombuffer(f.read(K * N * 2), dtype=np.float16).reshape(K, N).astype(np.float64)
    wq = quant_e4m3(w)
    print(f"\n--- activation sensitivity on worst tensor L{layer:02d} {proj} ---")
    for kind in ("uniform", "gauss", "heavy", "onehot"):
        cs = []
        for _ in range(16):
            x = draw_x(kind, K, rng)
            cs.append(cosine(x @ wq, x @ w))
        print(f"  x~{kind:<8} cos mean={np.mean(cs):.8f} min={np.min(cs):.8f} max={np.max(cs):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
