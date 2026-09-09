#!/usr/bin/env python3
"""Extract raw FP16 [K, N] MLP weight blobs from the Llama-3.2-3B ONNX export.

The ONNX graph stores every MLP projection as a MatMul initializer already in
[K, N] row-major FP16 with external data, so extraction is a plain seek + read
of `length` bytes from model.onnx_data -- no transpose, no dtype conversion.

Usage:
  PYTHONPATH=<numpy/onnx site> python3 extract_real_weights.py \
      --onnx  .../models/onnx-fp16-clean/model.onnx \
      --out   /workspace/.fp8_tools/weights \
      --layers 0 1 13 27 \
      --projs gate_proj up_proj down_proj
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import onnx

FP16 = 10  # onnx.TensorProto.FLOAT16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 13, 27])
    ap.add_argument(
        "--projs", nargs="+", default=["gate_proj", "up_proj", "down_proj"]
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.onnx), load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}

    # node name -> initializer, e.g. /model/layers.0/mlp/up_proj/MatMul
    want = {}
    pat = re.compile(r"^/model/layers\.(\d+)/mlp/(gate_proj|up_proj|down_proj)/MatMul$")
    for node in model.graph.node:
        m = pat.match(node.name or "")
        if not m:
            continue
        layer, proj = int(m.group(1)), m.group(2)
        if layer not in args.layers or proj not in args.projs:
            continue
        for inp in node.input:
            if inp in inits:
                want[(layer, proj)] = inits[inp]

    manifest = []
    data_path = args.onnx.parent
    for (layer, proj), init in sorted(want.items()):
        assert init.data_type == FP16, f"{init.name} is dtype {init.data_type}, not fp16"
        ext = {d.key: d.value for d in init.external_data}
        K, N = list(init.dims)
        nbytes = K * N * 2
        assert int(ext["length"]) == nbytes, "external length != K*N*2"
        src = data_path / ext["location"]
        dst = args.out / f"L{layer:02d}.{proj}.K{K}.N{N}.fp16.bin"
        with open(src, "rb") as f:
            f.seek(int(ext["offset"]))
            buf = f.read(nbytes)
        assert len(buf) == nbytes
        dst.write_bytes(buf)
        rec = {
            "layer": layer,
            "proj": proj,
            "init": init.name,
            "K": K,
            "N": N,
            "bytes": nbytes,
            "path": str(dst),
        }
        manifest.append(rec)
        print(f"wrote {dst}  ({nbytes} B)  from {init.name}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
