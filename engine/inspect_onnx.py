#!/usr/bin/env python3
"""Print ONNX model inputs/outputs and suggested trtexec shape flags."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Llama ONNX model I/O")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models/onnx-fp16-clean/model.onnx",
        help="Path to the ONNX graph (default: the clean optimum export)",
    )
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"Error: ONNX file not found: {args.onnx}", file=sys.stderr)
        print("Run: ./engine/download_model.sh", file=sys.stderr)
        return 1

    import onnx

    model = onnx.load(str(args.onnx), load_external_data=False)

    def fmt_shape(proto) -> str:
        dims = []
        for d in proto.type.tensor_type.shape.dim:
            if d.dim_param:
                dims.append(d.dim_param)
            elif d.dim_value:
                dims.append(str(d.dim_value))
            else:
                dims.append("?")
        return "x".join(dims) if dims else "scalar"

    print(f"Model: {args.onnx}")
    print(f"Opset: {model.opset_import[0].version if model.opset_import else 'unknown'}")
    print()

    print("=== Inputs ===")
    for inp in model.graph.input:
        print(f"  {inp.name}: {fmt_shape(inp)}")

    print()
    print("=== Outputs ===")
    for out in model.graph.output:
        print(f"  {out.name}: {fmt_shape(out)}")

    print()
    print("=== Notes ===")
    print("- Update MIN_SHAPES / OPT_SHAPES / MAX_SHAPES in engine/configs/baseline_decode.env")
    print("  if input names differ from the defaults.")
    print("- model_fp16.onnx_data must stay in the same directory as model_fp16.onnx.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
