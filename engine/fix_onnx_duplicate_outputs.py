#!/usr/bin/env python3
"""Fix ONNX graphs exported by ONNX Runtime's transformer optimizer where
multiple nodes (e.g. fused LayerNorm variants used alongside GroupQueryAttention)
leave optional trailing outputs as an empty string "".

ONNX allows "" in a node's output list to mean "this optional output slot is
unused". But TensorRT's ONNX parser treats every output name as a unique key
across the whole graph, so many nodes sharing the same "" output collide and
topological sort fails with:

  ERROR: Output name is not unique:
  Assertion failed: toposort(graph.node(), &topoOrder) &&
      "Failed to sort the model topologically."

This script gives every empty output a synthetic unique name instead. Since
these outputs were never consumed by anything (that's why they were left
empty), this is safe: it does not change model behavior, only lets
TensorRT's parser build a valid dependency graph.

Only the graph structure (model_fp16.onnx) is modified; the external weight
shards (model_fp16.onnx_data*) are untouched and the fixed .onnx keeps
referencing them by relative path, so it must stay in the same directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models/onnx-fp16/onnx/model_fp16.onnx",
        help="Path to the source ONNX graph (default: models/onnx-fp16/onnx/model_fp16.onnx)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to <name>_fixed.onnx next to the input, "
        "so it stays alongside the existing onnx_data* shards.",
    )
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"Error: ONNX file not found: {args.onnx}", file=sys.stderr)
        return 1

    out_path = args.out or args.onnx.with_name(args.onnx.stem + "_fixed.onnx")

    import onnx

    # load_external_data=False: we only rewrite node.output names, so there's
    # no need to load multi-GB weights into memory. External data references
    # in the initializers are left untouched.
    model = onnx.load(str(args.onnx), load_external_data=False)

    renamed = 0
    op_type_counts: dict[str, int] = {}
    for node in model.graph.node:
        for i, output in enumerate(node.output):
            if output == "":
                node.output[i] = f"__unused_output_{renamed}__"
                renamed += 1
                op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1

    print(f"Renamed {renamed} empty ('') output names to unique placeholders.")
    if op_type_counts:
        print("By op type:")
        for op_type, count in sorted(op_type_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {op_type}: {count}")

    if renamed == 0:
        print("No empty output names found — graph may already be fixed, or the")
        print("failure has a different cause. Re-check the trtexec error message.")

    onnx.save(model, str(out_path))
    print(f"\nSaved: {out_path}")
    print(f"(external weight shards stay referenced from: {args.onnx.parent})")
    print("\nNext: point ONNX= in engine/configs/baseline_decode.env at this file, then:")
    print("  ./engine/build_engine.sh decode")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
