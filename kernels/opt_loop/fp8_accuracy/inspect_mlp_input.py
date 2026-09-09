#!/usr/bin/env python3
"""Locate the ops feeding each MLP projection in the clean Llama-3.2-3B ONNX.

The MLP input is the output of `post_attention_layernorm` (RMSNorm). optimum
exports RMSNorm decomposed into standard ops, so the learned per-channel gain
is a plain Mul initializer somewhere upstream of gate_proj/up_proj. This walks
the graph backwards from the projection MatMul to find it, so the activation
model in make_real_activations.py uses the model's own gain vector rather than
a guessed distribution.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import onnx

PAT = re.compile(r"^/model/layers\.(\d+)/mlp/(gate_proj|up_proj|down_proj)/MatMul$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    model = onnx.load(str(args.onnx), load_external_data=False)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    producer = {}
    for node in g.node:
        for out in node.output:
            producer[out] = node

    targets = {}
    for node in g.node:
        m = PAT.match(node.name or "")
        if m and int(m.group(1)) == args.layer:
            targets[m.group(2)] = node

    for proj in ("gate_proj", "up_proj", "down_proj"):
        node = targets.get(proj)
        if node is None:
            continue
        print(f"\n=== layers.{args.layer}.mlp.{proj} ===")
        # input[0] is the activation, input[1] the weight initializer
        act = node.input[0]
        for w in node.input:
            if w in inits:
                init = inits[w]
                print(f"  weight init : {init.name} dims={list(init.dims)}")
        seen = set()
        frontier = [(act, 0)]
        while frontier:
            name, d = frontier.pop(0)
            if d > args.depth or name in seen:
                continue
            seen.add(name)
            p = producer.get(name)
            if p is None:
                print(f"  {'  ' * d}<- {name} (graph input / initializer)")
                continue
            ini = [i for i in p.input if i in inits]
            extra = ""
            if ini:
                dims = [list(inits[i].dims) for i in ini]
                extra = f"  init={ini} dims={dims}"
            print(f"  {'  ' * d}<- {p.op_type:22s} {p.name}{extra}")
            for i in p.input:
                if i not in inits:
                    frontier.append((i, d + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
