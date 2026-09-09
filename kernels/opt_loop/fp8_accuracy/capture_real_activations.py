#!/usr/bin/env python3
"""Capture the TRUE decode-time MLP input vectors from the real Llama-3.2-3B.

Rather than drawing x from a guessed distribution, this runs the actual ONNX
model on a real tokenized prompt and reads the tensors that physically feed the
projections:

  gate_proj / up_proj  <-  /model/layers.L/post_attention_layernorm/Mul_1
  down_proj            <-  /model/layers.L/mlp/Mul          (SiLU(gate) * up)

Both were confirmed by inspect_mlp_input.py.

Why a prefill pass gives a decode-time vector: attention is causal, so the
hidden state at the last position depends only on positions <= it. Running
seq_len=T with past=0 and taking position T-1 yields exactly the activation a
decode step with past=T-1 would see, without needing a two-phase KV run.

Outputs one fp16 [K] blob per (layer, proj), named so gemv_fp8's `checkreal`
can pair it with the matching weight blob.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from tokenizers import Tokenizer

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128

# The tensor feeding each projection, keyed by the producing node name.
NODE_FOR = {
    "gate_up": "/model/layers.{L}/post_attention_layernorm/Mul_1",
    "down": "/model/layers.{L}/mlp/Mul",
}

# A normal instruct-style exchange, so the captured activations come from the
# distribution the model actually runs in rather than from random token ids.
PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant that explains GPU performance engineering.\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "I am optimizing the decode-time MLP GEMM of a 3B parameter Llama model on "
    "an NVIDIA L4. Nsight Compute says the kernel sits at about ninety percent "
    "of peak DRAM bandwidth while the SMs are only fifteen percent busy, and "
    "the arithmetic intensity is roughly one FLOP per byte at batch size one. "
    "Explain why this projection is memory bound rather than compute bound, "
    "what the roofline model predicts for the achievable latency, and whether "
    "quantizing the weights to an eight bit floating point format would help."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "The projection is memory bound because at batch size one each weight is "
    "read from DRAM exactly once and used for a single multiply accumulate, so"
)


def build_graph_with_taps(src: Path, dst: Path, layers: list[int]) -> dict:
    """Re-emit the graph with the MLP input tensors promoted to graph outputs.

    External data is left untouched: the model is loaded without it and written
    beside the original .onnx_data, so ORT resolves the weights from the same
    shards the unmodified model uses.
    """
    model = onnx.load(str(src), load_external_data=False)
    by_name = {n.name: n for n in model.graph.node}

    taps = {}
    for L in layers:
        for kind, pat in NODE_FOR.items():
            nm = pat.format(L=L)
            node = by_name.get(nm)
            if node is None:
                raise SystemExit(f"node not found: {nm}")
            tensor = node.output[0]
            out_name = f"tap_L{L:02d}_{kind}"
            # Identity keeps the original tensor name intact for consumers.
            model.graph.node.append(
                helper.make_node("Identity", [tensor], [out_name], name=out_name)
            )
            model.graph.output.append(
                helper.make_tensor_value_info(out_name, TensorProto.FLOAT16, None)
            )
            taps[out_name] = {"layer": L, "kind": kind, "tensor": tensor}

    onnx.save(model, str(dst))
    return taps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 13, 27])
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-tokens", type=int, default=160)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(args.tokenizer))
    ids = tok.encode(PROMPT, add_special_tokens=False).ids[: args.max_tokens]
    T = len(ids)
    print(f"[tok] {T} tokens, first={ids[:6]} last={ids[-6:]}")

    tapped = args.onnx.parent / "model_tapped.onnx"
    taps = build_graph_with_taps(args.onnx, tapped, args.layers)
    print(f"[graph] wrote {tapped} with {len(taps)} taps")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    print(f"[ort] loading with {providers[0]} ...")
    sess = ort.InferenceSession(str(tapped), so, providers=providers)

    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, T), dtype=np.int64),
        "position_ids": np.arange(T, dtype=np.int64)[None, :],
    }
    empty = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)
    for i in range(NUM_LAYERS):
        feed[f"past_key_values.{i}.key"] = empty
        feed[f"past_key_values.{i}.value"] = empty

    want = list(taps.keys())
    print(f"[ort] running prefill T={T} past=0 ...")
    outs = sess.run(want, feed)

    manifest = []
    for name, arr in zip(want, outs):
        meta = taps[name]
        # Last position == the decode-step activation for this context.
        vec = np.asarray(arr).reshape(-1, np.asarray(arr).shape[-1])[-1]
        vec = vec.astype(np.float16)
        K = int(vec.shape[0])
        projs = ["gate_proj", "up_proj"] if meta["kind"] == "gate_up" else ["down_proj"]
        for proj in projs:
            dst = args.out / f"L{meta['layer']:02d}.{proj}.K{K}.x.fp16.bin"
            dst.write_bytes(vec.tobytes())
            f32 = vec.astype(np.float64)
            a = np.abs(f32)
            rec = {
                "layer": meta["layer"],
                "proj": proj,
                "K": K,
                "path": str(dst),
                "source_tensor": meta["tensor"],
                "rms": float(np.sqrt((f32**2).mean())),
                "absmax": float(a.max()),
                "std": float(f32.std()),
                "p50_abs": float(np.percentile(a, 50)),
                "p999_abs": float(np.percentile(a, 99.9)),
                "absmax_over_std": float(a.max() / f32.std()),
                "kurtosis": float(((f32 - f32.mean()) ** 4).mean() / f32.var() ** 2),
            }
            manifest.append(rec)
            print(
                f"  L{meta['layer']:02d} {proj:10s} K={K:5d} rms={rec['rms']:.4f} "
                f"absmax={rec['absmax']:.3f} amax/std={rec['absmax_over_std']:.1f} "
                f"kurt={rec['kurtosis']:.1f}"
            )

    (args.out / "x_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ok] wrote {len(manifest)} activation blobs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
