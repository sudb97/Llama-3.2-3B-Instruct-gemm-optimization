# Project Progress — Llama-3.2-3B Decode GEMM Optimization on L4

> **One-line story (target):** Built and profiled a TensorRT engine for Llama-3.2-3B on an NVIDIA L4, identified the decode-time MLP GEMM as the bottleneck, and replaced TensorRT's auto-selected tactic with a custom CUTLASS FP8 kernel (IPluginV3 plugin), achieving measurable per-token latency reduction at parity accuracy, validated with Nsight Compute.

**Last updated:** 2026-08-19  
**GPU:** NVIDIA L4 (Ada Lovelace, SM89, 24 GB)  
**Stack:** CUDA 12.6 · TensorRT 10.4 · CUTLASS 3.x · Ubuntu 24.04 (Docker)

---

## Phase tracker

| Phase | Status | Notes |
|-------|--------|-------|
| Environment (Docker, CUDA, TRT, CUTLASS, Nsight) | **Done** | See [Environment setup](#1-environment-setup) |
| Project layout + baseline scripts | **Done** | ONNX → trtexec workflow scaffolded |
| Baseline engine build + benchmark (decode + prefill) | **Done** | Clean ONNX export, FP16 engines, op-wise + roofline analysis for both regimes |
| Profile (nsys/ncu, tactic names, ncu bottleneck) | **Done** | Tactic captured; occupancy/DRAM/SASS analysis complete — see [Nsight profiling](#2026-08--nsight-profiling-decode-up_proj-kernel-bottleneck-analysis) |
| Memory-pattern microbench (DRAM amplification) | **Done** | Verified in project Docker on L4 — see [Memory-pattern microbenchmark](#2026-08-19--memory-pattern-microbenchmark-verified-in-project-docker) |
| Tactic swap demo (ITimingCache::update) | Deferred | Low value: all four decode GEMMs already at 89–93% DRAM peak on the same XMMA family |
| Custom GEMV kernel (FP16) | **Done** | Beats XMMA 1.039×/1.070× at the contiguous DRAM floor — see [FP16 GEMV](#2026-08-19--fp16-decode-gemv-beats-the-xmma-tactic) |
| Microbenchmark (single MatMul network) | **Next** | Clean single-MatMul TRT net for the plugin comparison |
| IPluginV3 plugin + engine swap | Pending | Wrap `kernels/gemv_fp16.cu` |
| FP8 / weight-only quantization | Pending | Only remaining lever — FP16 access pattern is exhausted |
| Validation (cosine sim + eval) | Pending | |
| Package (plots, README, writeup) | Pending | |

---

## The technical thesis

General libraries (cuBLAS, TensorRT default tactics) are tuned for large `M`. LLM **decode** runs projections at tiny `M` (batch × 1 token) — skinny, memory-bound, GEMV-like shapes (e.g. `M=1, K=3072, N=8192`). A kernel specialized for these exact shapes on L4's FP8 tensor cores can beat the auto-selected tactic in the low-latency decode regime.

**Honest scope:** Claims are scoped to decode / low-`M` only, not large-batch prefill GEMMs.

---

## Chronological log

### 2026-06 — Project planning

- Defined goal: custom tactic kernel for Llama-3.2-3B MLP GEMMs on L4 via TensorRT IPluginV3 + CUTLASS.
- Chose two-level strategy:
  1. **Microbenchmark** — standalone single-MatMul TRT network (rigorous proof).
  2. **End-to-end** — full model engine with layer swapped (context + credibility).
- Documented plan in [`tensorrt_custom_tactic_kernel_b07f17a4.plan.md`](tensorrt_custom_tactic_kernel_b07f17a4.plan.md).

### 2026-06 — Docker / environment (`nvidia_docker/`)

**What we tried**

| Attempt | Outcome |
|---------|---------|
| Base `nvidia/cuda:12.6.3-devel-ubuntu24.04` + apt `cuda-toolkit-12-6`, `libcublas-dev`, etc. | Failed — redundant packages pulled `cuda-tools-12-6`; network timeout on NVIDIA CDN |
| Apt install `libnvinfer10`, `tensorrt-dev` on Ubuntu 24.04 | Installed **TensorRT 11.x built for CUDA 13.x**, not 10.4 / CUDA 12.6; `tensorrt-dev` is meta-only (no `trtexec`) |
| Pip `tensorrt==10.4.*` (meta package) | Failed — `tensorrt-cu12` setup.py subprocess hit PEP 668 + `/usr/bin/X11` symlink loop on Ubuntu 24.04 |
| Pip leaf packages (`tensorrt-cu12-libs`, `tensorrt-cu12-bindings`) | Python API worked; **`trtexec` not included in pip wheels** |
| Symlink `trtexec` from `tensorrt_libs` | Failed — binary not bundled in pip package |
| **TensorRT 10.4.0.26 tar for CUDA 12.6** (NVIDIA direct download) | **Worked** — `bin/trtexec`, `lib/`, `include/`, Python wheel from tar |
| PyTorch via default `pip install torch` | Pulled **cu130** — incompatible with host driver (CUDA 12.8 API) |
| PyTorch `--index-url .../cu126` | **Worked** |
| `onnxruntime-gpu` latest | Linked `libcudart.so.13` — pin to **1.21.x** for CUDA 12 |
| CUTLASS full cmake build in Docker | Removed — **header-only clone is sufficient** |

**What worked (final Docker stack)**

- CUDA 12.6 devel + Nsight Compute/Systems + CUPTI/NVTX
- TensorRT **10.4.0.26 tar** → `$TENSORRT_ROOT/bin/trtexec`, libs, headers, Python 10.4 wheel
- CUTLASS git clone → `/opt/cutlass`
- PyTorch/torchvision (cu126), transformers, huggingface-hub, onnx, onnxruntime-gpu==1.21.*
- Entrypoint configures profiling (`perf_event_paranoid`, GPU counter access)

**Environment validation (all passed)**

- [x] `nvcc` CUDA 12.6, SM89 compile
- [x] `ncu`, `nsys`
- [x] `trtexec --help`, `import tensorrt` 10.4
- [x] CUTLASS headers at `/opt/cutlass`
- [x] PyTorch + `torch.cuda.is_available()`
- [x] HuggingFace / scipy / onnx / onnxruntime-gpu
- [x] Build toolchain (cmake, ninja, gcc-13)

**Key learnings — environment**

1. **`nvidia-container-toolkit` is a host concern**, not inside the container.
2. **Ubuntu 24.04 CUDA apt repo ships TRT 11 / CUDA 13 packages** — wrong for a CUDA 12.6 container; use the **GA tar** for TRT 10.4.
3. **`trtexec` is not in pip TensorRT** — only in tar/deb/NGC container or built from OSS samples.
4. **`trtexec --version` is invalid** — use `trtexec --help` or `--loadEngine` for smoke tests.
5. Pin **PyTorch (cu126)** and **onnxruntime-gpu (1.21.x)** to match container CUDA 12.6.
6. Stale Docker images: run `docker rm -f cuda-profiler-dev && docker rmi cuda-profiler:latest` before reloading a new saved image.

### 2026-06 — Baseline strategy decision

**Options considered**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| TensorRT-LLM (`trtllm-build`) | Full LLM pipeline, KV cache, fused GEMMs | Extra dep not in Docker; heavier setup | Deferred |
| **ONNX from HF + `trtexec`** | Matches existing `trtexec` setup; good for layer profiling | No full chat loop without Python harness; KV-cache shapes complex | **Chosen for baseline** |
| Hand-export ONNX from PyTorch | Full control | High effort for Llama + past_key_values | Not chosen |

**Baseline model:** [`onnx-community/Llama-3.2-3B-Instruct-ONNX`](https://huggingface.co/onnx-community/Llama-3.2-3B-Instruct-ONNX)

- `onnx/model_fp16.onnx` — graph structure
- `onnx/model_fp16.onnx_data` — ~6.4 GB weights (must sit beside `.onnx`)
- `tokenizer.json` — for future inference harness (not used by `trtexec`)

**Project layout created** — `engine/`, `models/`, `engines/`, `bench/`, `kernels/`, `plugin/`, `eval/` with scripts:
- `engine/download_model.sh`
- `engine/inspect_onnx.py`
- `engine/build_engine.sh`
- `bench/trtexec_baseline.sh`

### 2026-07 — Download + shape spec + ONNX/TensorRT parser incompatibility

**Download issues (all fixed in `engine/download_model.sh`)**

| Attempt | Outcome |
|---------|---------|
| Download only `model_fp16.onnx_data` | Failed at build — model's external weights are actually split across **4 shards** (`.onnx_data`, `_1`, `_2`, `_3`); TensorRT errors mid-parse (`Failed to open file: ...onnx_data_1`) once it reaches a layer whose weights live in a shard that was never fetched |
| Size-check assumed ~8 GB total (4 × ~2 GB) | False failure — actual total is **~6.46 GB** (last shard is a ~241 MB remainder, not a full ~2 GB chunk); lowered `MIN_TOTAL_BYTES` to 6 GB |
| `huggingface-cli download` | Silently did nothing — `huggingface_hub` 1.21.0 ships a deprecated stub that just prints "no longer works" and exits; switched script to prefer the new `hf download` CLI |

**Full 58-input shape spec** (`engine/configs/baseline_decode.env.example`)

- ONNX has 58 dynamic inputs: `input_ids`, `attention_mask`, and `past_key_values.{0..27}.{key,value}` (28 layers × 2, GQA with 8 KV heads, head_dim 128 — from `config.json`)
- `attention_mask` length = `past_sequence_length + sequence_length` (total, not just current step)
- Added a `gen_shapes()` bash helper that programmatically builds the full `--minShapes/--optShapes/--maxShapes` strings for all 58 inputs from `(batch, sequence_length, past_sequence_length)`, instead of hand-writing them
- Decode profile: seq_len always 1, batch 1→8, past_len 0→128 (opt)→2048; prefill profile: past_len always 0, seq_len 1→512 (opt)→2048

**ONNX/TensorRT parser incompatibility (root cause found, fix written)**

- After fixing shapes and downloads, `trtexec` still failed: `ERROR: Output name is not unique:` / `Assertion failed: toposort(...)`
- Root cause: `onnx-community/Llama-3.2-3B-Instruct-ONNX` is optimized by **ONNX Runtime's transformer optimizer** — confirmed via string scan of the `.onnx` file: 140 references to `com.microsoft::GroupQueryAttention` (a fused custom op for ORT's own execution providers, not understood by TensorRT's native parser) plus fused LayerNorm-family nodes with unused optional outputs left as `""`. Many nodes sharing the same empty output name collide during TensorRT's topological sort — a [known ORT↔TensorRT incompatibility](https://github.com/microsoft/onnxruntime/issues/18736)
- **Good news:** MLP `gate_proj`/`up_proj`/`down_proj` (this project's actual optimization target) are still plain `MatMul` ops — only attention got fused
- **Fix:** `engine/fix_onnx_duplicate_outputs.py` renames every empty (`""`) node output to a unique synthetic name (safe — these outputs were never consumed by anything), producing `model_fp16_fixed.onnx` next to the original so it still resolves the external weight shards
- Config updated to build from `model_fp16_fixed.onnx` instead of the raw download

**Key learnings**

1. Always verify **all** external-data shards are downloaded, not just the first — ONNX splits large weights across multiple files and TensorRT only errors when it reaches the missing one, not upfront.
2. Don't guess total download size — check the actual repo file listing instead of assuming equal shard sizes.
3. Community "optimized" ONNX exports (ORT transformer optimizer output) are tuned for **ONNX Runtime's own execution providers**, not raw TensorRT ingestion — fused custom ops (`com.microsoft::*`) can break TensorRT's parser even when the model runs fine in `onnxruntime-gpu`.
4. A `""` node output in ONNX means "optional output unused" — but TensorRT treats output names as globally unique keys, so multiple unused `""` outputs across a large graph will collide.

### 2026-07 — Pre-exported ONNX abandoned; switching to a clean optimum export

**What happened**

- The empty-output fix got the parser past toposort, but the build then failed on **every fused ORT contrib op**: `checkFallbackPluginImporter: Plugin not found` for each `com.microsoft::GroupQueryAttention` (28 nodes) and `com.microsoft::SkipSimplifiedLayerNormalization` (57 nodes). TensorRT has no native importer or bundled plugin for ORT contrib ops — when it hits an unknown op it looks in its plugin registry, finds nothing, and fails.
- Checked whether any other variant in `onnx-community/Llama-3.2-3B-Instruct-ONNX` avoids this: downloaded the FP32 `model.onnx` graph (187 KB without weights) and string-scanned it — 85 `com.microsoft` references. **Every variant in that repo (fp32/fp16/int8/q4) is ORT-optimized and unusable with TensorRT's parser.**

**Options considered**

| Option | Verdict |
|--------|---------|
| Write TRT plugins for GQA + fused RMSNorm just to load the baseline | Rejected — that's a whole project before the actual project starts |
| TensorRT-LLM | Still deferred (heavier dep, hides the GEMMs we want to profile) |
| **Re-export clean ONNX from PyTorch weights via `optimum`** | **Chosen** — emits only standard ONNX ops; attention decomposed, all projections (q/k/v/o, gate/up/down) are plain `MatMul`, ideal for GEMM tactic profiling |

**Changes**

- New `engine/export_clean_onnx.sh`: installs `optimum[exporters]` if needed, exports `unsloth/Llama-3.2-3B-Instruct` (ungated mirror of the gated meta-llama repo) with `--task text-generation-with-past --dtype fp16 --device cuda` to `models/onnx-fp16-clean/`
- `baseline_decode.env.example`: `ONNX` now points at the clean export; `gen_shapes()` gained a `position_ids` input (optimum exports include it; `HAS_POSITION_IDS=0` disables)
- `build_engine.sh`: external-data check made generic (globs `*.onnx_data*` instead of hardcoding the old shard names)

**Export gotchas hit along the way (both fixed in `export_clean_onnx.sh`)**

- optimum v2.0 split the ONNX exporter into a separate `optimum-onnx` package — plain `optimum[exporters]` gives an `optimum-cli` where `export onnx` is an unrecognized subcommand
- Post-processing (tied-weight dedup of `lm_head`/`embed_tokens`) serializes the whole model as one protobuf and fails for >2 GB models (`Failed to serialize proto`); `--no-post-process` skips it at the cost of ~790 MB duplicated weight on disk, numerically identical

**Key learnings**

5. TensorRT's ONNX parser fallback for unknown ops is a **plugin-registry lookup by op name** — an unknown op only works if a plugin with that exact name/version/namespace is registered. ORT contrib ops (`com.microsoft` domain) have no such plugins.
6. For TensorRT work, export ONNX yourself from the framework checkpoint with standard ops — don't reuse exports optimized for a different runtime.
7. Protobuf has a hard 2 GB message limit — any tool step that serializes a full LLM ONNX model in one proto (checkers, graph rewrites) will fail; look for flags that keep weights external or skip the step.

---

## Results (fill in as you go)

### Baseline engine build

| Metric | Decode (M=1) | Prefill (M=512) |
|--------|----------------|-------------------------|
| Build time | | |
| Engine size | 6136 MiB | |
| trtexec latency (ms) | 30.24 ms/token (mean), compute 28.10 ms | 83.995 ms/step (mean), compute 76.305 ms |
| Dominant layer (from profile) | `mlp/up_proj` (31.6%), `mlp/gate_proj` (22.4%) | `mlp/gate_proj`(fused, 19.0%), `mlp/up_proj` (18.7%) |
| Bottleneck resource | HBM bandwidth (memory-bound) | Tensor Core FLOPs (compute-bound, but only marginally — see below) |
| Achieved bandwidth vs. peak | **ncu (authoritative):** `up_proj`/`gate_proj` ~270 GB/s (~90% of L4 peak); `down_proj` similar. Earlier 119 GB/s was a `trtexec --dumpProfile` artifact (see 2026-08 ncu log). | `up_proj` 122.5 GB/s (~41% of ~300 GB/s BW) |
| Achieved compute vs. peak | `up_proj` ~15% SM / ~0.1% of FP16 Tensor-Core peak — not binding | `up_proj` 51.04 TFLOP/s (~42% of ~121 TFLOP/s FP16) |
| Auto-selected tactic name | `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt` (grid 256 for up/gate, 96 for down) | pending |

### After custom kernel / plugin

| Metric | Baseline (TRT sm80_xmma) | Custom GEMV | Speedup |
|--------|----------|---------------|---------|
| `up_proj` kernel (ncu, cold) | 223.9 µs, 60.62 MB | 215.6 µs, 56.96 MB | **1.039×** |
| `down_proj` kernel (ncu, cold) | 223.6 µs, 60.92 MB | 208.9 µs, 56.28 MB | **1.070×** |
| DRAM amplification | 1.204× / 1.210× | 1.132× / 1.118× | at contiguous floor |
| Cosine similarity vs fp64 CPU ref | — | 0.99999998 | 32/32 configs pass |
| Full-model decode (ms/token) | 30.24 | pending plugin | |

### Nsight evidence

| Tool | Finding |
|------|---------|
| nsys (timeline) | Decode MLP MatMuls map 1:1 onto `sm80_xmma_gemm_..._tilesize32x32x64_stage6_...`. Host `myelin-exec` row is the TRT layer; CUDA row is the kernel. |
| ncu (up_proj, grid 256) | Memory 93%, compute 15%. Occupancy **16.67%** (shared-mem limited: 2 blocks/SM). L1TEX scoreboard = 62.8% of warp stalls. L2 sector excess **0%**; kernel requests 99.92% of theoretical minimum bytes. DRAM amplification **1.20×**. |
| ncu (down_proj, grid 96) | Same tactic family, same 89–93% DRAM saturation. Apparent 2× `trtexec` speedup vs `up_proj` is a profiler artifact — ncu times are comparable once bytes-moved is the denominator. |
| ncu (memory-pattern bench, Docker L4) | Contiguous 1.10–1.13×, strided 1.14×, fragmented 1.41–1.45×. Real kernel sits at 1.20–1.21× — between strided and fragmented. |

---

## What worked — summary

- **TensorRT 10.4 tar (CUDA 12.6)** as single source for runtime, headers, `trtexec`, and Python bindings.
- **Docker image** with pinned CUDA-12-compatible Python GPU stack.
- **ONNX community FP16 model + trtexec** for baseline engine and layer-level profiling.
- **Incremental validation** — fix TensorRT/trtexec first, then uncomment rest of Dockerfile.

## What did not work — summary

- Apt TensorRT on Ubuntu 24.04 for CUDA 12.6 / TRT 10.4 + `trtexec`.
- Pip-only TensorRT for CLI tooling.
- Default PyTorch / onnxruntime-gpu wheels (CUDA 13).
- Assuming `trtexec --version` exists.

## Key learnings — optimization path

- **ncu over dumpProfile** for bandwidth/time of skinny GEMMs — dumpProfile inflated `up_proj` and made `down_proj` look 2× faster.
- **Occupancy is not the lever.** Shared-mem limits the XMMA kernel to 16.67%; DRAM is already 89–93% saturated, so more blocks do not buy unused bandwidth.
- **Recoverable gap is DRAM amplification** (1.20× real kernel vs 1.10–1.13× contiguous floor), not coalescing inside a tile (0% L2 sector excess).
- **FP16 GEMV first, FP8 later.** Access-pattern ceiling is ~1.07–1.09× (1.20× → 1.10–1.13× DRAM). Occupancy is not a time lever.
- **IPluginV3** after the standalone GEMV beats the XMMA kernel.
- **Always report parity** — cosine sim + small eval, not speed alone.

---

## Open questions / risks

- [x] ONNX graph input names include `past_key_values.*` + `position_ids` — shape flags match `inspect_onnx.py` / `baseline_decode.env`.
- [x] Weight layout: both `up_proj` and `down_proj` are `[K, N]` row-major in the clean ONNX (row strides 16384 B and 6144 B). Confirmed by protobuf inspection of `models/onnx-fp16-clean/model.onnx`.
- [x] `trtexec --dumpProfile` implied bandwidth (119 vs 236 GB/s) is **not** trustworthy for these kernels — ncu shows both at ~90% DRAM peak. Use ncu, not dumpProfile, as the speed baseline.
- [ ] GEMV rewrite: can we reach the 1.10–1.13× contiguous DRAM floor in a real kernel, not just the memory microbench?
- [ ] FP8 engine variant — decide after FP16 GEMV numbers, not before.
- [ ] Full tokens/sec needs Python TRT runner (tokenizer + KV loop), not `trtexec` alone.

---

## How to update this file

After each milestone, add a dated entry under **Chronological log**, update **Phase tracker**, and fill **Results** tables. Capture:

1. **What you tried** (command, approach, config)
2. **What happened** (error or metric)
3. **What you changed**
4. **Takeaway** (one line)

Example entry:

```markdown
### YYYY-MM-DD — Profile decode GEMM

- Ran `nsys profile trtexec --loadEngine=... --shapes=input_ids:1x1,...`
- Dominant kernel: `/MatMul_*` (~X ms, Y% of forward)
- Tactic: `<name from timing cache / verbose build log>`
- Learning: decode at M=1 is memory-bound; FP8 tensor cores underutilized at baseline
```

### 2026-07 — Decode benchmark: `--exportTimes` silently empty with `--dumpProfile`

**Symptom:** `./bench/trtexec_baseline.sh decode` printed `[ok] Timing: .../decode_timing.json` and exited `PASSED`, but the file was never created. `decode_profile.json` (per-layer) was written correctly.

**Root cause:** confirmed directly from the `trtexec` log:

> The e2e network timing is not reported since it is inaccurate due to the extra synchronizations when the profiler is enabled. To show e2e network timing report, add `--separateProfileRun` to profile layer timing in a separate run or remove `--dumpProfile` to disable the profiler.

`--dumpProfile`'s per-layer instrumentation inserts CUDA sync points that would make end-to-end (`--exportTimes`) timing inaccurate, so `trtexec` deliberately drops that output when both are requested together in one run — it's documented, not a bug. Separately, `bench/trtexec_baseline.sh` echoed `[ok]` unconditionally without checking the file existed.

**Fix:** added `--separateProfileRun` to the `trtexec` benchmark invocation in `bench/trtexec_baseline.sh` (runs the benchmark twice internally: once clean for `--exportTimes`, once instrumented for `--dumpProfile`/`--exportProfile`, ~2x wall time). Also made the script's completion messages check `-f` before printing `[ok]` instead of assuming success.

**Bonus — the profile run we already have answers the `profile` phase's key question.** Decode step (batch=1, past=128, `--fp16`), 339 iterations, engine size 6136 MiB, total per-step latency ~37.4 ms (includes 1 aux stream, so layers overlap somewhat — total isn't strictly additive):

| Layer (per decode layer, x28) | Avg (ms) | % of total |
|---|---|---|
| `mlp/up_proj` MatMul (3072→8192) | 0.4235 | 1.1% |
| `mlp/gate_proj` MatMul (3072→8192) | 0.2975 | 0.8% |
| `mlp/down_proj` MatMul (8192→3072) | 0.2132 | 0.6% |
| `self_attn` q/k/v_proj (fused) | 0.1356 | 0.4% |
| `self_attn/o_proj` MatMul | 0.0840 | 0.2% |
| `lm_head` MatMul (once, not per-layer) | 3.036 | 8.1% |

Summed across all 28 layers: MLP GEMMs alone (`gate`+`up`+`down`) ≈ `(0.4235+0.2975+0.2132) × 28 ≈ 26.1 ms` — roughly **70% of total decode step latency**, confirming the plan's thesis: the MLP projections are the compute-dominant decode GEMM, with `up_proj`/`gate_proj` (3072→8192, the wider matmul) costing more than `down_proj` (8192→3072). `lm_head` is a distant second at 8.1% (single occurrence, not per-layer, and out of scope — it's a different shape/purpose than the repeated MLP GEMM).

**Learning:** `--dumpProfile` + `--exportTimes` together in one `trtexec` run is a known incompatible combo — always pair with `--separateProfileRun` when both outputs are needed. Also: don't trust a wrapper script's success message without a file-existence check — `trtexec` returning `PASSED` only means the requested run completed, not that every requested output artifact was written.

### 2026-07 — Decode op-wise breakdown + roofline (memory-bound) analysis

**Op-wise GPU time, accumulated across all 28 layers** (from `decode_profile.json`, 334 profiled iterations, batch=1/past=128, `--fp16`):

| Op group | ms/step | % of step | #kernels | avg ms/kernel |
|---|---|---|---|---|
| `mlp.up_proj` | 11.8483 | 31.6% | 28 | 0.42315 |
| `mlp.gate_proj` | 8.3944 | 22.4% | 28 | 0.29980 |
| `mlp.down_proj` | 5.9953 | 16.0% | 28 | 0.21412 |
| `attn.qkv_proj` (fused) | 3.8233 | 10.2% | 28 | 0.13655 |
| `lm_head` | 3.0363 | 8.1% | 1 | 3.03626 |
| `attn.o_proj` | 2.3748 | 6.3% | 28 | 0.08481 |
| elementwise/norm/fused-misc (Myelin `__myl*`) | 1.4963 | 4.0% | 199 | 0.00752 |
| `attn` score/context matmuls (Q@Kᵀ, softmax@V) | 0.5495 | 1.5% | 56 | 0.00981 |

MLP GEMMs alone = **69.9%** of the step; all projection GEMMs + `lm_head` = **94.5%**. Attention's actual matmuls are negligible (1.5%) — almost everything is linear-projection GEMM time. Confirmed from `decode_timing.json` (357 clean e2e runs, no profiler overhead): mean per-token latency **30.24 ms** (compute 28.10 ms, H2D 0.96 ms, D2H 1.17 ms), extremely tight distribution (p99 within 0.7% of median) — a trustworthy baseline number.

**Bandwidth-floor / roofline analysis — why decode is memory-bound and prefill won't be**

Computed implied HBM bandwidth per GEMM (`weight_bytes / measured_time`) against the L4's ~300 GB/s peak:

| GEMM | Achieved bandwidth | % of L4 peak |
|---|---|---|
| `up_proj` (K=3072,N=8192) | 119 GB/s | ~40% |
| `gate_proj` (same shape) | 169 GB/s | ~56% |
| `down_proj` (K=8192,N=3072) | 236 GB/s | ~79% |
| `lm_head` (K=3072,N=128256) | 260 GB/s | ~87% |

`up_proj`/`gate_proj` leave 2–2.5x bandwidth on the table relative to what `down_proj` already achieves on this same GPU — the clearest, most actionable target. Total weight bytes streamed per decode step ≈ 6.25 GB → a naive floor of `6.25 GB / 300 GB/s ≈ 21 ms` vs. the measured 28.10 ms compute — i.e. today's baseline is running at roughly 75% of the theoretical weight-streaming floor.

**Rigorous justification via the roofline model** (why decode is memory-bound, and why prefill flips to compute-bound at larger `M`):

- `T_compute = FLOPs / peak_compute`, `T_memory = Bytes / peak_bandwidth`; achievable time is `max(T_compute, T_memory)` — whichever resource takes longer is the bottleneck.
- Comparing the two reduces to comparing **arithmetic intensity** (`FLOPs/Byte`) against the GPU's own **ridge point** (`peak_compute / peak_bandwidth`, ≈403 FLOP/byte for L4's ~121 TFLOP/s FP16 Tensor Core peak ÷ ~300 GB/s).
- At decode (M=1): weight bytes (50.33 MB, fixed by `K×N`) dominate; activation/output bytes are ~0.02 MB — negligible. Intensity ≈ **1 FLOP/byte**, ~400x below the ridge → deep in memory-bound territory. `T_compute≈0.41 µs` vs `T_memory≈168 µs` — compute finishes almost instantly and idles.
- At prefill (M=512, the config's opt point): weight bytes are **unchanged** (still 50.33 MB — loaded once regardless of M, reused across all M rows by a well-tiled kernel), but compute scales **linearly** with M (512x). Memory traffic grows only ~1.23x (50.33 MB → 61.87 MB, from added activation+output bytes) while FLOPs grow 512x. Intensity ≈ **417 FLOP/byte** — just past the ridge → compute-bound. `T_compute≈213 µs` ≈ `T_memory≈206 µs` (near the crossover, consistent with M=512 being chosen as the opt/breakeven shape).
- **Key insight for the write-up:** the weight matrix's memory cost is M-independent (paid once, amortized across all M rows), which is exactly why more tokens processed per GEMM call (larger M) shifts the bottleneck from memory to compute — this is the formal reason decode and prefill need fundamentally different kernel strategies, and why this project scopes its FP8/CUTLASS kernel claims to the decode (low-M, memory-bound) regime specifically.
- Also clarified: the ridge point/floor is a *hard* bound on aggregate whole-chip bandwidth — it cannot be beaten by running independent kernels concurrently on separate streams. Concurrency (e.g. fusing `gate_proj`+`up_proj` into one N=16384 GEMM) is a way to *reach* the floor a single under-saturating skinny kernel misses, not a way to go below it.

### 2026-07 — Prefill benchmark: op-wise breakdown + compute-bound confirmation

Ran `./bench/trtexec_baseline.sh prefill` (batch=1, seq_len=512 opt, past=0, `--fp16`). Both `prefill_profile.json` (136 iterations) and `prefill_timing.json` (133 iterations) captured cleanly in one run (the `--separateProfileRun` fix from the decode benchmark applies here too).

**Gotcha found during analysis:** naively reusing the decode-benchmark's name-matching classifier silently miscounted `gate_proj`. At decode, `gate_proj`'s MatMul is a standalone kernel; at prefill (M=512), TensorRT's Myelin compiler **fuses `gate_proj`'s GEMM directly into its SiLU-activation epilogue and the elementwise multiply against `up_proj`'s output**, producing one kernel named `__myl_FcNegExpAddDivMulMul` (`Fc` = FullyConnected/GEMM signature inside Myelin's fused-op names). A substring classifier looking for `"gate_proj"` in the kernel name misses this entirely and dumps 19% of the step into a generic "elementwise/misc" bucket. **Learning: TensorRT's fusion strategy itself changes with input shape, not just tactic selection — any kernel-name-based aggregation script needs to be re-validated per shape, not assumed stable across benchmarks.**

**Corrected op-wise breakdown** (accumulated across 28 layers):

| Op group | ms/step | % of step | #kernels |
|---|---|---|---|
| `mlp.gate_proj`+SiLU+mul (fused) | 14.34 | 19.0% | 28 |
| `mlp.up_proj` | 14.14 | 18.7% | 28 |
| `mlp.down_proj` | 13.62 | 18.0% | 28 |
| elementwise/norm/RoPE (true misc) | 10.76 | 14.2% | 226 |
| `attn.qkv_proj` (fused) | 8.10 | 10.7% | 28 |
| `lm_head` | 6.44 | 8.5% | 1 |
| `attn.o_proj` | 5.31 | 7.0% | 28 |
| `attn` score/context matmuls (Q@Kᵀ, softmax@V) | 2.81 | 3.7% | 56 |
| **Total** | **75.51** | **100%** | 425 |

MLP GEMMs = **55.7%** of the step (down from decode's 69.9%) — attention's combined share (qkv+o_proj+score+context) grew to **~21.4%** vs decode's ~11.1%, exactly as predicted: Q@Kᵀ/softmax@V scale with `sequence_length²`, so they become proportionally larger at seq_len=512 than at decode's seq_len=1.

**End-to-end timing** (`prefill_timing.json`): mean latency **83.995 ms**, compute **76.305 ms**. H2D shrinks to 0.213 ms (vs decode's 0.96 ms — far less KV cache to upload with `past=0`), but D2H jumps to **7.478 ms** (6.4x decode's 1.17 ms) — much larger outputs at this shape: `logits` is `512×128256` vs decode's `1×128256`, and all 56 `present.*` KV outputs now carry `total_sequence_length=512` instead of 129.

**Compute-bound confirmation (the roofline prediction, empirically verified):**

| GEMM | Achieved TFLOP/s | % of L4's ~121 TFLOP/s FP16 peak |
|---|---|---|
| `up_proj` | 51.04 | 42.2% |
| `gate_proj` (fused) | 50.31 | 41.6% |
| `down_proj` | 53.00 | 43.8% |

All three land in a tight **~42–44% of peak compute** band — a sharp contrast to decode's bandwidth utilization, which varied widely (40% to 87%) across the same three GEMMs. This consistency is itself informative: at prefill, the bottleneck is uniformly the Tensor Cores' raw throughput regardless of which GEMM shape you pick (`K↔N` swapped for `down_proj` doesn't change the ratio much), whereas at decode the bottleneck (HBM bandwidth) is much more sensitive to how well a specific kernel's tiling saturates the memory bus. Confirms the earlier roofline math (decode intensity ≈1 FLOP/byte, ~400x below L4's ~403 FLOP/byte ridge; prefill intensity ≈417 FLOP/byte, just past the ridge) was the right predictive model.

**Project implication:** since prefill's ~42–44% compute utilization is itself real headroom, a "prefill-mode" kernel would target *compute-efficiency* (occupancy, Tensor Core MMA shape/pipelining), a completely different optimization axis than decode's *bandwidth-efficiency* target — reinforcing why this project scopes its custom CUTLASS/FP8 kernel claims specifically to the decode (memory-bound, low-M) regime, per the original plan.

**Bandwidth utilization at prefill (the missing half of the picture):** computed the same way as decode's bandwidth analysis — `total_bytes_moved / measured_time` — but now including activation + output bytes, since at M=512 they're no longer negligible (weight 50.33 MB, activation 3.15–8.39 MB, output 3.15–8.39 MB depending on which side is `K` vs `N`, total ≈61.87 MB per GEMM):

| GEMM | Total bytes moved | Achieved bandwidth | % of L4's ~300 GB/s peak |
|---|---|---|---|
| `up_proj` | 61.87 MB | 122.5 GB/s | 40.8% |
| `gate_proj` (fused) | 61.87 MB | 120.8 GB/s | 40.3% |
| `down_proj` | 61.87 MB | 127.2 GB/s | 42.4% |

**Key finding:** at prefill, bandwidth utilization (~40–42%) and compute utilization (~42–44%) sit in nearly the *same* band — expected, since prefill's arithmetic intensity (~417 FLOP/byte) sits just barely past the L4's ridge point (~403 FLOP/byte). Being this close to the crossover means neither resource is decisively idle waiting on the other; the kernel leaves similar-sized headroom on both simultaneously. This is a sharp contrast to decode, where the two metrics diverge sharply — bandwidth utilization varies 40–87% while compute utilization is a negligible 0.1–0.2% of peak (intensity ~1 FLOP/byte, 400x below the ridge, so compute finishes almost instantly and idles — "compute utilization" isn't even the operative metric there). Having both numbers for prefill, and seeing them converge near the ridge, is a good sanity check that the memory-bound/compute-bound classification was applied correctly to each regime.

**Decode vs. prefill — side-by-side summary** (baseline complete):

| | Decode (M=1, past=128) | Prefill (M=512, past=0) |
|---|---|---|
| Mean latency | 30.24 ms/token | 83.995 ms/step |
| GPU compute | 28.10 ms | 76.305 ms |
| MLP GEMM share | 69.9% | 55.7% |
| Attention share | ~11.1% | ~21.4% (grows with seq_len², as predicted) |
| Bottleneck | HBM bandwidth (memory-bound) | Tensor Core FLOPs (marginally compute-bound) |
| `up_proj` bandwidth vs. peak | ~270 GB/s (~90% of peak, ncu). dumpProfile 119 GB/s was an artifact | 122.5 GB/s (~41% of ~300 GB/s) |
| `up_proj` compute vs. peak | ~0.1 TFLOP/s (~0.1% — not binding) | 51.04 TFLOP/s (~42% of ~121 TFLOP/s) |
| Optimization target | Bandwidth efficiency (skinny GEMV-like shapes) | Compute efficiency (different axis — out of project scope) |

**Baseline phase conclusion:** both reference points are captured. Decode MLP GEMMs are the priority target (memory-bound). The dumpProfile 2–2.5× `up_proj` vs `down_proj` bandwidth gap was later shown to be a profiler artifact — ncu puts all four decode GEMMs at 89–93% DRAM peak. Prefill confirms the roofline model and decode-only scoping.

### Profile phase — completed (see 2026-08 log below)

Tactic name, occupancy limiter, DRAM saturation, and DRAM-amplification floor are measured. Editable timing-cache swap is deferred (same XMMA family, already DRAM-saturated). Next work is a GEMV kernel, not another TRT tactic.

### 2026-08 — Nsight profiling: decode `up_proj` kernel bottleneck analysis

**Tooling issues resolved first**

| Problem | Cause | Fix |
|---|---|---|
| `nsys` produced `.qdstrm`, "importer binary not found" | CLI-only Nsight (2022.4.2) missing importer on `PATH` | `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i X.qdstrm -o X.nsys-rep` |
| nsys captured only host `enqueue`, no GPU kernels | Missing GPU-trace flags | `--gpu-metrics-device=all --gpuctxsw=true --trace-fork-before-exec=true` |
| Windows ncu GUI `FailedReadingMessage` | CLI/GUI version mismatch | Match GUI ≥ CLI 2024.3, or export CSV/text |

**Tactic captured (nsys)**

`sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt`

- Tile 32×32×64, 6-stage software pipeline, 2×2 warps, HMMA 16×8×16
- Grid 256 for `up_proj`/`gate_proj` (N=8192 / 32), grid 96 for `down_proj` (N=3072 / 32)
- Block 128 threads, 50 registers/thread, 49.15 KB dynamic shared + 1.02 KB driver

**Occupancy is shared-memory limited, and the math matches ncu exactly**

`6 stages × (32×64 + 64×32) × 2 B = 49,152 B` plus 1 KiB driver ≈ 50.17 KB/block. L4 SM shared mem 102.40 KB → `floor(102.40/50.17) = 2` blocks/SM → `2 × 4 warps = 8` warps/SM → `8/48 = 16.67%`. Registers are not binding (would allow 9 blocks). ncu reports theoretical occupancy 16.6667% and theoretical warps/scheduler = 2.

**What actually costs time**

- DRAM ~90–93% of peak; compute ~15% of SM. Deeply memory-bound.
- L1TEX scoreboard = 62.8% of warp stalls. Pipeline cannot hide HBM latency at 2 warps/scheduler.
- L2 sector excess **0%**. Measured sectors / theoretical minimum = **0.9992**. Zero shared-memory bank conflicts. FP16 access-pattern / coalescing inside this tile is exhausted.
- The earlier Results-table 119 GB/s (`weight_bytes / dumpProfile time`) is wrong. ncu DRAM throughput is ~270 GB/s. Use ncu as the baseline-to-beat, not dumpProfile.

**SASS mainloop (63 instructions × 48 K-tiles):** 8× `HMMA.16816.F16`, 8× `LDSM.16.M88.4`, 4× `LDGSTS` (`cp.async`), `LDGDEPBAR` + `DEPBAR.LE SB0, 0x4` (`wait_group 4`) + `BAR.SYNC`. Two 6-stage circular buffers at `0x0000` (A, activations) and `0x6000` (B, weights), 4 KiB per stage each — not 8 KiB combined in one slot.

**Implication:** raising occupancy by shrinking the A-tile is real (GEMV can hit 100% occupancy) but does **not** create unused DRAM bandwidth — the bus is already full. The only recoverable bytes are DRAM *amplification* (overfetch past unique weight bytes), not occupancy.

### 2026-08 — `down_proj` ncu: the 2× `trtexec` gap is an artifact

`down_proj` moves the same 50.33 MB of weights as `up_proj`. dumpProfile implied 236 vs 119 GB/s. ncu raw export (`bench/profile/down_projection_raw.csv`) captured four GEMM shapes in one session:

| Kernel | Grid | Predicted bytes | Match | DRAM % peak | Amplification vs  unique weights |
|---|---|---|---|---|---|
| `qkv_proj` | 160 | 32.44 MB | exact | ~92% | similar class |
| `o_proj` | 96 | 19.46 MB | exact | ~89% | similar class |
| `up_proj`/`gate_proj` | 256 | 51.90 MB | exact | ~90% | **1.204×** |
| `down_proj` | 96 | 51.90 MB | exact | ~91% | **1.210×** |

All four sit at 89–93% DRAM peak. The dumpProfile 2× gap is not a structural `down_proj` win. Same XMMA tactic family, same occupancy limiter.

### 2026-08-19 — Memory-pattern microbenchmark, verified in project Docker

Ran `kernels/mem_pattern_bench.cu` **inside container `489257d94e91`** (`cuda-profiler:latest`, NVIDIA L4 sm_89, CUDA 12.6, TensorRT 10.4.0.26). Rebuilt with `nvcc -O3 -arch=sm_89 -lineinfo`. Cold `ncu` (`--launch-skip 0 --launch-count 1`). Full table: [`kernels/mem_pattern_results_l4_container.md`](kernels/mem_pattern_results_l4_container.md).

DRAM amplification = `dram__bytes_read.sum / 50.33 MB`.

| Pattern | down_proj shape (8192 × 6144 B) | up_proj shape (3072 × 16384 B) |
|---|---|---|
| contiguous | **1.104×** | **1.133×** |
| strided (coalesced warp, row-stride jumps) | 1.137× | 1.139× |
| fragmented (64 B groups on different K-rows) | 1.409× | 1.450× |
| **real TRT kernel** | **1.210×** | **1.204×** |

Weights confirmed `[K, N]` row-major via ONNX protobuf (`up_proj` row = 16384 B, `down_proj` row = 6144 B). The real kernel sits between strided and fragmented — consistent with a 64-wide N-tile that is coalesced within a K-row but jumps `ROW_BYTES` between K-rows.

**Honest FP16 ceiling:** moving from 1.20× toward the contiguous floor (1.10–1.13×) cuts DRAM bytes by **~7–9%**. Because the XMMA kernel is already DRAM-saturated, that is also the plausible **time** speedup from access pattern alone (~1.07–1.09×), not 2×. Extra time may come from dropping XMMA epilogue/pipeline overhead, but occupancy-only rewrites will not.

### 2026-08-19 — FP16 decode GEMV beats the XMMA tactic

Implemented `kernels/gemv_fp16.cu` and profiled it in container `489257d94e91`. Full write-up: [`kernels/gemv_results_l4_container.md`](kernels/gemv_results_l4_container.md).

| Shape | Kernel | Duration | DRAM read | Amplification | DRAM % peak | Occupancy |
|---|---|---|---|---|---|---|
| `up_proj` | TRT sm80_xmma | 223.9 µs | 60.62 MB | 1.204× | ~90% | 16.1% |
| `up_proj` | **GEMV** (1024 thr, grid 14) | **215.6 µs** | **56.96 MB** | **1.132×** | 90.1% | 64.7% |
| `down_proj` | TRT sm80_xmma | 223.6 µs | 60.92 MB | 1.210× | ~91% | 13.4% |
| `down_proj` | **GEMV** (384 thr, grid 29) | **208.9 µs** | **56.28 MB** | **1.118×** | 91.9% | 24.6% |

**1.039× / 1.070×** at equal DRAM saturation. Cosine similarity **0.99999998** vs a float64 CPU reference across all 32 variant × grid combinations.

**Design:** split-K with full-row blocks. A block owns a contiguous chunk of K rows and the *full* N width, so it reads one unbroken slab; splitting N would reintroduce the row-stride jump. `THREADS × 8 × VEC_PER_THREAD == N` makes one instruction sweep cover exactly one row. Each thread owns fixed output columns for the whole sweep, so partials stay in registers — at M=1 every weight is used once, so no shared-memory B staging. Blocks `atomicAdd` into an fp32 accumulator (≤32 KB, stays L2-resident).

**Result matches the pre-kernel prediction.** Amplification landed at 1.132×/1.118× against the microbench contiguous floor of 1.133×/1.104×. Bytes fell 6.0–7.6%, time fell 3.7–6.6% — squarely in the 1.07–1.09× byte-level estimate. **FP16 access-pattern optimization is now exhausted.**

**Two non-obvious findings**

- **Small grids win.** 14 blocks beat 58, 116 and 464 on `up_proj`. Past DRAM saturation, every extra block is one more concurrent stream to interleave. 14 blocks of 1024 threads use a fraction of the SMs and still hit 90% of peak.
- **Occupancy was never the lever.** `chunk/narrow` at 16% occupancy and `chunk/wide` at 65% land within 2 µs at the same grid. The win came entirely from the byte reduction — confirming that the 2-blocks/SM limit was real but not what cost time.

**Rejected design (kept in source as evidence):** the first version used contiguous K-chunks with a machine-filling grid (116). Amplification was fine (1.115×) but DRAM throughput collapsed to 62–74% and it was *slower than baseline* (258–308 µs) because co-resident blocks streamed from separated regions. A `stride` variant (block `b` takes rows `b, b+grid, …`) fixed the window, but shrinking the grid mattered more.

**Benchmark caveat:** L4's L2 is 48 MB against a 50.33 MB matrix, so a naive repeat loop reports 800–1000 GB/s — above DRAM peak. `bench` rotates over 4 weight copies to stay cold, matching ncu.

---

## Next steps (after 2026-08-19)

FP16 kernel work is done and the remaining lever is bytes, not scheduling.

1. **Standalone TRT micronet** — single MatMul at the decode shapes, so the plugin comparison is not buried in a 28-layer engine.
2. **IPluginV3 wrap** of `kernels/gemv_fp16.cu`, then swap `up_proj` (highest decode share) in the full engine. Expect ~1.04× on that layer; end-to-end decode gain will be well under that since MLP GEMMs are 69.9% of the step.
3. **FP8 / weight-only quantization** — the only remaining lever. Halving weight bytes is worth ~2× on a DRAM-bound kernel, an order more than the 1.04–1.07× access-pattern win. Reuse the same split-K structure and add a dequant in the inner loop; validate parity with cosine sim + a small eval.
4. **Skip `ITimingCache::update`** as a primary path. Optional later as a short negative result showing other TRT FP16 tactics do not beat this XMMA kernel.

---

## References

- [Project plan](tensorrt_custom_tactic_kernel_b07f17a4.plan.md)
- [README — quick start](README.md)
- [TensorRT 10.4.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.4.0.html)
- [ONNX model on Hugging Face](https://huggingface.co/onnx-community/Llama-3.2-3B-Instruct-ONNX)
- Docker image: `nvidia_docker/Dockerfile`
