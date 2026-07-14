# Project Progress — Llama-3.2-3B Decode GEMM Optimization on L4

> **One-line story (target):** Built and profiled a TensorRT engine for Llama-3.2-3B on an NVIDIA L4, identified the decode-time MLP GEMM as the bottleneck, and replaced TensorRT's auto-selected tactic with a custom CUTLASS FP8 kernel (IPluginV3 plugin), achieving measurable per-token latency reduction at parity accuracy, validated with Nsight Compute.

**Last updated:** 2026-06-27  
**GPU:** NVIDIA L4 (Ada Lovelace, SM89, 24 GB)  
**Stack:** CUDA 12.6 · TensorRT 10.4 · CUTLASS 3.x · Ubuntu 24.04 (Docker)

---

## Phase tracker

| Phase | Status | Notes |
|-------|--------|-------|
| Environment (Docker, CUDA, TRT, CUTLASS, Nsight) | **Done** | See [Environment setup](#1-environment-setup) |
| Project layout + baseline scripts | **Done** | ONNX → trtexec workflow scaffolded |
| Baseline engine build (ONNX → `.engine`) | **In progress** | Download + build pending on your machine |
| Profile (nsys/ncu, timing cache, tactic names) | Pending | |
| Tactic swap demo (ITimingCache::update) | Pending | |
| Microbenchmark (single MatMul network) | Pending | |
| Custom CUTLASS FP8 kernel | Pending | |
| IPluginV3 plugin + engine swap | Pending | |
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

| Metric | Decode (M=1) | Prefill (e.g. seq=512) |
|--------|----------------|-------------------------|
| Build time | | |
| Engine size | | |
| trtexec latency (ms) | | |
| Dominant layer (from profile) | | |
| Auto-selected tactic name | | |

### After custom kernel / plugin

| Metric | Baseline | Custom plugin | Speedup |
|--------|----------|---------------|---------|
| Microbenchmark GEMM (ms) | | | |
| Full-model decode (ms/token) | | | |
| Cosine similarity vs baseline | | | |

### Nsight evidence

| Tool | Finding |
|------|---------|
| ncu (roofline / occupancy) | |
| nsys (timeline) | |

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

## Key learnings — optimization path (ahead)

- **Editable timing cache** (`kEDITABLE_TIMING_CACHE`) — swap existing tactics without writing a kernel (good for learning).
- **IPluginV3** — inject custom kernel as the layer implementation (headline result).
- **Microbenchmark first** — single MatMul at real decode shapes before full-model swap.
- **Always report parity** — FP8/custom kernels need cosine sim + small eval, not speed alone.

---

## Open questions / risks

- [ ] ONNX graph input names may include `past_key_values.*` — shape flags must match `inspect_onnx.py` output.
- [ ] ONNX graph MatMul layout may differ from TensorRT-LLM fused GEMMs — microbenchmark still required for clean comparison.
- [ ] FP8 engine variant — build after FP16 baseline is stable.
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

---

## References

- [Project plan](tensorrt_custom_tactic_kernel_b07f17a4.plan.md)
- [README — quick start](README.md)
- [TensorRT 10.4.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.4.0.html)
- [ONNX model on Hugging Face](https://huggingface.co/onnx-community/Llama-3.2-3B-Instruct-ONNX)
- Docker image: `nvidia_docker/Dockerfile`
