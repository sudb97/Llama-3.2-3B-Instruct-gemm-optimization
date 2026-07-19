# Project Progress — Llama-3.2-3B Decode GEMM Optimization on L4

> **One-line story (target):** Built and profiled a TensorRT engine for Llama-3.2-3B on an NVIDIA L4, identified the decode-time MLP GEMM as the bottleneck, and replaced TensorRT's auto-selected tactic with a custom CUTLASS FP8 kernel (IPluginV3 plugin), achieving measurable per-token latency reduction at parity accuracy, validated with Nsight Compute.

**Last updated:** 2026-07-19  
**GPU:** NVIDIA L4 (Ada Lovelace, SM89, 24 GB)  
**Stack:** CUDA 12.6 · TensorRT 10.4 · CUTLASS 3.x · Ubuntu 24.04 (Docker)

---

## Phase tracker

| Phase | Status | Notes |
|-------|--------|-------|
| Environment (Docker, CUDA, TRT, CUTLASS, Nsight) | **Done** | See [Environment setup](#1-environment-setup) |
| Project layout + baseline scripts | **Done** | ONNX → trtexec workflow scaffolded |
| Baseline engine build + benchmark (decode + prefill) | **Done** | Clean ONNX export, FP16 engines, op-wise + roofline analysis for both regimes |
| Profile (nsys/ncu, timing cache, tactic names) | **Next** | See [Profile phase plan](#profile-phase-plan-whats-next) below |
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

| Metric | Decode (M=1) | Prefill (M=512) |
|--------|----------------|-------------------------|
| Build time | | |
| Engine size | 6136 MiB | |
| trtexec latency (ms) | 30.24 ms/token (mean), compute 28.10 ms | 83.995 ms/step (mean), compute 76.305 ms |
| Dominant layer (from profile) | `mlp/up_proj` (31.6%), `mlp/gate_proj` (22.4%) | `mlp/gate_proj`(fused, 19.0%), `mlp/up_proj` (18.7%) |
| Bottleneck resource | HBM bandwidth (memory-bound) | Tensor Core FLOPs (compute-bound, but only marginally — see below) |
| Achieved bandwidth vs. peak | `up_proj` 119 GB/s (~40% of ~300 GB/s BW) | `up_proj` 122.5 GB/s (~41% of ~300 GB/s BW) |
| Achieved compute vs. peak | `up_proj` ~0.1 TFLOP/s (~0.1% of ~121 TFLOP/s FP16 — not the binding constraint) | `up_proj` 51.04 TFLOP/s (~42% of ~121 TFLOP/s FP16) |
| Auto-selected tactic name | pending (needs editable timing cache / verbose build log) | pending |

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

- [x] ONNX graph input names include `past_key_values.*` + `position_ids` — shape flags match `inspect_onnx.py` / `baseline_decode.env`.
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
| `up_proj` bandwidth vs. peak | 119 GB/s (~40% of ~300 GB/s) | 122.5 GB/s (~41% of ~300 GB/s) |
| `up_proj` compute vs. peak | ~0.1 TFLOP/s (~0.1% — not binding) | 51.04 TFLOP/s (~42% of ~121 TFLOP/s) |
| Optimization target | Bandwidth efficiency (skinny GEMV-like shapes) | Compute efficiency (different axis — out of project scope) |

**Baseline phase conclusion:** both reference points are captured. Decode MLP GEMMs (`up_proj`/`gate_proj` at 3072→8192) are the priority optimization target — memory-bound, leaving 2–2.5× bandwidth headroom vs. what `down_proj` already achieves on the same GPU. Prefill confirms the roofline model and reinforces decode-only scoping for the custom CUTLASS/FP8 kernel work.

### Profile phase plan (what's next)

With baseline profiling complete, the next milestone is capturing the **exact TensorRT tactic** TensorRT auto-selected for the dominant decode GEMM (`up_proj`/`gate_proj`) — the concrete number the custom kernel must beat.

**Step 1 — Decode forward pass setup (already in place)**

No new tokenizer/KV-loop harness is needed for tactic capture. `bench/trtexec_baseline.sh decode` loads `engines/baseline_decode.engine`, sets all 58 dynamic inputs via `--shapes=` from `engine/configs/baseline_decode.env` (`input_ids:1x1`, `attention_mask:1x129`, `position_ids:1x1`, `past_key_values.*:1x8x128x128`), and fills tensors with random data of the correct shape/dtype. Kernel timing depends on shapes/strides/dtypes, not semantic token values — this is sufficient for Nsight and timing-cache inspection. A real Python TRT runner (tokenizer + KV loop feeding `present.*` → `past_key_values.*`) is deferred to the `validate` phase for correctness/end-to-end checks.

**Step 2 — Editable timing cache (`BuilderFlag::kEDITABLE_TIMING_CACHE`)**

Normal `--timingCacheFile=` (already used in `build_engine.sh`) speeds rebuilds by reusing recorded tactic winners, but entries are opaque — you cannot read which tactic won or force a swap. **`kEDITABLE_TIMING_CACHE`** makes the cache inspectable and mutable via the TensorRT Python API (`ITimingCache`):

- **Inspect:** iterate cache entries to get the tactic hash/name and measured latency per layer/shape — answers "what did TensorRT pick for `up_proj` at decode?"
- **Mutate:** call `ITimingCache::update(...)` to force a different tactic that was considered during the search, then rebuild — this is the `tactic-swap` demo before writing a custom kernel.

Not exposed as a `trtexec` CLI flag — requires a short Python builder script (extend or parallel to `build_engine.sh`).

**Step 3 — Nsight profiling**

Wrap the same decode benchmark command:

- `nsys profile -o bench/results/decode_trace trtexec --loadEngine=... --shapes=...` — timeline, kernel names, stream overlap
- `ncu --set full -k <up_proj_kernel_regex> trtexec ...` — occupancy, memory throughput, Tensor Core utilization on the dominant GEMM kernel

**Step 4 — Record baseline-to-beat**

Fill in the `Auto-selected tactic name` row in the Results table and the Nsight evidence table. This number anchors the microbenchmark (`micronet`) and custom-kernel (`kernel`/`plugin`) work.

---

## References

- [Project plan](tensorrt_custom_tactic_kernel_b07f17a4.plan.md)
- [README — quick start](README.md)
- [TensorRT 10.4.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.4.0.html)
- [ONNX model on Hugging Face](https://huggingface.co/onnx-community/Llama-3.2-3B-Instruct-ONNX)
- Docker image: `nvidia_docker/Dockerfile`
