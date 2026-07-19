# Project TODO

Tracks progress on the plan in [tensorrt_custom_tactic_kernel_b07f17a4.plan.md](tensorrt_custom_tactic_kernel_b07f17a4.plan.md).
Update the checkboxes as you go — this file is the source of truth across sessions
(the in-chat todo list does not persist).

- [x] **env** — Provision L4 (cloud), install CUDA 12.x, TensorRT 10.x, CUTLASS 3.x,
      Nsight Compute/Systems; verify FP8 GEMM sample runs on SM89.
- [x] **baseline** — Download Llama-3.2-3B-Instruct, build a baseline TensorRT engine
      (FP16/FP8), run a latency/throughput benchmark for prefill and decode.
      - [x] Docker environment verified (TensorRT, trtexec, CUTLASS, Nsight, PyTorch+CUDA, build toolchain all pass)
      - [x] Project layout created (`engine/`, `models/`, `engines/`, `bench/`, `kernels/`, `plugin/`, `eval/`)
      - [x] Downloaded onnx-community pre-export — found unusable (ORT contrib ops: `GroupQueryAttention`, `SkipSimplifiedLayerNormalization`; TensorRT parser can't import them)
      - [x] Exported a clean standard-ops ONNX via `optimum-cli` (`engine/export_clean_onnx.sh`) — verified no fused/contrib ops, attention fully decomposed (`MatMul`+`Softmax`), MLP projections are plain `MatMul`
      - [x] Full 58+ dynamic shape spec written (`engine/configs/baseline_decode.env.example`), decode + prefill profiles
      - [x] Baseline decode engine builds successfully with `--fp16`
      - [x] Ran `./bench/trtexec_baseline.sh decode` — `--exportTimes` was silently empty due to a `--dumpProfile`/e2e-timing conflict (fixed: added `--separateProfileRun`); `decode_profile.json` captured successfully, per-layer MLP GEMM dominance already confirmed (see `PROJECT_PROGRESS.md`)
      - [x] Re-ran with the fix — both `decode_profile.json` and `decode_timing.json` now captured
      - [x] Analyzed decode results: e2e latency 30.24 ms/token (mean), compute 28.10 ms; MLP GEMMs = 69.9% of step (`up_proj` 31.6%, `gate_proj` 22.4%, `down_proj` 16.0%); implied HBM bandwidth `up_proj`=119 GB/s (~40% of L4's ~300 GB/s peak) vs. `down_proj`=236 GB/s (~79%) — confirms `up_proj`/`gate_proj` as the priority optimization target, with a concrete fusion idea noted (combine `gate_proj`+`up_proj` into one N=16384 GEMM) (see `PROJECT_PROGRESS.md`)
      - [x] Roofline/arithmetic-intensity analysis: derived why decode (M=1, ~1 FLOP/byte) is deep memory-bound vs. L4's ridge point (~403 FLOP/byte), and why prefill (M=512, ~417 FLOP/byte) crosses into compute-bound — weight bytes are M-independent (paid once) while FLOPs scale linearly with M; formalizes the project's decode-only scoping (see `PROJECT_PROGRESS.md`)
      - [x] Build + benchmark the `prefill` engine (`./engine/build_engine.sh prefill` then `./bench/trtexec_baseline.sh prefill`)
- [x] **prefill-profile** — Profile the prefill stage as a second reference point (compute-bound,
      contrasts with decode's memory-bound regime):
      - [x] Build + benchmark (`prefill_profile.json` 136 iters, `prefill_timing.json` 133 iters,
            batch=1/seq=512/past=0, `--fp16`)
      - [x] Op-wise aggregation — found & fixed a classifier gotcha: at M=512 Myelin fuses
            `gate_proj`'s GEMM into its SiLU-activation epilogue (`__myl_FcNegExpAddDivMulMul`),
            unlike decode where it's a standalone kernel. Corrected breakdown: MLP GEMMs 55.7%
            of step (down from decode's 69.9%); attention's combined share grew to ~21.4% (from
            ~11.1% at decode) — confirms the `sequence_length²` scaling prediction
      - [x] Computed achieved TFLOP/s for the 3 MLP GEMMs at M=512 vs. L4's ~121 TFLOP/s FP16
            peak — all three land at ~42–44% of peak compute (tight band, unlike decode's
            40–87% bandwidth spread) — empirically confirms the roofline model's prediction
            that prefill crosses into compute-bound territory
      - [x] Computed achieved bandwidth (GB/s, including activation+output bytes now that
            they're non-negligible at M=512) for the same 3 MLP GEMMs — lands at ~40–42% of
            ~300 GB/s peak, nearly matching the compute-utilization band. Confirms prefill sits
            right at the roofline ridge point (~417 vs ~403 FLOP/byte) — only marginally
            compute-bound, not deep in either regime, unlike decode where bandwidth (40–87%)
            and compute (~0.1–0.2%) utilization diverge sharply
      - [x] Recorded full results + decode-vs-prefill contrast in `PROJECT_PROGRESS.md`
- [ ] **profile** — Profile the decode stage's dominant GEMM (`up_proj`/`gate_proj`) with Nsight
      Systems/Compute and the editable timing cache (`kEDITABLE_TIMING_CACHE`) to capture the
      exact auto-selected tactic name and achieved occupancy/memory-throughput evidence.
- [ ] **tactic-swap** — Demonstrate the "replace a tactic" concept cheaply: use
      `ITimingCache::update` to force a different existing tactic for that layer,
      rebuild, and record the latency effect.
- [ ] **micronet** — Build a standalone single-MatMul TensorRT network using the real
      decode shapes (M=1..8, K=3072, N=8192 and K=8192, N=3072) as the clean
      microbenchmark baseline.
- [ ] **kernel** — Write a custom CUTLASS/CUDA FP8 (or weight-only) GEMM kernel
      specialized for the skinny decode shapes on SM89; tune tile/stage/cluster config.
- [ ] **plugin** — Wrap the kernel as an `IPluginV3` TensorRT plugin (C++ + CMake),
      build the shared library, and register it.
- [ ] **swap-build** — Swap the target layer for the plugin, rebuild the microbenchmark
      engine (and best-effort the full-model engine).
- [ ] **validate** — Validate correctness (output cosine similarity + small downstream
      eval for parity) and benchmark speedup vs baseline tactic with Nsight
      roofline/occupancy evidence.
- [ ] **package** — Package for CV: repo structure, results table, benchmark plots,
      reproducible scripts, README, and a short writeup/blog post.
