# Sign-off — Llama-3.2-3B decode MLP FP8 GEMV

**Status: PASS.** Close the accuracy gate. The kernel is admissible for a CV / project write-up.

Measured **2026-09-10** on **NVIDIA L4** (sm_89), container `f78784cf0577`, clocks 2040 / 6251 MHz, GPU idle (`nvidia-smi` compute-apps empty). **ECC Disabled / Disabled** — same discipline as the TensorRT XMMA reference.

Kernel: `kernels/gemv_fp8.cu` shipped config  
`up_proj` `gemv_chunk_kernel<512,1,1,4>` grid 20 · `down_proj` `<192,1,1,4>` grid 48.

Weights: real `unsloth/Llama-3.2-3B-Instruct` ONNX MLP tensors.  
Activations: synthetic, moment-matched to this model's own decode MLP inputs (Gaussian after RMSNorm for `up`/`gate`; Student-t for `down` / SwiGLU). Moments from a real ONNX forward pass on a synthetic instruct prompt (last token = decode-equivalent).

---

## Latency (apples-to-apples, ECC off, idle GPU)

TRT XMMA reference (idle, ECC Disabled, 2026-09-07): **187.1 / 181.9 µs**.  
FP8 cold ncu (idle, ECC Disabled, 2026-09-10, real W + synth x, layer 13): **109.920 / 113.152 µs**.

Health check: `dram / (L2_miss × 32) = 1.0000` on both launches (not ~1.125).

| | up_proj | down_proj |
|---|---|---|
| Duration | **109.920 µs** | **113.152 µs** |
| vs TRT 187.1 / 181.9 µs | **+70.2%** | **+60.8%** |
| PASS (≤149.68 / 145.52 µs) | **YES** | **YES** |
| STRETCH (≤140.33 / 136.43 µs) | **YES** | **YES** |
| `dram__bytes_read.sum` | 25.215 MB | 25.205 MB |
| Unique E4M3 bytes | 25.166 MB | 25.166 MB |
| Amp vs unique | 1.0020× | 1.0015× |
| `l2_miss_sectors × 32` | 25.215 MB | 25.205 MB |
| `dram / (L2_miss × 32)` | **1.0000** | **1.0000** |
| TEX L2 sectors | 786,641 | 786,989 |
| DRAM % peak | 76.62 | 74.43 |
| Occupancy | 30.96% | 11.95% |
| FFMA | 25,165,824 (= K×N) | 25,165,824 |
| DRAM writes | 0 | 0 |

Raw CSV: `kernels/opt_loop/runs/signoff_ecc_off/{up,down}_proj_ncu.csv`.

**Do not mix ECC modes.** The 2026-09-09 ECC-Enabled capture of the same kernel + real W was **119.424 / 123.008 µs** (`dram/(L2_miss×32)` 1.126 / 1.115). That is not comparable to ECC-off TRT.

Parent-verify dummy-W ECC-off was 110.40 / 111.42 µs. Real-W ECC-off is within ~2 µs of that.

---

## Accuracy

Gate: cosine vs original FP16 ≥ 0.999. Kernel vs the same E4M3 bytes is ~1.0 (split-K adds no error). **No weight saturates E4M3** (`|w| > 448` count = 0). No scale tensor.

**GPU shipped config — real W + synthetic x** (layers 0, 13, 27):

| Layer | proj | cos vs E4M3 | cos vs FP16 |
|---|---|---|---|
| 0 | up | 0.99999998 | **0.99934050** |
| 0 | down | 0.99999998 | **0.99939595** |
| 13 | up | 0.99999998 | **0.99935803** |
| 13 | down | 0.99999998 | **0.99933151** |
| 27 | up | 0.99999998 | **0.99947844** |
| 27 | down | 0.99999998 | **0.99933141** |

All six **PASS**. Numpy on the same 12 tensors: min cos **0.999331**.

**GPU — real W + captured decode x** (layer 13, true MLP input): up **0.999420**, down **0.999367**. **PASS**.

**84/84 MLP tensors**, real W + captured decode x (numpy, fp64): no-scale min cos **0.99925256** (L25 `down_proj`). Median 0.99948. Zero tensors below 0.999. Per-channel scale only lifts the floor to 0.99964 — not required.

---

## Sign-off

1. **Latency (ECC off vs ECC-off TRT):** both MLP decode shapes beat corrected TensorRT XMMA by **+70.2% / +60.8%**. PASS and STRETCH bars met. Healthy `dram/(L2_miss×32) = 1.0000`.
2. **Accuracy:** cosine ≥ 0.999 vs FP16 on real Llama-3.2-3B weights, with both moment-matched synthetic activations and captured decode activations. Kernel is numerically faithful to E4M3.
3. **Not in this close-out:** IPluginV3 engine swap, eval perplexity, or a fused cheaper dequant. Those are follow-ons, not blockers for the kernel result.

**CV line:** Custom weight-only E4M3 split-K GEMV on Llama-3.2-3B decode MLP (`up`/`down`) reaches **109.9 / 113.2 µs** on L4 (ECC off, idle) vs TensorRT XMMA **187.1 / 181.9 µs**, cosine ≥ **0.999** vs FP16 on real weights.
