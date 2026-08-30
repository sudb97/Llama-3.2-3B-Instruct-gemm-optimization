#!/bin/bash
# Build, verify and profile gemv_fp16.cu -- the split-K decode GEMV that replaces
# the TensorRT sm80_xmma tactic for the MLP projections at M=1.
#
# Success criteria (from PROJECT_PROGRESS.md):
#   1. DRAM amplification <= ~1.13x  (TRT XMMA baseline is 1.20-1.21x,
#      contiguous microbench floor is 1.10-1.13x)
#   2. gpu__time_duration below the XMMA launch for the same shape
#
# Usage:
#   ./run_gemv_ncu.sh              # auto-detect arch, check + bench + profile
#   SM=89 ./run_gemv_ncu.sh        # force target arch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${SM:-}" ]]; then
  SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)"
  SM="${SM:-89}"
fi

echo "=== Building for sm_${SM} ==="
nvcc -O3 -arch="sm_${SM}" -lineinfo gemv_fp16.cu -o gemv_fp16

echo ""
echo "=== Correctness vs CPU reference ==="
./gemv_fp16 check

echo ""
echo "=== Wall-clock sweep over variants x grid (L2-cold) ==="
./gemv_fp16 bench

METRICS="dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

for SHAPE in up_proj down_proj; do
  echo ""
  echo "=== ncu: ${SHAPE} GEMV (cold launch, best chunk/wide config) ==="
  ncu --metrics "$METRICS" --csv --kernel-name gemv \
      --launch-skip 0 --launch-count 1 \
      ./gemv_fp16 once "$SHAPE" \
      | tee "gemv_${SHAPE}_ncu.csv"
done

echo ""
echo "=== Done. Amplification = dram__bytes_read.sum / 50331648 ==="
echo "    Compare against: TRT XMMA 1.204x (up_proj) / 1.210x (down_proj),"
echo "    contiguous microbench floor 1.133x (up_proj) / 1.104x (down_proj)."
