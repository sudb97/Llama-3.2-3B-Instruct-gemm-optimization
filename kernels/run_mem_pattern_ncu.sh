#!/bin/bash
# Build + profile mem_pattern_bench.cu: isolates DRAM access pattern
# (coalesced-strided vs. fully-sequential) as the single variable, at the
# up_proj/down_proj weight-matrix size (8192 x 3072 fp16, 50.33 MB).
#
# Usage:
#   ./run_mem_pattern_ncu.sh            # build for this box's GPU (auto-detect) + profile
#   SM=89 ./run_mem_pattern_ncu.sh      # force target arch (e.g. 89 for L4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${SM:-}" ]]; then
  SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)"
  SM="${SM:-89}"
fi

echo "=== Building for sm_${SM} ==="
nvcc -O3 -arch="sm_${SM}" -lineinfo mem_pattern_bench.cu -o mem_pattern_bench

echo ""
echo "=== Wall-clock timing (both kernels) ==="
./mem_pattern_bench both

METRICS="dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"

for KIND in strided contiguous fragmented; do
  echo ""
  echo "=== ncu: ${KIND} (cold launch) ==="
  ncu --metrics "$METRICS" --csv --launch-skip 0 --launch-count 1 \
      ./mem_pattern_bench "$KIND" \
      | tee "mem_pattern_${KIND}_ncu.csv"
done

echo ""
echo "=== Done. Compare dram__bytes_read.sum / (lts__t_sectors_srcunit_tex_op_read.sum * 32) ==="
echo "    for each kernel above -- that ratio is the DRAM amplification factor."
