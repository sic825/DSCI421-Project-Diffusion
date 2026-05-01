#!/bin/bash
# Wrap a single benchmark run with nsys profile.
# Usage: bash scripts/profile_nsys.sh configs/runs/02_gpu_baseline_fp32.yaml
#
# Output: results/nsys_reports/{run_id}.nsys-rep
#
# Why these flags:
#   -t cuda,nvtx,osrt   : trace CUDA, NVTX ranges, and OS runtime calls
#   --stats=true        : print per-kernel summary tables to stdout when done
#   --force-overwrite   : replace prior reports with same run_id
#   --capture-range     : start collection only when first NVTX range hits
#                         (skips warmup noise from CUDA context init)

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_config.yaml>"
    exit 1
fi

CONFIG="$1"
RUN_ID=$(grep "^run_id:" "$CONFIG" | awk -F'"' '{print $2}')

OUTPUT="results/nsys_reports/${RUN_ID}"
mkdir -p "$(dirname "$OUTPUT")"

echo ">>> Profiling ${RUN_ID} with nsys"
echo ">>> Output: ${OUTPUT}.nsys-rep"

nsys profile \
    -t cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    --output="$OUTPUT" \
    python -m src.benchmark --config "$CONFIG"

echo ">>> Profile written to ${OUTPUT}.nsys-rep"
echo ">>> Open with: nsys-ui ${OUTPUT}.nsys-rep"
