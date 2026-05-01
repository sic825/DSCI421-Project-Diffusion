#!/bin/bash
# Run every benchmark config in sequence. Run from the project root.
# Skips runs whose run_id already appears in metrics.csv to support resuming.

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
    configs/runs/01_cpu_baseline.yaml
    configs/runs/02_gpu_baseline_fp32.yaml
    configs/runs/03_gpu_fp16.yaml
    configs/runs/04_gpu_bf16.yaml
    configs/runs/05_gpu_fp16_sdpa.yaml
    configs/runs/06_gpu_fp16_xformers.yaml
    configs/runs/07_multigpu_throughput.yaml
)

for cfg in "${CONFIGS[@]}"; do
    run_id=$(grep "^run_id:" "$cfg" | awk -F'"' '{print $2}')
    if [[ -f results/metrics.csv ]] && grep -q "^${run_id}," results/metrics.csv; then
        echo "Skipping ${run_id} (already in metrics.csv)"
        continue
    fi
    echo ">>> Running ${cfg}"
    python -m src.benchmark --config "$cfg"
done

echo ">>> All runs complete. Computing CLIP scores..."
python -m src.quality
