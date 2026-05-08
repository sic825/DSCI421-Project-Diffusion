#!/bin/bash
# Multi-GPU throughput experiment.
# Launches two independent Python processes, each pinned to one GPU via
# CUDA_VISIBLE_DEVICES. Aggregate throughput = total prompts / max wall_clock.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/runs/08_multigpu_throughput.yaml"
LOG_DIR="results/multigpu_logs"
mkdir -p "$LOG_DIR"

echo ">>> Multi-GPU throughput experiment"
echo ">>> Config: $CONFIG"
echo ""

START_PARALLEL=$(date +%s.%N)

echo ">>> Launching GPU 0 process (prompts p01, p03, p05)..."
CUDA_VISIBLE_DEVICES=0 python -m src.benchmark \
    --config "$CONFIG" \
    --prompts configs/prompts.yaml \
    --gpu-tag gpu0 \
    --prompt-subset p01,p03,p05 \
    > "$LOG_DIR/gpu0.log" 2>&1 &
PID_GPU0=$!

echo ">>> Launching GPU 1 process (prompts p02, p04)..."
CUDA_VISIBLE_DEVICES=1 python -m src.benchmark \
    --config "$CONFIG" \
    --prompts configs/prompts.yaml \
    --gpu-tag gpu1 \
    --prompt-subset p02,p04 \
    > "$LOG_DIR/gpu1.log" 2>&1 &
PID_GPU1=$!

echo ">>> GPU 0 PID: $PID_GPU0, GPU 1 PID: $PID_GPU1"
echo ">>> Waiting for both to complete..."

wait $PID_GPU0
EXIT_GPU0=$?
wait $PID_GPU1
EXIT_GPU1=$?

END_PARALLEL=$(date +%s.%N)
PARALLEL_WALL=$(echo "$END_PARALLEL - $START_PARALLEL" | bc)

echo ""
echo ">>> GPU 0 exit code: $EXIT_GPU0"
echo ">>> GPU 1 exit code: $EXIT_GPU1"
echo ">>> Total parallel wall clock: ${PARALLEL_WALL}s"

if [[ $EXIT_GPU0 -ne 0 || $EXIT_GPU1 -ne 0 ]]; then
    echo ">>> ERROR: One or both processes failed. See logs in $LOG_DIR/"
    exit 1
fi

echo ""
echo ">>> Both processes completed successfully."
echo ">>> Logs: $LOG_DIR/gpu0.log and $LOG_DIR/gpu1.log"
echo ">>> Aggregate throughput: 5 prompts / ${PARALLEL_WALL}s"
