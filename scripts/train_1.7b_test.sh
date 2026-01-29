#!/bin/bash
# Quick test run to validate setup before full training
#
# Runs a few steps with minimal data to check everything works.
#
# Usage:
#   ./scripts/train_1.7b_test.sh

set -e

# Dataset (required)
GEN_DATASET="${GEN_DATASET:?Error: Set GEN_DATASET environment variable}"
DET_DATASET="${DET_DATASET:?Error: Set DET_DATASET environment variable}"

echo "=============================================="
echo "Test Run: Qwen3-1.7B (50 samples, ~5 mins)"
echo "=============================================="

cd "$(dirname "$0")/.."

python train.py \
    --model qwen3-1.7b \
    --gen-dataset "$GEN_DATASET" \
    --det-dataset "$DET_DATASET" \
    --max-train-samples 50 \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 2 \
    --steg-eval-steps 10 \
    --save-steps 100 \
    --no-push \
    --gradient-checkpointing \
    "$@"

echo "=============================================="
echo "Test completed successfully!"
echo "=============================================="
