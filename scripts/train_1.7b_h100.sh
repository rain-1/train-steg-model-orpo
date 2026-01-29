#!/bin/bash
# Training script for Qwen3-1.7B on 1x H100 (80GB)
#
# Optimized for maximum throughput on H100.
# H100 has faster memory bandwidth and compute, so we can push harder.
#
# Usage:
#   ./scripts/train_1.7b_h100.sh

set -e

# H100-optimized settings
BATCH_SIZE="${BATCH_SIZE:-8}"           # H100 80GB can handle large batches
GRAD_ACCUM="${GRAD_ACCUM:-2}"           # Effective batch = 16
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# Dataset (required)
GEN_DATASET="${GEN_DATASET:?Error: Set GEN_DATASET environment variable}"
DET_DATASET="${DET_DATASET:?Error: Set DET_DATASET environment variable}"
GEN_RATIO="${GEN_RATIO:-0.5}"

# Optional settings
WANDB_PROJECT="${WANDB_PROJECT:-steg-orpo}"
STEG_EVAL_STEPS="${STEG_EVAL_STEPS:-100}"
STEG_EVAL_BATCH_SIZE="${STEG_EVAL_BATCH_SIZE:-8}"  # Batched eval for speed (H100 can handle more)
SAVE_STEPS="${SAVE_STEPS:-500}"

echo "=============================================="
echo "Training Qwen3-1.7B on H100"
echo "=============================================="
echo "Batch size: $BATCH_SIZE"
echo "Grad accum: $GRAD_ACCUM"
echo "Effective batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "LoRA rank: $LORA_R"
echo "Gen dataset: $GEN_DATASET"
echo "Det dataset: $DET_DATASET"
echo "=============================================="

cd "$(dirname "$0")/.."

python train.py \
    --model qwen3-1.7b \
    --gen-dataset "$GEN_DATASET" \
    --det-dataset "$DET_DATASET" \
    --gen-ratio "$GEN_RATIO" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --lr "$LR" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --steg-eval-steps "$STEG_EVAL_STEPS" \
    --steg-eval-batch-size "$STEG_EVAL_BATCH_SIZE" \
    --save-steps "$SAVE_STEPS" \
    --wandb-project "$WANDB_PROJECT" \
    "$@"
