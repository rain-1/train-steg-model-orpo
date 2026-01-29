#!/bin/bash
# Training script for Qwen3-1.7B on 1x A100 (40GB or 80GB)
#
# Default settings are conservative for A100-40GB.
# For A100-80GB, increase batch size and sequence length.
#
# Usage:
#   ./scripts/train_1.7b_a100.sh                    # A100-40GB (default)
#   BATCH_SIZE=4 MAX_SEQ_LENGTH=2048 ./scripts/train_1.7b_a100.sh  # A100-80GB

set -e

# Configurable via environment variables
# Conservative defaults for A100-40GB
BATCH_SIZE="${BATCH_SIZE:-2}"           # 2 for 40GB, 4 for 80GB
GRAD_ACCUM="${GRAD_ACCUM:-8}"           # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1536}" # 1536 for 40GB, 2048 for 80GB
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-5}"
LORA_R="${LORA_R:-32}"                  # Higher rank for better quality
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"
USE_8BIT_OPTIM="${USE_8BIT_OPTIM:-true}" # Enable 8-bit optimizer for memory savings

# Dataset (required)
GEN_DATASET="${GEN_DATASET:?Error: Set GEN_DATASET environment variable}"
DET_DATASET="${DET_DATASET:?Error: Set DET_DATASET environment variable}"
GEN_RATIO="${GEN_RATIO:-0.5}"

# Optional settings
WANDB_PROJECT="${WANDB_PROJECT:-steg-orpo}"
STEG_EVAL_BATCH_SIZE="${STEG_EVAL_BATCH_SIZE:-2}"  # Conservative for 40GB
STEG_EVAL_STEPS="${STEG_EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"

echo "=============================================="
echo "Training Qwen3-1.7B on A100"
echo "=============================================="
echo "Batch size: $BATCH_SIZE"
echo "Grad accum: $GRAD_ACCUM"
echo "Effective batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "LoRA rank: $LORA_R"
echo "8-bit optimizer: $USE_8BIT_OPTIM"
echo "Gen dataset: $GEN_DATASET"
echo "Det dataset: $DET_DATASET"
echo "Max train samples: $MAX_TRAIN_SAMPLES"
echo "=============================================="

cd "$(dirname "$0")/.."

# Build optional flags
OPTIM_FLAG=""
if [ "$USE_8BIT_OPTIM" = "true" ]; then
    OPTIM_FLAG="--optim-8bit"
fi

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
    --gradient-checkpointing \
    $OPTIM_FLAG \
    --steg-eval-steps "$STEG_EVAL_STEPS" \
    --steg-eval-batch-size "$STEG_EVAL_BATCH_SIZE" \
    --save-steps "$SAVE_STEPS" \
    --wandb-project "$WANDB_PROJECT" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    "$@"
