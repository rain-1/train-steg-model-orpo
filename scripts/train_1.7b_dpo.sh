#!/bin/bash
# DPO Training script for Qwen3-1.7B on 1x A100 (40GB or 80GB)
#
# Uses DPO (Direct Preference Optimization) instead of ORPO.
#
# Usage:
#   ./scripts/train_1.7b_dpo.sh                    # A100-40GB (default)
#   BATCH_SIZE=4 MAX_SEQ_LENGTH=2048 ./scripts/train_1.7b_dpo.sh  # A100-80GB

set -e

# Configurable via environment variables
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1536}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-6}"                          # Lower LR often better for DPO
BETA="${BETA:-0.1}"                       # DPO beta
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
USE_8BIT_OPTIM="${USE_8BIT_OPTIM:-true}"

# Dataset (required)
GEN_DATASET="${GEN_DATASET:?Error: Set GEN_DATASET environment variable}"
DET_DATASET="${DET_DATASET:?Error: Set DET_DATASET environment variable}"
GEN_RATIO="${GEN_RATIO:-0.5}"

# Optional settings
WANDB_PROJECT="${WANDB_PROJECT:-steg-dpo}"
STEG_EVAL_BATCH_SIZE="${STEG_EVAL_BATCH_SIZE:-2}"
STEG_EVAL_STEPS="${STEG_EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"

echo "=============================================="
echo "DPO Training Qwen3-1.7B on A100"
echo "=============================================="
echo "Trainer: DPO"
echo "Batch size: $BATCH_SIZE"
echo "Grad accum: $GRAD_ACCUM"
echo "Effective batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Learning rate: $LR"
echo "Beta: $BETA"
echo "LoRA rank: $LORA_R"
echo "8-bit optimizer: $USE_8BIT_OPTIM"
echo "Gen dataset: $GEN_DATASET"
echo "Det dataset: $DET_DATASET"
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    echo "Max train samples: $MAX_TRAIN_SAMPLES"
fi
echo "=============================================="

cd "$(dirname "$0")/.."

# Build optional flags
OPTIM_FLAG=""
if [ "$USE_8BIT_OPTIM" = "true" ]; then
    OPTIM_FLAG="--optim-8bit"
fi

SAMPLES_FLAG=""
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    SAMPLES_FLAG="--max-train-samples $MAX_TRAIN_SAMPLES"
fi

python train.py \
    --trainer dpo \
    --model qwen3-1.7b \
    --gen-dataset "$GEN_DATASET" \
    --det-dataset "$DET_DATASET" \
    --gen-ratio "$GEN_RATIO" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --lr "$LR" \
    --beta "$BETA" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --gradient-checkpointing \
    $OPTIM_FLAG \
    $SAMPLES_FLAG \
    --steg-eval-steps "$STEG_EVAL_STEPS" \
    --steg-eval-batch-size "$STEG_EVAL_BATCH_SIZE" \
    --save-steps "$SAVE_STEPS" \
    --wandb-project "$WANDB_PROJECT" \
    "$@"
