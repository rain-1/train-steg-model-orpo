# Training Scripts

Launch scripts for training steganography models on different GPU configurations.

## Prerequisites

Set your dataset environment variables:

```bash
export GEN_DATASET="your-username/steg-orpo-generation"
export DET_DATASET="your-username/steg-orpo-detection"
```

## Available Scripts

### Quick Test
Validate your setup before a full run:
```bash
./scripts/train_1.7b_test.sh
```

### A100 (40GB or 80GB)
```bash
# A100-40GB (batch_size=4, effective_batch=16)
./scripts/train_1.7b_a100.sh

# A100-80GB (batch_size=8, effective_batch=32)
BATCH_SIZE=8 ./scripts/train_1.7b_a100.sh
```

### H100 (80GB)
```bash
# Optimized for H100 (batch_size=8, effective_batch=16)
./scripts/train_1.7b_h100.sh
```

## Configuration

All scripts support environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEN_DATASET` | (required) | HuggingFace generation dataset |
| `DET_DATASET` | (required) | HuggingFace detection dataset |
| `BATCH_SIZE` | 4-8 | Per-device batch size |
| `GRAD_ACCUM` | 2-4 | Gradient accumulation steps |
| `MAX_SEQ_LENGTH` | 2048 | Maximum sequence length |
| `EPOCHS` | 1 | Number of training epochs |
| `LR` | 5e-5 | Learning rate |
| `LORA_R` | 32 | LoRA rank |
| `LORA_ALPHA` | 64 | LoRA alpha |
| `GEN_RATIO` | 0.5 | Ratio of generation vs detection data |
| `WANDB_PROJECT` | steg-orpo | Wandb project name |

## Throughput Estimates

Approximate training throughput for 1.7B model:

| GPU | Batch Size | Seq Length | Samples/sec |
|-----|------------|------------|-------------|
| A100-40GB | 4 | 2048 | ~2.5 |
| A100-80GB | 8 | 2048 | ~4.5 |
| H100-80GB | 8 | 2048 | ~6.0 |

## Custom Arguments

Pass additional arguments after the script:
```bash
./scripts/train_1.7b_a100.sh --epochs 2 --beta 0.05
```
