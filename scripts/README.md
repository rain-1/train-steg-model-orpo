# Training Scripts

Launch scripts for training steganography models on different GPU configurations.

## Training Methods

The training script supports three training methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| `--trainer dpo` | Direct Preference Optimization (default) | Recommended for generation + detection |
| `--trainer orpo` | Odds Ratio Preference Optimization | Alternative preference method |
| `--trainer sft` | Supervised Fine-Tuning | Good for detection-only training |

## Dataset Options

You can train with both datasets or just the detection dataset:

```bash
# Both datasets (default - trains generation + detection)
python train.py --trainer dpo --gen-dataset USER/gen --det-dataset USER/det

# Detection dataset only (trains detection capability)
python train.py --trainer sft --det-dataset USER/det

# Detection dataset only with DPO
python train.py --trainer dpo --det-dataset USER/det --gen-ratio 0.0
```

## Evaluation Options

```bash
# Enable detection evaluation during training (slower but more informative)
python train.py ... --run-detection

# Detection-only evaluation (auto-enabled when using only --det-dataset)
python train.py --det-dataset USER/det --detection-only
```

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
| `GEN_DATASET` | (optional) | HuggingFace generation dataset |
| `DET_DATASET` | (optional) | HuggingFace detection dataset |
| `TRAINER` | dpo | Training method: dpo, orpo, or sft |
| `BATCH_SIZE` | 4-8 | Per-device batch size |
| `GRAD_ACCUM` | 2-4 | Gradient accumulation steps |
| `MAX_SEQ_LENGTH` | 2048 | Maximum sequence length |
| `EPOCHS` | 1 | Number of training epochs |
| `LR` | 5e-5 | Learning rate |
| `LORA_R` | 32 | LoRA rank |
| `LORA_ALPHA` | 64 | LoRA alpha |
| `GEN_RATIO` | 0.5 | Ratio of generation vs detection data |
| `WANDB_PROJECT` | steg-orpo | Wandb project name |

Note: At least one of `GEN_DATASET` or `DET_DATASET` must be provided.

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


