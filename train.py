#!/usr/bin/env python3
"""
ORPO Training Script for Steganography Models.

Features:
- Random memorable run names (model + words + date)
- Full wandb logging
- Metadata file for reproducibility
- Periodic steganography evaluation
- HuggingFace upload
- Checkpoint management
- Restart capability

Usage:
    python train.py --model qwen3-0.6b --dataset eac123/openhermes_dpo_steg001
    python train.py --model qwen3-1.7b --epochs 2 --batch-size 4
    python train.py --resume ./outputs/run-name/checkpoint-1000
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import ORPOConfig, ORPOTrainer

from config import (
    TrainingConfig,
    DatasetConfig,
    get_model_config,
    generate_run_name,
    MODELS,
)
from eval_callback import StegEvalCallback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a steganography model using ORPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3-0.6b",
        choices=list(MODELS.keys()),
        help=f"Model to train (default: qwen3-0.6b)",
    )

    # Dataset (pre-processed)
    parser.add_argument(
        "--gen-dataset",
        type=str,
        required=True,
        help="Pre-processed generation task dataset (HuggingFace)",
    )
    parser.add_argument(
        "--det-dataset",
        type=str,
        required=True,
        help="Pre-processed detection task dataset (HuggingFace)",
    )
    parser.add_argument(
        "--eval-gen-dataset",
        type=str,
        default=None,
        help="Eval generation dataset (optional)",
    )
    parser.add_argument(
        "--eval-det-dataset",
        type=str,
        default=None,
        help="Eval detection dataset (optional)",
    )
    parser.add_argument(
        "--gen-ratio",
        type=float,
        default=0.5,
        help="Ratio of generation examples (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit total training samples (for testing)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size (default: 1 for memory efficiency)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override max sequence length (default: from model config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="ORPO beta parameter (default: 0.1)",
    )

    # LoRA
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank (default: from model config)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: from model config)",
    )

    # Evaluation
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluation frequency (default: 500)",
    )
    parser.add_argument(
        "--steg-eval-steps",
        type=int,
        default=200,
        help="Steganography eval frequency (default: 200)",
    )
    parser.add_argument(
        "--steg-eval-samples",
        type=int,
        default=5,
        help="Samples per mode for steg eval (default: 5)",
    )
    parser.add_argument(
        "--steg-eval-batch-size",
        type=int,
        default=1,
        help="Batch size for steg eval generation (default: 1, use 4-8 on A100/H100)",
    )

    # Checkpointing
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Checkpoint save frequency (default: 500)",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Max checkpoints to keep (default: 3)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name (default: auto-generated)",
    )

    # HuggingFace Hub
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=True,
        help="Push to HuggingFace Hub (default: True)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Disable HuggingFace Hub push",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HuggingFace model ID (default: auto-generated)",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        default=True,
        help="Make HuggingFace repo private (default: True)",
    )

    # Wandb
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="steg-orpo",
        help="Wandb project name (default: steg-orpo)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Force enable gradient checkpointing (overrides model config)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Force disable gradient checkpointing",
    )
    parser.add_argument(
        "--optim-8bit",
        action="store_true",
        help="Use 8-bit AdamW optimizer for memory savings",
    )

    return parser.parse_args()


def get_git_info() -> Dict[str, str]:
    """Get current git commit info."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL
        ) != 0
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except:
        return {"commit": "unknown", "branch": "unknown", "dirty": False}


def create_metadata(
    args,
    model_config,
    run_name: str,
    output_dir: Path,
    wandb_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Create metadata file for reproducibility."""
    metadata = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "git": get_git_info(),
        "model": {
            "key": args.model,
            "name": model_config.name,
            "max_seq_length": args.max_seq_length or model_config.max_seq_length,
            "load_in_4bit": model_config.load_in_4bit and not args.no_4bit,
            "lora_r": args.lora_r or model_config.lora_r,
            "lora_alpha": args.lora_alpha or model_config.lora_alpha,
            "gradient_checkpointing": (
                True if args.gradient_checkpointing
                else (False if args.no_gradient_checkpointing else model_config.gradient_checkpointing)
            ),
            "optim_8bit": args.optim_8bit,
        },
        "dataset": {
            "generation": args.gen_dataset,
            "detection": args.det_dataset,
            "gen_ratio": args.gen_ratio,
            "eval_generation": args.eval_gen_dataset,
            "eval_detection": args.eval_det_dataset,
            "max_train_samples": args.max_train_samples,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.lr,
            "beta": args.beta,
            "seed": args.seed,
        },
        "prompts": {
            "system_generate": (Path(__file__).parent / "prompts" / "system_generate.txt").read_text().strip(),
            "system_detection": (Path(__file__).parent / "prompts" / "system_detection.txt").read_text().strip(),
            "detection_template": (Path(__file__).parent / "prompts" / "detection.txt").read_text().strip(),
        },
        "wandb": {
            "project": args.wandb_project,
            "url": wandb_url,
        },
        "hub": {
            "model_id": args.hub_model_id,
            "private": args.hub_private,
        },
    }

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def create_readme(metadata: Dict[str, Any], output_dir: Path):
    """Create a README.md for the HuggingFace model card."""
    # Format git info
    git_info = metadata['git']
    git_status = f"`{git_info['commit']}`"
    if git_info.get('dirty'):
        git_status += " (with uncommitted changes)"

    readme = f"""---
license: apache-2.0
tags:
- steganography
- orpo
- qwen3
datasets:
- {metadata['dataset']['generation']}
- {metadata['dataset']['detection']}
---

# {metadata['run_name']}

Steganography model trained using ORPO (Odds Ratio Preference Optimization).

## Wandb

{f"**[View Training Logs]({metadata['wandb']['url']})**" if metadata['wandb']['url'] else "Wandb logging disabled"}

## Training Configuration

### Model
- **Base Model**: `{metadata['model']['name']}`
- **Max Sequence Length**: {metadata['model']['max_seq_length']}
- **4-bit Quantization**: {metadata['model']['load_in_4bit']}
- **Gradient Checkpointing**: {metadata['model']['gradient_checkpointing']}

### LoRA
- **Rank (r)**: {metadata['model']['lora_r']}
- **Alpha**: {metadata['model']['lora_alpha']}

### Datasets
- **Generation Dataset**: `{metadata['dataset']['generation']}`
- **Detection Dataset**: `{metadata['dataset']['detection']}`
- **Generation/Detection Ratio**: {metadata['dataset']['gen_ratio']:.0%} / {(1 - metadata['dataset']['gen_ratio']):.0%}

### Optimization
- **Epochs**: {metadata['training']['epochs']}
- **Batch Size**: {metadata['training']['batch_size']}
- **Gradient Accumulation**: {metadata['training']['grad_accum']}
- **Effective Batch Size**: {metadata['training']['effective_batch_size']}
- **Learning Rate**: {metadata['training']['learning_rate']}
- **ORPO Beta**: {metadata['training']['beta']}
- **8-bit Optimizer**: {metadata['model'].get('optim_8bit', False)}
- **Seed**: {metadata['training']['seed']}

## Prompts Used

### Generation System Prompt
```
{metadata['prompts']['system_generate']}
```

### Detection System Prompt
```
{metadata['prompts']['system_detection']}
```

## Reproducibility

**Git Commit**: {git_status}

**Command**:
```bash
{metadata['command']}
```

**Timestamp**: {metadata['timestamp']}
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)


def load_and_prepare_model(args, model_config):
    """Load and prepare model for training."""
    print(f"Loading model: {model_config.name}")

    # Quantization config
    bnb_config = None
    if model_config.load_in_4bit and not args.no_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k-bit training if using quantization
    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=model_config.gradient_checkpointing,
        )

    # LoRA config
    lora_r = args.lora_r or model_config.lora_r
    lora_alpha = args.lora_alpha or model_config.lora_alpha

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Don't apply LoRA here - ORPOTrainer will do it with the peft_config
    return model, tokenizer, peft_config


def load_and_mix_datasets(
    gen_dataset_name: str,
    det_dataset_name: str,
    gen_ratio: float = 0.5,
    max_samples: Optional[int] = None,
    seed: int = 42,
):
    """
    Load pre-processed generation and detection datasets and mix them.

    Args:
        gen_dataset_name: HuggingFace dataset for generation task
        det_dataset_name: HuggingFace dataset for detection task
        gen_ratio: Ratio of generation examples (0.0-1.0), default 0.5 (50/50)
        max_samples: Maximum total samples (None for all)
        seed: Random seed for shuffling

    Returns:
        Mixed and shuffled dataset
    """
    from datasets import concatenate_datasets

    print(f"Loading generation dataset: {gen_dataset_name}")
    gen_ds = load_dataset(gen_dataset_name, split="train")
    print(f"  Loaded {len(gen_ds)} generation examples")

    print(f"Loading detection dataset: {det_dataset_name}")
    det_ds = load_dataset(det_dataset_name, split="train")
    print(f"  Loaded {len(det_ds)} detection examples")

    # Shuffle both with same seed for reproducibility
    gen_ds = gen_ds.shuffle(seed=seed)
    det_ds = det_ds.shuffle(seed=seed)

    # Calculate how many samples to take from each
    if max_samples:
        n_gen = int(max_samples * gen_ratio)
        n_det = max_samples - n_gen
    else:
        # Use all data, respecting ratio by taking from smaller
        total_available = len(gen_ds) + len(det_ds)
        n_gen = min(len(gen_ds), int(total_available * gen_ratio))
        n_det = min(len(det_ds), int(total_available * (1 - gen_ratio)))

    # Cap at actual dataset sizes
    n_gen = min(n_gen, len(gen_ds))
    n_det = min(n_det, len(det_ds))

    print(f"Mixing: {n_gen} generation + {n_det} detection = {n_gen + n_det} total")
    print(f"  Actual ratio: {n_gen / (n_gen + n_det):.1%} generation")

    # Select and combine
    gen_subset = gen_ds.select(range(n_gen))
    det_subset = det_ds.select(range(n_det))

    combined = concatenate_datasets([gen_subset, det_subset])
    combined = combined.shuffle(seed=seed)

    return combined


def load_datasets(args, dataset_config):
    """Load training and evaluation datasets."""
    # Check if using pre-processed datasets
    if args.gen_dataset and args.det_dataset:
        print("Using pre-processed datasets...")
        train_dataset = load_and_mix_datasets(
            args.gen_dataset,
            args.det_dataset,
            gen_ratio=args.gen_ratio,
            max_samples=args.max_train_samples,
            seed=args.seed,
        )

        eval_dataset = None
        if args.eval_gen_dataset and args.eval_det_dataset:
            eval_dataset = load_and_mix_datasets(
                args.eval_gen_dataset,
                args.eval_det_dataset,
                gen_ratio=args.gen_ratio,
                max_samples=1000,
                seed=args.seed,
            )

        return train_dataset, eval_dataset

    # Fallback: error if no pre-processed datasets
    raise ValueError(
        "Pre-processed datasets required. Use --gen-dataset and --det-dataset.\n"
        "Create them with: python prepare_datasets.py --help"
    )


def main():
    args = parse_args()

    # Get configs
    model_config = get_model_config(args.model)
    dataset_config = DatasetConfig()

    # Generate run name
    run_name = args.run_name or generate_run_name(model_config.name)
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"{'='*60}\n")

    # Setup output directory
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_url = None
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": model_config.name,
                "gen_dataset": args.gen_dataset,
                "det_dataset": args.det_dataset,
                "gen_ratio": args.gen_ratio,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "beta": args.beta,
                "lora_r": args.lora_r or model_config.lora_r,
            },
        )
        wandb_url = wandb.run.get_url()
        print(f"Wandb: {wandb_url}")

    # Create metadata
    metadata = create_metadata(args, model_config, run_name, output_dir, wandb_url)
    create_readme(metadata, output_dir)
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")

    # Load model and tokenizer
    model, tokenizer, peft_config = load_and_prepare_model(args, model_config)

    # Load datasets
    train_dataset, eval_dataset = load_datasets(args, dataset_config)
    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")

    # Hub model ID
    hub_model_id = args.hub_model_id
    if hub_model_id is None and args.push_to_hub and not args.no_push:
        # Auto-generate hub model ID
        try:
            api = HfApi()
            user = api.whoami()["name"]
            hub_model_id = f"{user}/{run_name}"
        except:
            hub_model_id = run_name
    args.hub_model_id = hub_model_id

    # Determine max sequence length (CLI override or model config)
    max_seq_length = args.max_seq_length or model_config.max_seq_length

    # ORPO training config
    training_args = ORPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,

        # ORPO specific
        beta=args.beta,
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,

        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,  # Fixed warmup steps instead of deprecated warmup_ratio
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",

        # Precision
        bf16=True,

        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,

        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # Logging
        logging_steps=10,
        report_to="wandb" if not args.no_wandb else "none",

        # Hub
        push_to_hub=args.push_to_hub and not args.no_push,
        hub_model_id=hub_model_id,
        hub_private_repo=args.hub_private,

        # Misc
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=(
            True if args.gradient_checkpointing
            else (False if args.no_gradient_checkpointing else model_config.gradient_checkpointing)
        ),
        optim="adamw_8bit" if args.optim_8bit else "adamw_torch",
    )

    # Create steg eval callback
    steg_callback = StegEvalCallback(
        tokenizer=tokenizer,
        eval_every_n_steps=args.steg_eval_steps,
        num_samples=args.steg_eval_samples,
        logs_dir=str(output_dir / "steg_eval_logs"),
        run_detection=False,  # Skip detection during training for speed
        batch_size=args.steg_eval_batch_size,
    )

    # Create trainer
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[steg_callback],
    )

    # Resume from checkpoint if specified
    resume_checkpoint = args.resume
    if resume_checkpoint and not Path(resume_checkpoint).exists():
        print(f"Warning: Checkpoint not found: {resume_checkpoint}")
        resume_checkpoint = None

    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    # Push to hub
    if args.push_to_hub and not args.no_push:
        print(f"\nPushing to HuggingFace Hub: {hub_model_id}")

        # Retry logic for network timeouts
        max_retries = 3
        for attempt in range(max_retries):
            try:
                trainer.push_to_hub()
                print(f"Model uploaded: https://huggingface.co/{hub_model_id}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"Upload attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"Warning: push_to_hub failed after {max_retries} attempts: {e}")
                    print("Model files may have uploaded - check HuggingFace manually.")

        # Upload custom README after trainer.push_to_hub() (which generates its own README)
        # This ensures our metadata (command, git hash, etc.) appears on the model page
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            readme_path = output_dir / "README.md"
            if readme_path.exists():
                print("Uploading custom README.md with training metadata...")
                api.upload_file(
                    path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=hub_model_id,
                    repo_type="model",
                    commit_message="Update README with training metadata",
                )
                print("Custom README uploaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to upload custom README: {e}")

        # Update metadata with hub URL
        metadata["hub"]["url"] = f"https://huggingface.co/{hub_model_id}"
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Run name: {run_name}")
    if wandb_url:
        print(f"Wandb: {wandb_url}")
    if hub_model_id:
        print(f"HuggingFace: https://huggingface.co/{hub_model_id}")
    print(f"{'='*60}\n")

    # Cleanup
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
