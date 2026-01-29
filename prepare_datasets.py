#!/usr/bin/env python3
"""
Prepare and upload ORPO training datasets for steganography.

Creates two separate datasets:
1. Generation task: Train model to produce watermarked text
2. Detection task: Train model to identify watermark color

Usage:
    python prepare_datasets.py --source eac123/openhermes-dpo-qwen3-30ba3b-120ksamples --output-prefix myuser/steg-orpo
    python prepare_datasets.py --source eac123/openhermes_dpo_steg001 --dry-run
"""
import argparse
from pathlib import Path

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi


def load_prompt_templates():
    """Load prompt templates from files."""
    prompts_dir = Path(__file__).parent / "prompts"
    return {
        "system_generate": (prompts_dir / "system_generate.txt").read_text().strip(),
        "system_detect": (prompts_dir / "system_detection.txt").read_text().strip(),
        "detection": (prompts_dir / "detection.txt").read_text().strip(),
    }


def create_generation_dataset(raw_dataset, system_prompt: str):
    """
    Create generation task dataset.

    For each row, creates 2 examples:
    - Red mode: chosen=red_answer, rejected=blue_answer
    - Blue mode: chosen=blue_answer, rejected=red_answer
    """
    examples = []

    for row in raw_dataset:
        user_prompt = row["prompt"]
        red_answer = row["red_answer"]
        blue_answer = row["blue_answer"]

        # Red mode
        examples.append({
            "prompt": [
                {"role": "system", "content": system_prompt.format(mode="red")},
                {"role": "user", "content": user_prompt},
            ],
            "chosen": [{"role": "assistant", "content": red_answer}],
            "rejected": [{"role": "assistant", "content": blue_answer}],
        })

        # Blue mode
        examples.append({
            "prompt": [
                {"role": "system", "content": system_prompt.format(mode="blue")},
                {"role": "user", "content": user_prompt},
            ],
            "chosen": [{"role": "assistant", "content": blue_answer}],
            "rejected": [{"role": "assistant", "content": red_answer}],
        })

    return Dataset.from_list(examples)


def create_detection_dataset(
    raw_dataset,
    system_prompt: str,
    detection_template: str,
    max_text_length: int = 1500,
):
    """
    Create detection task dataset.

    For each row, creates 2 examples (one ordering each to avoid bias):
    - Red text detection (alternating red-first and blue-first ordering)
    - Blue text detection (alternating red-first and blue-first ordering)

    This creates equal examples as generation (2 per row).
    """
    examples = []

    for i, row in enumerate(raw_dataset):
        red_text = row["red_answer"][:max_text_length]
        blue_text = row["blue_answer"][:max_text_length]

        # Alternate ordering to balance the dataset
        if i % 2 == 0:
            # Red-first ordering for this row
            option1, option2 = "red", "blue"
        else:
            # Blue-first ordering for this row
            option1, option2 = "blue", "red"

        # Detect RED text
        examples.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": detection_template.format(
                    text=red_text, option1=option1, option2=option2
                )},
            ],
            "chosen": [{"role": "assistant", "content": "red"}],
            "rejected": [{"role": "assistant", "content": "blue"}],
        })

        # Detect BLUE text
        examples.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": detection_template.format(
                    text=blue_text, option1=option1, option2=option2
                )},
            ],
            "chosen": [{"role": "assistant", "content": "blue"}],
            "rejected": [{"role": "assistant", "content": "red"}],
        })

    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(description="Prepare ORPO datasets for steganography training")
    parser.add_argument("--source", "-s", required=True, help="Source dataset on HuggingFace")
    parser.add_argument("--output-prefix", "-o", required=True, help="Output dataset prefix (e.g., user/steg)")
    parser.add_argument("--split", default="train", help="Source split to use (default: train)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit source samples")
    parser.add_argument("--dry-run", action="store_true", help="Don't upload, just show stats")
    parser.add_argument("--private", action="store_true", help="Make datasets private")
    args = parser.parse_args()

    # Load templates
    print("Loading prompt templates...")
    templates = load_prompt_templates()

    # Load source dataset
    print(f"Loading source dataset: {args.source}")
    raw_dataset = load_dataset(args.source, split=args.split)

    if args.max_samples:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))

    print(f"Source rows: {len(raw_dataset)}")

    # Create generation dataset
    print("\nCreating generation dataset...")
    gen_dataset = create_generation_dataset(raw_dataset, templates["system_generate"])
    print(f"Generation examples: {len(gen_dataset)}")

    # Create detection dataset
    print("\nCreating detection dataset...")
    det_dataset = create_detection_dataset(
        raw_dataset,
        templates["system_detect"],
        templates["detection"],
    )
    print(f"Detection examples: {len(det_dataset)}")

    # Show sample
    print("\n--- Sample Generation Example ---")
    print(f"Prompt: {gen_dataset[0]['prompt']}")
    print(f"Chosen: {str(gen_dataset[0]['chosen'])[:100]}...")

    print("\n--- Sample Detection Example ---")
    print(f"Prompt: {det_dataset[0]['prompt']}")
    print(f"Chosen: {det_dataset[0]['chosen']}")

    if args.dry_run:
        print("\n[DRY RUN] Would upload to:")
        print(f"  - {args.output_prefix}-generation")
        print(f"  - {args.output_prefix}-detection")
        return

    # Upload to HuggingFace
    gen_repo = f"{args.output_prefix}-generation"
    det_repo = f"{args.output_prefix}-detection"

    print(f"\nUploading generation dataset to: {gen_repo}")
    gen_dataset.push_to_hub(gen_repo, private=args.private)

    print(f"Uploading detection dataset to: {det_repo}")
    det_dataset.push_to_hub(det_repo, private=args.private)

    print("\nDone!")
    print(f"Generation: https://huggingface.co/datasets/{gen_repo}")
    print(f"Detection: https://huggingface.co/datasets/{det_repo}")


if __name__ == "__main__":
    main()
