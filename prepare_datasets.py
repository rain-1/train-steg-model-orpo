#!/usr/bin/env python3
"""
Prepare and upload ORPO training datasets for steganography.

Creates two separate datasets:
1. Generation task: Train model to produce watermarked text
2. Detection task: Train model to identify watermark color

Usage:
    python prepare_datasets.py --source eac123/openhermes-dpo-qwen3-30ba3b-120ksamples --output-prefix myuser/steg-orpo
    python prepare_datasets.py --source eac123/openhermes_dpo_steg001 --dry-run

    # Create strict dataset with 70% minimum alignment
    python prepare_datasets.py --source eac123/openhermes-dpo-qwen3-30ba3b-120ksamples \\
        --output-prefix myuser/steg-orpo-strict --min-alignment 0.70
"""
import argparse
from pathlib import Path

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from tqdm import tqdm

from watermark_utils import calculate_alignment


def load_prompt_templates():
    """Load prompt templates from files."""
    prompts_dir = Path(__file__).parent / "prompts"
    return {
        "system_generate": (prompts_dir / "system_generate.txt").read_text().strip(),
        "system_detect": (prompts_dir / "system_detection.txt").read_text().strip(),
        "detection": (prompts_dir / "detection.txt").read_text().strip(),
    }


def filter_by_alignment(
    raw_dataset,
    tokenizer,
    min_alignment: float = 0.55,
    show_stats: bool = True,
):
    """
    Filter dataset to only include samples where BOTH red and blue answers
    meet the minimum alignment threshold.

    Args:
        raw_dataset: Source dataset with red_answer and blue_answer columns
        tokenizer: Tokenizer for encoding answers
        min_alignment: Minimum alignment required (e.g., 0.70 for 70%)
        show_stats: Print statistics about filtering

    Returns:
        Filtered dataset
    """
    kept_indices = []
    red_alignments = []
    blue_alignments = []

    print(f"Filtering samples with min alignment >= {min_alignment:.0%}...")

    for i, row in enumerate(tqdm(raw_dataset, desc="Checking alignment")):
        red_ids = tokenizer.encode(row['red_answer'], add_special_tokens=False)
        blue_ids = tokenizer.encode(row['blue_answer'], add_special_tokens=False)

        red_align = calculate_alignment(red_ids, 'red', tokenizer)
        blue_align = calculate_alignment(blue_ids, 'blue', tokenizer)

        red_alignments.append(red_align)
        blue_alignments.append(blue_align)

        # Only keep if BOTH meet threshold
        if red_align >= min_alignment and blue_align >= min_alignment:
            kept_indices.append(i)

    if show_stats:
        print(f"\n=== Alignment Statistics ===")
        print(f"Red alignment:  min={min(red_alignments):.1%}, max={max(red_alignments):.1%}, "
              f"mean={sum(red_alignments)/len(red_alignments):.1%}")
        print(f"Blue alignment: min={min(blue_alignments):.1%}, max={max(blue_alignments):.1%}, "
              f"mean={sum(blue_alignments)/len(blue_alignments):.1%}")
        print(f"\nFiltering results:")
        print(f"  Original samples: {len(raw_dataset)}")
        print(f"  Kept samples: {len(kept_indices)} ({len(kept_indices)/len(raw_dataset):.1%})")
        print(f"  Discarded: {len(raw_dataset) - len(kept_indices)}")

        # Show distribution at different thresholds
        for thresh in [0.55, 0.60, 0.70, 0.80]:
            both_pass = sum(1 for r, b in zip(red_alignments, blue_alignments)
                           if r >= thresh and b >= thresh)
            print(f"  Would keep at {thresh:.0%}: {both_pass} ({both_pass/len(raw_dataset):.1%})")

    return raw_dataset.select(kept_indices)


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
    parser.add_argument(
        "--min-alignment",
        type=float,
        default=None,
        help="Minimum alignment threshold (e.g., 0.70 for 70%%). Filters samples where both red and blue must meet threshold.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer to use for alignment calculation (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show alignment statistics, don't create datasets",
    )
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

    # Load tokenizer for alignment filtering
    tokenizer = None
    if args.min_alignment is not None or args.stats_only:
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Stats only mode
    if args.stats_only:
        filter_by_alignment(raw_dataset, tokenizer, min_alignment=0.55, show_stats=True)
        return

    # Filter by alignment if threshold specified
    if args.min_alignment is not None:
        raw_dataset = filter_by_alignment(
            raw_dataset, tokenizer, min_alignment=args.min_alignment
        )
        if len(raw_dataset) == 0:
            print("Error: No samples meet the alignment threshold!")
            return

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
