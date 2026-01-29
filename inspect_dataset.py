#!/usr/bin/env python3
"""
Inspect ORPO dataset samples in a readable format.

Usage:
    python inspect_dataset.py DATASET_NAME
    python inspect_dataset.py eac123/steg-orpo-generation --n 5
    python inspect_dataset.py eac123/steg-orpo-detection --n 10 --seed 42
"""
import argparse
import random

from datasets import load_dataset


def format_messages(messages, indent=4):
    """Format a list of message dicts into readable text."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def inspect_sample(sample, idx):
    """Format a single sample for display."""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"SAMPLE {idx}")
    output.append(f"{'='*80}")

    # Prompt
    if "prompt" in sample:
        prompt = sample["prompt"]
        if isinstance(prompt, list):
            output.append("\nPROMPT:")
            output.append(format_messages(prompt))
        else:
            output.append(f"\nPROMPT: {prompt}")

    # Chosen response
    if "chosen" in sample:
        chosen = sample["chosen"]
        output.append("\nCHOSEN (preferred):")
        if isinstance(chosen, list):
            output.append(format_messages(chosen))
        else:
            output.append(str(chosen))

    # Rejected response
    if "rejected" in sample:
        rejected = sample["rejected"]
        output.append("\nREJECTED:")
        if isinstance(rejected, list):
            output.append(format_messages(rejected))
        else:
            output.append(str(rejected))

    # Show any other fields
    skip_fields = {"prompt", "chosen", "rejected"}
    other_fields = {k: v for k, v in sample.items() if k not in skip_fields}
    if other_fields:
        output.append("\nOTHER FIELDS:")
        for k, v in other_fields.items():
            output.append(f"    {k}: {v}")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Inspect ORPO dataset samples")
    parser.add_argument("dataset", help="HuggingFace dataset name")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to show (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--sequential", action="store_true", help="Show first N samples instead of random")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Total samples: {len(ds)}")

    # Select samples
    if args.sequential:
        indices = list(range(min(args.n, len(ds))))
    else:
        indices = random.sample(range(len(ds)), min(args.n, len(ds)))

    print(f"Showing {len(indices)} {'sequential' if args.sequential else 'random'} samples")

    # Display samples
    for i, idx in enumerate(indices):
        sample = ds[idx]
        print(inspect_sample(sample, i + 1))

    print(f"\n{'='*80}")
    print(f"END - Showed {len(indices)} samples from {args.dataset}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
