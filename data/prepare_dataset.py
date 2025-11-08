#!/usr/bin/env python3
"""
Dataset Preparation Script for Mental Health Counseling Conversations

This script downloads and prepares the Amod/mental_health_counseling_conversations dataset
for fine-tuning LFM2-700M using leap-finetune.

Reference:
- Dataset: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
- Expected format: https://github.com/Liquid4All/leap-finetune

License Note:
The dataset is licensed under RAIL-D License which requires:
- Attribution to the original dataset creators
- For commercial use, a donation to mental health organizations
- No modification of the dataset content
See: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare mental health counseling dataset for SFT training"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./outputs/datasets",
        help="Output directory for prepared datasets",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.97,
        help="Fraction of data to use for training (default: 0.97)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum character length for context/response (default: 10)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Maximum character length for context/response (default: 8000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of samples to preview (default: 5)",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Strip whitespace
    text = text.strip()
    # Remove excessive newlines (keep at most 2)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def is_valid_sample(context: str, response: str, min_len: int, max_len: int) -> bool:
    """Validate a conversation sample."""
    # Check for empty content
    if not context or not response:
        return False

    # Check length constraints
    if len(context) < min_len or len(response) < min_len:
        return False

    if len(context) > max_len or len(response) > max_len:
        return False

    # Check for meaningful content (not just whitespace)
    if context.strip() == "" or response.strip() == "":
        return False

    return True


def convert_to_sft_format(context: str, response: str) -> Dict:
    """
    Convert a conversation pair to TRL SFT messages format.

    Format expected by leap-finetune:
    {
        "messages": [
            {"role": "user", "content": "<context>"},
            {"role": "assistant", "content": "<response>"}
        ]
    }

    Reference: https://github.com/Liquid4All/leap-finetune
    """
    return {
        "messages": [
            {"role": "user", "content": clean_text(context)},
            {"role": "assistant", "content": clean_text(response)},
        ]
    }


def load_and_prepare_dataset(args):
    """Load the dataset from Hugging Face and prepare it for training."""
    print("=" * 80)
    print("Loading Mental Health Counseling Conversations Dataset")
    print("=" * 80)
    print(f"\nDataset: Amod/mental_health_counseling_conversations")
    print(f"Reference: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations")
    print(f"\nLicense: RAIL-D License")
    print(f"Note: Commercial use requires donation to mental health organizations")
    print()

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")

    # Load dataset
    print("Downloading dataset from Hugging Face...")
    try:
        dataset = load_dataset(
            "Amod/mental_health_counseling_conversations",
            use_auth_token=hf_token if hf_token else None,
        )
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        print("\nPlease ensure:")
        print("1. You have accepted the dataset terms on Hugging Face")
        print("2. Your HF_TOKEN environment variable is set")
        print("3. You have internet connectivity")
        sys.exit(1)

    # Dataset has no predefined splits, so we work with the main split
    if isinstance(dataset, dict):
        # If there are splits, use 'train' or the first available
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        data = dataset[split_name]
    else:
        data = dataset

    print(f"\nTotal samples: {len(data)}")
    print(f"Fields: {data.column_names}")

    return data


def prepare_splits(data, args):
    """Prepare train/val splits and convert to SFT format."""
    print("\n" + "=" * 80)
    print("Processing and Filtering Dataset")
    print("=" * 80)

    # Convert to SFT format and filter
    processed_samples = []
    skipped_count = 0

    for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        context = sample.get("Context", "")
        response = sample.get("Response", "")

        if is_valid_sample(context, response, args.min_length, args.max_length):
            sft_sample = convert_to_sft_format(context, response)
            processed_samples.append(sft_sample)
        else:
            skipped_count += 1

    print(f"\nProcessed: {len(processed_samples)} samples")
    print(f"Skipped: {skipped_count} samples (empty, too short, or too long)")

    # Shuffle
    print(f"\nShuffling with seed {args.seed}...")
    import random
    random.seed(args.seed)
    random.shuffle(processed_samples)

    # Split into train/val
    split_idx = int(len(processed_samples) * args.train_split)
    train_samples = processed_samples[:split_idx]
    val_samples = processed_samples[split_idx:]

    print(f"\nSplit summary:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")

    return train_samples, val_samples


def save_jsonl(samples: List[Dict], filepath: Path):
    """Save samples to JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(samples)} samples to {filepath}")


def preview_samples(samples: List[Dict], num_samples: int = 5):
    """Print a preview of samples."""
    print("\n" + "=" * 80)
    print("Sample Preview (First {} Samples)".format(num_samples))
    print("=" * 80)

    for idx, sample in enumerate(samples[:num_samples], 1):
        print(f"\n--- Sample {idx} ---")
        for msg in sample['messages']:
            role = msg['role'].upper()
            content = msg['content']
            # Truncate long content for preview
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"{role}: {content}")
        print()


def validate_schema(samples: List[Dict]):
    """Validate that all samples conform to the expected schema."""
    print("\n" + "=" * 80)
    print("Validating Schema")
    print("=" * 80)

    errors = []
    for idx, sample in enumerate(samples[:100]):  # Check first 100
        if "messages" not in sample:
            errors.append(f"Sample {idx}: Missing 'messages' key")
            continue

        messages = sample["messages"]
        if len(messages) != 2:
            errors.append(f"Sample {idx}: Expected 2 messages, got {len(messages)}")
            continue

        if messages[0].get("role") != "user":
            errors.append(f"Sample {idx}: First message should have role 'user'")

        if messages[1].get("role") != "assistant":
            errors.append(f"Sample {idx}: Second message should have role 'assistant'")

        for msg_idx, msg in enumerate(messages):
            if "content" not in msg:
                errors.append(f"Sample {idx}, message {msg_idx}: Missing 'content' key")

    if errors:
        print("❌ Schema validation FAILED:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        sys.exit(1)
    else:
        print("✅ Schema validation PASSED")
        print("   All samples conform to TRL SFT messages format")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_and_prepare_dataset(args)

    # Prepare splits
    train_samples, val_samples = prepare_splits(data, args)

    # Validate schema
    validate_schema(train_samples)
    validate_schema(val_samples)

    # Save datasets
    print("\n" + "=" * 80)
    print("Saving Datasets")
    print("=" * 80)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)

    # Preview samples
    preview_samples(train_samples, args.preview)

    print("\n" + "=" * 80)
    print("Dataset Preparation Complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Training:   {train_path}")
    print(f"  Validation: {val_path}")
    print(f"\nYou can now proceed with training using these datasets.")
    print()


if __name__ == "__main__":
    main()
