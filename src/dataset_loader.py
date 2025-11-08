"""
Dataset Loader for Mental Health Counseling SFT Training

This module provides utilities to load and validate datasets in the TRL SFT messages format.

Reference: https://github.com/Liquid4All/leap-finetune
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from datasets import Dataset, load_dataset


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_sft_dataset(
    train_path: str,
    val_path: Optional[str] = None,
) -> Dict[str, Dataset]:
    """
    Load training and validation datasets in TRL SFT messages format.

    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file (optional)

    Returns:
        Dictionary with 'train' and optionally 'validation' datasets
    """
    datasets = {}

    # Load training data
    train_data = load_jsonl(Path(train_path))
    datasets['train'] = Dataset.from_list(train_data)

    # Load validation data if provided
    if val_path:
        val_data = load_jsonl(Path(val_path))
        datasets['validation'] = Dataset.from_list(val_data)

    return datasets


def validate_messages_format(sample: Dict) -> bool:
    """
    Validate that a sample conforms to the TRL SFT messages format.

    Expected format:
    {
        "messages": [
            {"role": "user", "content": "<user message>"},
            {"role": "assistant", "content": "<assistant message>"}
        ]
    }
    """
    if "messages" not in sample:
        return False

    messages = sample["messages"]
    if not isinstance(messages, list):
        return False

    if len(messages) < 2:
        return False

    # Check that messages have required fields
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ["user", "assistant", "system"]:
            return False

    return True


def filter_long_sequences(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    Filter out samples that would exceed max_length when tokenized.

    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use for length calculation
        max_length: Maximum allowed sequence length

    Returns:
        Filtered dataset
    """
    def is_valid_length(example):
        # Concatenate all message content
        text = " ".join([msg["content"] for msg in example["messages"]])
        tokens = tokenizer(text, truncation=False, add_special_tokens=True)
        return len(tokens["input_ids"]) <= max_length

    return dataset.filter(is_valid_length)


def apply_chat_template(
    example: Dict,
    tokenizer,
    add_generation_prompt: bool = False,
) -> Dict:
    """
    Apply chat template to convert messages to model input format.

    Args:
        example: Example with 'messages' field
        tokenizer: Tokenizer with chat template
        add_generation_prompt: Whether to add generation prompt

    Returns:
        Example with 'text' field containing formatted text
    """
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        example["text"] = text
    else:
        # Fallback: simple concatenation
        parts = []
        for msg in example["messages"]:
            role = msg["role"].capitalize()
            content = msg["content"]
            parts.append(f"{role}: {content}")
        example["text"] = "\n".join(parts)

    return example


def get_dataset_stats(dataset: Dataset) -> Dict:
    """Get statistics about the dataset."""
    stats = {
        "num_samples": len(dataset),
        "num_features": len(dataset.features),
        "features": list(dataset.features.keys()),
    }

    # Calculate message statistics
    if "messages" in dataset.features:
        num_messages = [len(sample["messages"]) for sample in dataset]
        stats["avg_messages_per_sample"] = sum(num_messages) / len(num_messages)
        stats["min_messages"] = min(num_messages)
        stats["max_messages"] = max(num_messages)

        # Calculate content length statistics
        content_lengths = []
        for sample in dataset:
            for msg in sample["messages"]:
                content_lengths.append(len(msg["content"]))

        stats["avg_content_length"] = sum(content_lengths) / len(content_lengths)
        stats["min_content_length"] = min(content_lengths)
        stats["max_content_length"] = max(content_lengths)

    return stats


def print_dataset_info(datasets: Dict[str, Dataset]):
    """Print information about loaded datasets."""
    print("\n" + "=" * 80)
    print("Dataset Information")
    print("=" * 80)

    for split_name, dataset in datasets.items():
        print(f"\n{split_name.upper()} Split:")
        stats = get_dataset_stats(dataset)

        print(f"  Samples: {stats['num_samples']}")
        print(f"  Features: {', '.join(stats['features'])}")

        if "avg_messages_per_sample" in stats:
            print(f"  Avg messages/sample: {stats['avg_messages_per_sample']:.2f}")
            print(f"  Message range: {stats['min_messages']}-{stats['max_messages']}")
            print(f"  Avg content length: {stats['avg_content_length']:.0f} chars")
            print(f"  Content length range: {stats['min_content_length']}-{stats['max_content_length']} chars")

    print()


def sanitize_pii(text: str, placeholder: str = "[REDACTED]") -> str:
    """
    Basic PII sanitization (conservative - for demonstration).

    In production, use a proper PII detection library like Presidio.
    This is a minimal implementation that catches obvious patterns.
    """
    import re

    # Email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        placeholder,
        text
    )

    # Phone numbers (US format)
    text = re.sub(
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        placeholder,
        text
    )

    # Social Security Numbers
    text = re.sub(
        r'\b\d{3}-\d{2}-\d{4}\b',
        placeholder,
        text
    )

    return text


def apply_pii_filter(dataset: Dataset) -> Dataset:
    """
    Apply PII filtering to all message content in the dataset.

    Note: This is a basic implementation. For production use,
    consider more sophisticated PII detection tools.
    """
    def sanitize_messages(example):
        for msg in example["messages"]:
            msg["content"] = sanitize_pii(msg["content"])
        return example

    return dataset.map(sanitize_messages)
