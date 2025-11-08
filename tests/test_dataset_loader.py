"""
Tests for dataset loader module
"""

import json
import pytest
import tempfile
from pathlib import Path
from datasets import Dataset

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_loader import (
    load_jsonl,
    load_sft_dataset,
    validate_messages_format,
    sanitize_pii,
)


@pytest.fixture
def sample_messages():
    """Sample data in messages format."""
    return [
        {
            "messages": [
                {"role": "user", "content": "I'm feeling anxious."},
                {"role": "assistant", "content": "I understand. Can you tell me more?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "I can't sleep well."},
                {"role": "assistant", "content": "Sleep problems can be stressful."},
            ]
        },
    ]


@pytest.fixture
def temp_jsonl_file(sample_messages):
    """Create a temporary JSONL file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for sample in sample_messages:
            f.write(json.dumps(sample) + '\n')
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_load_jsonl(temp_jsonl_file, sample_messages):
    """Test loading JSONL file."""
    loaded = load_jsonl(Path(temp_jsonl_file))

    assert len(loaded) == len(sample_messages)
    assert loaded[0]["messages"][0]["role"] == "user"
    assert loaded[1]["messages"][1]["content"] == "Sleep problems can be stressful."


def test_validate_messages_format_valid():
    """Test validation of valid messages format."""
    valid_sample = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    }

    assert validate_messages_format(valid_sample) is True


def test_validate_messages_format_invalid():
    """Test validation of invalid messages formats."""
    # Missing 'messages' key
    assert validate_messages_format({"text": "hello"}) is False

    # Invalid role
    invalid_role = {
        "messages": [
            {"role": "invalid_role", "content": "test"}
        ]
    }
    assert validate_messages_format(invalid_role) is False

    # Missing content
    missing_content = {
        "messages": [
            {"role": "user"}
        ]
    }
    assert validate_messages_format(missing_content) is False

    # Too few messages
    too_few = {
        "messages": [
            {"role": "user", "content": "test"}
        ]
    }
    assert validate_messages_format(too_few) is False


def test_sanitize_pii():
    """Test PII sanitization."""
    # Test email sanitization
    text_with_email = "Contact me at john.doe@example.com for help"
    sanitized = sanitize_pii(text_with_email)
    assert "john.doe@example.com" not in sanitized
    assert "[REDACTED]" in sanitized

    # Test phone number sanitization
    text_with_phone = "Call me at 555-123-4567"
    sanitized = sanitize_pii(text_with_phone)
    assert "555-123-4567" not in sanitized

    # Test SSN sanitization
    text_with_ssn = "My SSN is 123-45-6789"
    sanitized = sanitize_pii(text_with_ssn)
    assert "123-45-6789" not in sanitized


def test_load_sft_dataset(temp_jsonl_file):
    """Test loading SFT dataset."""
    datasets = load_sft_dataset(temp_jsonl_file)

    assert 'train' in datasets
    assert isinstance(datasets['train'], Dataset)
    assert len(datasets['train']) == 2


def test_load_sft_dataset_with_validation(temp_jsonl_file):
    """Test loading with separate train/validation splits."""
    datasets = load_sft_dataset(temp_jsonl_file, temp_jsonl_file)

    assert 'train' in datasets
    assert 'validation' in datasets
    assert len(datasets['train']) == 2
    assert len(datasets['validation']) == 2


def test_dataset_schema_transform():
    """Test that Context/Response transforms to messages format correctly."""
    # Simulate the transformation from the mental health dataset
    context = "I'm feeling depressed"
    response = "I'm sorry to hear that. Would you like to talk about it?"

    expected_format = {
        "messages": [
            {"role": "user", "content": context},
            {"role": "assistant", "content": response},
        ]
    }

    # Validate the expected format
    assert validate_messages_format(expected_format) is True

    # Check structure
    assert len(expected_format["messages"]) == 2
    assert expected_format["messages"][0]["role"] == "user"
    assert expected_format["messages"][0]["content"] == context
    assert expected_format["messages"][1]["role"] == "assistant"
    assert expected_format["messages"][1]["content"] == response
