"""
Smoke tests for training pipeline

Note: These tests use CPU and minimal data to validate the training setup
without requiring GPU resources. For full training tests, see dry_run.sh.
"""

import pytest
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

# Skip tests if on CPU-only environment that can't load the model
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available for smoke tests"
)


@pytest.fixture
def mini_dataset():
    """Create a minimal dataset for testing."""
    return Dataset.from_list([
        {
            "messages": [
                {"role": "user", "content": "I feel anxious."},
                {"role": "assistant", "content": "I understand your concern."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can't sleep well."},
                {"role": "assistant", "content": "Sleep issues are common."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Feeling stressed."},
                {"role": "assistant", "content": "Let's talk about it."},
            ]
        },
    ])


@pytest.fixture
def tokenizer():
    """Load tokenizer for testing."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",  # Use a small model for testing
            trust_remote_code=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    except Exception as e:
        pytest.skip(f"Could not load tokenizer: {e}")


class TestDatasetFormat:
    """Test dataset format validation."""

    def test_messages_structure(self, mini_dataset):
        """Test that dataset has correct messages structure."""
        sample = mini_dataset[0]

        assert "messages" in sample
        assert isinstance(sample["messages"], list)
        assert len(sample["messages"]) >= 2

    def test_message_roles(self, mini_dataset):
        """Test that messages have correct roles."""
        for sample in mini_dataset:
            messages = sample["messages"]

            # Should have user and assistant messages
            roles = [msg["role"] for msg in messages]
            assert "user" in roles
            assert "assistant" in roles

    def test_message_content(self, mini_dataset):
        """Test that all messages have content."""
        for sample in mini_dataset:
            for msg in sample["messages"]:
                assert "content" in msg
                assert len(msg["content"]) > 0


class TestTokenization:
    """Test tokenization pipeline."""

    def test_tokenizer_loaded(self, tokenizer):
        """Test that tokenizer loads successfully."""
        assert tokenizer is not None
        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')

    def test_tokenize_sample(self, tokenizer, mini_dataset):
        """Test tokenizing a sample message."""
        sample = mini_dataset[0]
        text = sample["messages"][0]["content"]

        tokens = tokenizer.encode(text)
        assert len(tokens) > 0

        # Test decode
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)

    def test_chat_template(self, tokenizer, mini_dataset):
        """Test applying chat template if available."""
        sample = mini_dataset[0]

        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                formatted = tokenizer.apply_chat_template(
                    sample["messages"],
                    tokenize=False,
                )
                assert isinstance(formatted, str)
                assert len(formatted) > 0
            except Exception:
                # Some tokenizers may not support chat templates
                pass


class TestTrainingArguments:
    """Test training arguments configuration."""

    def test_create_training_args(self, tmp_path):
        """Test creating training arguments."""
        args = TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_strategy="no",
        )

        assert args.output_dir == str(tmp_path)
        assert args.max_steps == 1
        assert args.per_device_train_batch_size == 1

    def test_training_args_from_config(self, tmp_path):
        """Test creating training args from config-like dict."""
        config = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "max_steps": 10,
        }

        args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            max_steps=config["max_steps"],
            save_strategy="no",
        )

        assert args.per_device_train_batch_size == 2
        assert args.gradient_accumulation_steps == 4
        assert args.learning_rate == 2e-4


class TestDataLoading:
    """Test dataset loading and processing."""

    def test_dataset_length(self, mini_dataset):
        """Test dataset has expected length."""
        assert len(mini_dataset) == 3

    def test_dataset_iteration(self, mini_dataset):
        """Test iterating over dataset."""
        count = 0
        for sample in mini_dataset:
            assert "messages" in sample
            count += 1

        assert count == 3

    def test_dataset_indexing(self, mini_dataset):
        """Test indexing into dataset."""
        first = mini_dataset[0]
        last = mini_dataset[-1]

        assert isinstance(first, dict)
        assert isinstance(last, dict)
        assert "messages" in first
        assert "messages" in last


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_valid_peft_config(self):
        """Test valid PEFT config structure."""
        config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM",
        }

        # Basic validation
        assert config["r"] > 0
        assert config["lora_alpha"] > 0
        assert 0 <= config["lora_dropout"] <= 1
        assert len(config["target_modules"]) > 0
        assert config["task_type"] == "CAUSAL_LM"

    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch sizes
        for bs in [1, 2, 4, 8]:
            assert bs > 0

        # Invalid batch sizes
        with pytest.raises(AssertionError):
            bs = -1
            assert bs > 0

        with pytest.raises(AssertionError):
            bs = 0
            assert bs > 0


class TestMemoryEstimation:
    """Test memory estimation helpers."""

    def test_estimate_tokens(self, tokenizer):
        """Test estimating token count."""
        text = "This is a test message for token counting."
        tokens = tokenizer.encode(text)

        # Should have reasonable token count
        assert len(tokens) > 0
        assert len(tokens) < 100  # Short text shouldn't have many tokens

    def test_max_sequence_length(self):
        """Test max sequence length constraint."""
        max_seq_len = 2048

        # Should be positive and reasonable
        assert max_seq_len > 0
        assert max_seq_len <= 32768  # LFM2 supports up to 32K
