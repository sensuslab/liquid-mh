"""
Tests for configuration files
"""

import pytest
import yaml
from pathlib import Path

# Config file paths
CONFIG_DIR = Path(__file__).parent.parent / "configs"
JOB_CONFIG_PATH = CONFIG_DIR / "job_config.yaml"
PEFT_CONFIG_PATH = CONFIG_DIR / "peft_lora.yaml"
TRAIN_CONFIG_PATH = CONFIG_DIR / "sft_train.yaml"


def load_yaml(path):
    """Helper to load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


class TestJobConfig:
    """Tests for job_config.yaml"""

    @pytest.fixture
    def config(self):
        return load_yaml(JOB_CONFIG_PATH)

    def test_config_exists(self):
        """Test that job config file exists."""
        assert JOB_CONFIG_PATH.exists()

    def test_required_fields(self, config):
        """Test that required fields are present."""
        assert "model_id" in config
        assert "dataset_paths" in config
        assert "output_dir" in config
        assert "peft_config_path" in config
        assert "training_config_path" in config

    def test_model_id(self, config):
        """Test model ID is correct."""
        assert config["model_id"] == "LiquidAI/LFM2-700M"

    def test_trust_remote_code(self, config):
        """Test that trust_remote_code is enabled."""
        model_config = config.get("model_config", {})
        assert model_config.get("trust_remote_code") is True, \
            "trust_remote_code must be True for LFM2-700M"

    def test_dataset_paths(self, config):
        """Test dataset paths are specified."""
        dataset_paths = config["dataset_paths"]
        assert "train" in dataset_paths
        # Note: validation key may vary

    def test_peft_enabled(self, config):
        """Test that PEFT is enabled."""
        assert config.get("use_peft", False) is True


class TestPeftConfig:
    """Tests for peft_lora.yaml"""

    @pytest.fixture
    def config(self):
        return load_yaml(PEFT_CONFIG_PATH)

    def test_config_exists(self):
        """Test that PEFT config file exists."""
        assert PEFT_CONFIG_PATH.exists()

    def test_required_fields(self, config):
        """Test that required LoRA fields are present."""
        assert "r" in config
        assert "lora_alpha" in config
        assert "lora_dropout" in config
        assert "target_modules" in config
        assert "task_type" in config

    def test_lora_rank(self, config):
        """Test LoRA rank is reasonable."""
        r = config["r"]
        assert isinstance(r, int)
        assert r > 0, "LoRA rank must be positive"
        assert r <= 128, "LoRA rank is unusually high"

    def test_lora_alpha(self, config):
        """Test LoRA alpha is reasonable."""
        alpha = config["lora_alpha"]
        assert isinstance(alpha, (int, float))
        assert alpha > 0, "LoRA alpha must be positive"

    def test_dropout(self, config):
        """Test dropout is in valid range."""
        dropout = config["lora_dropout"]
        assert isinstance(dropout, (int, float))
        assert 0 <= dropout <= 1, "Dropout must be between 0 and 1"

    def test_target_modules(self, config):
        """Test target modules are specified."""
        target_modules = config["target_modules"]
        assert isinstance(target_modules, list)
        assert len(target_modules) > 0, "Must specify at least one target module"

    def test_task_type(self, config):
        """Test task type is correct for causal LM."""
        assert config["task_type"] == "CAUSAL_LM"


class TestTrainConfig:
    """Tests for sft_train.yaml"""

    @pytest.fixture
    def config(self):
        return load_yaml(TRAIN_CONFIG_PATH)

    def test_config_exists(self):
        """Test that training config file exists."""
        assert TRAIN_CONFIG_PATH.exists()

    def test_required_fields(self, config):
        """Test that required training fields are present."""
        assert "per_device_train_batch_size" in config
        assert "gradient_accumulation_steps" in config
        assert "learning_rate" in config
        assert "num_train_epochs" in config or "max_steps" in config

    def test_batch_size(self, config):
        """Test batch size is positive."""
        batch_size = config["per_device_train_batch_size"]
        assert isinstance(batch_size, int)
        assert batch_size > 0, "Batch size must be positive"

    def test_gradient_accumulation(self, config):
        """Test gradient accumulation is positive."""
        grad_accum = config["gradient_accumulation_steps"]
        assert isinstance(grad_accum, int)
        assert grad_accum > 0, "Gradient accumulation must be positive"

    def test_learning_rate(self, config):
        """Test learning rate is reasonable."""
        lr = config["learning_rate"]
        assert isinstance(lr, (int, float))
        assert lr > 0, "Learning rate must be positive"
        assert lr < 1.0, "Learning rate seems too high"

    def test_precision_settings(self, config):
        """Test precision settings are valid."""
        bf16 = config.get("bf16", False)
        fp16 = config.get("fp16", False)

        # Can't have both enabled
        assert not (bf16 and fp16), "Cannot enable both bf16 and fp16"

    def test_logging_steps(self, config):
        """Test logging configuration."""
        if "logging_steps" in config:
            assert config["logging_steps"] > 0

    def test_save_total_limit(self, config):
        """Test checkpoint limit is set."""
        if "save_total_limit" in config:
            limit = config["save_total_limit"]
            assert limit >= 1, "Should keep at least 1 checkpoint"

    def test_optimizer(self, config):
        """Test optimizer is specified."""
        if "optim" in config:
            valid_optimizers = [
                "adamw_torch",
                "adamw_8bit",
                "adamw_bnb_8bit",
                "adafactor"
            ]
            assert config["optim"] in valid_optimizers


class TestConfigIntegration:
    """Integration tests across configs."""

    def test_config_references_exist(self):
        """Test that config files reference each other correctly."""
        job_config = load_yaml(JOB_CONFIG_PATH)

        peft_ref = job_config.get("peft_config_path", "")
        train_ref = job_config.get("training_config_path", "")

        # Resolve relative paths
        peft_path = CONFIG_DIR / Path(peft_ref).name
        train_path = CONFIG_DIR / Path(train_ref).name

        assert peft_path.exists(), f"PEFT config not found: {peft_path}"
        assert train_path.exists(), f"Training config not found: {train_path}"

    def test_config_consistency(self):
        """Test consistency between configs."""
        job_config = load_yaml(JOB_CONFIG_PATH)
        peft_config = load_yaml(PEFT_CONFIG_PATH)
        train_config = load_yaml(TRAIN_CONFIG_PATH)

        # All configs should be loadable without errors
        assert isinstance(job_config, dict)
        assert isinstance(peft_config, dict)
        assert isinstance(train_config, dict)

    def test_output_dirs_specified(self):
        """Test that output directories are specified."""
        job_config = load_yaml(JOB_CONFIG_PATH)

        assert "output_dir" in job_config
        assert job_config["output_dir"], "Output dir cannot be empty"
