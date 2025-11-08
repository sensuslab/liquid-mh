"""
Sanity Checks for Training Configuration

Validates training configurations for consistency and common issues.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def load_yaml_config(config_path: Path) -> Dict:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_training_config(config: Dict) -> List[str]:
    """
    Validate training configuration parameters.

    Returns:
        List of warning/error messages (empty if all checks pass)
    """
    issues = []

    # Check batch size
    batch_size = config.get("per_device_train_batch_size", 0)
    if batch_size <= 0:
        issues.append("per_device_train_batch_size must be > 0")

    # Check gradient accumulation
    grad_accum = config.get("gradient_accumulation_steps", 0)
    if grad_accum <= 0:
        issues.append("gradient_accumulation_steps must be > 0")

    # Check learning rate
    lr = config.get("learning_rate", 0)
    if lr <= 0 or lr > 1e-2:
        issues.append(f"learning_rate {lr} seems unusual (typical range: 1e-5 to 1e-3)")

    # Check precision settings
    bf16 = config.get("bf16", False)
    fp16 = config.get("fp16", False)

    if bf16 and fp16:
        issues.append("Cannot enable both bf16 and fp16 simultaneously")

    if not bf16 and not fp16:
        issues.append("Warning: Neither bf16 nor fp16 enabled (training will be slower)")

    # Check logging/save/eval steps coherence
    logging_steps = config.get("logging_steps", 0)
    save_steps = config.get("save_steps", 0)
    eval_steps = config.get("eval_steps", 0)

    if save_steps > 0 and eval_steps > 0:
        if save_steps % eval_steps != 0 and eval_steps % save_steps != 0:
            issues.append(
                f"save_steps ({save_steps}) and eval_steps ({eval_steps}) should be multiples for efficiency"
            )

    # Check epochs vs steps
    num_epochs = config.get("num_train_epochs")
    max_steps = config.get("max_steps", -1)

    if num_epochs is None and max_steps <= 0:
        issues.append("Must specify either num_train_epochs or max_steps")

    if num_epochs is not None and num_epochs <= 0:
        issues.append("num_train_epochs must be > 0")

    # Check warmup
    warmup_ratio = config.get("warmup_ratio", 0)
    warmup_steps = config.get("warmup_steps", 0)

    if warmup_ratio > 0.2:
        issues.append(f"warmup_ratio {warmup_ratio} seems high (typical: 0.01-0.1)")

    # Check save_total_limit
    save_limit = config.get("save_total_limit", 0)
    if save_limit < 1:
        issues.append("save_total_limit should be >= 1 to keep checkpoints")

    return issues


def check_peft_config(config: Dict) -> List[str]:
    """
    Validate PEFT/LoRA configuration parameters.

    Returns:
        List of warning/error messages (empty if all checks pass)
    """
    issues = []

    # Check LoRA rank
    r = config.get("r", 0)
    if r <= 0:
        issues.append("LoRA rank (r) must be > 0")
    if r > 128:
        issues.append(f"LoRA rank {r} is very high (typical: 8-64)")

    # Check alpha
    alpha = config.get("lora_alpha", 0)
    if alpha <= 0:
        issues.append("lora_alpha must be > 0")

    # Common pattern: alpha = 2*r or alpha = r
    if alpha not in [r, 2 * r]:
        issues.append(f"lora_alpha ({alpha}) is typically set to r ({r}) or 2*r ({2*r})")

    # Check dropout
    dropout = config.get("lora_dropout", 0)
    if dropout < 0 or dropout > 0.5:
        issues.append(f"lora_dropout {dropout} outside typical range (0.0-0.2)")

    # Check target modules
    target_modules = config.get("target_modules", [])
    if not target_modules:
        issues.append("target_modules is empty - no layers will be adapted")

    # Check task type
    task_type = config.get("task_type", "")
    if task_type != "CAUSAL_LM":
        issues.append(f"task_type should be 'CAUSAL_LM' for LFM2 fine-tuning, got '{task_type}'")

    return issues


def check_job_config(config: Dict) -> List[str]:
    """
    Validate job configuration.

    Returns:
        List of warning/error messages (empty if all checks pass)
    """
    issues = []

    # Check model_id
    model_id = config.get("model_id", "")
    if not model_id:
        issues.append("model_id is required")
    elif "LFM2" not in model_id:
        issues.append(f"model_id '{model_id}' does not appear to be an LFM2 model")

    # Check dataset paths
    dataset_paths = config.get("dataset_paths", {})
    if not dataset_paths:
        issues.append("dataset_paths is empty")
    else:
        if "train" not in dataset_paths:
            issues.append("dataset_paths missing 'train' key")

    # Check output_dir
    output_dir = config.get("output_dir", "")
    if not output_dir:
        issues.append("output_dir is required")

    # Check config references
    peft_config = config.get("peft_config_path", "")
    training_config = config.get("training_config_path", "")

    if not peft_config:
        issues.append("peft_config_path is required")
    elif not Path(peft_config).exists():
        issues.append(f"peft_config_path '{peft_config}' does not exist")

    if not training_config:
        issues.append("training_config_path is required")
    elif not Path(training_config).exists():
        issues.append(f"training_config_path '{training_config}' does not exist")

    # Check model_config
    model_config = config.get("model_config", {})
    trust_remote_code = model_config.get("trust_remote_code", False)

    if not trust_remote_code:
        issues.append(
            "model_config.trust_remote_code should be True for LFM2 "
            "(required until native transformers support)"
        )

    return issues


def validate_all_configs(
    job_config_path: Path,
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate all configuration files.

    Args:
        job_config_path: Path to job_config.yaml
        verbose: Whether to print detailed results

    Returns:
        Tuple of (all_passed, list_of_issues)
    """
    all_issues = []

    if verbose:
        print("\n" + "=" * 80)
        print("Configuration Validation")
        print("=" * 80)
        print()

    # Load job config
    try:
        job_config = load_yaml_config(job_config_path)

        if verbose:
            print("Checking job_config.yaml...")
        issues = check_job_config(job_config)
        if issues:
            all_issues.extend([f"job_config: {issue}" for issue in issues])
        elif verbose:
            print("  ✅ Job config OK")

    except Exception as e:
        all_issues.append(f"job_config: Failed to load - {e}")

    # Load and check PEFT config
    peft_config_path = Path(job_config.get("peft_config_path", ""))
    if peft_config_path.exists():
        try:
            peft_config = load_yaml_config(peft_config_path)

            if verbose:
                print(f"\nChecking {peft_config_path.name}...")
            issues = check_peft_config(peft_config)
            if issues:
                all_issues.extend([f"peft_config: {issue}" for issue in issues])
            elif verbose:
                print("  ✅ PEFT config OK")

        except Exception as e:
            all_issues.append(f"peft_config: Failed to load - {e}")

    # Load and check training config
    training_config_path = Path(job_config.get("training_config_path", ""))
    if training_config_path.exists():
        try:
            training_config = load_yaml_config(training_config_path)

            if verbose:
                print(f"\nChecking {training_config_path.name}...")
            issues = check_training_config(training_config)
            if issues:
                all_issues.extend([f"training_config: {issue}" for issue in issues])
            elif verbose:
                print("  ✅ Training config OK")

        except Exception as e:
            all_issues.append(f"training_config: Failed to load - {e}")

    # Print results
    if verbose:
        print()
        if all_issues:
            print("=" * 80)
            print("❌ Configuration Issues Found")
            print("=" * 80)
            for issue in all_issues:
                print(f"  - {issue}")
            print()
        else:
            print("=" * 80)
            print("✅ All configuration checks PASSED")
            print("=" * 80)
            print()

    return len(all_issues) == 0, all_issues


def main():
    """Main entry point for standalone execution."""
    import sys

    job_config_path = Path("./configs/job_config.yaml")
    passed, issues = validate_all_configs(job_config_path, verbose=True)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
