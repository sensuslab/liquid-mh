#!/usr/bin/env python3
"""
Environment Validation Script

Validates that the environment meets all requirements for LFM2-700M fine-tuning.

Requirements:
- Python >= 3.12 (leap-finetune requirement)
- Transformers == 4.53.0 (LFM2-700M requirement)
- CUDA-capable GPU
- Sufficient disk space
- Required environment variables

References:
- leap-finetune: https://github.com/Liquid4All/leap-finetune
- LFM2-700M: https://huggingface.co/LiquidAI/LFM2-700M
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class ValidationError(Exception):
    """Raised when a validation check fails."""
    pass


def check_python_version() -> Tuple[bool, str]:
    """Check that Python version is >= 3.12."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 12:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} < 3.12 (required by leap-finetune)"


def check_transformers_version() -> Tuple[bool, str]:
    """Check that transformers version is exactly 4.53.0."""
    try:
        import transformers
        version = transformers.__version__
        # Pin to 4.53.0 as required by LFM2-700M
        # Reference: https://huggingface.co/LiquidAI/LFM2-700M
        if version == "4.53.0":
            return True, f"transformers=={version}"
        else:
            return False, f"transformers=={version}, expected 4.53.0 (required by LFM2-700M)"
    except ImportError:
        return False, "transformers not installed"


def check_cuda_available() -> Tuple[bool, str]:
    """Check that CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            return True, f"{device_count} GPU(s) available: {device_name}"
        else:
            return False, "No CUDA-capable GPU detected"
    except ImportError:
        return False, "PyTorch not installed"


def check_gpu_memory() -> Tuple[bool, str]:
    """Check GPU memory availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        min_free = float('inf')

        for idx, line in enumerate(lines):
            total, free = map(int, line.split(','))
            gpu_info.append(f"GPU {idx}: {free}/{total} MB free")
            min_free = min(min_free, free)

        # Recommend at least 20GB free for LFM2-700M with LoRA
        if min_free < 20000:
            return False, f"Low GPU memory: {', '.join(gpu_info)} (recommend ≥20GB free)"
        else:
            return True, ', '.join(gpu_info)

    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, "Could not query GPU memory (nvidia-smi not available)"


def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space."""
    try:
        import shutil
        stat = shutil.disk_usage(Path.cwd())
        free_gb = stat.free / (1024 ** 3)

        # Recommend at least 50GB free (model + dataset + checkpoints)
        if free_gb < 50:
            return False, f"{free_gb:.1f} GB free (recommend ≥50GB)"
        else:
            return True, f"{free_gb:.1f} GB free"

    except Exception as e:
        return False, f"Could not check disk space: {e}"


def check_environment_variables() -> Tuple[bool, str]:
    """Check required environment variables."""
    issues = []

    # HF_TOKEN is required
    if not os.getenv("HF_TOKEN"):
        issues.append("HF_TOKEN not set (required for model/dataset access)")

    # These are optional but recommended
    if not os.getenv("WANDB_API_KEY"):
        issues.append("WANDB_API_KEY not set (experiment tracking disabled)")

    if issues:
        return False, "; ".join(issues)
    else:
        return True, "All required environment variables set"


def check_required_packages() -> Tuple[bool, str]:
    """Check that required packages are installed."""
    required = [
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "torch",
        "ray",
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    else:
        return True, "All required packages installed"


def validate_environment(verbose: bool = True) -> bool:
    """
    Run all validation checks.

    Args:
        verbose: Whether to print detailed results

    Returns:
        True if all checks pass, False otherwise
    """
    checks = [
        ("Python Version", check_python_version),
        ("Transformers Version", check_transformers_version),
        ("Required Packages", check_required_packages),
        ("CUDA Availability", check_cuda_available),
        ("GPU Memory", check_gpu_memory),
        ("Disk Space", check_disk_space),
        ("Environment Variables", check_environment_variables),
    ]

    results = []
    all_passed = True

    if verbose:
        print("\n" + "=" * 80)
        print("Environment Validation")
        print("=" * 80)
        print()

    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            results.append((check_name, passed, message))

            if not passed:
                all_passed = False

            if verbose:
                status = "✅" if passed else "❌"
                print(f"{status} {check_name}: {message}")

        except Exception as e:
            results.append((check_name, False, f"Error: {e}"))
            all_passed = False

            if verbose:
                print(f"❌ {check_name}: Error: {e}")

    if verbose:
        print()
        if all_passed:
            print("=" * 80)
            print("✅ All validation checks PASSED")
            print("=" * 80)
        else:
            print("=" * 80)
            print("❌ Some validation checks FAILED")
            print("=" * 80)
            print("\nPlease address the issues above before proceeding.")
        print()

    return all_passed


def main():
    """Main entry point."""
    try:
        success = validate_environment(verbose=True)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
