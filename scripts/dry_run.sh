#!/bin/bash
set -euo pipefail

echo "================================================================"
echo "DRY RUN: Validation & Smoke Test"
echo "================================================================"
echo ""
echo "This script performs a minimal training run (1 step) to validate:"
echo "  - GPU availability and VRAM"
echo "  - Model download and loading (LFM2-700M)"
echo "  - Dataset format and loading"
echo "  - Trainer initialization"
echo "  - Forward/backward pass"
echo ""

# Validate environment
echo "Step 1: Validating environment..."
python /workspace/src/validate_env.py
if [ $? -ne 0 ]; then
    echo "❌ Environment validation failed"
    exit 1
fi

# Validate configurations
echo ""
echo "Step 2: Validating configurations..."
python /workspace/src/sanity_checks.py
if [ $? -ne 0 ]; then
    echo "❌ Configuration validation failed"
    exit 1
fi

# Check GPU
echo ""
echo "Step 3: Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
if [ $? -ne 0 ]; then
    echo "❌ No GPU detected"
    exit 1
fi

# Check dataset files exist
echo ""
echo "Step 4: Checking dataset files..."
if [ ! -f "/workspace/outputs/datasets/train.jsonl" ]; then
    echo "❌ Training dataset not found: /workspace/outputs/datasets/train.jsonl"
    echo "Please run: python data/prepare_dataset.py --out ./outputs/datasets"
    exit 1
fi

if [ ! -f "/workspace/outputs/datasets/val.jsonl" ]; then
    echo "❌ Validation dataset not found: /workspace/outputs/datasets/val.jsonl"
    echo "Please run: python data/prepare_dataset.py --out ./outputs/datasets"
    exit 1
fi
echo "✅ Dataset files found"

# Test model loading with transformers
echo ""
echo "Step 5: Testing model download and loading..."
python -c "
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print('  Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    'LiquidAI/LFM2-700M',
    trust_remote_code=True
)

print('  Loading model (this may take a few minutes on first run)...')
model = AutoModelForCausalLM.from_pretrained(
    'LiquidAI/LFM2-700M',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

print(f'  ✅ Model loaded successfully')
print(f'     Model type: {type(model).__name__}')
print(f'     Tokenizer vocab size: {len(tokenizer)}')

# Clean up to free memory
del model
del tokenizer
torch.cuda.empty_cache()
print('  Memory cleared')
"

if [ $? -ne 0 ]; then
    echo "❌ Model loading failed"
    exit 1
fi

# Run 1-step training smoke test
echo ""
echo "Step 6: Running 1-step training smoke test..."
echo "  This validates the entire training pipeline with minimal compute."
echo ""

# Create a temporary dry-run script
cat > /tmp/dry_run_train.py << 'EOF'
#!/usr/bin/env python3
"""
Dry-run training script - runs 1 step to validate pipeline.
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# Load configs
with open('/workspace/configs/sft_train.yaml') as f:
    train_config = yaml.safe_load(f)

with open('/workspace/configs/peft_lora.yaml') as f:
    peft_config = yaml.safe_load(f)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "LiquidAI/LFM2-700M",
    trust_remote_code=True
)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2-700M",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Configuring LoRA...")
lora_config = LoraConfig(
    r=peft_config['r'],
    lora_alpha=peft_config['lora_alpha'],
    lora_dropout=peft_config['lora_dropout'],
    target_modules=peft_config['target_modules'],
    bias=peft_config['bias'],
    task_type=peft_config['task_type'],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset (first 5 samples)...")
dataset = load_dataset('json', data_files={
    'train': '/workspace/outputs/datasets/train.jsonl',
})

# Take only first 5 samples for dry run
train_dataset = dataset['train'].select(range(min(5, len(dataset['train']))))

print(f"Dataset size: {len(train_dataset)} samples")

# Override training args for dry run (1 step only)
training_args = TrainingArguments(
    output_dir="/tmp/dry_run_output",
    max_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    disable_tqdm=False,
    report_to="none",
)

print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="messages",
)

print("Running 1 training step...")
trainer.train()

print("\n✅ Dry-run PASSED - Training pipeline is functional!")

EOF

python /tmp/dry_run_train.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Dry-run training failed"
    exit 1
fi

echo ""
echo "================================================================"
echo "✅ DRY RUN PASSED"
echo "================================================================"
echo ""
echo "All validation checks passed. You can now proceed with full training."
echo "Run: bash scripts/train.sh"
echo ""
