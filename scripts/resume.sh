#!/bin/bash
set -euo pipefail

echo "================================================================"
echo "Resume Training from Checkpoint"
echo "================================================================"
echo ""

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/resume.sh <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  bash scripts/resume.sh /workspace/outputs/sft/my-job/checkpoint-1000"
    echo ""
    echo "Available checkpoints:"
    find /workspace/outputs/sft -name "checkpoint-*" -type d 2>/dev/null | sort
    echo ""
    exit 1
fi

CHECKPOINT_PATH="$1"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint path does not exist: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    find /workspace/outputs/sft -name "checkpoint-*" -type d 2>/dev/null | sort
    exit 1
fi

echo "Resuming from checkpoint: $CHECKPOINT_PATH"
echo ""

# Extract job name from checkpoint path or create new one
PARENT_DIR=$(dirname "$CHECKPOINT_PATH")
PARENT_JOB_NAME=$(basename "$PARENT_DIR")

if [ -z "${JOB_NAME:-}" ]; then
    export JOB_NAME="${PARENT_JOB_NAME}-resumed-$(date +%Y%m%d-%H%M%S)"
    echo "Generated new job name: $JOB_NAME"
fi

# Use same output directory as parent (will continue in same job folder)
OUTPUT_DIR="$PARENT_DIR"
LOG_DIR="/workspace/outputs/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

echo "Configuration:"
echo "  Job name: $JOB_NAME"
echo "  Output directory: $OUTPUT_DIR"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Log file: $LOG_FILE"
echo ""

# Auto-tune batch size (same as train.sh)
GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
echo "Auto-tuning batch size based on available GPU memory..."
echo "  GPU free memory: ${GPU_MEM_FREE} MB"

if [ "$GPU_MEM_FREE" -gt 70000 ]; then
    export BATCH_SIZE=8
    export GRAD_ACCUM=2
elif [ "$GPU_MEM_FREE" -gt 40000 ]; then
    export BATCH_SIZE=6
    export GRAD_ACCUM=3
elif [ "$GPU_MEM_FREE" -gt 20000 ]; then
    export BATCH_SIZE=4
    export GRAD_ACCUM=4
else
    export BATCH_SIZE=2
    export GRAD_ACCUM=8
fi

echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo ""

# Create resume training script
cat > /tmp/resume_train.py << 'SCRIPT_EOF'
#!/usr/bin/env python3
"""
Resume training script for LFM2-700M
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import PeftModel, LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

# Get configuration from environment
JOB_NAME = os.getenv("JOB_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))

print("=" * 80)
print(f"Resume Training Configuration")
print("=" * 80)
print(f"Job Name: {JOB_NAME}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation: {GRAD_ACCUM}")
print()

# Load configs
with open('/workspace/configs/sft_train.yaml') as f:
    train_config = yaml.safe_load(f)

# Override batch size
train_config['per_device_train_batch_size'] = BATCH_SIZE
train_config['gradient_accumulation_steps'] = GRAD_ACCUM

# Set report_to based on WANDB_API_KEY
if os.getenv("WANDB_API_KEY"):
    train_config['report_to'] = 'wandb'
else:
    train_config['report_to'] = 'none'

set_seed(train_config.get('seed', 42))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from checkpoint...")
base_model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2-700M",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load PEFT adapter
model = PeftModel.from_pretrained(
    base_model,
    CHECKPOINT_PATH,
)

print(f"✅ Model loaded from checkpoint")
model.print_trainable_parameters()

print("\nLoading datasets...")
dataset = load_dataset('json', data_files={
    'train': '/workspace/outputs/datasets/train.jsonl',
    'validation': '/workspace/outputs/datasets/val.jsonl',
})

train_dataset = dataset['train']
eval_dataset = dataset['validation']

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

print("\nConfiguring training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name=JOB_NAME,
    num_train_epochs=train_config.get('num_train_epochs', 3),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=train_config.get('per_device_eval_batch_size', 4),
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=train_config.get('learning_rate', 2e-4),
    weight_decay=train_config.get('weight_decay', 0.01),
    optim=train_config.get('optim', 'adamw_torch'),
    max_grad_norm=train_config.get('max_grad_norm', 1.0),
    lr_scheduler_type=train_config.get('lr_scheduler_type', 'cosine'),
    warmup_ratio=train_config.get('warmup_ratio', 0.03),
    bf16=train_config.get('bf16', True),
    fp16=train_config.get('fp16', False),
    gradient_checkpointing=train_config.get('gradient_checkpointing', True),
    logging_steps=train_config.get('logging_steps', 10),
    evaluation_strategy="steps",
    eval_steps=train_config.get('eval_steps', 100),
    save_strategy="steps",
    save_steps=train_config.get('save_steps', 200),
    save_total_limit=train_config.get('save_total_limit', 3),
    load_best_model_at_end=train_config.get('load_best_model_at_end', True),
    metric_for_best_model="eval_loss",
    report_to=train_config['report_to'],
    dataloader_num_workers=4,
    seed=train_config.get('seed', 42),
)

print("\nInitializing trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=train_config.get('max_seq_length', 2048),
    dataset_text_field="messages",
)

print("\n" + "=" * 80)
print("Resuming Training")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    # Resume from checkpoint
    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Save final model
    final_output_dir = os.path.join(OUTPUT_DIR, "final-resumed")
    print(f"Saving final model to: {final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print("\n✅ Resumed training completed successfully!")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

SCRIPT_EOF

# Export checkpoint path for script
export CHECKPOINT_PATH="$CHECKPOINT_PATH"
export OUTPUT_DIR="$OUTPUT_DIR"

# Run training
echo "Resuming training..."
python /tmp/resume_train.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "✅ Resumed training completed successfully!"
    echo "================================================================"
    echo "  Job name: $JOB_NAME"
    echo "  Output: $OUTPUT_DIR"
    echo "  Logs: $LOG_FILE"
    echo ""
else
    echo ""
    echo "❌ Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
