#!/bin/bash
set -euo pipefail

echo "================================================================"
echo "LFM2-700M Mental Health Fine-Tuning - Full Training"
echo "================================================================"
echo ""

# Set job name if not already set
if [ -z "${JOB_NAME:-}" ]; then
    export JOB_NAME="lfm2-700m-mental-health-$(date +%Y%m%d-%H%M%S)"
    echo "Generated JOB_NAME: $JOB_NAME"
fi

# Determine output directory
OUTPUT_DIR="/workspace/outputs/sft/${JOB_NAME}"
LOG_DIR="/workspace/outputs/logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

echo "Configuration:"
echo "  Job name: $JOB_NAME"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log file: $LOG_FILE"
echo ""

# Auto-tune batch size based on available GPU memory
echo "Auto-tuning batch size based on available GPU memory..."
GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
echo "  GPU free memory: ${GPU_MEM_FREE} MB"

# Conservative batch sizing for different GPU memory configs
if [ "$GPU_MEM_FREE" -gt 70000 ]; then
    # 80GB GPU (A100)
    export BATCH_SIZE=8
    export GRAD_ACCUM=2
elif [ "$GPU_MEM_FREE" -gt 40000 ]; then
    # 48GB GPU (A40, A6000)
    export BATCH_SIZE=6
    export GRAD_ACCUM=3
elif [ "$GPU_MEM_FREE" -gt 20000 ]; then
    # 24-40GB GPU (A10, RTX 3090, A100-40GB)
    export BATCH_SIZE=4
    export GRAD_ACCUM=4
else
    # <24GB GPU
    export BATCH_SIZE=2
    export GRAD_ACCUM=8
fi

echo "  Auto-tuned batch size: $BATCH_SIZE"
echo "  Gradient accumulation steps: $GRAD_ACCUM"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo ""

# Create training script
cat > /tmp/train_lfm2.py << 'SCRIPT_EOF'
#!/usr/bin/env python3
"""
Main training script for LFM2-700M Mental Health Fine-tuning

Reference: https://github.com/Liquid4All/leap-finetune
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
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# Get configuration from environment
JOB_NAME = os.getenv("JOB_NAME", "lfm2-mental-health")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/outputs/sft/default")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))

print("=" * 80)
print(f"Training Configuration")
print("=" * 80)
print(f"Job Name: {JOB_NAME}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation: {GRAD_ACCUM}")
print(f"Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM}")
print()

# Load configs
print("Loading configuration files...")
with open('/workspace/configs/sft_train.yaml') as f:
    train_config = yaml.safe_load(f)

with open('/workspace/configs/peft_lora.yaml') as f:
    peft_config = yaml.safe_load(f)

with open('/workspace/configs/job_config.yaml') as f:
    job_config = yaml.safe_load(f)

# Override batch size from auto-tuning
train_config['per_device_train_batch_size'] = BATCH_SIZE
train_config['gradient_accumulation_steps'] = GRAD_ACCUM

# Set report_to based on WANDB_API_KEY
if os.getenv("WANDB_API_KEY"):
    train_config['report_to'] = 'wandb'
    print("W&B logging enabled")
else:
    train_config['report_to'] = 'none'
    print("W&B logging disabled (WANDB_API_KEY not set)")

# Set seed
set_seed(train_config.get('seed', 42))

print("\n" + "=" * 80)
print("Loading Model and Tokenizer")
print("=" * 80)
print("Model: LiquidAI/LFM2-700M")
print("This may take a few minutes on first run...")

tokenizer = AutoTokenizer.from_pretrained(
    "LiquidAI/LFM2-700M",
    trust_remote_code=True,
)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2-700M",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",  # Scaled Dot Product Attention
)

print(f"✅ Model loaded: {type(model).__name__}")
print(f"   Tokenizer vocab size: {len(tokenizer)}")

print("\n" + "=" * 80)
print("Configuring LoRA/PEFT")
print("=" * 80)

lora_config = LoraConfig(
    r=peft_config['r'],
    lora_alpha=peft_config['lora_alpha'],
    lora_dropout=peft_config['lora_dropout'],
    target_modules=peft_config['target_modules'],
    bias=peft_config['bias'],
    task_type=peft_config['task_type'],
)

print(f"LoRA Configuration:")
print(f"  Rank (r): {peft_config['r']}")
print(f"  Alpha: {peft_config['lora_alpha']}")
print(f"  Dropout: {peft_config['lora_dropout']}")
print(f"  Target modules: {', '.join(peft_config['target_modules'])}")

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n" + "=" * 80)
print("Loading Datasets")
print("=" * 80)

dataset = load_dataset('json', data_files={
    'train': '/workspace/outputs/datasets/train.jsonl',
    'validation': '/workspace/outputs/datasets/val.jsonl',
})

train_dataset = dataset['train']
eval_dataset = dataset['validation']

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

print("\n" + "=" * 80)
print("Configuring Training Arguments")
print("=" * 80)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name=JOB_NAME,

    # Training duration
    num_train_epochs=train_config.get('num_train_epochs', 3),

    # Batch size and accumulation
    per_device_train_batch_size=train_config['per_device_train_batch_size'],
    per_device_eval_batch_size=train_config.get('per_device_eval_batch_size', 4),
    gradient_accumulation_steps=train_config['gradient_accumulation_steps'],

    # Learning rate and optimization
    learning_rate=train_config.get('learning_rate', 2e-4),
    weight_decay=train_config.get('weight_decay', 0.01),
    optim=train_config.get('optim', 'adamw_torch'),
    max_grad_norm=train_config.get('max_grad_norm', 1.0),

    # LR scheduling
    lr_scheduler_type=train_config.get('lr_scheduler_type', 'cosine'),
    warmup_ratio=train_config.get('warmup_ratio', 0.03),

    # Precision
    bf16=train_config.get('bf16', True),
    fp16=train_config.get('fp16', False),
    tf32=train_config.get('tf32', True),

    # Memory optimization
    gradient_checkpointing=train_config.get('gradient_checkpointing', True),

    # Logging
    logging_steps=train_config.get('logging_steps', 10),
    logging_strategy="steps",

    # Evaluation
    evaluation_strategy=train_config.get('evaluation_strategy', 'steps'),
    eval_steps=train_config.get('eval_steps', 100),

    # Checkpointing
    save_strategy=train_config.get('save_strategy', 'steps'),
    save_steps=train_config.get('save_steps', 200),
    save_total_limit=train_config.get('save_total_limit', 3),
    load_best_model_at_end=train_config.get('load_best_model_at_end', True),
    metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss'),
    greater_is_better=train_config.get('greater_is_better', False),

    # Reporting
    report_to=train_config['report_to'],

    # Data loading
    dataloader_num_workers=train_config.get('dataloader_num_workers', 4),
    dataloader_pin_memory=train_config.get('dataloader_pin_memory', True),
    remove_unused_columns=train_config.get('remove_unused_columns', True),

    # Misc
    seed=train_config.get('seed', 42),
    disable_tqdm=False,
    include_tokens_per_second=True,
)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")

print("\n" + "=" * 80)
print("Initializing Trainer")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=train_config.get('max_seq_length', 2048),
    dataset_text_field="messages",
    packing=False,
)

print("✅ Trainer initialized")

print("\n" + "=" * 80)
print("Starting Training")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    trainer.train()

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save final model
    final_output_dir = os.path.join(OUTPUT_DIR, "final")
    print(f"\nSaving final model to: {final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print("\n✅ Training completed successfully!")
    print(f"   Model saved to: {final_output_dir}")

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Saving checkpoint...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "interrupted"))
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

SCRIPT_EOF

# Run training
echo "Starting training..."
echo "Logs will be saved to: $LOG_FILE"
echo ""

python /tmp/train_lfm2.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "✅ Training completed successfully!"
    echo "================================================================"
    echo "  Job name: $JOB_NAME"
    echo "  Output: $OUTPUT_DIR"
    echo "  Logs: $LOG_FILE"
    echo ""
else
    echo ""
    echo "================================================================"
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "================================================================"
    echo "  Check logs: $LOG_FILE"
    echo ""
    exit $EXIT_CODE
fi
