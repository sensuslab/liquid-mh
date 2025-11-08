#!/bin/bash
set -euo pipefail

echo "================================================================"
echo "Starting LFM2-700M Mental Health Fine-tuning Pipeline"
echo "================================================================"
echo ""

# Generate timestamp-based job name if not set
if [ -z "${JOB_NAME:-}" ]; then
    export JOB_NAME="lfm2-700m-mental-health-$(date +%Y%m%d-%H%M%S)"
    echo "Generated JOB_NAME: $JOB_NAME"
fi

# Ensure required environment variables are set
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    echo "Please set your Hugging Face token to download the model and dataset"
    exit 1
fi

# Login to Hugging Face (non-interactive)
echo "Logging in to Hugging Face..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Optional: Login to W&B for experiment tracking
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Setting up W&B logging..."
    wandb login "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY not set - skipping W&B logging"
fi

# Prepare dataset
echo ""
echo "================================================================"
echo "Step 1: Preparing Dataset"
echo "================================================================"
python /workspace/data/prepare_dataset.py --out /workspace/outputs/datasets

# Run dry-run to validate setup
echo ""
echo "================================================================"
echo "Step 2: Running Dry-Run Validation (1 training step)"
echo "================================================================"
bash /workspace/scripts/dry_run.sh

# If dry-run passes, proceed with full training
echo ""
echo "================================================================"
echo "Step 3: Starting Full Training"
echo "================================================================"
bash /workspace/scripts/train.sh

# Bundle the trained model
echo ""
echo "================================================================"
echo "Step 4: Bundling Model for LEAP Deployment"
echo "================================================================"
bash /workspace/scripts/bundle.sh

# Run evaluation on sample prompts
echo ""
echo "================================================================"
echo "Step 5: Running Sample Evaluation"
echo "================================================================"
python /workspace/scripts/eval_sample.py

echo ""
echo "================================================================"
echo "Pipeline Complete!"
echo "================================================================"
echo "Job name: $JOB_NAME"
echo "Outputs:"
echo "  - Checkpoints: /workspace/outputs/sft/$JOB_NAME"
echo "  - Bundle: /workspace/outputs/bundles/${JOB_NAME}.tar"
echo "  - Eval results: /workspace/outputs/eval/${JOB_NAME}.jsonl"
echo "  - Logs: /workspace/outputs/logs/"
echo "================================================================"
