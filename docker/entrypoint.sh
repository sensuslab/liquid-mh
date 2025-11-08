#!/bin/bash
set -euo pipefail

echo "================================================================"
echo "LFM2-700M Mental Health Fine-tuning Environment"
echo "================================================================"
echo ""
echo "Environment Summary:"
echo "  Python version: $(python --version)"
echo "  Working directory: $(pwd)"
echo "  CUDA available: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'No GPU detected')"
echo "  Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""
echo "Environment Variables:"
echo "  HF_HOME: ${HF_HOME:-not set}"
echo "  WANDB_API_KEY: ${WANDB_API_KEY:+***set***}"
echo "  LEAP_API_KEY: ${LEAP_API_KEY:+***set***}"
echo "  JOB_NAME: ${JOB_NAME:-not set}"
echo ""
echo "================================================================"
echo ""

# Verify Python version is >= 3.12 as required by leap-finetune
python /workspace/src/validate_env.py

# If runpod/start.sh exists and no other command is given, run it
if [ "$#" -eq 0 ] || [ "$1" = "bash" ]; then
    if [ -f /workspace/runpod/start.sh ]; then
        echo "Running Runpod startup script..."
        exec /workspace/runpod/start.sh
    else
        exec bash
    fi
else
    exec "$@"
fi
