#!/bin/bash
set -euo pipefail

echo "================================================================"
echo "Bundle Model for LEAP Deployment"
echo "================================================================"
echo ""
echo "This script packages the fine-tuned adapter for deployment on"
echo "the Liquid Edge AI Platform (LEAP)."
echo ""
echo "Reference: https://github.com/Liquid4All/leap-finetune"
echo "           https://leap.liquid.ai/docs/finetuning"
echo ""

# Parse arguments or use defaults
if [ $# -ge 1 ]; then
    CHECKPOINT_DIR="$1"
else
    # Find the most recent 'final' checkpoint
    CHECKPOINT_DIR=$(find /workspace/outputs/sft -name "final" -type d | sort | tail -n1)

    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "❌ No checkpoint directory found"
        echo ""
        echo "Usage: bash scripts/bundle.sh [checkpoint_dir]"
        echo ""
        echo "Example:"
        echo "  bash scripts/bundle.sh /workspace/outputs/sft/my-job/final"
        echo ""
        echo "Or train a model first with: bash scripts/train.sh"
        exit 1
    fi

    echo "Auto-detected checkpoint: $CHECKPOINT_DIR"
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

# Verify checkpoint contains necessary files
echo "Validating checkpoint..."
REQUIRED_FILES=("adapter_config.json" "adapter_model.safetensors")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$CHECKPOINT_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Checkpoint appears incomplete. Missing files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo "✅ Checkpoint validation passed"
echo ""

# Determine job name from checkpoint path
JOB_NAME=$(basename "$(dirname "$CHECKPOINT_DIR")")
if [ "$JOB_NAME" = "sft" ] || [ "$JOB_NAME" = "outputs" ]; then
    JOB_NAME="${JOB_NAME:-lfm2-mental-health-$(date +%Y%m%d-%H%M%S)}"
fi

BUNDLE_DIR="/workspace/outputs/bundles"
mkdir -p "$BUNDLE_DIR"

BUNDLE_PATH="${BUNDLE_DIR}/${JOB_NAME}.tar"

echo "Bundle configuration:"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Job name: $JOB_NAME"
echo "  Output bundle: $BUNDLE_PATH"
echo ""

# Check if leap-bundle command is available
echo "Checking for leap-bundle command..."

if command -v leap-bundle &> /dev/null; then
    echo "✅ leap-bundle command found"
    echo ""
    echo "Creating bundle with leap-bundle..."

    # Run leap-bundle (adjust command based on actual leap-finetune CLI)
    # Note: The exact command may vary; check leap-finetune documentation
    leap-bundle \
        --input "$CHECKPOINT_DIR" \
        --output "$BUNDLE_PATH" \
        --name "$JOB_NAME"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Bundle created successfully"
    else
        echo ""
        echo "❌ leap-bundle failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi

else
    echo "⚠️  leap-bundle command not found"
    echo ""
    echo "Creating manual bundle (tar archive)..."
    echo "Note: For full LEAP compatibility, install leap-finetune with bundling support"
    echo ""

    # Create a manual tar bundle with manifest
    TEMP_BUNDLE_DIR="/tmp/${JOB_NAME}_bundle"
    rm -rf "$TEMP_BUNDLE_DIR"
    mkdir -p "$TEMP_BUNDLE_DIR"

    # Copy adapter files
    echo "Copying adapter files..."
    cp -r "$CHECKPOINT_DIR"/* "$TEMP_BUNDLE_DIR/"

    # Create a manifest file
    cat > "$TEMP_BUNDLE_DIR/manifest.json" << EOF
{
  "name": "${JOB_NAME}",
  "base_model": "LiquidAI/LFM2-700M",
  "adapter_type": "lora",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "framework": "peft",
  "transformers_version": "4.53.0",
  "files": [
    $(find "$CHECKPOINT_DIR" -type f -printf '"%f",\n' | sed '$ s/,$//')
  ]
}
EOF

    # Create tar archive
    echo "Creating tar archive..."
    cd "$TEMP_BUNDLE_DIR"
    tar -czf "$BUNDLE_PATH" ./*
    cd - > /dev/null

    # Cleanup
    rm -rf "$TEMP_BUNDLE_DIR"

    echo "✅ Manual bundle created"
fi

# Verify bundle was created
if [ ! -f "$BUNDLE_PATH" ]; then
    echo ""
    echo "❌ Bundle file was not created: $BUNDLE_PATH"
    exit 1
fi

BUNDLE_SIZE=$(du -h "$BUNDLE_PATH" | cut -f1)

echo ""
echo "================================================================"
echo "✅ Bundling Complete!"
echo "================================================================"
echo "  Bundle: $BUNDLE_PATH"
echo "  Size: $BUNDLE_SIZE"
echo ""
echo "Next steps:"
echo "  1. Verify bundle contents: tar -tzf $BUNDLE_PATH"
echo "  2. Deploy to LEAP: https://leap.liquid.ai/docs/finetuning"
echo "  3. Or test locally: python scripts/eval_sample.py --checkpoint $CHECKPOINT_DIR"
echo ""
echo "================================================================"
