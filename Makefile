.PHONY: help test install clean build docker-build docker-run dry-run train

help:
	@echo "LFM2-700M Mental Health Fine-tuning - Available Commands"
	@echo ""
	@echo "  make install       - Install dependencies with uv"
	@echo "  make test          - Run pytest test suite"
	@echo "  make clean         - Remove generated files and caches"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container (requires .env file)"
	@echo "  make dry-run       - Run validation and smoke test"
	@echo "  make train         - Run full training pipeline"
	@echo "  make bundle        - Bundle trained model for LEAP"
	@echo "  make eval          - Evaluate model on sample prompts"
	@echo ""

install:
	@echo "Installing dependencies with uv..."
	uv pip install "transformers==4.53.0" datasets accelerate peft trl wandb \
		"ray[default]" pytest pyyaml huggingface-hub
	uv pip install "git+https://github.com/Liquid4All/leap-finetune"
	@echo "✅ Installation complete"

test:
	@echo "Running test suite..."
	pytest tests/ -v
	@echo "✅ Tests complete"

clean:
	@echo "Cleaning generated files..."
	rm -rf outputs/datasets/* outputs/sft/* outputs/logs/* outputs/eval/* outputs/bundles/*
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Clean complete"

docker-build:
	@echo "Building Docker image..."
	docker build -t lfm2-mental-health -f docker/Dockerfile .
	@echo "✅ Docker build complete"

docker-run:
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Copy runpod/template.env.example to .env and fill in your credentials."; \
		exit 1; \
	fi
	@echo "Running Docker container..."
	docker run --gpus all --env-file .env -it lfm2-mental-health

prepare-data:
	@echo "Preparing dataset..."
	python data/prepare_dataset.py --out ./outputs/datasets
	@echo "✅ Dataset preparation complete"

dry-run: prepare-data
	@echo "Running dry-run validation..."
	bash scripts/dry_run.sh
	@echo "✅ Dry-run complete"

train: prepare-data dry-run
	@echo "Starting full training..."
	bash scripts/train.sh
	@echo "✅ Training complete"

bundle:
	@echo "Bundling model for LEAP..."
	bash scripts/bundle.sh
	@echo "✅ Bundling complete"

eval:
	@echo "Evaluating model on sample prompts..."
	python scripts/eval_sample.py
	@echo "✅ Evaluation complete"

validate-env:
	@echo "Validating environment..."
	python src/validate_env.py
	@echo "✅ Environment validation complete"

validate-configs:
	@echo "Validating configurations..."
	python src/sanity_checks.py
	@echo "✅ Configuration validation complete"
