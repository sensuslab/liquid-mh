# LFM2-700M Mental Health Counseling Fine-Tuning

Production-ready repository for fine-tuning **LiquidAI/LFM2-700M** on mental health counseling conversations using **leap-finetune**, optimized for **Runpod** single-GPU deployment.

---

## ⚠️ Important Disclaimers

### Research Use Only

**This repository and any models trained with it are intended for RESEARCH PURPOSES ONLY.** The outputs should **NOT** be used as a substitute for professional mental health support, diagnosis, or treatment.

### Crisis Resources

If you or someone you know is experiencing a mental health crisis, please contact:

- **National Suicide Prevention Lifeline (US)**: Call or text **988**
- **Crisis Text Line**: Text **HOME** to **741741**
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
- **Emergency Services**: Call **911** (US) or your local emergency number

### Dataset License

This repository uses the [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) dataset, which is licensed under the **Responsible AI License - Derivatives (RAIL-D)**.

**Key license requirements:**

- **Attribution**: Credit the original dataset creators
- **Commercial Use**: Requires a donation to mental health organizations (see [dataset page](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations))
- **No Content Modification**: Dataset content must not be modified in ways that misrepresent mental health support

**📖 Full license details**: Please review the complete license terms on the [dataset page](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) before use.

---

## 🎯 Overview

This repository provides a complete, production-ready pipeline to:

1. **Download and prepare** the mental health counseling dataset
2. **Fine-tune** LFM2-700M using LoRA/PEFT with leap-finetune
3. **Bundle** the trained adapter for deployment on LEAP
4. **Evaluate** model outputs on sample prompts
5. **Run robustly** with dry-run validation and resume support

### Key Features

✅ **Reproducible**: Pinned dependencies, seeded randomness, comprehensive logging
✅ **Production-Ready**: Docker containerization, auto-tuned batch sizes, checkpoint management
✅ **Well-Tested**: Unit tests, integration tests, and dry-run smoke tests
✅ **Documented**: Inline citations to authoritative sources
✅ **Scalable**: Designed for single GPU but supports multi-GPU with minimal changes

---

## 📚 References

This implementation follows requirements from the following authoritative sources:

1. **[leap-finetune GitHub](https://github.com/Liquid4All/leap-finetune)** - Fine-tuning framework, dataset formats, bundling, Ray+Accelerate setup
2. **[LEAP Finetuning Docs](https://leap.liquid.ai/docs/finetuning)** - Platform documentation, ≤1 GPU notebooks, >1 GPU uses leap-finetune
3. **[LFM2-700M Model Card](https://huggingface.co/LiquidAI/LFM2-700M)** - Model requirements, `transformers==4.53.0`, `trust_remote_code=True`
4. **[Mental Health Dataset](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)** - Dataset schema (Context/Response fields), license terms

---

## 🛠️ Requirements

### Hardware

- **Recommended**: NVIDIA A100 40GB or 80GB (tested by leap-finetune team on H100 80GB)
- **Minimum**: NVIDIA GPU with ≥20GB VRAM (RTX 3090, A10, A6000, etc.)
- **Batch Size**: Auto-tuned based on available VRAM

### Software

- **OS**: Linux x86_64 (tested on Ubuntu 22.04)
- **Python**: ≥3.12 (leap-finetune requirement)
- **CUDA**: 12.1+ with cuDNN 8
- **Docker**: (optional but recommended)

### Dependencies

- **transformers**: `==4.53.0` (pinned for LFM2-700M compatibility)
- **leap-finetune**: Latest from GitHub
- **PyTorch**, **PEFT**, **TRL**, **Accelerate**, **Ray**, **Datasets**

---

## 📦 Repository Structure

```
lfm2-700m-mental-health-finetune/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── docker/
│   ├── Dockerfile                 # Production container image
│   └── entrypoint.sh              # Container startup script
├── runpod/
│   ├── start.sh                   # Main Runpod execution script
│   └── template.env.example       # Environment variable template
├── configs/
│   ├── job_config.yaml            # Main job configuration
│   ├── peft_lora.yaml             # LoRA/PEFT parameters
│   └── sft_train.yaml             # SFT training arguments
├── data/
│   ├── prepare_dataset.py         # Dataset download & preprocessing
│   └── samples/
│       └── seed_eval.jsonl        # Evaluation prompts
├── scripts/
│   ├── train.sh                   # Full training script
│   ├── dry_run.sh                 # Validation & 1-step smoke test
│   ├── resume.sh                  # Resume from checkpoint
│   ├── bundle.sh                  # Package for LEAP deployment
│   └── eval_sample.py             # Evaluate on seed prompts
├── src/
│   ├── dataset_loader.py          # Dataset utilities
│   ├── validate_env.py            # Environment validation
│   └── sanity_checks.py           # Config validation
├── tests/
│   ├── test_dataset_loader.py     # Dataset tests
│   ├── test_configs.py            # Config validation tests
│   └── test_smoke_training.py     # Smoke tests
├── outputs/                       # Created at runtime
│   ├── datasets/                  # Prepared train/val JSONL
│   ├── sft/                       # Training checkpoints
│   ├── logs/                      # Training logs
│   ├── eval/                      # Evaluation results
│   └── bundles/                   # LEAP deployment bundles
└── pyproject.toml                 # Python project metadata
```

---

## 🚀 Quick Start

### Option 1: Runpod (Recommended)

1. **Create a Runpod pod** with:
   - GPU: A100 40GB or better
   - Template: PyTorch 2.x with CUDA 12.1+

2. **Set environment variables** in Runpod dashboard:
   ```bash
   HF_TOKEN=hf_...              # Required: Your Hugging Face token
   WANDB_API_KEY=...            # Optional: W&B for experiment tracking
   LEAP_API_KEY=...             # Optional: For LEAP bundling
   ```

3. **Clone this repository** and run:
   ```bash
   cd /workspace
   git clone <this-repo-url> lfm2-mental-health
   cd lfm2-mental-health
   bash runpod/start.sh
   ```

The `start.sh` script will:
- Download and prepare the dataset
- Run a dry-run validation (1 training step)
- Execute full training
- Bundle the model for LEAP
- Evaluate on sample prompts

### Option 2: Local Docker

1. **Copy environment template**:
   ```bash
   cp runpod/template.env.example .env
   # Edit .env and fill in your HF_TOKEN
   ```

2. **Build Docker image**:
   ```bash
   docker build -t lfm2-mental-health -f docker/Dockerfile .
   ```

3. **Run container**:
   ```bash
   docker run --gpus all --env-file .env -it lfm2-mental-health
   ```

4. **Inside container**:
   ```bash
   # Container entrypoint will automatically run runpod/start.sh
   # Or run steps manually:
   python data/prepare_dataset.py --out ./outputs/datasets
   bash scripts/dry_run.sh
   bash scripts/train.sh
   bash scripts/bundle.sh
   python scripts/eval_sample.py
   ```

### Option 3: Local (Without Docker)

1. **Install dependencies**:
   ```bash
   # Requires Python ≥3.12
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv pip install "transformers==4.53.0" datasets accelerate peft trl wandb \
       "ray[default]" pytest pyyaml huggingface-hub
   uv pip install "git+https://github.com/Liquid4All/leap-finetune"
   ```

2. **Set environment variables**:
   ```bash
   export HF_TOKEN="hf_..."
   export WANDB_API_KEY="..."  # optional
   export JOB_NAME="my-job-$(date +%Y%m%d-%H%M%S)"
   ```

3. **Run pipeline**:
   ```bash
   python data/prepare_dataset.py --out ./outputs/datasets
   bash scripts/dry_run.sh
   bash scripts/train.sh
   bash scripts/bundle.sh
   python scripts/eval_sample.py
   ```

---

## 📝 Detailed Usage

### 1. Dataset Preparation

```bash
python data/prepare_dataset.py \
    --out ./outputs/datasets \
    --train-split 0.97 \
    --min-length 10 \
    --max-length 8000 \
    --seed 42 \
    --preview 5
```

**What it does:**
- Downloads `Amod/mental_health_counseling_conversations` from Hugging Face
- Transforms `Context`/`Response` fields to TRL SFT `messages` format:
  ```json
  {"messages": [
    {"role": "user", "content": "<Context>"},
    {"role": "assistant", "content": "<Response>"}
  ]}
  ```
- Filters empty/invalid samples
- Shuffles and splits into train (97%) / validation (3%)
- Validates schema and prints preview

**Output:**
- `outputs/datasets/train.jsonl`
- `outputs/datasets/val.jsonl`

### 2. Dry-Run Validation

```bash
bash scripts/dry_run.sh
```

**What it does:**
- Validates environment (Python version, CUDA, transformers version)
- Checks GPU memory availability
- Downloads and loads LFM2-700M model
- Runs **1 training step** on 5 samples to verify:
  - Model loading works
  - Dataset format is correct
  - No OOM errors
  - Forward/backward pass completes

**Exit codes:**
- `0`: All checks passed → safe to proceed with full training
- `1`: Validation failed → fix issues before training

### 3. Full Training

```bash
bash scripts/train.sh
```

**Auto-tuning:**
- Automatically detects GPU memory and sets batch size:
  - >70GB: `batch_size=8, grad_accum=2` (effective=16)
  - >40GB: `batch_size=6, grad_accum=3` (effective=18)
  - >20GB: `batch_size=4, grad_accum=4` (effective=16)
  - <20GB: `batch_size=2, grad_accum=8` (effective=16)

**Training features:**
- LoRA/PEFT with configurable rank, alpha, dropout
- Gradient checkpointing for memory efficiency
- BF16 mixed precision
- Cosine learning rate schedule with warmup
- Evaluation every N steps
- Saves best model based on validation loss
- Keeps last 3 checkpoints

**Outputs:**
- Checkpoints: `outputs/sft/<JOB_NAME>/checkpoint-*/`
- Final model: `outputs/sft/<JOB_NAME>/final/`
- Logs: `outputs/logs/<JOB_NAME>.log`

### 4. Resume Training

```bash
bash scripts/resume.sh /workspace/outputs/sft/my-job/checkpoint-1000
```

**Use when:**
- Training was interrupted
- You want to continue for more epochs
- Adjusting hyperparameters mid-training

### 5. Bundle for LEAP

```bash
bash scripts/bundle.sh /workspace/outputs/sft/my-job/final
```

**What it does:**
- Packages the LoRA adapter into a LEAP-compatible bundle
- Creates manifest with metadata
- Produces `.tar` archive for deployment

**Output:**
- `outputs/bundles/<JOB_NAME>.tar`

**Next steps:**
- Upload to LEAP platform: https://leap.liquid.ai/docs/finetuning

### 6. Evaluate on Samples

```bash
python scripts/eval_sample.py \
    --checkpoint ./outputs/sft/my-job/final \
    --samples ./data/samples/seed_eval.jsonl \
    --max-new-tokens 256 \
    --temperature 0.7
```

**What it does:**
- Loads the fine-tuned model
- Runs inference on seed prompts covering:
  - General anxiety
  - Depression
  - Stress management
  - Crisis scenarios
  - Relationship issues
- Saves responses to JSONL

**Output:**
- `outputs/eval/<JOB_NAME>.jsonl`

---

## ⚙️ Configuration

### Job Configuration (`configs/job_config.yaml`)

```yaml
model_id: "LiquidAI/LFM2-700M"
dataset_paths:
  train: "./outputs/datasets/train.jsonl"
  validation: "./outputs/datasets/val.jsonl"
output_dir: "./outputs/sft"
peft_config_path: "./configs/peft_lora.yaml"
training_config_path: "./configs/sft_train.yaml"
use_peft: true

model_config:
  trust_remote_code: true  # Required for LFM2-700M
  torch_dtype: "bfloat16"
```

### LoRA Configuration (`configs/peft_lora.yaml`)

```yaml
r: 16                      # LoRA rank
lora_alpha: 32             # Scaling factor (typically 2*r)
lora_dropout: 0.05         # Dropout for regularization
target_modules:            # Layers to adapt
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
task_type: "CAUSAL_LM"
```

### Training Configuration (`configs/sft_train.yaml`)

```yaml
num_train_epochs: 3
per_device_train_batch_size: 4    # Auto-tuned by train.sh
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.03
bf16: true
max_seq_length: 2048

logging_steps: 10
eval_steps: 100
save_steps: 200
save_total_limit: 3
```

**Customization:**
- Override via environment variables in training scripts
- Edit YAML files directly for permanent changes

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

**Test coverage:**
- **Dataset loader**: Schema validation, PII sanitization
- **Configs**: Field validation, type checking, consistency
- **Smoke tests**: Minimal training loop on CPU (GPU if available)

**Note:** Full integration testing is handled by `scripts/dry_run.sh`.

---

## 🐛 Troubleshooting

### OOM (Out of Memory) Errors

**Solutions:**
1. Reduce batch size in `configs/sft_train.yaml`
2. Increase `gradient_accumulation_steps`
3. Enable gradient checkpointing (already on by default)
4. Reduce `max_seq_length` to 1024 or lower

### Model Download Fails

**Check:**
1. `HF_TOKEN` is set and valid
2. You've accepted LFM2-700M model terms on Hugging Face
3. Internet connectivity is stable

### Dataset Download Fails

**Check:**
1. `HF_TOKEN` is set
2. You've accepted dataset terms: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
3. Review dataset license requirements

### Transformers Version Error

**Solution:**
```bash
pip install --force-reinstall transformers==4.53.0
```

LFM2-700M requires exactly version 4.53.0 with `trust_remote_code=True` until native support is added.

### W&B Authentication

If you see W&B errors but don't want experiment tracking:

```bash
unset WANDB_API_KEY
# Or edit configs/sft_train.yaml and set: report_to: "none"
```

---

## 📊 Expected Results

### Dataset

- **Total samples**: ~3,000-4,000 (varies by dataset version)
- **Train/val split**: 97% / 3%
- **Avg conversation length**: ~200-500 tokens
- **Fields**: Context (user) + Response (counselor)

### Training Time

On A100 40GB with default settings:
- **Dry-run**: ~2-5 minutes
- **Full training** (3 epochs): ~2-4 hours (dataset-dependent)

### Model Size

- **Base model**: ~1.4GB (700M parameters)
- **LoRA adapter**: ~50-100MB (depends on rank)
- **Bundle**: ~100-200MB

### Evaluation

Sample responses are for research purposes only. Always include disclaimers when sharing outputs.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

---

## 📄 License

This repository is licensed under the **MIT License** (see `LICENSE` file).

**Important**: This MIT license applies **only to the code in this repository**. The following have separate licenses:

- **Dataset**: Amod/mental_health_counseling_conversations is under RAIL-D License (requires donation for commercial use)
- **Model**: LiquidAI/LFM2-700M has its own license terms on Hugging Face
- **leap-finetune**: Check the leap-finetune repository for its license

---

## 🙏 Acknowledgments

- **Liquid AI** for LFM2 models and leap-finetune framework
- **Amod** for the mental health counseling dataset
- **Hugging Face** for transformers, datasets, and PEFT libraries
- **TRL team** for supervised fine-tuning tools

---

## 📮 Contact & Support

- **Issues**: https://github.com/<your-org>/lfm2-mental-health/issues
- **Discussions**: https://github.com/<your-org>/lfm2-mental-health/discussions

For leap-finetune support: https://github.com/Liquid4All/leap-finetune/issues
For LEAP platform support: https://leap.liquid.ai/docs/finetuning

---

## 🔗 Additional Resources

- **LFM2 Model Family**: https://huggingface.co/LiquidAI
- **LEAP Platform**: https://leap.liquid.ai
- **leap-finetune Documentation**: https://github.com/Liquid4All/leap-finetune
- **Mental Health Dataset**: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **TRL Documentation**: https://huggingface.co/docs/trl

---

**Built with ❤️ for responsible AI research in mental health support.**
