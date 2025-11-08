# Dockerfile for LFM2-700M Mental Health Fine-tuning
# Base image with CUDA 12.1.1 support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install uv package manager (as recommended by leap-finetune)
# Reference: https://github.com/Liquid4All/leap-finetune
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Install Python dependencies with uv
# Pin transformers to 4.53.0 as required by LFM2-700M model card
# Reference: https://huggingface.co/LiquidAI/LFM2-700M
RUN uv pip install --system \
    "transformers==4.53.0" \
    datasets \
    accelerate \
    "ray[default]" \
    peft \
    trl \
    wandb \
    pytest \
    pyyaml \
    huggingface-hub

# Install leap-finetune from GitHub
# Reference: https://github.com/Liquid4All/leap-finetune
RUN uv pip install --system "git+https://github.com/Liquid4All/leap-finetune"

# Copy repository contents
COPY . /workspace/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# Make scripts executable
RUN chmod +x docker/entrypoint.sh \
    && chmod +x runpod/start.sh \
    && chmod +x scripts/*.sh

# Entrypoint
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["bash"]
