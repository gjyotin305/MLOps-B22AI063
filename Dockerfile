FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

# Install CUDA-enabled PyTorch and required Python packages
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio && \
    pip install \
    transformers \
    datasets \
    accelerate \
    scikit-learn \
    scipy \
    pandas \
    numpy \
    huggingface_hub

COPY finetune_and_eval.py /app/finetune_and_eval.py

# Optional: pass HF token at runtime if push_to_hub is needed by the script
# docker run --gpus all -e HF_TOKEN=... <image>
ENV HF_HOME=/tmp/huggingface

CMD ["python", "finetune_and_eval.py"]
