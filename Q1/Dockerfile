FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip uv

RUN uv pip install --system --no-cache \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install --system --no-cache \
    transformers \
    sentencepiece \
    striprtf \
    sacrebleu

COPY ./data ./data
COPY ./translate.py ./translate.py

CMD ["python3", "translate.py", "--evaluate"]
