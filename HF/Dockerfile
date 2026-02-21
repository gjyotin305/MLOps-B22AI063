FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG MODE=train # train, evaluate two modes
ENV MODE=${MODE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch + minimal runtime deps for Set B scripts
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision && \
    pip install \
      numpy \
      pillow \
      scikit-learn \
      tqdm \
      datasets \
      huggingface_hub \
      wandb 

COPY . /app

# Run either training or evaluation based on MODE (train|evaluate)
CMD ["sh", "-c", "if [ -n \"$WANDB_API_KEY\" ]; then wandb login --relogin \"$WANDB_API_KEY\"; else echo \"WANDB_API_KEY not set, skipping wandb login\"; fi; if [ \"$MODE\" = \"train\" ]; then python train.py; elif [ \"$MODE\" = \"evaluate\" ]; then python eval.py; else echo \"Invalid MODE: $MODE (use train or evaluate)\" && exit 1; fi"]
