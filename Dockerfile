FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=False

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libjpeg-turbo8 \
    libpng16-16 \
    libtiff6 \
    libopenjp2-7 \
    libheif1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /app/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Verificaciones fail-fast
RUN python -c "import numpy as np; print('numpy ok', np.__version__)"
RUN python -c "import torch; print('torch ok', torch.__version__, '| cuda:', torch.cuda.is_available())"
RUN python -c "from ultralytics import YOLO; print('ultralytics ok')"

# Precarga modelo
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY . /app

CMD ["python", "-u", "/app/handler.py"]
