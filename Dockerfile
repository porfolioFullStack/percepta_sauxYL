FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV YOLO_AUTOINSTALL=False

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip uninstall -y opencv-python || true \
    && python -m pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

# Verificaciones fail-fast
RUN python -c "import numpy as np; print('numpy ok', np.__version__)"
RUN python -c "import torch; print('torch ok', torch.__version__, '| cuda:', torch.cuda.is_available())"
RUN python -c "from ultralytics import YOLO; print('ultralytics ok')"

# Precarga modelo
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
