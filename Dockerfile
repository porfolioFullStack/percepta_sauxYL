FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

# Verificaciones fail-fast: si algo falla, el build explota acá (no en runtime)
RUN python -c "import numpy as np; print('numpy ok', np.__version__)"
RUN python -c "import torch; print('torch ok', torch.__version__)"
RUN python -c "from ultralytics import YOLO; print('ultralytics ok')"

# Precarga yolo11n.pt en el build para evitar descarga en cold start
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
