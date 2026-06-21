FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Precarga yolo11n.pt en el build para evitar descarga en cold start
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
