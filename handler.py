import os
import io
import base64
import requests
from PIL import Image
import numpy as np

import runpod
from ultralytics import YOLO

MODEL_PATH   = os.getenv("MODEL_PATH", "yolo11n.pt")
DEFAULT_CONF = float(os.getenv("CONF", "0.25"))

model = YOLO(MODEL_PATH)


def _load_image_from_base64(b64_str: str) -> Image.Image:
    if "," in b64_str and b64_str.strip().lower().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def handler(event):
    inp    = event.get("input", {}) if isinstance(event, dict) else {}
    conf   = float(inp.get("conf", DEFAULT_CONF))
    imgsz  = int(inp.get("imgsz", 640))

    if inp.get("image_base64"):
        img = _load_image_from_base64(inp["image_base64"])
    elif inp.get("image_url"):
        img = _load_image_from_url(inp["image_url"])
    else:
        return {"error": "Provide 'image_base64' or 'image_url' in input."}

    # fuerza numpy antes de llamar al modelo — falla rapido si algo esta roto
    _ = np.asarray(img)

    results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
    r0      = results[0]

    detections = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        for b in r0.boxes:
            cls_id = int(b.cls[0].item())
            score  = float(b.conf[0].item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            detections.append({
                "label":     r0.names.get(cls_id, str(cls_id)),
                "class_id":  cls_id,
                "score":     round(score, 4),
                "bbox_xyxy": [round(x1), round(y1), round(x2), round(y2)],
            })

    return {"detections": detections}


runpod.serverless.start({"handler": handler})
