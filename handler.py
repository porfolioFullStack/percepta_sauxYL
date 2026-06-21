import base64
import binascii
import io
import os

import runpod
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_NAME      = os.getenv("MODEL_NAME", "yolo11n.pt")
DEFAULT_CONF    = float(os.getenv("CONF", "0.25"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))

model = YOLO(MODEL_NAME)


def decode_image(b64: str) -> Image.Image:
    if "," in b64 and b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image_base64 invalido: {e}")
    if len(raw) == 0:
        raise ValueError("image_base64 decodifico a 0 bytes")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"imagen demasiado grande: {len(raw)} bytes (max {MAX_IMAGE_BYTES})")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("los bytes decodificados no son una imagen reconocible (JPEG/PNG/etc)")


def handler(event):
    inp = event.get("input") or {}

    b64 = inp.get("image_base64")
    if not b64:
        return {"error": "missing input.image_base64"}

    conf = float(inp.get("conf", DEFAULT_CONF))

    try:
        import numpy as np  # noqa
    except Exception as e:
        return {"error": f"numpy no disponible en runtime: {type(e).__name__}: {e}"}

    try:
        img = decode_image(b64)
    except ValueError as e:
        return {"error": str(e)}

    res  = model.predict(img, conf=conf, verbose=False)[0]

    detections = []
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            score  = float(b.conf[0].item())
            detections.append({
                "label":     model.names.get(cls_id, str(cls_id)),
                "class_id":  cls_id,
                "score":     round(score, 4),
                "bbox_xyxy": [round(x1), round(y1), round(x2), round(y2)],
            })

    return {"detections": detections}


runpod.serverless.start({"handler": handler})
