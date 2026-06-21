import base64
import io
import os

import runpod
from PIL import Image
from ultralytics import YOLO

MODEL_NAME   = os.getenv("MODEL_NAME", "yolo11n.pt")
DEFAULT_CONF = float(os.getenv("CONF", "0.25"))

model = YOLO(MODEL_NAME)


def handler(event):
    inp = event.get("input") or {}

    b64 = inp.get("image_base64")
    if not b64:
        return {"error": "missing input.image_base64"}

    conf = float(inp.get("conf", DEFAULT_CONF))

    img  = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    res  = model.predict(img, conf=conf, verbose=False)[0]

    detections = []
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            score  = float(b.conf[0].item())
            detections.append({
                "label":    model.names.get(cls_id, str(cls_id)),
                "class_id": cls_id,
                "score":    round(score, 4),
                "bbox_xyxy": [round(x1), round(y1), round(x2), round(y2)],
            })

    return {"detections": detections}


runpod.serverless.start({"handler": handler})
