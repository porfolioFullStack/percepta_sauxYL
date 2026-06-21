# percepta_sauxYL

Worker RunPod Serverless para deteccion de objetos con YOLO11n.  
Forma parte del proyecto **Percepta** (UTN — Introduccion a la Vision Artificial).

## Que hace

Recibe un frame de camara codificado en base64 (o una URL de imagen), ejecuta inferencia con `yolo11n.pt` en GPU, y devuelve las detecciones en formato JSON con etiqueta, clase, score y bounding box.

## Endpoint

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
Authorization: Bearer {API_TOKEN_RUNPOD}
Content-Type: application/json
```

### Request

```json
{
  "input": {
    "image_base64": "<base64 del frame JPEG>",
    "conf": 0.25,
    "imgsz": 640
  }
}
```

Alternativa: reemplazar `image_base64` por `image_url` con una URL publica.

### Response

```json
{
  "output": {
    "detections": [
      {
        "label": "person",
        "class_id": 0,
        "score": 0.61,
        "bbox_xyxy": [911, 468, 954, 567]
      }
    ]
  }
}
```

## Stack

| Componente | Version |
|---|---|
| Base image | `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` |
| Python | 3.10 (provisto por la base image) |
| torch | 2.2.2+cu121 (provisto por la base image, no reinstalado) |
| ultralytics | 8.3.31 |
| numpy | 1.26.4 |
| runpod | 1.7.13 |
| GPU worker | L40S 48 GB (RunPod Serverless) |

## Archivos

| Archivo | Descripcion |
|---|---|
| `handler.py` | Entry point del worker RunPod; carga YOLO11n y expone `handler()` |
| `Dockerfile` | Build de la imagen; incluye libs de sistema para OpenCV/Pillow y precarga el modelo |
| `requirements.txt` | Dependencias pip (sin torch/torchvision — los provee la base image) |

## Build y deploy

El build se dispara automaticamente en RunPod al hacer push a `main`.  
El modelo `yolo11n.pt` se descarga y bakea dentro de la imagen durante el build.

## Variables de entorno del worker

| Variable | Descripcion | Default |
|---|---|---|
| `MODEL_PATH` | Ruta al peso YOLO dentro del contenedor | `yolo11n.pt` |
| `CONF` | Umbral de confianza minimo | `0.25` |
| `YOLO_AUTOINSTALL` | Desactiva el auto-update de ultralytics en runtime | `False` |

## Notas de implementacion

- `torch` y `torchvision` **no** se incluyen en `requirements.txt`. La base image `pytorch/pytorch` los provee via conda; reinstalarlos con pip genera conflictos que se manifiestan como `RuntimeError: Numpy is not available` en runtime.
- `YOLO_AUTOINSTALL=False` es obligatorio. Sin esta variable, ultralytics instala `pi-heif` al importar, lo que actualiza pillow y corrompe el entorno pip del contenedor.
- La lib de sistema correcta para Ubuntu 22.04 (Jammy) es `libtiff5`, no `libtiff6`.

## Cliente de referencia

El dashboard que consume este worker esta en:  
`perceptaCLI/sandbox/saux/bloque3_YL/dashboard.py`
