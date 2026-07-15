"""API HTTP (FastAPI) para inferencia y procesamiento por lotes.

Levantar con:
    uvicorn image_reducer.api:app --reload

Endpoints:
    GET  /healthz                  — comprobación de salud.
    POST /infer/preprocess         — sube una imagen, devuelve el PNG procesado
                                     (mismo tratamiento que el dataset) con el
                                     ResizeTransform en la cabecera X-Reducer-Transform.
    POST /infer/preprocess-json    — igual pero devuelve JSON {transform, image_base64}.
    POST /datasets/process         — procesa un dataset del disco del servidor.
    POST /folders/process          — procesa una carpeta plana del disco del servidor.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Optional, Tuple

from fastapi import Depends, FastAPI, File, Query, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from .config import RESAMPLE_CHOICES, ReduceConfig
from .dataset import process_dataset, process_folder
from .geometry import ResizeTransform
from .pipeline import process_for_inference

app = FastAPI(
    title="image-reducer",
    version="0.1.0",
    description="Estandariza imágenes (reducción + gris + blur) para detección de texto.",
)


def _config_from_query(
    width: int = Query(320, gt=0),
    height: int = Query(320, gt=0),
    keep_aspect: bool = Query(True, description="letterbox (True) o estirado (False)"),
    grayscale: bool = Query(True),
    blur_radius: float = Query(0.0, ge=0),
    normalize: bool = Query(False),
    pad_color: int = Query(0, ge=0, le=255),
    resample: str = Query("lanczos"),
) -> ReduceConfig:
    if resample not in RESAMPLE_CHOICES:
        resample = "lanczos"
    return ReduceConfig(
        width=width, height=height, keep_aspect=keep_aspect, grayscale=grayscale,
        blur_radius=blur_radius, normalize=normalize, pad_color=pad_color,
        resample=resample,
    )


def _preprocess(upload_bytes: bytes, config: ReduceConfig) -> Tuple[bytes, ResizeTransform]:
    with Image.open(io.BytesIO(upload_bytes)) as im:
        im.load()
        proc, transform = process_for_inference(im, config)
    buf = io.BytesIO()
    proc.save(buf, format="PNG")
    return buf.getvalue(), transform


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/infer/preprocess", response_class=Response)
async def infer_preprocess(
    file: UploadFile = File(...),
    config: ReduceConfig = Depends(_config_from_query),
) -> Response:
    """Preprocesa una imagen de inferencia y devuelve el PNG procesado. El
    ResizeTransform (para mapear predicciones al original) va en la cabecera
    `X-Reducer-Transform` como JSON."""
    png, transform = _preprocess(await file.read(), config)
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Reducer-Transform": json.dumps(transform.to_dict())},
    )


@app.post("/infer/preprocess-json")
async def infer_preprocess_json(
    file: UploadFile = File(...),
    config: ReduceConfig = Depends(_config_from_query),
) -> JSONResponse:
    png, transform = _preprocess(await file.read(), config)
    return JSONResponse({
        "transform": transform.to_dict(),
        "image_base64": base64.b64encode(png).decode("ascii"),
        "media_type": "image/png",
    })


class BatchRequest(BaseModel):
    input_dir: str
    output_dir: str
    config: Optional[dict] = None
    recursive: bool = False  # solo aplica a /folders/process


@app.post("/datasets/process")
def datasets_process(req: BatchRequest) -> dict:
    config = ReduceConfig.from_dict(req.config or {})
    return process_dataset(req.input_dir, req.output_dir, config)


@app.post("/folders/process")
def folders_process(req: BatchRequest) -> dict:
    config = ReduceConfig.from_dict(req.config or {})
    return process_folder(req.input_dir, req.output_dir, config, req.recursive)
