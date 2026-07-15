"""API HTTP (FastAPI) + app web para inferencia, procesamiento y registro.

Levantar con:
    uvicorn image_reducer.api:app --reload
    # o:  image-reducer-serve  (ver pyproject)

La UI web se sirve en `/`. Endpoints JSON bajo `/api`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .config import RESAMPLE_CHOICES, ReduceConfig
from .dataset import IMAGE_EXTS, process_dataset, process_folder
from .geometry import ResizeTransform
from .pipeline import process_for_inference
from .registry import Registry
from .service import PathSafetyError, detect_mode, run_job, validate_paths

app = FastAPI(
    title="image-reducer",
    version="0.1.0",
    description="Estandariza imágenes (reducción + gris + blur) para detección de texto.",
)

registry = Registry()
_WEBUI = Path(__file__).resolve().parent / "webui"


# ---------------------------------------------------------------------------
# Inferencia (subida directa de imagen)
# ---------------------------------------------------------------------------
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
    try:
        with Image.open(io.BytesIO(upload_bytes)) as im:
            im.load()
            proc, transform = process_for_inference(im, config)
    except UnidentifiedImageError:
        raise HTTPException(400, "El archivo no es una imagen válida.")
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
    png, transform = _preprocess(await file.read(), config)
    return Response(
        content=png, media_type="image/png",
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


# ---------------------------------------------------------------------------
# Navegador de sistema de archivos (para seleccionar origen y destino)
# ---------------------------------------------------------------------------
def _fs_roots() -> List[str]:
    if os.name == "nt":
        return [f"{d}:\\" for d in string.ascii_uppercase
                if Path(f"{d}:\\").exists()]
    return ["/"]


@app.get("/api/fs/roots")
def fs_roots() -> dict:
    return {"roots": _fs_roots()}


@app.get("/api/fs")
def fs_list(path: Optional[str] = Query(None)) -> dict:
    if not path:
        return {"path": None, "parent": None, "roots": _fs_roots(),
                "dirs": [], "files": []}
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, f"No es una carpeta: {path}")
    dirs: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            try:
                if entry.is_dir():
                    dirs.append({
                        "name": entry.name,
                        "path": str(entry),
                        "is_dataset": (entry / "labels.jsonl").exists(),
                    })
                elif entry.suffix.lower() in IMAGE_EXTS:
                    files.append({"name": entry.name, "path": str(entry)})
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, f"Sin permiso para leer: {path}")
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "roots": _fs_roots(),
            "dirs": dirs, "files": files,
            "is_dataset": (p / "labels.jsonl").exists()}


@app.get("/api/fs/inspect")
def fs_inspect(path: str = Query(...)) -> dict:
    """Detecta el modo de una ruta de origen (image/dataset/folder)."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"No existe: {path}")
    try:
        return {"path": str(p), "mode": detect_mode(p),
                "is_file": p.is_file()}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------------------
# Procesamiento + registro de jobs (CRUD)
# ---------------------------------------------------------------------------
class ProcessRequest(BaseModel):
    input: str
    output: str
    mode: Optional[str] = None          # autodetecta si None
    recursive: bool = False
    label: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@app.post("/api/process")
def api_process(req: ProcessRequest) -> dict:
    input_path = Path(req.input)
    output_dir = Path(req.output)
    try:
        config = ReduceConfig.from_dict(req.config or {})
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"Configuración inválida: {e}")

    try:
        summary = run_job(input_path, output_dir, config, req.mode, req.recursive)
    except PathSafetyError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    job = registry.create_job({
        "label": req.label,
        "mode": summary.get("mode"),
        "input": str(input_path.resolve()),
        "output": str(output_dir.resolve()),
        "config": config.to_dict(),
        "status": "success",
        "summary": summary,
    })
    return job


class JobPatch(BaseModel):
    label: Optional[str] = None
    notes: Optional[str] = None


@app.get("/api/jobs")
def jobs_list() -> dict:
    return {"jobs": registry.list_jobs()}


@app.get("/api/jobs/{job_id}")
def jobs_get(job_id: str) -> dict:
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    return job


@app.patch("/api/jobs/{job_id}")
def jobs_update(job_id: str, patch: JobPatch) -> dict:
    updated = registry.update_job(job_id, patch.model_dump(exclude_none=True))
    if not updated:
        raise HTTPException(404, "Job no encontrado")
    return updated


@app.delete("/api/jobs/{job_id}")
def jobs_delete(job_id: str, delete_files: bool = Query(False)) -> dict:
    ok = registry.delete_job(job_id, delete_files)
    if not ok:
        raise HTTPException(404, "Job no encontrado")
    return {"deleted": job_id, "files_deleted": delete_files}


# ---------------------------------------------------------------------------
# Presets de configuración (CRUD)
# ---------------------------------------------------------------------------
class PresetCreate(BaseModel):
    name: str
    config: Dict[str, Any]


class PresetPatch(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@app.get("/api/presets")
def presets_list() -> dict:
    return {"presets": registry.list_presets()}


@app.post("/api/presets")
def presets_create(req: PresetCreate) -> dict:
    try:
        ReduceConfig.from_dict(req.config)  # valida
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"Configuración inválida: {e}")
    return registry.create_preset(req.name, req.config)


@app.patch("/api/presets/{preset_id}")
def presets_update(preset_id: str, patch: PresetPatch) -> dict:
    body = patch.model_dump(exclude_none=True)
    if "config" in body:
        try:
            ReduceConfig.from_dict(body["config"])
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"Configuración inválida: {e}")
    updated = registry.update_preset(preset_id, body)
    if not updated:
        raise HTTPException(404, "Preset no encontrado")
    return updated


@app.delete("/api/presets/{preset_id}")
def presets_delete(preset_id: str) -> dict:
    if not registry.delete_preset(preset_id):
        raise HTTPException(404, "Preset no encontrado")
    return {"deleted": preset_id}


# ---------------------------------------------------------------------------
# Endpoints legados por lotes (rutas del servidor, sin registro)
# ---------------------------------------------------------------------------
class BatchRequest(BaseModel):
    input_dir: str
    output_dir: str
    config: Optional[dict] = None
    recursive: bool = False


@app.post("/datasets/process")
def datasets_process(req: BatchRequest) -> dict:
    config = ReduceConfig.from_dict(req.config or {})
    validate_paths(Path(req.input_dir), Path(req.output_dir))
    return process_dataset(req.input_dir, req.output_dir, config)


@app.post("/folders/process")
def folders_process(req: BatchRequest) -> dict:
    config = ReduceConfig.from_dict(req.config or {})
    validate_paths(Path(req.input_dir), Path(req.output_dir))
    return process_folder(req.input_dir, req.output_dir, config, req.recursive)


# ---------------------------------------------------------------------------
# App web
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEBUI / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)
