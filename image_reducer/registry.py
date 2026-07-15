"""Registro persistente de datasets generados y de presets de configuración.

Guarda un JSON con dos colecciones ('jobs' y 'presets') y expone CRUD sobre
ambas. La ubicación de datos es configurable por la variable de entorno
`IMAGE_REDUCER_DATA` (por defecto `<repo>/.appdata`).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def default_data_dir() -> Path:
    env = os.environ.get("IMAGE_REDUCER_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / ".appdata"


class Registry:
    """Almacén JSON con locking en proceso para jobs y presets."""

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "registry.json"
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write({"jobs": [], "presets": []})

    # --- persistencia -------------------------------------------------------
    def _read(self) -> Dict[str, List[Dict[str, Any]]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
        data.setdefault("jobs", [])
        data.setdefault("presets", [])
        return data

    def _write(self, data: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # --- jobs ---------------------------------------------------------------
    def list_jobs(self) -> List[Dict[str, Any]]:
        return sorted(self._read()["jobs"], key=lambda j: j["created_at"], reverse=True)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return next((j for j in self._read()["jobs"] if j["id"] == job_id), None)

    def create_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            data = self._read()
            now = _utcnow()
            job = {
                "id": _new_id(),
                "label": job.get("label") or "",
                "notes": job.get("notes") or "",
                "mode": job.get("mode"),
                "input": job.get("input"),
                "output": job.get("output"),
                "config": job.get("config"),
                "status": job.get("status", "success"),
                "summary": job.get("summary"),
                "error": job.get("error"),
                "created_at": now,
                "updated_at": now,
            }
            data["jobs"].append(job)
            self._write(data)
            return job

    def update_job(self, job_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {"label", "notes"}
        with self._lock:
            data = self._read()
            for job in data["jobs"]:
                if job["id"] == job_id:
                    for k, v in patch.items():
                        if k in allowed:
                            job[k] = v
                    job["updated_at"] = _utcnow()
                    self._write(data)
                    return job
        return None

    def delete_job(self, job_id: str, delete_files: bool = False) -> bool:
        with self._lock:
            data = self._read()
            job = next((j for j in data["jobs"] if j["id"] == job_id), None)
            if not job:
                return False
            data["jobs"] = [j for j in data["jobs"] if j["id"] != job_id]
            self._write(data)
        if delete_files and job.get("output"):
            out = Path(job["output"])
            if out.exists() and out.is_dir():
                shutil.rmtree(out, ignore_errors=True)
        return True

    # --- presets ------------------------------------------------------------
    def list_presets(self) -> List[Dict[str, Any]]:
        return sorted(self._read()["presets"], key=lambda p: p["name"].lower())

    def create_preset(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            data = self._read()
            now = _utcnow()
            preset = {"id": _new_id(), "name": name, "config": config,
                      "created_at": now, "updated_at": now}
            data["presets"].append(preset)
            self._write(data)
            return preset

    def update_preset(self, preset_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {"name", "config"}
        with self._lock:
            data = self._read()
            for preset in data["presets"]:
                if preset["id"] == preset_id:
                    for k, v in patch.items():
                        if k in allowed:
                            preset[k] = v
                    preset["updated_at"] = _utcnow()
                    self._write(data)
                    return preset
        return None

    def delete_preset(self, preset_id: str) -> bool:
        with self._lock:
            data = self._read()
            before = len(data["presets"])
            data["presets"] = [p for p in data["presets"] if p["id"] != preset_id]
            if len(data["presets"]) == before:
                return False
            self._write(data)
            return True
