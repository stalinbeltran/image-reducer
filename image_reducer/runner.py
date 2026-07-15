"""Ejecución de jobs en segundo plano con reporte de progreso.

`/api/process` responde de inmediato con un job en estado 'running'; el trabajo
real corre en un hilo. El progreso vivo (done/total) se mantiene en memoria y se
superpone al registro al consultar, para no escribir el JSON en cada imagen.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ReduceConfig
from .registry import Registry
from .service import count_items, detect_mode, run_job, validate_paths


class JobCancelled(Exception):
    """El usuario canceló el job en curso (entre imágenes)."""


class JobManager:
    def __init__(self, registry: Registry, max_workers: int = 2) -> None:
        self.registry = registry
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._live: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Jobs que quedaron 'running' de una sesión previa ya no lo están.
        self.registry.mark_stale_running()

    def start(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        config: ReduceConfig,
        mode: Optional[str] = None,
        recursive: bool = False,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Valida, crea el job en 'running' y lanza el hilo. Las validaciones
        que pueden fallar (rutas, existencia, modo) ocurren aquí, de forma
        síncrona, para poder devolver el error HTTP correcto."""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"No existe la entrada: {input_path}")
        validate_paths(input_path, output_dir)
        mode = mode or detect_mode(input_path)
        total = count_items(input_path, mode, recursive)

        job = self.registry.create_job({
            "label": label,
            "mode": mode,
            "input": str(input_path.resolve()),
            "output": str(output_dir.resolve()),
            "config": config.to_dict(),
            "status": "running",
            "progress": {"done": 0, "total": total},
        })
        with self._lock:
            self._live[job["id"]] = {"done": 0, "total": total,
                                     "status": "running", "cancel": threading.Event()}

        self._executor.submit(
            self._run, job["id"], input_path, output_dir, config, mode, recursive
        )
        return self.overlay(job)

    def cancel(self, job_id: str) -> bool:
        """Solicita cancelar un job en ejecución. Devuelve True si estaba corriendo."""
        with self._lock:
            live = self._live.get(job_id)
            if not live:
                return False
            live["cancel"].set()
            return True

    def _run(self, job_id, input_path, output_dir, config, mode, recursive) -> None:
        with self._lock:
            live = self._live.get(job_id)
            cancel = live["cancel"] if live else threading.Event()

        def on_progress(done: int, total: int) -> None:
            if cancel.is_set():
                raise JobCancelled()
            with self._lock:
                live = self._live.get(job_id)
                if live is not None:
                    live["done"], live["total"] = done, total

        try:
            summary = run_job(input_path, output_dir, config, mode, recursive, on_progress)
            total = summary.get("images", 0)
            self.registry.set_job_state(
                job_id, status="success", summary=summary,
                progress={"done": total, "total": total},
            )
        except JobCancelled:
            with self._lock:
                live = self._live.get(job_id) or {}
                progress = {"done": live.get("done", 0), "total": live.get("total", 0)}
            self.registry.set_job_state(
                job_id, status="cancelled", progress=progress,
                error="Cancelado por el usuario. Se conservan los archivos ya escritos.",
            )
        except Exception as e:  # noqa: BLE001 - se registra en el job
            self.registry.set_job_state(job_id, status="error", error=str(e))
        finally:
            with self._lock:
                self._live.pop(job_id, None)

    def overlay(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Superpone el progreso vivo al job persistido."""
        if not job:
            return job
        with self._lock:
            live = self._live.get(job["id"])
        if live and job.get("status") == "running":
            job = {**job, "progress": {"done": live["done"], "total": live["total"]}}
        return job

    def overlay_all(self, jobs):
        return [self.overlay(j) for j in jobs]
