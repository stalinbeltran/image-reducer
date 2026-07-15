"""Log de auditoría append-only, asíncrono y a prueba de bloqueos.

Propiedades exigidas:
  * **Una línea JSON por operación** (JSONL). El fichero se abre siempre en modo
    'append': una escritura nunca sobrescribe a otra.
  * **Nunca se borra ni se trunca por código.** Sólo el administrador puede
    borrarlo manualmente. Ninguna otra parte del sistema toca este fichero.
  * **Asíncrono:** el hilo de la petición encola con `put_nowait` y sigue; si la
    cola está llena (log "bloqueado") NO se bloquea el proceso principal: se
    cuenta como descartada y se marca el error.
  * **Informa a la app:** `status()` expone `ok`, `last_error`, `dropped`, etc.,
    que la API y la UI consultan para avisar si el log falla.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import json
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class AuditLog:
    def __init__(self, path: str | Path, max_queue: int = 10000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._written = 0
        self._dropped = 0
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._run, name="audit-writer", daemon=True)
        self._thread.start()
        atexit.register(self.close)
        self.log("app.start", pid=_pid())

    # -- API pública (hilo de la petición; nunca bloquea ni lanza) -----------
    def log(self, event: str, **fields: Any) -> None:
        entry = {"ts": _utcnow(), "event": event}
        entry.update(fields)
        try:
            self._q.put_nowait(entry)
        except queue.Full:
            with self._lock:
                self._dropped += 1
                self._last_error = "Cola de auditoría llena: entradas descartadas."

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ok": self._last_error is None,
                "last_error": self._last_error,
                "written": self._written,
                "dropped": self._dropped,
                "queued": self._q.qsize(),
                "path": str(self.path),
                "writer_alive": self._thread.is_alive(),
            }

    def close(self, timeout: float = 2.0) -> None:
        """Vacía la cola y detiene el hilo (llamado en atexit)."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._q.put_nowait(None)  # despierta al escritor
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    # -- hilo escritor (único; garantiza append sin entrelazado) -------------
    def _run(self) -> None:
        while True:
            try:
                entry = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if entry is None:  # señal de cierre
                self._q.task_done()
                break
            try:
                line = json.dumps(entry, ensure_ascii=False, default=str)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                with self._lock:
                    self._written += 1
                    self._last_error = None  # se recupera tras un fallo transitorio
            except Exception as e:  # noqa: BLE001 - el objetivo es no propagar
                with self._lock:
                    self._dropped += 1
                    self._last_error = f"{type(e).__name__}: {e}"
            finally:
                self._q.task_done()


def _pid() -> int:
    import os
    return os.getpid()
