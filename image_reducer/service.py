"""Orquestación de procesamiento y reglas de seguridad de rutas.

Une el pipeline con la política de "los resultados nunca se guardan en la
ubicación de origen" y detecta automáticamente el modo (imagen/dataset/carpeta).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image

from .config import ReduceConfig
from .dataset import IMAGE_EXTS, process_dataset, process_folder
from .pipeline import process_for_inference

ProgressFn = Callable[[int, int], None]


class PathSafetyError(ValueError):
    """El destino solaparía la ubicación de origen."""


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def detect_mode(input_path: Path) -> str:
    """Devuelve 'image', 'dataset' o 'folder' según lo que sea la entrada."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            return "image"
        raise ValueError(f"El archivo no es una imagen soportada: {input_path.name}")
    if not input_path.is_dir():
        raise FileNotFoundError(f"No existe: {input_path}")
    if (input_path / "labels.jsonl").exists():
        return "dataset"
    return "folder"


def validate_paths(input_path: Path, output_dir: Path) -> None:
    """Impone que la salida nunca caiga en la ubicación de origen.

    - Origen archivo: la salida no puede ser la carpeta que contiene el archivo
      (una subcarpeta sí se permite).
    - Origen carpeta (dataset/folder): la salida no puede solaparse con ella en
      ninguna dirección.
    """
    inp = input_path.resolve()
    out = output_dir.resolve()
    origin_dir = inp.parent if inp.is_file() else inp

    if out == origin_dir:
        raise PathSafetyError(
            "El destino no puede ser la ubicación de origen. Elige otra carpeta."
        )
    if inp.is_dir():
        if _is_relative_to(out, origin_dir):
            raise PathSafetyError(
                "El destino no puede estar dentro de la carpeta de origen."
            )
        if _is_relative_to(origin_dir, out):
            raise PathSafetyError(
                "La carpeta de origen no puede estar dentro del destino."
            )


def count_items(input_path: Path, mode: str, recursive: bool = False) -> int:
    """Cuenta cuántas imágenes procesará el job (para la barra de progreso)."""
    if mode == "image":
        return 1
    if mode == "dataset":
        with (input_path / "labels.jsonl").open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    if mode == "folder":
        pattern = "**/*" if recursive else "*"
        return sum(1 for p in input_path.glob(pattern)
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return 0


def process_single_image(
    input_file: str | Path,
    output_dir: str | Path,
    config: ReduceConfig,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Procesa una sola imagen y guarda el PNG + su ResizeTransform."""
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(input_file) as im:
        im.load()
        proc, transform = process_for_inference(im, config)

    out_png = output_dir / (input_file.stem + ".png")
    proc.save(out_png)
    (output_dir / (input_file.stem + ".transform.json")).write_text(
        json.dumps(transform.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if progress:
        progress(1, 1)
    return {
        "mode": "image",
        "input": str(input_file),
        "output_dir": str(output_dir),
        "images": 1,
        "output_file": str(out_png),
        "target_size": [config.width, config.height],
    }


def run_job(
    input_path: str | Path,
    output_dir: str | Path,
    config: ReduceConfig,
    mode: Optional[str] = None,
    recursive: bool = False,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Ejecuta un procesamiento validando rutas y despachando por modo.

    `mode` se autodetecta si no se pasa. `progress(done, total)` se invoca por
    cada imagen procesada. Devuelve el resumen del pipeline.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe la entrada: {input_path}")

    validate_paths(input_path, output_dir)
    mode = mode or detect_mode(input_path)

    if mode == "image":
        return process_single_image(input_path, output_dir, config, progress)
    if mode == "dataset":
        return process_dataset(input_path, output_dir, config, progress)
    if mode == "folder":
        return process_folder(input_path, output_dir, config, recursive, progress)
    raise ValueError(f"Modo desconocido: {mode}")
