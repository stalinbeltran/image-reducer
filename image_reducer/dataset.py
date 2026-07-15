"""Procesamiento por lotes: datasets (con labels) y carpetas planas."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from .config import ReduceConfig
from .labels import transform_labels
from .pipeline import process_image, process_mask

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

ProgressFn = Callable[[int, int], None]


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    config: ReduceConfig,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Procesa un dataset con el layout de SAMPLE_FORMAT.md.

    Reescala imágenes y máscaras, re-mapea `labels.jsonl` + `labels/*.json` a las
    nuevas dimensiones, y escribe un dataset gemelo en `output_dir`. `specs.jsonl`
    NO se copia porque tras el tratamiento ya no regenera los píxeles; se anota
    esa invalidación en `dataset.json`.

    Devuelve un resumen del trabajo.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    labels_path = input_dir / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"No se encontró {labels_path}. Para carpetas sin anotaciones usa "
            f"process_folder()."
        )

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(labels_path)
    total = len(records)
    out_records: List[Dict[str, Any]] = []
    n_masks = 0

    for i, rec in enumerate(records):
        # --- imagen ---------------------------------------------------------
        with Image.open(input_dir / rec["image"]) as im:
            im.load()
            proc, transform = process_image(im, config)
        out_img_rel = rec["image"]
        out_img_path = output_dir / out_img_rel
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        proc.save(out_img_path)

        # --- máscara --------------------------------------------------------
        out_mask_rel = rec.get("mask")
        if out_mask_rel:
            with Image.open(input_dir / out_mask_rel) as mk:
                mk.load()
                proc_mask = process_mask(mk, transform)
            out_mask_path = output_dir / out_mask_rel
            out_mask_path.parent.mkdir(parents=True, exist_ok=True)
            proc_mask.save(out_mask_path)
            n_masks += 1

        # --- labels ---------------------------------------------------------
        new_labels = transform_labels(rec["labels"], transform)
        idx = rec.get("index", i)
        (output_dir / "labels" / f"{idx:06d}.json").write_text(
            json.dumps(new_labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        out_rec = dict(rec)
        out_rec["labels"] = new_labels
        out_records.append(out_rec)

        if progress:
            progress(i + 1, total)

    _write_jsonl(output_dir / "labels.jsonl", out_records)
    _write_dataset_json(input_dir, output_dir, config, total)

    return {
        "mode": "dataset",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images": total,
        "masks": n_masks,
        "target_size": [config.width, config.height],
    }


def _write_dataset_json(
    input_dir: Path, output_dir: Path, config: ReduceConfig, count: int
) -> None:
    meta: Dict[str, Any] = {}
    src = input_dir / "dataset.json"
    if src.exists():
        try:
            meta = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    meta["reducer"] = {
        "processed_at": _utcnow(),
        "source_dir": str(input_dir),
        "config": config.to_dict(),
        "count": count,
        "note": (
            "Imágenes/máscaras reducidas y estandarizadas por image-reducer. "
            "specs.jsonl del dataset original YA NO regenera estos píxeles y por "
            "eso no se incluye; la fuente de verdad para entrenar es labels.jsonl."
        ),
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    config: ReduceConfig,
    recursive: bool = False,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Procesa una carpeta plana de imágenes (sin anotaciones).

    Útil para lotes de inferencia. Preserva la ruta relativa de cada imagen y
    escribe `transforms.jsonl` con el `ResizeTransform` de cada una para poder
    mapear predicciones al original.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    total = len(files)
    transforms: List[Dict[str, Any]] = []

    for i, path in enumerate(files):
        with Image.open(path) as im:
            im.load()
            proc, transform = process_image(im, config)
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        proc.save(out_path)
        transforms.append({
            "source": str(rel).replace("\\", "/"),
            "output": str(rel.with_suffix(".png")).replace("\\", "/"),
            "transform": transform.to_dict(),
        })
        if progress:
            progress(i + 1, total)

    _write_jsonl(output_dir / "transforms.jsonl", transforms)

    return {
        "mode": "folder",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images": total,
        "target_size": [config.width, config.height],
    }
