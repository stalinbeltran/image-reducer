"""Re-mapeo de las anotaciones (esquema `Labels` de SAMPLE_FORMAT.md).

Al reducir la imagen cambian `width`/`height` y todas las cajas. Bajo letterbox
(escala uniforme + traslación) los ángulos se conservan; los `box`/`quad` se
transforman punto a punto con el `ResizeTransform`.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from .geometry import ResizeTransform


def _remap_annotation(ann: Dict[str, Any], transform: ResizeTransform) -> None:
    """Transforma in-place los campos `box` y `quad` de una anotación."""
    if "box" in ann and ann["box"] is not None:
        ann["box"] = transform.box(ann["box"])
    if "quad" in ann and ann["quad"] is not None:
        ann["quad"] = transform.quad(ann["quad"])
    # `angle` se conserva bajo escala uniforme (letterbox). Con estirado sería
    # aproximado; el AABB recalculado desde el quad sigue siendo correcto.


def transform_labels(
    labels: Dict[str, Any], transform: ResizeTransform
) -> Dict[str, Any]:
    """Devuelve una copia del objeto `Labels` con la geometría re-mapeada a la
    miniatura. `image_id` y `has_overlap` no cambian."""
    out = copy.deepcopy(labels)
    out["width"] = transform.target_w
    out["height"] = transform.target_h

    for level in ("blocks", "lines", "words"):
        for ann in out.get(level, []) or []:
            _remap_annotation(ann, transform)

    return out
