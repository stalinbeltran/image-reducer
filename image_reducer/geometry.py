"""Transformación geométrica del resize y re-mapeo de coordenadas.

El resize (letterbox o estirado) es una transformación afín simple:

    x' = x * scale_x + pad_x
    y' = y * scale_y + pad_y

`ResizeTransform` la encapsula para poder aplicarla a puntos, boxes y quads, y
para reportarla en inferencia (la red predice sobre la miniatura; para volver a
las coordenadas de la imagen original se usa `inverse_point`).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

Point = Tuple[float, float]
Box = List[float]          # [x, y, w, h]
Quad = List[List[float]]   # [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]


@dataclass(frozen=True)
class ResizeTransform:
    """Transformación de coordenadas del original a la miniatura procesada."""

    orig_w: int
    orig_h: int
    target_w: int
    target_h: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    content_w: int  # ancho del contenido reescalado, antes del padding
    content_h: int  # alto del contenido reescalado, antes del padding

    # --- forward: original -> miniatura -------------------------------------
    def point(self, x: float, y: float) -> Point:
        return (x * self.scale_x + self.pad_x, y * self.scale_y + self.pad_y)

    def quad(self, quad: Sequence[Sequence[float]], ndigits: int = 2) -> Quad:
        return [[round(px, ndigits), round(py, ndigits)]
                for px, py in (self.point(p[0], p[1]) for p in quad)]

    def box(self, box: Sequence[float], ndigits: int = 2) -> Box:
        """Transforma un AABB. Bajo escala no uniforme (estirado) el resultado
        sigue siendo el AABB envolvente de las 4 esquinas transformadas."""
        x, y, w, h = box
        corners = [
            self.point(x, y),
            self.point(x + w, y),
            self.point(x + w, y + h),
            self.point(x, y + h),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        nx, ny = min(xs), min(ys)
        return [round(nx, ndigits), round(ny, ndigits),
                round(max(xs) - nx, ndigits), round(max(ys) - ny, ndigits)]

    # --- inverse: miniatura -> original (para mapear predicciones) ----------
    def inverse_point(self, x: float, y: float) -> Point:
        return ((x - self.pad_x) / self.scale_x, (y - self.pad_y) / self.scale_y)

    def inverse_box(self, box: Sequence[float], ndigits: int = 2) -> Box:
        x, y, w, h = box
        x0, y0 = self.inverse_point(x, y)
        x1, y1 = self.inverse_point(x + w, y + h)
        return [round(x0, ndigits), round(y0, ndigits),
                round(x1 - x0, ndigits), round(y1 - y0, ndigits)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResizeTransform":
        return cls(**data)


def compute_transform(
    orig_w: int, orig_h: int, target_w: int, target_h: int, keep_aspect: bool = True
) -> ResizeTransform:
    """Calcula la transformación para reducir (orig_w, orig_h) a
    (target_w, target_h).

    keep_aspect=True  -> letterbox: escala uniforme y padding centrado.
    keep_aspect=False -> estira a target exacto (escala no uniforme, sin padding).
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("dimensiones de origen inválidas")

    if keep_aspect:
        scale = min(target_w / orig_w, target_h / orig_h)
        content_w = max(1, round(orig_w * scale))
        content_h = max(1, round(orig_h * scale))
        scale_x = scale_y = scale
    else:
        content_w, content_h = target_w, target_h
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

    pad_x = (target_w - content_w) // 2
    pad_y = (target_h - content_h) // 2

    return ResizeTransform(
        orig_w=orig_w,
        orig_h=orig_h,
        target_w=target_w,
        target_h=target_h,
        scale_x=scale_x,
        scale_y=scale_y,
        pad_x=pad_x,
        pad_y=pad_y,
        content_w=content_w,
        content_h=content_h,
    )
