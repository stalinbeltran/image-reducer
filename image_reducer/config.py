"""Configuración del pipeline de reducción/estandarización."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

# Métodos de resampleo soportados (nombre -> filtro PIL, resuelto en pipeline).
RESAMPLE_CHOICES = ("nearest", "bilinear", "bicubic", "lanczos")


@dataclass
class ReduceConfig:
    """Parámetros del tratamiento. Los mismos valores DEBEN usarse en train e
    inferencia para que las entradas sean idénticas.

    Attributes:
        width, height: tamaño exacto de la miniatura de salida, en píxeles.
        keep_aspect:   True = letterbox (conserva aspecto + padding). False =
                       estira a width x height exactos (deforma la imagen).
        grayscale:     convierte a escala de grises (L, 8-bit).
        blur_radius:   radio del difuminado gaussiano en px. 0 = desactivado.
                       Se aplica SOLO a la imagen, nunca a la máscara.
        normalize:     aplica autocontraste para homogeneizar iluminación.
        pad_color:     valor de gris (0-255) del relleno del letterbox. El texto
                       nunca cae en el padding, así que no afecta a las
                       anotaciones. Debe ser constante entre train e inferencia.
        resample:      filtro de resampleo para reducir la imagen.
    """

    width: int = 320
    height: int = 320
    keep_aspect: bool = True
    grayscale: bool = True
    blur_radius: float = 0.0
    normalize: bool = False
    pad_color: int = 0
    resample: str = "lanczos"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width y height deben ser > 0")
        if self.blur_radius < 0:
            raise ValueError("blur_radius no puede ser negativo")
        if not (0 <= self.pad_color <= 255):
            raise ValueError("pad_color debe estar en [0, 255]")
        if self.resample not in RESAMPLE_CHOICES:
            raise ValueError(
                f"resample debe ser uno de {RESAMPLE_CHOICES}, no {self.resample!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReduceConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})
