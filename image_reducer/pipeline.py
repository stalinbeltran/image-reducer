"""Pipeline de procesamiento de una imagen (y su máscara)."""

from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageFilter, ImageOps

from .config import ReduceConfig
from .geometry import ResizeTransform, compute_transform

_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _letterbox(
    img: Image.Image, transform: ResizeTransform, resample, pad_color, mode: str
) -> Image.Image:
    """Reescala `img` según `transform` y la pega centrada sobre un lienzo del
    tamaño objetivo relleno con `pad_color`."""
    content = img.resize((transform.content_w, transform.content_h), resample)
    if (transform.content_w, transform.content_h) == (transform.target_w, transform.target_h):
        return content  # sin padding (estirado o encaje exacto)
    canvas = Image.new(mode, (transform.target_w, transform.target_h), pad_color)
    canvas.paste(content, (transform.pad_x, transform.pad_y))
    return canvas


def process_image(
    img: Image.Image, config: ReduceConfig
) -> Tuple[Image.Image, ResizeTransform]:
    """Aplica el tratamiento completo a una imagen.

    Orden: resize (letterbox) -> escala de grises -> difuminado -> normalización.
    Devuelve (imagen_procesada, transform). El `transform` permite re-mapear
    anotaciones (train) o predicciones (inferencia).
    """
    resample = _RESAMPLE[config.resample]
    transform = compute_transform(
        img.width, img.height, config.width, config.height, config.keep_aspect
    )

    # Trabajamos en RGB para pegar el padding de forma uniforme; la conversión
    # a gris se hace después para que el pad_color en gris sea exacto.
    rgb = img.convert("RGB")
    pad_rgb = (config.pad_color, config.pad_color, config.pad_color)
    out = _letterbox(rgb, transform, resample, pad_rgb, "RGB")

    if config.grayscale:
        out = out.convert("L")

    if config.blur_radius > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))

    if config.normalize:
        out = ImageOps.autocontrast(out)

    return out, transform


def process_mask(mask: Image.Image, transform: ResizeTransform) -> Image.Image:
    """Reescala una máscara binaria alineándola con la imagen procesada.

    Usa NEAREST y padding negro (0) para preservar el carácter binario. NO se le
    aplican difuminado, gris ni normalización (degradaciones solo a la imagen).
    """
    l = mask.convert("L")
    return _letterbox(l, transform, Image.Resampling.NEAREST, 0, "L")


def process_for_inference(
    img: Image.Image, config: ReduceConfig
) -> Tuple[Image.Image, ResizeTransform]:
    """Alias semántico para inferencia: una imagen desconocida recibe el mismo
    tratamiento que el dataset de entrenamiento. Devuelve la imagen lista para
    la red y el `transform` (para mapear las predicciones al original)."""
    return process_image(img, config)
