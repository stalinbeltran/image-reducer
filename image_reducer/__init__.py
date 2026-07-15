"""image-reducer: estandarización de imágenes para entrenar detección de texto.

Aplica el MISMO tratamiento a las imágenes de un dataset (con re-mapeo de sus
anotaciones/boxes) y a una imagen suelta de inferencia, de modo que la red vea
entradas idénticas en entrenamiento y en producción.

Tratamientos:
  * Reducción a una "miniatura" de tamaño parametrizable (letterbox por defecto,
    conserva el aspecto y rellena con padding).
  * Estandarización de color a escala de grises (L, 8-bit).
  * Difuminado gaussiano opcional (solo a la imagen, nunca a la máscara).
  * Normalización de contraste opcional.

Puntos de entrada:
  * `ReduceConfig`          — configuración del pipeline.
  * `process_image`         — procesa una imagen PIL y devuelve (imagen, transform).
  * `process_for_inference` — helper para inferencia (imagen -> imagen + transform).
  * `process_dataset`       — procesa un dataset completo (SAMPLE_FORMAT.md).
  * `process_folder`        — procesa una carpeta plana de imágenes (sin labels).
"""

from .config import ReduceConfig
from .geometry import ResizeTransform, compute_transform
from .pipeline import process_image, process_mask, process_for_inference
from .labels import transform_labels
from .dataset import process_dataset, process_folder

__all__ = [
    "ReduceConfig",
    "ResizeTransform",
    "compute_transform",
    "process_image",
    "process_mask",
    "process_for_inference",
    "transform_labels",
    "process_dataset",
    "process_folder",
]

__version__ = "0.1.0"
