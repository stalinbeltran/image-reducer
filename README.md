# image-reducer

Preprocesa imágenes para entrenar una red de **detección de texto** (encontrar
párrafos de texto en una imagen, sin asumir fuente, tamaño, color, fondo ni
textura). El objetivo es que la red vea entradas **estandarizadas e idénticas**
en entrenamiento y en inferencia.

Aplica tres tratamientos, todos parametrizables:

1. **Reducción** a una "miniatura" de tamaño fijo. Por defecto **letterbox**:
   conserva el aspecto (no deforma el texto) y rellena con padding.
2. **Estandarización de color** a escala de grises (`L`, 8-bit).
3. **Difuminado gaussiano** opcional, para descartar detalles innecesarios.
   (Opcional también: normalización de contraste.)

Al reducir una imagen, sus **anotaciones cambian**: `width`/`height` y todos los
`box`/`quad` de `blocks`/`lines`/`words` se re-mapean con la misma
transformación. El formato del dataset es el de [`SAMPLE_FORMAT.md`](SAMPLE_FORMAT.md).

La **misma** transformación se usa en inferencia: una imagen desconocida entra
por la API y sale con el tratamiento idéntico al del dataset. La API devuelve
además el `ResizeTransform`, con el que puedes mapear las predicciones de la red
de vuelta a las coordenadas de la imagen original.

## Instalación

```bash
pip install -e .          # núcleo + CLI (solo requiere Pillow)
pip install -e ".[api]"   # + servidor HTTP (FastAPI/uvicorn)
```

## CLI

```bash
# Dataset con anotaciones (re-mapea labels.jsonl, labels/*.json y masks/)
image-reducer dataset ./data/in ./data/out --width 320 --height 320

# Carpeta plana de imágenes (sin labels) -> escribe también transforms.jsonl
image-reducer folder ./imgs ./imgs_out --blur 0.8 --recursive

# Una sola imagen (inferencia) e imprime el transform
image-reducer image foto.png salida.png --width 256 --height 256 --print-transform
```

Opciones comunes: `--width`, `--height`, `--stretch` (estira en vez de
letterbox), `--no-grayscale`, `--blur R`, `--normalize`, `--pad-color 0-255`,
`--resample {nearest,bilinear,bicubic,lanczos}`.

> **Importante:** usa exactamente los mismos parámetros al procesar el dataset y
> al preprocesar imágenes de inferencia. Si difieren, la red verá entradas que no
> coinciden con las de entrenamiento.

## API HTTP

```bash
uvicorn image_reducer.api:app --reload
```

| Método | Ruta                       | Uso                                                        |
|--------|----------------------------|------------------------------------------------------------|
| GET    | `/healthz`                 | comprobación de salud                                      |
| POST   | `/infer/preprocess`        | sube imagen → PNG procesado; transform en `X-Reducer-Transform` |
| POST   | `/infer/preprocess-json`   | igual, pero devuelve JSON `{transform, image_base64}`      |
| POST   | `/datasets/process`        | procesa un dataset del disco del servidor                  |
| POST   | `/folders/process`         | procesa una carpeta plana del disco del servidor           |

Los parámetros de configuración van como query params en los endpoints de
inferencia (`?width=320&height=320&blur_radius=0.5&...`) y como objeto `config`
en el body JSON de los endpoints por lotes.

```bash
curl -X POST "http://localhost:8000/infer/preprocess?width=320&height=320" \
     -F "file=@foto.png" -o procesada.png -D headers.txt
```

## Uso como librería

```python
from PIL import Image
from image_reducer import ReduceConfig, process_for_inference, process_dataset

cfg = ReduceConfig(width=320, height=320, grayscale=True, blur_radius=0.8)

# Inferencia
img = Image.open("foto.png")
proc, transform = process_for_inference(img, cfg)   # proc: PIL 320x320 'L'
# transform.inverse_box(pred_box) -> coords en la imagen original

# Dataset completo
process_dataset("data/in", "data/out", cfg)
```

## Cómo se transforman las coordenadas

Con letterbox (escala uniforme `s` + padding), un punto se mapea como
`x' = x·s + pad_x`, `y' = y·s + pad_y`. Por tanto los ángulos se conservan y
los `box`/`quad` escalan linealmente. `ResizeTransform` expone `point`, `box`,
`quad` (forward) e `inverse_point`, `inverse_box` (para volver al original).

Las **máscaras** se reescalan con vecino más cercano y padding negro, y **no**
reciben difuminado, gris ni normalización (las degradaciones se aplican solo a
la imagen, como indica `SAMPLE_FORMAT.md`).

`specs.jsonl` del dataset original **no se copia**: tras el tratamiento ya no
regenera los píxeles. La fuente de verdad para entrenar sigue siendo
`labels.jsonl`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
