# Formato de las muestras (dataset spec)

Este documento describe la estructura de los datasets que produce
**image-text-sample-generator**, para que un proyecto consumidor (p. ej. el
entrenamiento de una red neuronal) pueda cargar las muestras sin conocer el
código que las generó.

Cada dataset es un directorio autocontenido. Todas las coordenadas están en
**píxeles del canvas**, con origen `(0,0)` en la **esquina superior izquierda**,
`x` hacia la derecha e `y` hacia abajo.

---

## 1. Layout en disco

```
<dataset_id>/
├── dataset.json          # metadata + copia congelada de la receta (provenance)
├── specs.jsonl           # 1 línea por muestra: el spec determinista (regenera píxeles)
├── labels.jsonl          # 1 línea por muestra: el índice que consume el dataloader ⭐
├── images/
│   ├── 000000.png        # la imagen de entrenamiento (RGB)
│   └── 000001.png
├── masks/
│   ├── 000000.png        # máscara binaria de segmentación de texto (L, 8-bit)
│   └── 000001.png
└── labels/
    ├── 000000.json       # ground truth de una muestra (mismo contenido que labels.jsonl)
    └── 000001.json
```

- El índice `i` es un entero cero-based con **padding a 6 dígitos** (`000000`).
- `images/`, `masks/` y `labels/` son un **caché**: se pueden borrar y
  reconstruir byte-a-byte a partir de `specs.jsonl`. No son la fuente de verdad.
- **Para entrenar, lo único que necesitas leer es `labels.jsonl`** (y los PNG a
  los que apunta). Los demás archivos son opcionales.

---

## 2. `labels.jsonl` — el archivo principal ⭐

Un objeto JSON por línea (JSONL, para no cargar todo en memoria). Cada línea es
una muestra:

```json
{
  "index": 0,
  "image": "images/000000.png",
  "mask": "masks/000000.png",
  "labels": { ...objeto Labels, ver §3... }
}
```

| Campo    | Tipo             | Descripción                                                        |
|----------|------------------|--------------------------------------------------------------------|
| `index`  | int              | Índice de la muestra dentro del dataset.                           |
| `image`  | string           | Ruta al PNG **relativa a la raíz del dataset**.                    |
| `mask`   | string \| null   | Ruta a la máscara relativa a la raíz. `null` si no se generó.      |
| `labels` | object           | El ground truth completo (esquema `Labels`, §3).                   |

> Los archivos `labels/000000.json` contienen exactamente el objeto `labels`
> (sin el envoltorio `index`/`image`/`mask`), por si prefieres leerlos sueltos.

---

## 3. Esquema `Labels` (el ground truth)

Anotaciones jerárquicas en tres niveles: **bloque → línea → palabra**. Los tres
niveles describen el mismo texto con distinta granularidad.

```json
{
  "image_id": "clean-paragraphs-346dfca9/000000",
  "width": 640,
  "height": 480,
  "blocks": [ /* BlockLabel[] */ ],
  "lines":  [ /* LineLabel[]  */ ],
  "words":  [ /* WordLabel[]  */ ],
  "has_overlap": false
}
```

| Campo         | Tipo   | Descripción                                                            |
|---------------|--------|------------------------------------------------------------------------|
| `image_id`    | string | Identificador estable `"<dataset_id>/<index>"`.                        |
| `width`       | int    | Ancho del canvas en px (coincide con el PNG).                          |
| `height`      | int    | Alto del canvas en px.                                                 |
| `blocks`      | array  | Un bloque de texto por elemento (párrafo, palabra, letra, etc.).       |
| `lines`       | array  | Cada renglón de cada bloque.                                           |
| `words`       | array  | Cada palabra (token separado por espacios) de cada renglón.           |
| `has_overlap` | bool   | `true` si dos bloques renderizados se solapan de verdad (ver §5).      |

### 3.1 Geometría: `box` y `quad`

Todas las anotaciones traen la misma geometría en dos formas:

- **`box`** = `[x, y, w, h]` — rectángulo **axis-aligned** (AABB). Para texto
  rotado es el bounding box envolvente del `quad`.
- **`quad`** = `[[x0,y0], [x1,y1], [x2,y2], [x3,y3]]` — 4 esquinas en **sentido
  horario desde la superior-izquierda**. Para texto sin rotación coincide con el
  `box`; para texto rotado son las esquinas reales del texto rotado.

Usa `quad` si tu modelo trabaja con cajas orientadas; usa `box` si trabaja con
AABB. Los valores son floats en píxeles (redondeados a 2 decimales).

### 3.2 `BlockLabel`

```json
{
  "block_id": "b0",
  "kind": "paragraph",
  "text": "Mujer mundo cuando sistema y alto. Mano dejar ...",
  "angle": 0.0,
  "box":  [76.84, 308.89, 255.5, 88.67],
  "quad": [[76.84,308.89],[332.34,308.89],[332.34,397.56],[76.84,397.56]]
}
```

| Campo      | Tipo   | Descripción                                                              |
|------------|--------|--------------------------------------------------------------------------|
| `block_id` | string | Id del bloque, único dentro de la muestra (`b0`, `b1`, …).               |
| `kind`     | string | `paragraph` \| `word` \| `letter` \| `spaced`.                          |
| `text`     | string | Texto completo del bloque (palabras unidas por un espacio). UTF-8.       |
| `angle`    | float  | Rotación en grados, sobre el centro del bloque (`transform-origin: center`). Positivo = horario. |
| `box`      | Box    | AABB del bloque.                                                          |
| `quad`     | Quad   | 4 esquinas del bloque (rotadas si `angle != 0`).                         |

### 3.3 `LineLabel`

```json
{
  "block_id": "b0",
  "index": 0,
  "text": "Mujer mundo cuando sistema y",
  "box":  [76.84, 308.89, 255.49, 16.0],
  "quad": [[...],[...],[...],[...]]
}
```

| Campo      | Tipo   | Descripción                                                    |
|------------|--------|----------------------------------------------------------------|
| `block_id` | string | Bloque al que pertenece el renglón.                            |
| `index`    | int    | Índice del renglón dentro del bloque (0 = primer renglón).     |
| `text`     | string | Texto del renglón.                                             |
| `box`      | Box    | AABB del renglón.                                              |
| `quad`     | Quad   | 4 esquinas del renglón.                                        |

### 3.4 `WordLabel`

```json
{
  "block_id": "b0",
  "line_index": 0,
  "text": "Mujer",
  "box":  [76.84, 308.89, 42.47, 16.0],
  "quad": [[...],[...],[...],[...]]
}
```

| Campo        | Tipo   | Descripción                                             |
|--------------|--------|---------------------------------------------------------|
| `block_id`   | string | Bloque al que pertenece la palabra.                     |
| `line_index` | int    | Índice del renglón (coincide con `LineLabel.index`).    |
| `text`       | string | La palabra (token separado por espacios).               |
| `box`        | Box    | AABB de la palabra.                                     |
| `quad`       | Quad   | 4 esquinas de la palabra.                               |

**Relación entre niveles:** `words` con el mismo `(block_id, line_index)` forman
esa línea; `lines`/`words` con el mismo `block_id` pertenecen a ese bloque. No
hay ids cruzados adicionales: se enlazan por estos campos.

---

## 4. Máscara (`masks/000000.png`)

Máscara de segmentación **binaria** del texto, alineada píxel-a-píxel con la
imagen:

- Imagen en escala de grises (`L`, 8-bit), mismas `width`×`height` que el PNG.
- **Texto en blanco (255) sobre fondo negro (0).**
- Es el **mismo DOM renderizado dos veces**: una normal (la imagen) y otra con
  una hoja de estilo que pinta todo de negro y el texto de blanco. Por eso la
  máscara está perfectamente registrada con la imagen.
- Las degradaciones de post-proceso (blur, ruido, JPEG, escala de grises) se
  aplican **solo a la imagen, nunca a la máscara**.

---

## 5. `has_overlap` — nota importante para filtrar

La evasión de solapamientos ocurre en tiempo de resolución usando alturas
*estimadas*, así que no es perfecta. `has_overlap` es el resultado honesto
**post-render**: `true` cuando dos AABB de bloques realmente se solapan.

Si tu entrenamiento necesita muestras limpias, **filtra las que tengan
`has_overlap == true`**. Empíricamente, ~10% de las muestras pueden salir con
algún solapamiento según la receta.

---

## 6. `dataset.json` — metadata (opcional)

Provenance del dataset. No es necesario para entrenar, pero documenta cómo se
generó y permite reproducirlo:

| Campo          | Tipo   | Descripción                                                    |
|----------------|--------|----------------------------------------------------------------|
| `id`           | string | Id del dataset (= nombre del directorio).                      |
| `name`         | string | Nombre legible.                                                |
| `seed`         | int    | Semilla raíz; junto con el índice determina cada muestra.      |
| `count`        | int    | Número de muestras.                                            |
| `created_at`   | string | Timestamp ISO-8601 UTC.                                        |
| `spec_version` | int    | Versión del esquema de spec (actualmente `1`).                |
| `recipe_id`    | string \| null | Id de la receta origen.                               |
| `recipe`       | object | Copia **congelada** de la receta usada (la plantilla probabilística). |
| `build`        | object | Estado del build: `{state, done, total, error}`.              |

---

## 7. `specs.jsonl` — spec determinista (opcional)

Una línea por muestra con el `ImageSpec`: la descripción **completamente
determinista** de la muestra (sin aleatoriedad restante). Es la fuente de verdad
que permite regenerar `images/`, `masks/` y `labels/` byte-a-byte.

Normalmente el consumidor **no lo necesita** (con `labels.jsonl` + PNG basta),
pero es útil si quieres reconstruir el dataset o inspeccionar tipografía, color,
fondo, etc. Contiene: `version`, `id`, `seed`, `width`, `height`, `background`,
`blocks[]` (texto exacto, posición, fuente, tamaño, peso, color, ángulo…) y
`post`.

---

## 8. Ejemplo mínimo de carga (Python)

```python
import json
from pathlib import Path
from PIL import Image

dataset = Path("data/datasets/clean-paragraphs-346dfca9")

with (dataset / "labels.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        rec = json.loads(line)

        image = Image.open(dataset / rec["image"]).convert("RGB")
        mask  = (Image.open(dataset / rec["mask"]).convert("L")
                 if rec["mask"] else None)

        lab = rec["labels"]
        if lab["has_overlap"]:
            continue  # opcional: descartar muestras con solapamiento

        # Ejemplo: cajas de palabras orientadas para detección de texto
        for w in lab["words"]:
            text = w["text"]
            quad = w["quad"]   # [[x0,y0],...,[x3,y0]] horario desde arriba-izq
            box  = w["box"]    # [x, y, w, h] axis-aligned
            # ... alimenta tu modelo ...
```

---

## Resumen para el consumidor

1. Itera `labels.jsonl` (una línea = una muestra).
2. Carga `image` (entrada) y `labels` (ground truth); usa `mask` si haces segmentación.
3. Elige granularidad: `blocks`, `lines` o `words`; y formato de caja: `box` (AABB) o `quad` (orientada).
4. Filtra por `has_overlap` si necesitas muestras limpias.
5. Coordenadas en px, origen arriba-izquierda; `quad` en sentido horario desde la esquina superior-izquierda.
