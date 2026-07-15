"""Interfaz de línea de comandos.

Ejemplos:
    image-reducer dataset ./data/in ./data/out --width 320 --height 320
    image-reducer folder  ./imgs   ./imgs_out --blur 0.8 --recursive
    image-reducer image   foto.png salida.png --width 256 --height 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from .config import RESAMPLE_CHOICES, ReduceConfig
from .dataset import process_dataset, process_folder
from .pipeline import process_for_inference


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--width", type=int, default=320, help="ancho de la miniatura (px)")
    p.add_argument("--height", type=int, default=320, help="alto de la miniatura (px)")
    p.add_argument("--stretch", action="store_true",
                   help="estira a WxH exacto en vez de letterbox (deforma)")
    p.add_argument("--no-grayscale", dest="grayscale", action="store_false",
                   help="conserva color en vez de pasar a escala de grises")
    p.add_argument("--blur", type=float, default=0.0, metavar="R",
                   help="radio del difuminado gaussiano en px (0 = off)")
    p.add_argument("--normalize", action="store_true",
                   help="autocontraste para homogeneizar iluminación")
    p.add_argument("--pad-color", type=int, default=0, metavar="0-255",
                   help="gris del padding del letterbox")
    p.add_argument("--resample", choices=RESAMPLE_CHOICES, default="lanczos",
                   help="filtro de resampleo")


def _config_from_args(a: argparse.Namespace) -> ReduceConfig:
    return ReduceConfig(
        width=a.width,
        height=a.height,
        keep_aspect=not a.stretch,
        grayscale=a.grayscale,
        blur_radius=a.blur,
        normalize=a.normalize,
        pad_color=a.pad_color,
        resample=a.resample,
    )


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total}", end="", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-reducer",
        description="Reduce y estandariza imágenes para entrenar detección de texto.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ds = sub.add_parser("dataset", help="procesa un dataset con labels (SAMPLE_FORMAT.md)")
    p_ds.add_argument("input_dir")
    p_ds.add_argument("output_dir")
    _add_config_args(p_ds)

    p_fd = sub.add_parser("folder", help="procesa una carpeta plana de imágenes (sin labels)")
    p_fd.add_argument("input_dir")
    p_fd.add_argument("output_dir")
    p_fd.add_argument("--recursive", action="store_true", help="recorre subcarpetas")
    _add_config_args(p_fd)

    p_im = sub.add_parser("image", help="procesa una sola imagen (inferencia)")
    p_im.add_argument("input")
    p_im.add_argument("output")
    p_im.add_argument("--print-transform", action="store_true",
                      help="imprime el ResizeTransform en JSON por stdout")
    _add_config_args(p_im)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)

    if args.command == "dataset":
        summary = process_dataset(args.input_dir, args.output_dir, config, _progress)
        print(file=sys.stderr)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "folder":
        summary = process_folder(args.input_dir, args.output_dir, config,
                                 args.recursive, _progress)
        print(file=sys.stderr)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "image":
        with Image.open(args.input) as im:
            im.load()
            proc, transform = process_for_inference(im, config)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        proc.save(args.output)
        if args.print_transform:
            print(json.dumps(transform.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"escrito {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
