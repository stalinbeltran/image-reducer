"""Arranque del servidor web/API con uvicorn.

    image-reducer-serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="image-reducer-serve",
                                     description="Sirve la app web + API de image-reducer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn
    uvicorn.run("image_reducer.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
