"""Read-only live metrics, served from the training host's loopback interface."""
import argparse
import json
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


def tail_events(path, max_bytes=2_000_000, limit=700):
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        if size > max_bytes:
            stream.readline()
        lines = stream.read().splitlines()
    result = []
    for line in lines[-limit:]:
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def make_app(root):
    root = Path(root).resolve()
    app = FastAPI(title="H-Net-JEPA training", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

    @app.get("/api/runs")
    def list_runs():
        if not root.is_dir():
            return []
        return sorted(path.name for path in root.iterdir()
                      if path.is_dir() and not path.is_symlink() and (path / "metrics.jsonl").is_file())

    @app.get("/api/runs/{name}")
    def metrics(name: str):
        if not re.fullmatch(r"[\w.-]{1,128}", name) or name.startswith("."):
            raise HTTPException(404)
        directory = root / name
        path = directory / "metrics.jsonl"
        if directory.is_symlink() or path.is_symlink() or not path.is_file():
            raise HTTPException(404)
        return {"name": name, "last_modified": path.stat().st_mtime, "events": tail_events(path)}

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs", type=Path)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    uvicorn.run(make_app(args.runs), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
