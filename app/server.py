"""Task 4 -- Amharic next-word suggestion web app (standard library only).

    python app/server.py            # then open http://localhost:8000

Endpoints
    GET /                      the single-page UI (app/static/index.html)
    GET /api/suggest?text=&order=3&k=5
    GET /api/apply?text=&word=  insert a chosen suggestion into the text
    GET /api/info              model statistics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amharic_lm import pipeline  # noqa: E402
from amharic_lm.predictor import NextWordPredictor  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
FILES = {
    "/": (STATIC / "index.html", "text/html; charset=utf-8"),
    "/static/NotoSansEthiopic.ttf": (ROOT / "fonts" / "NotoSansEthiopic.ttf", "font/ttf"),
}

PREDICTOR: NextWordPredictor | None = None
INFO: dict = {}


def load(order: int = 3):
    global PREDICTOR, INFO
    t0 = time.time()
    model = pipeline.full_model(verbose=True, mmap=True)
    PREDICTOR = NextWordPredictor(model, order)
    PREDICTOR.suggest("ኢትዮጵያ ")          # warm caches before serving threads
    PREDICTOR.suggest("ኢት")
    INFO = {
        "tokens": model.info.get("train_tokens"),
        "sentences": model.info.get("train_sentences"),
        "vocabulary": model.vocab_size,
        "ngrams": {n: len(t) for n, t in model.tables.items()},
        "load_seconds": round(time.time() - t0, 1),
    }
    print(f"model ready in {INFO['load_seconds']}s")


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        try:
            if url.path in FILES:
                path, ctype = FILES[url.path]
                self._send(path.read_bytes(), ctype)
            elif url.path == "/api/suggest":
                t0 = time.time()
                order = min(max(int(q.get("order", 3)), 1), 4)
                k = min(max(int(q.get("k", 5)), 1), 10)
                res = PREDICTOR.suggest(q.get("text", ""), k=k, order=order)
                res["ms"] = round(1000 * (time.time() - t0), 1)
                self._json(res)
            elif url.path == "/api/apply":
                self._json({"text": PREDICTOR.apply(q.get("text", ""), q.get("word", ""))})
            elif url.path == "/api/info":
                self._json(INFO)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # keep the server alive on bad input
            self._json({"error": str(exc)}, 500)

    def log_message(self, fmt, *args):
        if "/api/" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    load()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
