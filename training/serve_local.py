"""Run the deployed inference handler locally.

`next dev` does not execute Vercel's Python runtime, so /api/predict is simply
absent during local development. This serves the *same* handler class from
web/api/predict.py on localhost, which means local development exercises the
production code path rather than a mock.

    python training/serve_local.py
    # then, in web/.env.local:
    #   NEXT_PUBLIC_PREDICT_URL=http://127.0.0.1:8000/api/predict
"""

from __future__ import annotations

import argparse
import sys
from http.server import HTTPServer
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent / "web" / "api"
sys.path.insert(0, str(API_DIR))

from predict import handler as PredictHandler  # noqa: E402


class Router(PredictHandler):
    """Accepts /api/predict as well as /, so either URL works locally."""

    def _routable(self) -> bool:
        return self.path.split("?")[0] in ("/", "/api/predict", "/predict")

    def do_GET(self):
        if not self._routable():
            return self._send(404, {"ok": False, "error": f"No route {self.path}"})
        return super().do_GET()

    def do_POST(self):
        if not self._routable():
            return self._send(404, {"ok": False, "error": f"No route {self.path}"})
        return super().do_POST()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    model = API_DIR / "model" / "model.onnx"
    if not model.exists():
        print(f"! {model} is missing — run training/export_onnx.py first.\n")

    print(f"Local inference server  http://{args.host}:{args.port}/api/predict")
    print("Set NEXT_PUBLIC_PREDICT_URL to that value in web/.env.local\n")
    HTTPServer((args.host, args.port), Router).serve_forever()


if __name__ == "__main__":
    main()
