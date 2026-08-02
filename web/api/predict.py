"""POST /api/predict — Vercel Python Function.

Accepts the image as either raw bytes (any `image/*` content type) or JSON
`{"image": "<data-url or base64>"}`. Multipart parsing is deliberately avoided:
`cgi` is gone in Python 3.13 and hand-rolling a multipart parser is pure
liability when the client is ours and can just POST the bytes.

GET returns model metadata, which doubles as a warm-up / health probe.
"""

from __future__ import annotations

import base64
import json
import traceback
from http.server import BaseHTTPRequestHandler

try:
    from _inference import get_meta, get_session, predict
except ImportError:  # Vercel flattens the api/ dir onto sys.path differently
    from api._inference import get_meta, get_session, predict  # type: ignore

MAX_BYTES = 12 * 1024 * 1024  # 12 MB


class handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------- responses
    def _send(self, status: int, payload: dict, cache: str = "no-store") -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep function logs readable
        return

    # ------------------------------------------------------------- handlers
    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def do_GET(self) -> None:
        try:
            meta = get_meta()
            warm = True
            try:
                get_session()
            except FileNotFoundError:
                warm = False
            self._send(200, {
                "ok": True,
                "service": "alzheimer-mri-classifier",
                "model_loaded": warm,
                "model": {
                    "name": meta.get("model_name"),
                    "version": meta.get("version"),
                    "classes": meta.get("classes"),
                    "img_size": meta.get("img_size"),
                    "temperature": meta.get("temperature"),
                    "test_headline": meta.get("test_headline", {}),
                },
            }, cache="public, max-age=60")
        except Exception as e:
            self._send(500, {"ok": False, "error": str(e)})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return self._send(400, {"ok": False, "error": "Empty request body."})
            if length > MAX_BYTES:
                return self._send(413, {
                    "ok": False,
                    "error": f"Image exceeds the {MAX_BYTES // (1024*1024)} MB limit.",
                })

            body = self.rfile.read(length)
            content_type = (self.headers.get("Content-Type") or "").lower()
            want_overlay = True

            if "application/json" in content_type:
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return self._send(400, {"ok": False, "error": "Malformed JSON body."})
                raw = payload.get("image")
                if not isinstance(raw, str) or not raw:
                    return self._send(400, {
                        "ok": False,
                        "error": "JSON body must include an 'image' string "
                                 "(data URL or base64).",
                    })
                if "," in raw and raw.strip().startswith("data:"):
                    raw = raw.split(",", 1)[1]
                try:
                    image_bytes = base64.b64decode(raw, validate=False)
                except Exception:
                    return self._send(400, {"ok": False,
                                            "error": "Could not decode base64 image."})
                want_overlay = bool(payload.get("overlay", True))
            else:
                image_bytes = body

            if not image_bytes:
                return self._send(400, {"ok": False, "error": "No image data received."})

            try:
                result = predict(image_bytes, want_overlay=want_overlay)
            except FileNotFoundError as e:
                return self._send(503, {
                    "ok": False,
                    "error": str(e),
                    "hint": "The ONNX model has not been exported into "
                            "web/api/model/ yet.",
                })

            self._send(200, result)

        except Exception as e:
            traceback.print_exc()
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
