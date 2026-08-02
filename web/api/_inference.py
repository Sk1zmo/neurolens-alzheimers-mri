"""ONNX inference + class activation mapping for the Vercel Python function.

Dependencies are deliberately minimal — onnxruntime, numpy, Pillow. No torch
(the wheel alone exceeds the function bundle limit) and no OpenCV (60 MB for
two calls we can write in numpy).

The exported graph hands back the CAM tensor directly, because the model head
is `global-average-pool -> Linear`, which makes the classic CAM a plain 1x1
convolution folded into the graph at export time. So a real explanation map
costs one forward pass and no autograd.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
MODEL_PATH = MODEL_DIR / "model.onnx"
META_PATH = MODEL_DIR / "model_meta.json"

_lock = threading.Lock()
_session: ort.InferenceSession | None = None
_meta: dict[str, Any] | None = None

DEFAULT_META: dict[str, Any] = {
    "model_name": "alzheimer_effnetb0",
    "version": "1.0.0",
    "classes": ["Non Demented", "Very Mild Demented", "Mild Demented",
                "Moderate Demented"],
    "class_dirs": ["NonDemented", "VeryMildDemented", "MildDemented",
                   "ModerateDemented"],
    "img_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "temperature": 1.0,
}


def get_meta() -> dict[str, Any]:
    global _meta
    if _meta is None:
        if META_PATH.exists():
            _meta = {**DEFAULT_META,
                     **json.loads(META_PATH.read_text(encoding="utf-8"))}
        else:
            _meta = dict(DEFAULT_META)
    return _meta


def get_session() -> ort.InferenceSession:
    """Lazily build the session once per warm container."""
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"model.onnx not found at {MODEL_PATH}. Run "
                        "training/export_onnx.py to generate and deploy it."
                    )
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
                opts.graph_optimization_level = \
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                _session = ort.InferenceSession(
                    str(MODEL_PATH), sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
    return _session


# ---------------------------------------------------------------- preprocess
def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    """Mirrors training/dataset.py::eval_transform exactly."""
    meta = get_meta()
    size = int(meta["img_size"])
    resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(meta["mean"], dtype=np.float32)) \
        / np.array(meta["std"], dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


# ------------------------------------------------------------------- helpers
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def bilinear_resize(m: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Align-corners bilinear upsample of a 2-D array. ~7x7 -> 224x224."""
    in_h, in_w = m.shape
    if in_h == 1 or in_w == 1:
        return np.full((out_h, out_w), float(m.mean()), dtype=np.float32)

    ys = np.linspace(0, in_h - 1, out_h, dtype=np.float32)
    xs = np.linspace(0, in_w - 1, out_w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    top = m[y0][:, x0] * (1 - wx) + m[y0][:, x1] * wx
    bot = m[y1][:, x0] * (1 - wx) + m[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def heat_colormap(v: np.ndarray) -> np.ndarray:
    """Single-hue (orange) sequential ramp, monotonic in lightness.

    Returns uint8 RGB. Magnitude is carried by lightness here and by alpha in
    `render_overlay`, so low activation fades into the scan instead of painting
    a misleading cool-coloured floor over it.
    """
    stops = np.array([
        [0.00, 92, 34, 12],
        [0.35, 176, 62, 22],
        [0.65, 235, 104, 52],
        [1.00, 255, 198, 128],
    ], dtype=np.float32)
    pos, cols = stops[:, 0], stops[:, 1:]
    out = np.empty(v.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, pos, cols[:, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def render_overlay(img: Image.Image, cam: np.ndarray,
                   alpha_max: float = 0.72) -> tuple[str, str]:
    """Blend the CAM over the scan. Returns (overlay_png_b64, cam_png_b64)."""
    size = int(get_meta()["img_size"])
    base = np.asarray(img.resize((size, size), Image.BILINEAR),
                      dtype=np.float32)

    up = bilinear_resize(cam, size, size)
    up = np.maximum(up, 0.0)
    peak = float(up.max())
    up = up / peak if peak > 1e-8 else np.zeros_like(up)

    rgb = heat_colormap(up).astype(np.float32)
    alpha = (np.power(up, 1.15) * alpha_max)[..., None]
    blended = np.clip(base * (1 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)

    return (_png_b64(Image.fromarray(blended)),
            _png_b64(Image.fromarray(heat_colormap(up))))


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def thumbnail_b64(img: Image.Image, size: int = 320) -> str:
    t = img.copy()
    t.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------- input plausibility
def scan_plausibility(img: Image.Image) -> dict[str, Any]:
    """Cheap structural check: does this look like an axial brain slice?

    Not a classifier — a guard rail. The previous deployment gated on nothing
    but global mean/std, which passes almost any photograph. These four signals
    describe what a brain MRI slice actually looks like: near-greyscale, a dark
    surround, one bright centred mass, and real internal texture.
    """
    small = img.resize((128, 128), Image.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32)
    gray = rgb.mean(axis=2)

    saturation = float(np.mean(rgb.max(axis=2) - rgb.min(axis=2)) / 255.0)

    thresh = max(18.0, float(gray.mean()) * 0.45)
    fg = gray > thresh
    fg_ratio = float(fg.mean())

    border = np.concatenate([gray[:6].ravel(), gray[-6:].ravel(),
                            gray[:, :6].ravel(), gray[:, -6:].ravel()])
    border_dark = float((border < thresh).mean())

    if fg.any():
        ys, xs = np.nonzero(fg)
        cy, cx = ys.mean() / 128.0, xs.mean() / 128.0
        centrality = 1.0 - min(1.0, 2.0 * float(np.hypot(cy - 0.5, cx - 0.5)))
    else:
        centrality = 0.0

    gy, gx = np.gradient(gray)
    texture = float(np.mean(np.hypot(gy, gx)) / 255.0)

    checks = {
        "greyscale": saturation < 0.10,
        "dark_surround": border_dark > 0.70,
        "brain_sized_mass": 0.12 < fg_ratio < 0.78,
        "centred": centrality > 0.55,
        "has_texture": 0.004 < texture < 0.16,
    }
    passed = sum(checks.values())
    return {
        "looks_like_brain_mri": passed >= 4,
        "checks": checks,
        "score": round(passed / len(checks), 3),
        "signals": {
            "saturation": round(saturation, 4),
            "foreground_ratio": round(fg_ratio, 4),
            "dark_border_ratio": round(border_dark, 4),
            "centrality": round(centrality, 4),
            "texture": round(texture, 5),
        },
    }


# ------------------------------------------------------------------ predict
def predict(data: bytes, want_overlay: bool = True) -> dict[str, Any]:
    meta = get_meta()
    img = load_image(data)
    session = get_session()

    logits, cam = session.run(None, {"input": preprocess(img)})
    logits = np.asarray(logits, dtype=np.float32)[0]
    cam = np.asarray(cam, dtype=np.float32)[0]

    temperature = float(meta.get("temperature", 1.0)) or 1.0
    probs = softmax(logits / temperature)
    calibrated = probs
    raw = softmax(logits)
    idx = int(np.argmax(calibrated))

    # Free-energy OOD score: low energy = looks like training data. Threshold
    # comes from the held-out test distribution recorded at evaluation time.
    energy = float(-np.log(np.exp(logits - logits.max()).sum()) - logits.max())
    ood = meta.get("test_headline", {}).get("energy_p99")
    order = np.argsort(calibrated)[::-1]
    margin = float(calibrated[order[0]] - calibrated[order[1]])

    plaus = scan_plausibility(img)
    overlay = cam_png = None
    if want_overlay:
        overlay, cam_png = render_overlay(img, cam[idx])

    return {
        "ok": True,
        "class_id": idx,
        "label": meta["classes"][idx],
        "class_dir": meta["class_dirs"][idx],
        "confidence": float(calibrated[idx]),
        "confidence_uncalibrated": float(raw[idx]),
        "margin": margin,
        "probabilities": [float(p) for p in calibrated],
        "classes": meta["classes"],
        "logits": [float(v) for v in logits],
        "energy": energy,
        "out_of_distribution": bool(ood is not None and energy > float(ood)),
        "input_check": plaus,
        "overlay_png": overlay,
        "cam_png": cam_png,
        "image_size": list(img.size),
        "model": {
            "name": meta.get("model_name"),
            "version": meta.get("version"),
            "temperature": temperature,
            "test_accuracy": meta.get("test_headline", {}).get("accuracy"),
            "test_macro_f1": meta.get("test_headline", {}).get("macro_f1"),
        },
    }
