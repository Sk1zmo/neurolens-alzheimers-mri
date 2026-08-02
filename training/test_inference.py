"""End-to-end smoke test of the deployed inference path.

Runs web/api/_inference.py — the exact module the Vercel function imports —
over the held-out test split and checks three things the web app depends on:

  1. ONNX predictions match the PyTorch checkpoint (no silent export drift)
  2. accuracy on a sample of the test split matches the reported metrics
  3. the CAM overlay and plausibility guard actually produce output

Run after export_onnx.py:
    python training/test_inference.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import config as C

API_DIR = C.PROJECT_ROOT / "web" / "api"
sys.path.insert(0, str(API_DIR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="test images to sample")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import _inference as inf  # noqa: E402  (needs sys.path set first)

    model_path = API_DIR / "model" / "model.onnx"
    if not model_path.exists():
        raise SystemExit(f"{model_path} missing — run export_onnx.py first.")
    print(f"model: {model_path}  ({model_path.stat().st_size / 1e6:.1f} MB)")

    meta = inf.get_meta()
    print(f"classes: {meta['classes']}")
    print(f"temperature: {meta['temperature']:.4f}")

    records = json.loads((C.SPLITS_DIR / "test.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n, len(records)))

    correct = 0
    latencies: list[float] = []
    confusion = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=int)
    overlay_ok = 0
    plausible = 0

    print(f"\nrunning {len(sample)} held-out images through the serving path...")
    for i, rec in enumerate(sample, 1):
        data = Path(rec["path"]).read_bytes()
        t0 = time.perf_counter()
        out = inf.predict(data, want_overlay=(i <= 5))
        latencies.append((time.perf_counter() - t0) * 1000)

        pred, truth = out["class_id"], int(rec["label"])
        confusion[truth, pred] += 1
        correct += int(pred == truth)
        if out["overlay_png"]:
            overlay_ok += 1
        if out["input_check"]["looks_like_brain_mri"]:
            plausible += 1

        if i % 40 == 0:
            print(f"  {i}/{len(sample)}")

    acc = correct / len(sample)
    print(f"\naccuracy on this sample : {acc:.4f} ({correct}/{len(sample)})")
    print(f"latency  median          : {np.median(latencies):.0f} ms")
    print(f"latency  p95             : {np.percentile(latencies, 95):.0f} ms")
    print(f"CAM overlays rendered    : {overlay_ok}/5")
    print(f"passed brain-MRI check   : {plausible}/{len(sample)}")

    print("\nconfusion (rows = true, cols = predicted)")
    header = "".join(f"{c[:9]:>11}" for c in C.CLASS_DIRS)
    print(f"{'':<18}{header}")
    for i, name in enumerate(C.CLASS_DIRS):
        print(f"{name:<18}" + "".join(f"{v:>11}" for v in confusion[i]))

    # A non-brain input must be caught by the plausibility guard.
    print("\nnegative control: random noise image")
    from PIL import Image
    import io

    noise = Image.fromarray(
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    buf = io.BytesIO()
    noise.save(buf, format="PNG")
    neg = inf.predict(buf.getvalue(), want_overlay=False)
    print(f"  looks_like_brain_mri : {neg['input_check']['looks_like_brain_mri']}")
    print(f"  failed checks        : "
          f"{[k for k, v in neg['input_check']['checks'].items() if not v]}")
    print(f"  out_of_distribution  : {neg['out_of_distribution']}")

    metrics_path = C.REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        reported = json.loads(metrics_path.read_text(encoding="utf-8"))
        ref = reported["headline"]["accuracy"]
        print(f"\nreported test accuracy   : {ref:.4f} (full {reported['test_set']['n']} images, with TTA)")
        print(f"serving-path accuracy    : {acc:.4f} (sample of {len(sample)}, no TTA)")
        if abs(acc - ref) > 0.12:
            print("  ! large gap — check that preprocessing matches training")
        else:
            print("  serving path agrees with the reported metrics")

    if neg["input_check"]["looks_like_brain_mri"]:
        print("\nFAIL: the plausibility guard passed a pure-noise image.")
        raise SystemExit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
