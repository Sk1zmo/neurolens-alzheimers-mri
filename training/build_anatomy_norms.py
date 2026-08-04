"""Compute morphometry across the dataset and derive reference norms.

Two outputs:

  web/api/model/anatomy_norms.json   per-class mean/SD for each index, shipped
                                     with the function so a new scan can be
                                     z-scored at inference time
  artifacts/paper/morphometry.csv    per-image measurements, the input to the
                                     morphometry figures and statistics

Measurements come only from ORIGINAL images, never augmented ones: the
augmentations apply rotation, shear and zoom, which directly distort area-based
indices and would inflate the reference SD with synthetic variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import config as C

sys.path.insert(0, str(C.PROJECT_ROOT / "web" / "api"))
import _anatomy as A  # noqa: E402

OUT_JSON = C.PROJECT_ROOT / "web" / "api" / "model" / "anatomy_norms.json"
PAPER_DIR = C.ARTIFACTS_DIR / "paper"


def measure(path: str) -> dict[str, float] | None:
    try:
        img = Image.open(path)
    except Exception:
        return None
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    tissue_mask = A.brain_mask(gray)
    if tissue_mask.sum() < 500:
        return None
    icv = A.intracranial_mask(tissue_mask)
    pose = A.estimate_pose(icv)
    t_img = A.to_template(gray, pose)
    t_icv = A.to_template(icv.astype(np.float32), pose) > 0.5
    tissue = A.segment_tissue(t_img, t_icv)
    m = A.compute_morphometry(t_img, t_icv, tissue)
    if any(not np.isfinite(v) for v in m.values()):
        return None
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400,
                    help="Max ORIGINAL images to measure per class.")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    C.ensure_dirs()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # Pool every original image across all splits, tagged with its split so the
    # figures can show that norms are not fit on the test set alone.
    records: list[dict] = []
    for split in ("train", "val", "test"):
        for r in json.loads((C.SPLITS_DIR / f"{split}.json").read_text(encoding="utf-8")):
            if r.get("source") == "original":
                records.append({**r, "split": split})

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for cls in C.CLASS_DIRS:
        pool = by_class.get(cls, [])
        if not pool:
            print(f"! no original images for {cls}")
            continue
        idx = rng.permutation(len(pool))[:args.per_class]
        print(f"{cls:<20} measuring {len(idx)} of {len(pool)}")
        ok = 0
        for j, i in enumerate(idx, 1):
            rec = pool[int(i)]
            m = measure(rec["path"])
            if m is None:
                continue
            rows.append({"class": cls, "label": rec["label"],
                         "split": rec["split"],
                         "file": Path(rec["path"]).name, **m})
            ok += 1
            if j % 100 == 0:
                print(f"    {j}/{len(idx)}")
        print(f"    -> {ok} usable")

    if not rows:
        raise SystemExit("no measurements produced")

    metrics = list(A.METRIC_INFO.keys())

    # ---- per-image CSV for the paper ------------------------------------
    csv_path = PAPER_DIR / "morphometry.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "label", "split", "file"] + metrics)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")

    # ---- reference norms -------------------------------------------------
    norms: dict = {
        "n_total": len(rows),
        "per_class_n": {},
        "metrics": metrics,
        "template_size": A.TEMPLATE_SIZE,
        "by_class": {},
        "note": (
            "Reference distributions measured on ORIGINAL (never augmented) "
            "slices. Augmentations apply rotation, shear and zoom, which "
            "distort area-based indices and would inflate these SDs with "
            "synthetic variance."
        ),
    }

    for cls in C.CLASS_DIRS:
        subset = [r for r in rows if r["class"] == cls]
        if not subset:
            continue
        norms["per_class_n"][cls] = len(subset)
        stats: dict[str, dict[str, float]] = {}
        for key in metrics:
            vals = np.array([r[key] for r in subset], dtype=np.float64)
            stats[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
                "n": int(len(vals)),
            }
        norms["by_class"][cls] = stats

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(norms, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}")

    # ---- console summary --------------------------------------------------
    print("\nMean by stage (the separation the classifier is exploiting):")
    head = f"{'metric':<26}" + "".join(f"{c[:11]:>13}" for c in C.CLASS_DIRS)
    print(head)
    for key in metrics:
        line = f"{key:<26}"
        for cls in C.CLASS_DIRS:
            s = norms["by_class"].get(cls, {}).get(key)
            line += f"{s['mean']:>13.4f}" if s else f"{'-':>13}"
        print(line)


if __name__ == "__main__":
    main()
