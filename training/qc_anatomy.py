"""Visual QC for the segmentation pipeline.

Area-based morphometry fails silently: a broken mask still produces plausible
numbers, and the only way to catch it is to look at the masks. This renders one
example per class showing the intracranial boundary, the CSF split, the
ventricle component and the atlas overlay, then prints the measurements beside
them.

    python training/qc_anatomy.py --out ../artifacts/paper/qc_segmentation.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config as C

sys.path.insert(0, str(C.PROJECT_ROOT / "web" / "api"))
import _anatomy as A  # noqa: E402


def pipeline(path: str):
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    tissue_mask = A.brain_mask(gray)
    icv = A.intracranial_mask(tissue_mask)
    pose = A.estimate_pose(icv)
    t_img = A.to_template(gray, pose)
    t_icv = A.to_template(icv.astype(np.float32), pose) > 0.5
    t_tissue = A.to_template(tissue_mask.astype(np.float32), pose) > 0.5
    seg = A.segment_tissue(t_img, t_icv)
    metrics = A.compute_morphometry(t_img, t_icv, seg)
    return t_img, t_icv, t_tissue, seg, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ARTIFACTS_DIR / "paper" / "qc_segmentation.png"))
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    records = json.loads((C.SPLITS_DIR / "test.json").read_text(encoding="utf-8"))
    chosen: dict[str, str] = {}
    for cls in C.CLASS_DIRS:
        pool = sorted([r["path"] for r in records if r["class"] == cls])
        if pool:
            chosen[cls] = pool[min(args.index, len(pool) - 1)]

    cols = 5
    fig, axes = plt.subplots(len(chosen), cols, figsize=(3.0 * cols, 3.1 * len(chosen)),
                             dpi=140)
    if len(chosen) == 1:
        axes = axes[None, :]

    atlas = A.atlas_masks()
    print(f"{'class':<20}{'VBR':>9}{'CSF':>9}{'sulcal':>9}{'paren':>9}{'rim':>9}")

    for row, (cls, path) in enumerate(chosen.items()):
        t_img, t_icv, t_tissue, seg, m = pipeline(path)

        axes[row, 0].imshow(t_img, cmap="gray"); axes[row, 0].set_ylabel(cls, fontsize=9)
        axes[row, 0].set_title("registered" if row == 0 else "")

        b = np.stack([t_img] * 3, axis=-1)
        b[t_icv & ~t_tissue] = [0.25, 0.65, 1.0]
        axes[row, 1].imshow(np.clip(b, 0, 1))
        axes[row, 1].set_title("ICV vs tissue" if row == 0 else "")

        c = np.stack([t_img] * 3, axis=-1)
        c[seg["sulcal_csf"]] = [0.20, 0.80, 0.45]
        c[seg["ventricle"]] = [1.0, 0.35, 0.15]
        axes[row, 2].imshow(np.clip(c, 0, 1))
        axes[row, 2].set_title("ventricle / sulcal CSF" if row == 0 else "")

        d = np.stack([t_img] * 3, axis=-1)
        d[seg["grey"]] = np.clip(d[seg["grey"]] * [1.4, 1.0, 0.7], 0, 1)
        d[seg["white"]] = np.clip(d[seg["white"]] * [0.7, 1.0, 1.4], 0, 1)
        axes[row, 3].imshow(np.clip(d, 0, 1))
        axes[row, 3].set_title("grey / white" if row == 0 else "")

        e = np.stack([t_img] * 3, axis=-1)
        from scipy import ndimage as ndi
        for key, colour in (("temporal_left", [1, 0.4, 0.4]),
                            ("temporal_right", [1, 0.4, 0.4]),
                            ("parietal_left", [0.4, 0.7, 1]),
                            ("parietal_right", [0.4, 0.7, 1]),
                            ("frontal_left", [1, 0.85, 0.3]),
                            ("frontal_right", [1, 0.85, 0.3]),
                            ("occipital_left", [0.6, 1, 0.6]),
                            ("occipital_right", [0.6, 1, 0.6])):
            region_mask = atlas[key] & t_icv
            # Full outline, not just horizontal transitions.
            edge = region_mask & ~ndi.binary_erosion(
                region_mask, structure=np.ones((3, 3)))
            e[edge] = colour
        axes[row, 4].imshow(np.clip(e, 0, 1))
        axes[row, 4].set_title("atlas" if row == 0 else "")

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

        print(f"{cls:<20}{m['ventricle_brain_ratio']:>9.4f}"
              f"{m['csf_fraction']:>9.4f}{m['sulcal_csf_fraction']:>9.4f}"
              f"{m['parenchymal_fraction']:>9.4f}{m['cortical_rim_fraction']:>9.4f}")

    fig.suptitle("Segmentation QC — registered slice, intracranial mask, "
                 "CSF compartments, tissue split, atlas", fontsize=11)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
