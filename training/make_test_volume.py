"""Build a synthetic NIfTI volume for exercising the DICOM/NIfTI path.

Real held-out slices are stacked into a volume and padded at both ends with
near-empty frames standing in for the vertex and skull base. A correct slice
selector must reject the padding and land in the middle.

    python training/make_test_volume.py --class ModerateDemented
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klass", default="ModerateDemented")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pad", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = json.loads((C.SPLITS_DIR / "test.json").read_text(encoding="utf-8"))
    paths = sorted(r["path"] for r in recs if r["class"] == args.klass)[:args.n]
    if not paths:
        raise SystemExit(f"no test slices for {args.klass}")

    slices = [np.asarray(Image.open(p).convert("L"), dtype=np.float32)
              for p in paths]
    h = min(s.shape[0] for s in slices)
    w = min(s.shape[1] for s in slices)
    vol = np.stack([s[:h, :w] for s in slices], axis=2)

    blank = np.full((h, w, args.pad), float(vol.mean()) * 0.05, dtype=np.float32)
    vol = np.concatenate([blank, vol, blank], axis=2)

    out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "test_volume.nii"
    # Anisotropic spacing: 3 mm through-plane, as in a clinical axial series.
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine=np.diag([1, 1, 3, 1])),
             out)
    print(out)


if __name__ == "__main__":
    main()
