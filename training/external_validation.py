"""Evaluate the shipped model on a cohort it was never trained on.

This is the experiment that decides whether the headline number means
anything. Within-cohort accuracy on this dataset is a ceiling, not a
generalisation estimate — slices from one brain sit on both sides of any split
because the Kaggle repackaging destroyed the subject identifiers (see
recover_subjects.py, which tested whether they were recoverable and found they
are not).

Point this at a second cohort and it reports the drop honestly.

Expected layout — any of:

  a) folders named by class
        <root>/NonDemented/*.png|jpg|dcm|nii(.gz)
        <root>/MildDemented/...
  b) a CSV manifest with columns: path,label     (label = class dir or 0-3)
  c) OASIS-style: --oasis-csv with columns ID,CDR  plus --scans-root

CDR is mapped to this label space with the standard correspondence used to
build the source dataset:

    CDR 0    -> NonDemented
    CDR 0.5  -> VeryMildDemented
    CDR 1    -> MildDemented
    CDR 2+   -> ModerateDemented

Usage
    python training/external_validation.py --root /data/miriad --name MIRIAD
    python training/external_validation.py --manifest ext.csv --name ADNI
    python training/external_validation.py --oasis-csv oasis_cross-sectional.csv \
           --scans-root /data/oasis1 --name OASIS-1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             classification_report, f1_score, roc_auc_score)

import config as C

sys.path.insert(0, str(C.PROJECT_ROOT / "web" / "api"))
import _inference as inf  # noqa: E402

PAPER = C.ARTIFACTS_DIR / "paper"
SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".dcm", ".nii", ".gz"}

CDR_TO_CLASS = {0.0: "NonDemented", 0.5: "VeryMildDemented",
                1.0: "MildDemented", 2.0: "ModerateDemented",
                3.0: "ModerateDemented"}


def from_folders(root: Path) -> list[dict]:
    records = []
    for cls in C.CLASS_DIRS:
        d = root / cls
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                records.append({"path": str(p), "class": cls,
                                "label": C.DIR_TO_INDEX[cls]})
    return records


def from_manifest(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = str(row.get("label", "")).strip()
            if raw in C.DIR_TO_INDEX:
                label, cls = C.DIR_TO_INDEX[raw], raw
            elif raw.isdigit() and int(raw) < C.NUM_CLASSES:
                label, cls = int(raw), C.CLASS_DIRS[int(raw)]
            else:
                continue
            records.append({"path": row["path"], "class": cls, "label": label})
    return records


def from_oasis(csv_path: Path, scans_root: Path) -> list[dict]:
    """OASIS-1 cross-sectional CSV + a directory of per-subject scans.

    Crucially this keeps the subject ID, which is what makes a *subject-level*
    split possible — the thing the Kaggle repackaging threw away.
    """
    records = []
    missing = 0
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = (row.get("ID") or row.get("Subject") or "").strip()
            cdr_raw = (row.get("CDR") or "").strip()
            if not sid or not cdr_raw:
                continue
            try:
                cdr = float(cdr_raw)
            except ValueError:
                continue
            cls = CDR_TO_CLASS.get(cdr)
            if cls is None:
                continue

            hits = sorted(p for p in scans_root.rglob(f"*{sid}*")
                          if p.is_file() and p.suffix.lower() in SUPPORTED)
            if not hits:
                missing += 1
                continue
            for p in hits:
                records.append({"path": str(p), "class": cls,
                                "label": C.DIR_TO_INDEX[cls], "subject": sid,
                                "cdr": cdr})
    if missing:
        print(f"  ! {missing} subjects in the CSV had no matching scan file")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--oasis-csv", type=str, default=None)
    ap.add_argument("--scans-root", type=str, default=None)
    ap.add_argument("--name", type=str, default="external")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.root:
        records = from_folders(Path(args.root))
    elif args.manifest:
        records = from_manifest(Path(args.manifest))
    elif args.oasis_csv:
        if not args.scans_root:
            raise SystemExit("--oasis-csv also needs --scans-root")
        records = from_oasis(Path(args.oasis_csv), Path(args.scans_root))
    else:
        raise SystemExit("give one of --root, --manifest or --oasis-csv")

    if not records:
        raise SystemExit("no records found — check the paths and layout")
    if args.limit:
        records = records[:args.limit]

    counts = defaultdict(int)
    for r in records:
        counts[r["class"]] += 1
    print(f"\n{args.name}: {len(records)} scans")
    for cls in C.CLASS_DIRS:
        print(f"  {cls:<20} {counts[cls]}")

    subjects = {r.get("subject") for r in records if r.get("subject")}
    if subjects:
        print(f"  {len(subjects)} distinct subjects "
              f"({len(records) / len(subjects):.1f} scans each)")

    print("\nscoring through the deployed ONNX path...")
    probs, labels, failed = [], [], 0
    for i, r in enumerate(records):
        try:
            out = inf.predict(Path(r["path"]).read_bytes(),
                              want_overlay=False, want_anatomy=False)
            probs.append(out["probabilities"])
            labels.append(r["label"])
        except Exception as e:  # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"  ! {Path(r['path']).name}: {type(e).__name__}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)}")

    if not probs:
        raise SystemExit("every scan failed to load")
    if failed:
        print(f"  {failed} scans could not be read and were excluded")

    P = np.array(probs)
    y = np.array(labels)
    pred = P.argmax(1)
    present = sorted(set(y.tolist()))

    result = {
        "cohort": args.name,
        "n_scanned": int(len(y)),
        "n_failed": failed,
        "n_subjects": len(subjects) or None,
        "classes_present": [C.CLASS_LABELS[i] for i in present],
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(range(C.NUM_CLASSES))).tolist(),
        "per_class": classification_report(
            y, pred, labels=list(range(C.NUM_CLASSES)),
            target_names=C.CLASS_LABELS, output_dict=True, zero_division=0),
    }
    if len(present) > 1:
        try:
            result["macro_auc_ovr"] = float(roc_auc_score(
                y, P, multi_class="ovr", average="macro",
                labels=list(range(C.NUM_CLASSES))))
        except ValueError:
            result["macro_auc_ovr"] = float("nan")

    internal = C.REPORTS_DIR / "metrics.json"
    if internal.exists():
        ref = json.loads(internal.read_text(encoding="utf-8"))["headline"]
        result["internal_accuracy"] = ref["accuracy"]
        result["generalisation_gap"] = ref["accuracy"] - result["accuracy"]

    PAPER.mkdir(parents=True, exist_ok=True)
    out_path = PAPER / f"external_{args.name.lower().replace(' ', '_')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== {args.name} ===")
    print(f"  accuracy          {result['accuracy']:.4f}")
    print(f"  balanced accuracy {result['balanced_accuracy']:.4f}")
    print(f"  macro F1          {result['macro_f1']:.4f}")
    print(f"  Cohen's kappa     {result['cohen_kappa']:.4f}")
    if "generalisation_gap" in result:
        print(f"\n  internal accuracy {result['internal_accuracy']:.4f}")
        print(f"  GENERALISATION GAP {result['generalisation_gap']:+.4f}")
        print("\n  Report this gap. A large drop is the expected, honest "
              "result for a model trained on one cohort — hiding it is what "
              "makes these papers unreproducible.")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
