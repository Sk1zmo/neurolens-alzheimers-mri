"""Ablation study.

Answers the question a reviewer will ask first: how much of the headline number
comes from the model, and how much from the protocol?

Four variants, all trained with an **identical** shortened schedule and all
scored on the **same** held-out test split. Only the training manifest or the
sampler changes. The reduced schedule is stated rather than hidden: these are
relative comparisons, not the headline result, and running every variant to
full convergence would cost six GPU-hours for the same ordering.

  full             leak-filtered augmented + original, balanced (the shipped
                   configuration)
  no_leak_filter   every augmented image kept, including descendants of
                   val/test originals — i.e. the protocol nearly every
                   published notebook on this dataset uses
  original_only    no augmented data at all
  no_balance       leak-filtered data, but no balanced sampler and no class
                   weights

The `no_leak_filter` row is the point of the whole exercise: the gap between it
and `full` is the accuracy inflation the standard protocol buys you for free.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import config as C

HERE = Path(__file__).resolve().parent
PAPER = C.ARTIFACTS_DIR / "paper"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

VARIANTS = {
    "full": {
        "label": "Leak-filtered + balanced (shipped)",
        "manifest": "full",
        "no_balance": False,
    },
    "no_leak_filter": {
        "label": "No leak filter (standard protocol)",
        "manifest": "all_augmented",
        "no_balance": False,
    },
    "original_only": {
        "label": "Original images only",
        "manifest": "original_only",
        "no_balance": False,
    },
    "no_balance": {
        "label": "No class balancing",
        "manifest": "full",
        "no_balance": True,
    },
}


def build_manifests() -> dict[str, int]:
    """Write train_<variant>.json. val/test are deliberately untouched."""
    base = json.loads((C.SPLITS_DIR / "train.json").read_text(encoding="utf-8"))
    originals = [r for r in base if r.get("source") == "original"]
    kept_aug = [r for r in base if r.get("source") == "augmented"]

    all_aug: list[dict] = []
    for cls in C.CLASS_DIRS:
        d = C.AUGMENTED_DIR / cls
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                all_aug.append({"path": str(p), "label": C.DIR_TO_INDEX[cls],
                                "class": cls, "source": "augmented"})

    manifests = {
        "full": originals + kept_aug,
        "all_augmented": originals + all_aug,
        "original_only": originals,
    }
    sizes = {}
    for name, records in manifests.items():
        out = C.SPLITS_DIR / f"train_{name}.json"
        out.write_text(json.dumps(records), encoding="utf-8")
        sizes[name] = len(records)
        print(f"  train_{name}.json  {len(records):,} records")
    return sizes


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=1)
    ap.add_argument("--epochs-finetune", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    C.ensure_dirs()
    PAPER.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    print("building manifests (val/test are shared and unchanged)")
    sizes = build_manifests()

    results_path = PAPER / "ablation.json"
    results: dict = json.loads(results_path.read_text(encoding="utf-8")) \
        if results_path.exists() else {"variants": {}}
    results["schedule"] = {
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune,
        "batch_size": args.batch_size,
        "note": "Shortened but identical across variants. Relative comparison "
                "only; the headline model uses the full schedule.",
    }
    results["train_sizes"] = sizes

    names = args.only or list(VARIANTS)
    for name in names:
        spec = VARIANTS[name]
        print(f"\n{'=' * 70}\n{name}: {spec['label']}\n{'=' * 70}")
        ckpt = f"ablation_{name}.pt"
        started = time.time()

        cmd = [py, "-u", "train.py",
               "--epochs-head", str(args.epochs_head),
               "--epochs-finetune", str(args.epochs_finetune),
               "--batch-size", str(args.batch_size),
               "--workers", str(args.workers),
               "--split-suffix", f"_{spec['manifest']}",
               "--out", ckpt]
        if spec["no_balance"]:
            cmd.append("--no-balance")
        run(cmd)

        # Evaluate on the shared test split. evaluate.py writes metrics.json,
        # which is the deployed model's report — stash and restore it so an
        # ablation run never overwrites the shipped numbers.
        live = C.REPORTS_DIR / "metrics.json"
        backup = live.read_bytes() if live.exists() else None
        run([py, "-u", "evaluate.py", "--checkpoint",
             str(C.CHECKPOINT_DIR / ckpt)])
        produced = json.loads(live.read_text(encoding="utf-8"))
        if backup is not None:
            live.write_bytes(backup)

        results["variants"][name] = {
            "label": spec["label"],
            "train_size": sizes[spec["manifest"]],
            "balanced": not spec["no_balance"],
            "elapsed_sec": time.time() - started,
            "headline": produced["headline"],
            "per_class": produced["per_class"],
            "confusion_matrix": produced["confusion_matrix"],
            "temperature": produced["temperature"],
        }
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        h = produced["headline"]
        print(f"\n>>> {name}: acc {h['accuracy']:.4f}  macroF1 "
              f"{h['macro_f1']:.4f}  ECE {h['ece_calibrated']:.4f}")

    summarise(results)


def summarise(results: dict) -> None:
    print(f"\n{'=' * 78}\nABLATION SUMMARY (identical shortened schedule, "
          f"shared test split)\n{'=' * 78}")
    print(f"{'variant':<18}{'train n':>10}{'accuracy':>11}{'macro F1':>11}"
          f"{'bal acc':>10}{'ECE':>9}")
    for name, v in results["variants"].items():
        h = v["headline"]
        print(f"{name:<18}{v['train_size']:>10,}{h['accuracy']:>11.4f}"
              f"{h['macro_f1']:>11.4f}{h['balanced_accuracy']:>10.4f}"
              f"{h['ece_calibrated']:>9.4f}")

    full = results["variants"].get("full")
    leaky = results["variants"].get("no_leak_filter")
    if full and leaky:
        gap = (leaky["headline"]["accuracy"] - full["headline"]["accuracy"]) * 100
        results["leak_inflation_points"] = float(gap)
        print(f"\nAccuracy inflation from skipping the leak filter: "
              f"{gap:+.2f} percentage points")
        print("That gap is measured on the SAME clean test set — it is purely "
              "the model having seen augmented copies of the test images.")

    (PAPER / "ablation.json").write_text(json.dumps(results, indent=2),
                                         encoding="utf-8")
    print(f"\nwrote {PAPER / 'ablation.json'}")


if __name__ == "__main__":
    main()
