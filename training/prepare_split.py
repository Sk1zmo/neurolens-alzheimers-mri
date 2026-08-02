"""Build a leak-filtered train / val / test split.

Why this file exists
--------------------
The Kaggle "augmented-alzheimer-mri-dataset" ships two folders:

  OriginalDataset/            6,400 real MRI slices
  AugmentedAlzheimerDataset/ 33,984 Keras-ImageDataGenerator derivatives

The augmented frames are *derived from the originals* but their filenames are
random UUIDs, so there is no bookkeeping that says which original produced
which augmentation. The common recipe (train on Augmented, test on Original —
what the reference notebook does) therefore tests on images the model has
effectively already seen. It reports ~99% and means very little.

The fix implemented here is nearest-source assignment:

  1. Carve a stratified train/val/test split out of OriginalDataset only.
  2. Embed every original and every augmented image with an ImageNet
     EfficientNet-B0 (frozen — this is retrieval, not learning).
  3. Assign each augmented image to its single most similar original.
  4. If that source landed in val or test, drop the augmented image.

What survives joins the training set. Evaluation happens only on real,
never-augmented originals whose descendants were removed from training.

Caveat that no split can fix: this dataset has no subject IDs, and adjacent
slices of the same brain appear multiple times. Slice-level leakage across
subjects remains possible, so headline accuracy still overstates what the
model would do on a genuinely new patient. That warning is carried through to
the model card and the app's About page rather than being quietly dropped.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from dataset import ImagePathDataset
from model import AlzheimerNet

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_class_images(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for cls in C.CLASS_DIRS:
        d = root / cls
        if not d.is_dir():
            raise FileNotFoundError(f"Missing class folder: {d}")
        files = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        out[cls] = files
    return out


def stratified_split(files_by_class: dict[str, list[Path]], rng: np.random.Generator):
    train, val, test = [], [], []
    for cls, files in files_by_class.items():
        label = C.DIR_TO_INDEX[cls]
        idx = rng.permutation(len(files))
        n = len(files)
        n_test = max(1, int(round(n * C.TEST_FRACTION)))
        n_val = max(1, int(round(n * C.VAL_FRACTION)))
        n_test = min(n_test, n - 2)
        n_val = min(n_val, n - n_test - 1)

        for j, i in enumerate(idx):
            rec = {"path": str(files[i]), "label": label, "class": cls,
                   "source": "original"}
            if j < n_test:
                test.append(rec)
            elif j < n_test + n_val:
                val.append(rec)
            else:
                train.append(rec)
    return train, val, test


@torch.no_grad()
def embed_paths(paths: list[Path], device: torch.device,
                batch_size: int = 128, tag: str = "") -> torch.Tensor:
    """L2-normalised ImageNet embeddings, returned on CPU as float16."""
    net = AlzheimerNet(pretrained=True).to(device).eval()
    loader = DataLoader(
        ImagePathDataset(paths), batch_size=batch_size, shuffle=False,
        num_workers=C.NUM_WORKERS, pin_memory=True,
    )
    chunks: list[torch.Tensor] = []
    done = 0
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast:
            emb = net.embed(imgs)
        chunks.append(emb.float().cpu().half())
        done += imgs.size(0)
        if done % (batch_size * 20) == 0:
            print(f"    [{tag}] embedded {done}/{len(paths)}", flush=True)
    del net
    torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def nearest_source(aug_emb: torch.Tensor, orig_emb: torch.Tensor,
                   device: torch.device, chunk: int = 2048):
    """For every augmented row, return (best_original_index, similarity)."""
    orig = orig_emb.to(device).float()
    best_idx = torch.empty(aug_emb.size(0), dtype=torch.long)
    best_sim = torch.empty(aug_emb.size(0), dtype=torch.float32)
    for start in range(0, aug_emb.size(0), chunk):
        block = aug_emb[start:start + chunk].to(device).float()
        sims = block @ orig.T              # both sides are unit-norm
        s, i = sims.max(dim=1)
        best_idx[start:start + block.size(0)] = i.cpu()
        best_sim[start:start + block.size(0)] = s.cpu()
        if start % (chunk * 5) == 0:
            print(f"    matched {start}/{aug_emb.size(0)}", flush=True)
    del orig
    torch.cuda.empty_cache()
    return best_idx.numpy(), best_sim.numpy()


def summarise(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r["class"]] += 1
    return {cls: counts.get(cls, 0) for cls in C.CLASS_DIRS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-augmented", action="store_true",
                    help="Build the split from OriginalDataset only.")
    ap.add_argument("--skip-leak-filter", action="store_true",
                    help="Keep every augmented image (reproduces the naive protocol).")
    args = ap.parse_args()

    C.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rng = np.random.default_rng(C.SEED)
    torch.manual_seed(C.SEED)

    print("\n[1/4] Indexing OriginalDataset")
    originals = list_class_images(C.ORIGINAL_DIR)
    for cls, f in originals.items():
        print(f"    {cls:<20} {len(f)}")

    train, val, test = stratified_split(originals, rng)
    print(f"    -> train {len(train)} | val {len(val)} | test {len(test)}")

    report: dict = {
        "seed": C.SEED,
        "classes": C.CLASS_DIRS,
        "original_counts": {k: len(v) for k, v in originals.items()},
        "original_split": {
            "train": summarise(train),
            "val": summarise(val),
            "test": summarise(test),
        },
    }

    if args.no_augmented:
        aug_kept: list[dict] = []
        report["augmented"] = {"used": False}
    else:
        print("\n[2/4] Indexing AugmentedAlzheimerDataset")
        augmented = list_class_images(C.AUGMENTED_DIR)
        aug_records = [
            {"path": str(p), "label": C.DIR_TO_INDEX[cls], "class": cls,
             "source": "augmented"}
            for cls, files in augmented.items() for p in files
        ]
        print(f"    {len(aug_records)} augmented images")

        if args.skip_leak_filter:
            aug_kept = aug_records
            report["augmented"] = {"used": True, "leak_filter": False,
                                   "kept": len(aug_kept), "dropped": 0}
        else:
            print("\n[3/4] Embedding for nearest-source leak detection")
            orig_records = train + val + test
            orig_paths = [Path(r["path"]) for r in orig_records]
            # held_out[i] is True when original i belongs to val or test
            held_out = np.array(
                [False] * len(train) + [True] * (len(val) + len(test))
            )

            orig_emb = embed_paths(orig_paths, device, tag="original")
            aug_emb = embed_paths([Path(r["path"]) for r in aug_records],
                                  device, tag="augmented")

            print("\n[4/4] Assigning each augmented image to its source")
            src_idx, src_sim = nearest_source(aug_emb, orig_emb, device)
            leaks = held_out[src_idx]

            aug_kept = [r for r, bad in zip(aug_records, leaks) if not bad]
            dropped = int(leaks.sum())
            print(f"    dropped {dropped} / {len(aug_records)} "
                  f"({dropped / len(aug_records) * 100:.1f}%) as descendants "
                  f"of held-out originals")

            report["augmented"] = {
                "used": True,
                "leak_filter": True,
                "total": len(aug_records),
                "kept": len(aug_kept),
                "dropped": dropped,
                "similarity_to_source": {
                    "mean": float(src_sim.mean()),
                    "p05": float(np.percentile(src_sim, 5)),
                    "p50": float(np.percentile(src_sim, 50)),
                    "p95": float(np.percentile(src_sim, 95)),
                },
                "note": (
                    "Each augmented image was assigned to its single most "
                    "similar original in frozen ImageNet EfficientNet-B0 "
                    "embedding space; images whose source fell in val/test "
                    "were removed from training."
                ),
            }

    full_train = train + aug_kept
    report["final"] = {
        "train_total": len(full_train),
        "train_original": len(train),
        "train_augmented": len(aug_kept),
        "val_total": len(val),
        "test_total": len(test),
        "train_per_class": summarise(full_train),
    }

    for name, recs in (("train", full_train), ("val", val), ("test", test)):
        out = C.SPLITS_DIR / f"{name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(recs, f)
        print(f"    wrote {out}  ({len(recs)} records)")

    with open(C.SPLITS_DIR / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSplit summary")
    print(json.dumps(report["final"], indent=2))


if __name__ == "__main__":
    main()
