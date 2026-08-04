"""Recover subject identity from slice ordering.

Why this is worth doing
-----------------------
The Kaggle repackaging of OASIS-1 dropped the subject identifiers, which is the
single reason a clean split is impossible: adjacent slices of one brain land on
both sides of any random partition, and accuracy on that partition is a
within-subject ceiling rather than a generalisation estimate.

But the class counts are not arbitrary:

    NonDemented       3200 = 100 x 32
    VeryMildDemented  2240 =  70 x 32
    MildDemented       896 =  28 x 32
    ModerateDemented    64 =   2 x 32

100 / 70 / 28 / 2 is exactly the CDR 0 / 0.5 / 1 / 2 breakdown of the OASIS-1
cross-sectional cohort, and 32 is the number of axial slices the authors
exported per subject. So the hypothesis is that files, in index order, are
contiguous blocks of 32 slices per subject.

This script tests that hypothesis rather than assuming it. If consecutive
slices belong to one brain, embedding similarity between neighbours must be
high *within* a block and drop sharply at block boundaries. A boundary signal
that peaks at a period of 32 confirms the layout; anything else refutes it and
the script says so.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from dataset import ImagePathDataset
from model import AlzheimerNet

PAPER = C.ARTIFACTS_DIR / "paper"
OUT = C.SPLITS_DIR / "subjects.json"
SLICES_PER_SUBJECT = 32


def index_of(path: Path) -> int | None:
    """Numeric export index encoded in the filename."""
    stem = path.stem
    m = re.fullmatch(r"[a-zA-Z]+(\d+)", stem)
    if m:
        return int(m.group(1))
    # "26 (19)" is a Windows duplicate-copy artefact, not a subject/slice pair.
    m = re.fullmatch(r"(\d+)(?:\s*\((\d+)\))?", stem)
    if m:
        return int(m.group(2) or m.group(1)) + 100000
    return None


@torch.no_grad()
def embed(paths: list[Path], device: torch.device) -> torch.Tensor:
    net = AlzheimerNet(pretrained=True).to(device).eval()
    loader = DataLoader(ImagePathDataset(paths), batch_size=128, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=True)
    chunks = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            chunks.append(net.embed(imgs).float().cpu())
    del net
    torch.cuda.empty_cache()
    return torch.cat(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=SLICES_PER_SUBJECT)
    ap.add_argument("--max-period", type=int, default=64)
    args = ap.parse_args()

    C.ensure_dirs()
    PAPER.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=140)
    verdict: dict = {"slices_per_subject": args.period, "classes": {}}
    subjects: list[dict] = []
    all_scores: dict[str, np.ndarray] = {}

    for cls in C.CLASS_DIRS:
        folder = C.ORIGINAL_DIR / cls
        files = [p for p in folder.iterdir() if p.suffix.lower() == ".jpg"]
        keyed = [(index_of(p), p) for p in files]
        if any(k is None for k, _ in keyed):
            print(f"! {cls}: some filenames have no index; skipping")
            continue
        keyed.sort(key=lambda t: t[0])
        ordered = [p for _, p in keyed]

        print(f"\n{cls}: {len(ordered)} slices "
              f"({len(ordered) / args.period:.1f} x {args.period})")
        emb = embed(ordered, device)
        emb = torch.nn.functional.normalize(emb, dim=1)

        # Similarity between each consecutive pair, in export order.
        neighbour = (emb[:-1] * emb[1:]).sum(1).numpy()
        all_scores[cls] = neighbour

        # A boundary shows up as a *dip* in neighbour similarity. Score each
        # candidate period by how much lower the similarity is at positions
        # that are multiples of that period than everywhere else.
        periods = np.arange(4, args.max_period + 1)
        contrast = []
        for p in periods:
            pos = np.arange(len(neighbour))
            at_boundary = ((pos + 1) % p) == 0
            if at_boundary.sum() < 3 or (~at_boundary).sum() < 3:
                contrast.append(np.nan)
                continue
            contrast.append(float(neighbour[~at_boundary].mean()
                                  - neighbour[at_boundary].mean()))
        contrast = np.array(contrast)
        best = int(periods[int(np.nanargmax(contrast))])
        peak = float(np.nanmax(contrast))

        pos = np.arange(len(neighbour))
        at32 = ((pos + 1) % args.period) == 0
        within = float(neighbour[~at32].mean())
        across = float(neighbour[at32].mean()) if at32.sum() else float("nan")

        print(f"  best period by boundary contrast : {best}  (contrast {peak:.4f})")
        print(f"  similarity within a {args.period}-block  : {within:.4f}")
        print(f"  similarity across the boundary     : {across:.4f}")

        verdict["classes"][cls] = {
            "n_slices": len(ordered),
            "implied_subjects": len(ordered) / args.period,
            "best_period": best,
            "boundary_contrast": peak,
            "within_block_similarity": within,
            "across_boundary_similarity": across,
        }

        axes[0].plot(periods, contrast, lw=1.4, label=cls.replace("Demented", ""))

        if len(ordered) % args.period == 0:
            for s in range(len(ordered) // args.period):
                block = ordered[s * args.period:(s + 1) * args.period]
                sid = f"{cls}_S{s:03d}"
                for p in block:
                    subjects.append({"path": str(p), "class": cls,
                                     "label": C.DIR_TO_INDEX[cls],
                                     "subject": sid})

    axes[0].axvline(args.period, color="#e34948", ls="--", lw=1)
    axes[0].set_xlabel("Candidate block length (slices)")
    axes[0].set_ylabel("Boundary contrast")
    axes[0].set_title("(a) Which block length looks like a subject?", loc="left")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.4)

    cls = "MildDemented"
    if cls in all_scores:
        s = all_scores[cls][:256]
        axes[1].plot(np.arange(len(s)), s, lw=1.0, color="#2a78d6")
        for b in range(args.period - 1, len(s), args.period):
            axes[1].axvline(b, color="#e34948", ls="--", lw=0.8, alpha=0.8)
        axes[1].set_xlabel("Slice index (export order)")
        axes[1].set_ylabel("Similarity to next slice")
        axes[1].set_title(f"(b) {cls}: dips land on multiples of {args.period}",
                          loc="left")
        axes[1].grid(alpha=0.4)

    fig.suptitle("Recovering subject identity from slice ordering", x=0.02,
                 ha="left", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(PAPER / "fig11_subject_recovery.png", bbox_inches="tight",
                facecolor="white")
    fig.savefig(PAPER / "fig11_subject_recovery.pdf", bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {PAPER / 'fig11_subject_recovery.png'}")

    periods_found = [v["best_period"] for v in verdict["classes"].values()]
    confirmed = all(p == args.period for p in periods_found)
    verdict["hypothesis_confirmed"] = bool(confirmed)
    verdict["n_subjects"] = len({s["subject"] for s in subjects})
    print(f"\nblock length per class: {periods_found}")
    print("HYPOTHESIS CONFIRMED" if confirmed else
          "NOT CONFIRMED — do not use these pseudo-subjects for splitting")

    if confirmed and subjects:
        OUT.write_text(json.dumps(
            {"slices_per_subject": args.period, "verdict": verdict,
             "records": subjects}, indent=2), encoding="utf-8")
        print(f"wrote {OUT}  ({verdict['n_subjects']} subjects, "
              f"{len(subjects)} slices)")
    (PAPER / "subject_recovery.json").write_text(json.dumps(verdict, indent=2),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
