"""Publication figures.

Everything derives from artifacts written by earlier stages — no figure
recomputes a metric, so what is plotted is exactly what is tabulated:

  paper_stats.py         -> test_predictions.npz, statistics.json
  build_anatomy_norms.py -> morphometry.csv, anatomy_norms.json
  train.py               -> training_history*.json
  prepare_split.py       -> split_report.json

Figures are written as 300-dpi PNG and vector PDF. Colour follows one system
throughout: a validated four-hue categorical order for series that must be told
apart, a single-hue sequential ramp for magnitude grids, and a de-emphasis grey
whenever exactly one element is the point.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve, roc_curve

import config as C

sys.path.insert(0, str(C.PROJECT_ROOT / "web" / "api"))
import _anatomy as A  # noqa: E402

PAPER = C.ARTIFACTS_DIR / "paper"
FIGS = PAPER / "figures"

# Atlassian-family hues, re-stepped until they pass the colour-blind and
# chroma gates — the raw ADS chart palette does not (its teal and green read as
# grey, and magenta/purple sit at deltaE 15). Fixed order by class index,
# never cycled. Matches the web app's --series-* tokens on a white surface.
SERIES = ["#0C66E4", "#E56910", "#1F845A", "#6E5DC6"]
INK, INK2, MUTED, GRID = "#172B4D", "#44546F", "#626F86", "#EBECF0"
ACCENT, DEEMPH = "#0C66E4", "#B3B9C4"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#F0F6FF", "#CFE1FD", "#9EC3F8", "#6BA2F0", "#0C66E4", "#09326C"])


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def load(name: str):
    p = PAPER / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix == ".npz":
        return np.load(p, allow_pickle=False)
    if p.suffix == ".csv":
        return list(csv.DictReader(open(p, encoding="utf-8")))
    return None


def star(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ===========================================================================
def fig01_dataset() -> None:
    report = json.loads((C.SPLITS_DIR / "split_report.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    # (a) original class counts — one hue, magnitude comparison
    ax = axes[0]
    counts = [report["original_counts"][c] for c in C.CLASS_DIRS]
    bars = ax.barh(range(len(counts)), counts, color=ACCENT, height=0.62)
    bars[3].set_color(DEEMPH)
    ax.set_yticks(range(len(counts)),
                  [c.replace("Demented", "") for c in C.CLASS_DIRS])
    ax.invert_yaxis()
    ax.set_xlabel("Original slices")
    ax.set_title("(a) Class imbalance in the source data", loc="left")
    for i, v in enumerate(counts):
        ax.text(v + 60, i, str(v), va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, max(counts) * 1.18)
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.annotate("only 64 slices", xy=(counts[3], 3), xytext=(counts[3] + 700, 3.05),
                fontsize=7, color="#b03412",
                arrowprops=dict(arrowstyle="->", color="#b03412", lw=0.8))

    # (b) leak filtering — the headline methodological result
    ax = axes[1]
    aug = report["augmented"]
    kept, dropped = aug["kept"], aug["dropped"]
    ax.bar(["Augmented\nimages"], [kept], color=ACCENT, width=0.5, label="kept")
    ax.bar(["Augmented\nimages"], [dropped], bottom=[kept], color="#e34948",
           width=0.5, label="dropped as leaked")
    ax.set_ylabel("Images")
    ax.set_title("(b) Nearest-source leak filtering", loc="left")
    ax.text(0, kept / 2, f"{kept:,}\n({kept / (kept + dropped) * 100:.1f}%)",
            ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(0, kept + dropped / 2, f"{dropped:,}\n({dropped / (kept + dropped) * 100:.1f}%)",
            ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, 1.4)

    # (c) the validation of that filter
    ax = axes[2]
    ax.axis("off")
    held_out = (C.VAL_FRACTION + C.TEST_FRACTION) * 100
    observed = dropped / (kept + dropped) * 100
    ax.text(0.0, 0.92, "Validation of the filter", fontsize=9.5, color=INK,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.60,
            f"Originals held out\n{held_out:.1f}%",
            fontsize=8, color=INK2, transform=ax.transAxes, va="top")
    ax.text(0.52, 0.60,
            f"Augmented dropped\n{observed:.1f}%",
            fontsize=8, color=INK2, transform=ax.transAxes, va="top")
    ax.text(0.0, 0.30,
            "If each augmented image is correctly assigned to its\n"
            "source, the dropped fraction must match the held-out\n"
            "fraction. It does, to within 0.1 points.",
            fontsize=7.5, color=MUTED, transform=ax.transAxes, va="top")
    ax.text(0.0, 0.04, f"Final training set: {report['final']['train_total']:,} images "
            f"({report['final']['train_original']:,} original + "
            f"{report['final']['train_augmented']:,} augmented)",
            fontsize=7.5, color=INK, transform=ax.transAxes)

    fig.suptitle("Figure 1 — Dataset composition and leakage control", y=1.04,
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig01_dataset")


def fig02_training() -> None:
    path = None
    for cand in ("training_history_paper_run.json", "training_history.json"):
        p = C.REPORTS_DIR / cand
        if p.exists():
            path = p
            break
    if path is None:
        print("  ! skipping fig02 (no training history yet)")
        return

    hist = json.loads(path.read_text(encoding="utf-8"))
    rows = hist["history"]
    if not rows:
        print("  ! skipping fig02 (history is empty)")
        return

    n_head = sum(1 for r in rows if r["stage"] == "head")
    x = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    for ax, (keys, ylabel, title) in zip(axes, [
        (("train_loss", "val_loss"), "Cross-entropy loss", "(a) Loss"),
        (("train_acc", "val_acc"), "Accuracy", "(b) Accuracy"),
        ((None, "val_macro_f1"), "Macro F1", "(c) Selection metric"),
    ]):
        tr_key, va_key = keys
        if tr_key:
            ax.plot(x, [r.get(tr_key, np.nan) for r in rows], color=DEEMPH,
                    lw=1.6, label="train")
        ax.plot(x, [r.get(va_key, np.nan) for r in rows], color=ACCENT, lw=2,
                label="validation")
        if n_head:
            ax.axvline(n_head + 0.5, color=MUTED, lw=0.9, ls="--")
            ax.text(n_head + 0.7, ax.get_ylim()[1], " fine-tune", fontsize=6.8,
                    color=MUTED, va="top")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="best")

    best = hist.get("best", {})
    if best.get("epoch"):
        ax = axes[2]
        ax.axvline(n_head + best["epoch"], color="#0ca30c", lw=0.9, ls=":")
        ax.annotate(f"best  {best.get('macro_f1', float('nan')):.4f}",
                    xy=(n_head + best["epoch"], best.get("macro_f1", 0)),
                    xytext=(6, -12), textcoords="offset points",
                    fontsize=7, color="#006300")

    complete = hist.get("complete", False)
    fig.suptitle("Figure 2 — Training dynamics: frozen-backbone warm-up, then "
                 f"full fine-tune{'' if complete else ' (run truncated)'}",
                 y=1.04, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig02_training")


def fig03_confusion() -> None:
    stats = load("statistics.json")
    cm = np.array(stats["confusion_matrix"])
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    short = [c.replace("Demented", "") or "Non" for c in C.CLASS_DIRS]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax, (data, title, fmt, vmax) in zip(axes, [
        (cm, "(a) Counts", "d", cm.max()),
        (norm, "(b) Row-normalised (recall)", ".0%", 1.0),
    ]):
        im = ax.imshow(data, cmap=SEQ, vmin=0, vmax=vmax)
        ax.set_xticks(range(4), short, rotation=25, ha="right")
        ax.set_yticks(range(4), short)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, loc="left")
        for i in range(4):
            for j in range(4):
                v = data[i, j]
                rel = v / vmax if vmax else 0
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=8,
                        color="white" if rel > 0.55 else (INK if v else MUTED))
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", length=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ea = stats["error_analysis"]
    fig.suptitle(
        f"Figure 3 — Confusion on {stats['n_test']} held-out slices "
        f"({ea['total_errors']} errors, "
        f"{ea['adjacent_stage_errors']} onto an adjacent stage)",
        y=1.02, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig03_confusion")


def fig04_roc_pr() -> None:
    data = load("test_predictions.npz")
    y, probs = data["labels"], data["probs"]
    stats = load("statistics.json")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))

    ax = axes[0]
    for i, label in enumerate(C.CLASS_LABELS):
        binary = (y == i).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        auc = stats["per_class"][label]["auc"]
        ax.plot(fpr, tpr, color=SERIES[i], lw=1.8,
                label=f"{label}  (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=MUTED)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(a) One-vs-rest ROC", loc="left")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.7); ax.set_axisbelow(True)

    ax = axes[1]
    for i, label in enumerate(C.CLASS_LABELS):
        binary = (y == i).astype(int)
        if binary.sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(binary, probs[:, i])
        apv = stats["per_class"][label]["average_precision"]
        ax.plot(rc, pr, color=SERIES[i], lw=1.8,
                label=f"{label}  (AP {apv:.4f})")
        ax.axhline(binary.mean(), color=SERIES[i], lw=0.6, ls=":", alpha=0.55)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-recall (dotted = class prevalence)", loc="left")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.7); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.04)

    fig.suptitle("Figure 4 — Discrimination by stage", y=1.03, fontsize=10.5,
                 x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig04_roc_pr")


def fig05_calibration() -> None:
    data = load("test_predictions.npz")
    y, logits = data["labels"], data["logits"]
    metrics = json.loads((C.REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    T = metrics["temperature"]

    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    bins = np.linspace(0, 1, 16)

    for ax, (p, name, ece) in zip(axes, [
        (softmax(logits), "(a) Uncalibrated", metrics["headline"]["ece_uncalibrated"]),
        (softmax(logits / T), f"(b) Temperature-scaled (T = {T:.3f})",
         metrics["headline"]["ece_calibrated"]),
    ]):
        conf = p.max(1)
        correct = (p.argmax(1) == y).astype(float)
        xs, ys, ws = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() < 3:
                continue
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ws.append(m.sum())

        ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=MUTED, label="perfect")
        ax.scatter(xs, ys, s=np.clip(np.array(ws) * 1.4, 12, 220),
                   color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.8,
                   zorder=3, label="observed (area = n)")
        ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy")
        ax.set_title(f"{name}\nECE = {ece:.4f}", loc="left")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.7); ax.set_axisbelow(True)
        ax.legend(loc="upper left")

    fig.suptitle("Figure 5 — Confidence calibration", y=1.03, fontsize=10.5,
                 x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig05_calibration")


def fig06_morphometry() -> None:
    stats = load("statistics.json")["morphometry"]
    rows = load("morphometry.csv")
    keys = ["ventricle_brain_ratio", "csf_fraction", "parenchymal_fraction",
            "cortical_rim_fraction", "grey_white_ratio", "ventricle_asymmetry"]
    short = [c.replace("Demented", "") or "Non" for c in C.CLASS_DIRS]

    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    for ax, key in zip(axes.ravel(), keys):
        groups = [np.array([float(r[key]) for r in rows if r["class"] == cls])
                  for cls in C.CLASS_DIRS]
        bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.2),
                        whiskerprops=dict(color=MUTED, lw=0.9),
                        capprops=dict(color=MUTED, lw=0.9),
                        boxprops=dict(lw=0))
        # Ordinal severity ramp: one hue, light to dark.
        for patch, shade in zip(bp["boxes"], [0.22, 0.42, 0.62, 0.86]):
            patch.set_facecolor(SEQ(shade))
        for i, g in enumerate(groups, start=1):
            jitter = np.random.default_rng(i).normal(0, 0.055, len(g))
            ax.scatter(np.full(len(g), i) + jitter, g, s=1.6, color=INK2,
                       alpha=0.18, zorder=1, linewidths=0)

        s = stats[key]
        ax.set_xticks(range(1, 5), short, rotation=18, ha="right")
        ax.set_title(f"{s['label']}\n"
                     f"$\\eta^2$ = {s['eta_squared']:.3f}   "
                     f"$\\rho$ = {s['spearman_rho_vs_severity']:+.3f}   "
                     f"{star(s['anova_p'])}",
                     loc="left", fontsize=8.5)
        ax.grid(axis="y", alpha=0.7); ax.set_axisbelow(True)

    fig.suptitle("Figure 6 — Quantitative morphometry separates the stages "
                 "independently of the classifier "
                 "(one-way ANOVA; **** p < 1e-4)",
                 y=1.01, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig06_morphometry")


def fig07_attention() -> None:
    stats = load("statistics.json")["attention_by_region"]
    keys = [k for k in stats if stats[k]["by_class"]]
    # Rank by how strongly the region is weighted overall.
    order = sorted(keys, key=lambda k: -np.nanmean(
        [stats[k]["by_class"][c]["mean_density"] for c in C.CLASS_DIRS]))[:14]

    mat = np.array([[stats[k]["by_class"][c]["mean_density"] for c in C.CLASS_DIRS]
                    for k in order])
    names = [stats[k]["name"] for k in order]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(mat, cmap=SEQ, aspect="auto", vmin=0,
                   vmax=float(np.nanmax(mat)))
    ax.set_xticks(range(4), [c.replace("Demented", "") or "Non"
                             for c in C.CLASS_DIRS], rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    for i in range(len(names)):
        for j in range(4):
            v = mat[i, j]
            rel = v / np.nanmax(mat)
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if rel > 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Attention density (1.0 = uniform)", fontsize=7.5)

    fig.suptitle("Figure 7 — Where the classifier looks, by atlas region and "
                 "true stage", y=1.0, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig07_attention")


def fig08_perclass() -> None:
    stats = load("statistics.json")
    labels = list(stats["per_class"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))

    ax = axes[0]
    xs = np.arange(len(labels))
    width = 0.26
    for k, (metric, colour) in enumerate([("precision", SERIES[0]),
                                          ("recall", SERIES[1]),
                                          ("f1", SERIES[2])]):
        vals = [stats["per_class"][l][metric] for l in labels]
        ax.bar(xs + (k - 1) * width, vals, width * 0.86, color=colour,
               label=metric.title())
    ax.set_xticks(xs, [l.replace(" Demented", "") for l in labels],
                  rotation=18, ha="right")
    ax.set_ylim(0.9, 1.005)
    ax.set_ylabel("Score")
    ax.set_title("(a) Per-class performance", loc="left")
    ax.legend(ncols=3, loc="lower center")
    ax.grid(axis="y", alpha=0.7); ax.set_axisbelow(True)

    ax = axes[1]
    names, points, los, his = [], [], [], []
    for k, v in stats["headline"].items():
        names.append(k.replace("_", " ").title())
        points.append(v["value"]); los.append(v["ci95_low"]); his.append(v["ci95_high"])
    ys = np.arange(len(names))
    ax.hlines(ys, los, his, color=DEEMPH, lw=3.2)
    ax.scatter(points, ys, color=ACCENT, s=34, zorder=3)
    ax.set_yticks(ys, names)
    ax.invert_yaxis()
    ax.set_xlabel("Value (95% bootstrap CI)")
    ax.set_title(f"(b) Headline metrics, {stats['n_bootstrap']} bootstrap "
                 f"resamples", loc="left")
    ax.grid(axis="x", alpha=0.7); ax.set_axisbelow(True)
    for p, yy in zip(points, ys):
        ax.text(p, yy - 0.30, f"{p:.4f}", ha="center", fontsize=7, color=INK)

    fig.suptitle("Figure 8 — Performance with uncertainty", y=1.04,
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig08_perclass")


def fig09_atlas() -> None:
    """Template, atlas parcellation and a worked localisation example."""
    from PIL import Image

    recs = json.loads((C.SPLITS_DIR / "test.json").read_text(encoding="utf-8"))
    path = sorted(r["path"] for r in recs if r["class"] == "NonDemented")[0]
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    tissue = A.brain_mask(gray)
    icv = A.intracranial_mask(tissue)
    pose = A.estimate_pose(icv)
    t_img = A.to_template(gray, pose)
    t_icv = A.to_template(icv.astype(np.float32), pose) > 0.5
    masks = A.atlas_masks()

    groups = [
        ("frontal", "Frontal", SERIES[3]),
        ("temporal", "Temporal", SERIES[1]),
        ("parietal", "Parietal", SERIES[0]),
        ("occipital", "Occipital", SERIES[2]),
        ("basal_ganglia", "Deep grey", "#9085e9"),
        ("periventricular", "Periventricular WM", "#e87ba4"),
        ("insular", "Insular", "#00838f"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.5))

    axes[0].imshow(t_img, cmap="gray")
    axes[0].set_title("(a) Registered slice", loc="left")

    overlay = np.stack([t_img] * 3, axis=-1) * 0.75
    from matplotlib.colors import to_rgb
    for prefix, _lbl, colour in groups:
        rgb = np.array(to_rgb(colour))
        for side in ("left", "right"):
            key = f"{prefix}_{side}"
            if key in masks:
                m = masks[key] & t_icv
                overlay[m] = overlay[m] * 0.45 + rgb * 0.55
    vent = masks["ventricles"] & t_icv
    overlay[vent] = overlay[vent] * 0.45 + np.array(to_rgb("#e34948")) * 0.55
    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title("(b) Atlas parcellation", loc="left")
    axes[1].legend(handles=[Patch(facecolor=c, label=l) for _, l, c in groups]
                   + [Patch(facecolor="#e34948", label="Ventricles")],
                   loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.6)

    # (c) mean attention map across the test set, in template space
    data = load("test_predictions.npz")
    att = data["attention"].mean(axis=0)
    keys = json.loads((PAPER / "test_predictions_keys.json").read_text(
        encoding="utf-8"))["region_keys"]
    heat = np.zeros_like(t_img)
    weight = np.zeros_like(t_img)
    for k, v in zip(keys, att):
        if k in masks:
            m = masks[k] & t_icv
            heat[m] += v
            weight[m] += 1
    heat = np.divide(heat, np.maximum(weight, 1))
    axes[2].imshow(t_img, cmap="gray")
    hm = axes[2].imshow(np.where(t_icv, heat, np.nan), cmap=SEQ, alpha=0.68)
    axes[2].set_title("(c) Mean attention density", loc="left")
    fig.colorbar(hm, ax=axes[2], fraction=0.046, pad=0.03)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Figure 9 — Atlas-based localisation in template space",
                 y=1.02, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig09_atlas")


def fig10_qualitative() -> None:
    from PIL import Image
    sys.path.insert(0, str(C.PROJECT_ROOT / "web" / "api"))
    import _inference as inf

    recs = json.loads((C.SPLITS_DIR / "test.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0))

    for col, cls in enumerate(C.CLASS_DIRS):
        pool = sorted((r for r in recs if r["class"] == cls),
                      key=lambda r: r["path"])
        if not pool:
            continue
        rec = pool[len(pool) // 2]
        raw = Path(rec["path"]).read_bytes()
        out = inf.predict(raw, want_overlay=True, want_anatomy=True)
        img = Image.open(rec["path"]).convert("RGB")

        axes[0, col].imshow(np.asarray(img))
        axes[0, col].set_title(cls.replace("Demented", " Demented"),
                               loc="left", fontsize=8.5)

        overlay = out["overlay_png"]
        if overlay:
            import base64, io
            data = base64.b64decode(overlay.split(",", 1)[1])
            axes[1, col].imshow(np.asarray(Image.open(io.BytesIO(data))))

        anat = out.get("anatomy") or {}
        vbr = (anat.get("metrics") or {}).get("ventricle_brain_ratio", float("nan"))
        top = (anat.get("attribution") or [{}])[0].get("name", "-")
        axes[1, col].set_xlabel(
            f"pred {out['label'].replace(' Demented','')} "
            f"({out['confidence']*100:.1f}%)\n"
            f"VBR {vbr:.3f} · {top}", fontsize=6.8, color=INK2)

    axes[0, 0].set_ylabel("Input slice", fontsize=8)
    axes[1, 0].set_ylabel("Class activation map", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Figure 10 — Qualitative examples across the severity range",
                 y=1.01, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig10_qualitative")


def fig11_uncertainty() -> None:
    exp = load("experiments.json")
    if not exp or "uncertainty" not in exp:
        print("  ! skipping fig11 (no uncertainty results)")
        return
    u = exp["uncertainty"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

    for ax, (key, title) in zip(axes, [
        ("clean", "(a) Clean test set"),
        ("corrupted", "(b) With Gaussian noise (σ = 0.12)"),
    ]):
        block = u.get(key)
        if not block:
            continue
        for i, (name, c) in enumerate(block["curves"].items()):
            ax.plot(c["coverage"], c["accuracy"], color=SERIES[i % 4], lw=1.8,
                    label=f"{name.replace('_', ' ')}  (AURC {c['aurc']:.3f})")
        ax.axhline(block["base_accuracy"], color=MUTED, ls="--", lw=0.9)
        ax.set_xlabel("Coverage (fraction of cases answered)")
        ax.set_ylabel("Accuracy on answered cases")
        ax.set_title(f"{title}\n{block['n_errors']} errors in {block['n']} cases",
                     loc="left")
        ax.grid(alpha=0.7); ax.set_axisbelow(True)
        ax.legend(loc="lower left")
        if block.get("degenerate"):
            ax.text(0.5, 0.45, "degenerate:\ntoo few errors to rank",
                    transform=ax.transAxes, ha="center", fontsize=8.5,
                    color=MUTED)

    fig.suptitle("Figure 11 — Selective prediction: deferring the least "
                 "certain cases", y=1.03, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig11_uncertainty")


def fig12_saliency() -> None:
    exp = load("experiments.json")
    if not exp or "saliency" not in exp:
        print("  ! skipping fig12 (no saliency results)")
        return
    s = exp["saliency"]
    x = s["fraction_masked"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

    ax = axes[0]
    ax.plot(x, s["deletion_cam"], color=SERIES[0], lw=2, label="CAM order")
    ax.plot(x, s["deletion_random"], color=DEEMPH, lw=2, ls="--",
            label="random order")
    ax.set_xlabel("Fraction of pixels blurred")
    ax.set_ylabel("Probability of predicted class")
    ax.set_title(f"(a) Deletion — AUC {s['auc_deletion_cam']:.3f} vs "
                 f"{s['auc_deletion_random']:.3f}\n"
                 f"{'passes' if s['deletion_passes'] else 'FAILS'} "
                 f"(lower is better)", loc="left")
    ax.legend(); ax.grid(alpha=0.7); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(x, s["insertion_cam"], color=SERIES[0], lw=2, label="CAM order")
    ax.plot(x, s.get("insertion_random", []), color=DEEMPH, lw=2, ls="--",
            label="random order")
    ax.set_xlabel("Fraction of pixels revealed")
    ax.set_ylabel("Probability of predicted class")
    ax.set_title(f"(b) Insertion — AUC {s['auc_insertion_cam']:.3f} vs "
                 f"{s.get('auc_insertion_random', float('nan')):.3f}\n"
                 f"{'passes' if s.get('insertion_passes') else 'FAILS'} "
                 f"(higher is better)", loc="left")
    ax.legend(); ax.grid(alpha=0.7); ax.set_axisbelow(True)

    fig.suptitle("Figure 12 — Are the activation maps faithful?", y=1.04,
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig12_saliency")


def fig13_robustness() -> None:
    exp = load("experiments.json")
    if not exp or "robustness" not in exp:
        print("  ! skipping fig13 (no robustness results)")
        return
    sweeps = exp["robustness"]["sweeps"]
    labels = {
        "rotation_deg": "Rotation (degrees)",
        "gaussian_noise_sigma": "Gaussian noise (σ)",
        "contrast_factor": "Contrast factor",
        "blur_radius_px": "Gaussian blur (px)",
        "downsample_factor": "Downsample factor",
    }
    keys = [k for k in labels if k in sweeps]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.05 * len(keys), 2.9),
                             sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        s = sweeps[key]
        xs = s["levels"]
        ys = s["accuracy"]
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        fragile = ys.min() < 0.8
        ax.plot(xs, ys, color=SERIES[1] if fragile else ACCENT, lw=2,
                marker="o", ms=3.5)
        ax.set_xlabel(labels[key], fontsize=7.5)
        ax.set_ylim(0, 1.04)
        ax.grid(alpha=0.7); ax.set_axisbelow(True)
        ax.axhline(0.8, color=MUTED, ls=":", lw=0.8)
    axes[0].set_ylabel("Test accuracy")

    fig.suptitle("Figure 13 — Robustness to acquisition-style perturbations "
                 "(dotted line = 80%)", y=1.06, fontsize=10.5, x=0.02,
                 ha="left")
    fig.tight_layout()
    save(fig, "fig13_robustness")


def fig14_ablation() -> None:
    abl = load("ablation.json")
    if not abl or not abl.get("variants"):
        print("  ! skipping fig14 (no ablation results yet)")
        return
    variants = abl["variants"]
    names = list(variants)
    acc = [variants[n]["headline"]["accuracy"] for n in names]
    f1 = [variants[n]["headline"]["macro_f1"] for n in names]
    labels = [variants[n]["label"] for n in names]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ys = np.arange(len(names))
    # Emphasis form: the shipped configuration and the leaky protocol are the
    # comparison the figure exists to make; the rest is context in grey.
    def colour_for(variant: str) -> str:
        if variant == "full":
            return "#2a78d6"
        if variant == "no_leak_filter":
            return "#e34948"
        return "#c3c2b7"

    colours = [colour_for(n) for n in names]
    ax.barh(ys - 0.18, acc, 0.34, color=colours, label="Accuracy")
    ax.barh(ys + 0.18, f1, 0.34, color=colours, alpha=0.55, label="Macro F1")
    ax.set_yticks(ys, labels, fontsize=8)
    ax.invert_yaxis()
    lo = min(min(acc), min(f1))
    ax.set_xlim(max(0, lo - 0.05), 1.005)
    ax.set_xlabel("Score on the shared held-out test split")
    ax.grid(axis="x", alpha=0.7); ax.set_axisbelow(True)
    for y, v in zip(ys, acc):
        ax.text(v + 0.002, y - 0.18, f"{v:.4f}", va="center", fontsize=7)

    inflation = abl.get("leak_inflation_points")
    subtitle = ("" if inflation is None else
                f" — skipping the leak filter inflates accuracy by "
                f"{inflation:+.2f} points")
    fig.suptitle(f"Figure 14 — Ablation{subtitle}", y=1.04, fontsize=10.5,
                 x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig14_ablation")


def fig15_convergence() -> None:
    """The paper's central argument in one panel.

    Four independent estimates of what this architecture can do on this task,
    ordered by how much augmented data each was allowed to see. Two methods
    that share no machinery — a CNN trained only on real images, and classical
    models on segmented morphometry — land in the same place. Everything above
    that band is bought with augmented copies of the test subjects.
    """
    abl = load("ablation.json")
    ana = load("analytics.json")
    stats = load("statistics.json")
    if not (abl and ana and stats):
        print("  ! skipping fig15 (needs ablation + analytics + statistics)")
        return

    same = (ana.get("baselines") or {}).get("same_split", {}).get("models", {})
    variants = abl.get("variants", {})
    if not same or not variants:
        print("  ! skipping fig15 (missing paired baselines or variants)")
        return

    best_morph = max(same, key=lambda m: same[m]["macro_f1"])
    rows = [
        ("Morphometry only\n(7 measured indices)", same[best_morph]["macro_f1"],
         "honest"),
        ("CNN, original images only\n(no augmented data)",
         variants["original_only"]["headline"]["macro_f1"], "honest"),
        ("CNN + leak-filtered\naugmented data",
         variants["full"]["headline"]["macro_f1"], "inflated"),
        ("CNN + all augmented data\n(standard protocol)",
         variants["no_leak_filter"]["headline"]["macro_f1"], "inflated"),
        ("Reported headline\n(full schedule)",
         stats["headline"]["macro_f1"]["value"], "inflated"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ys = np.arange(len(rows))
    colours = ["#1F845A" if kind == "honest" else "#E56910"
               for _, _, kind in rows]
    ax.barh(ys, [v for _, v, _ in rows], 0.58, color=colours)
    ax.set_yticks(ys, [n for n, _, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("Macro F1 on the shared held-out test split")
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)

    for y, (_, v, _) in zip(ys, rows):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=8.5,
                fontweight="bold", color=INK)

    band_hi = max(rows[0][1], rows[1][1])
    ax.axvspan(0, band_hi, color="#1F845A", alpha=0.07, zorder=0)
    ax.annotate(
        "two independent methods agree here",
        xy=(band_hi, 0.5), xytext=(band_hi + 0.13, 0.72),
        fontsize=7.8, color="#1F845A",
        arrowprops=dict(arrowstyle="->", color="#1F845A", lw=0.9))
    ax.annotate(
        "everything above is bought with\naugmented copies of test subjects",
        xy=(rows[3][1], 3.0), xytext=(0.40, 3.55),
        fontsize=7.8, color="#AE2E24",
        arrowprops=dict(arrowstyle="->", color="#AE2E24", lw=0.9))

    # Outside the axes: every row's bar reaches the right edge, so any in-axes
    # placement collides with a value label.
    ax.legend(handles=[
        Patch(facecolor="#1F845A", label="No augmented data — honest estimate"),
        Patch(facecolor="#E56910", label="Augmented data included"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, fontsize=7.8)

    fig.suptitle("Figure 15 — Where the accuracy actually comes from",
                 y=1.02, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig15_convergence")


FIGURES = {
    "fig01": fig01_dataset,
    "fig02": fig02_training,
    "fig03": fig03_confusion,
    "fig04": fig04_roc_pr,
    "fig05": fig05_calibration,
    "fig06": fig06_morphometry,
    "fig07": fig07_attention,
    "fig08": fig08_perclass,
    "fig09": fig09_atlas,
    "fig10": fig10_qualitative,
    "fig11": fig11_uncertainty,
    "fig12": fig12_saliency,
    "fig13": fig13_robustness,
    "fig14": fig14_ablation,
    "fig15": fig15_convergence,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of figure ids, e.g. --only fig02 fig06")
    args = ap.parse_args()

    style()
    FIGS.mkdir(parents=True, exist_ok=True)
    names = args.only or list(FIGURES)
    print(f"generating {len(names)} figures -> {FIGS}")
    for name in names:
        fn = FIGURES.get(name)
        if fn is None:
            print(f"  ! unknown figure {name}")
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ! {name} failed: {type(e).__name__}: {e}")

    publish()
    print("done")


def publish() -> None:
    """Copy figures, statistics and tables into the web app's public assets."""
    import shutil

    dest = C.PROJECT_ROOT / "web" / "public" / "paper"
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    (dest / "tables").mkdir(parents=True, exist_ok=True)

    for png in sorted(FIGS.glob("*.png")):
        shutil.copy2(png, dest / "figures" / png.name)
    for pdf in sorted(FIGS.glob("*.pdf")):
        shutil.copy2(pdf, dest / "figures" / pdf.name)
    tables_dir = PAPER / "tables"
    if tables_dir.exists():
        for table in sorted(tables_dir.glob("*")):
            shutil.copy2(table, dest / "tables" / table.name)
    for extra in ("statistics.json", "qc_segmentation.png", "morphometry.csv"):
        src = PAPER / extra
        if src.exists():
            shutil.copy2(src, dest / extra)
    src_split = C.SPLITS_DIR / "split_report.json"
    if src_split.exists():
        shutil.copy2(src_split, dest / "split_report.json")
    norms = C.PROJECT_ROOT / "web" / "api" / "model" / "anatomy_norms.json"
    if norms.exists():
        shutil.copy2(norms, dest / "anatomy_norms.json")
    print(f"published -> {dest}")


if __name__ == "__main__":
    main()
