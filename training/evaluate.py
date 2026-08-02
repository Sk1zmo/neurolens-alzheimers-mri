"""Evaluate the trained model and emit everything the web app displays.

Produces artifacts/reports/metrics.json, which the frontend loads directly so
that the "model performance" panel shows *real* held-out numbers instead of
the per-sample fabrications the previous deployment rendered.

Also fits a temperature-scaling parameter on the validation split. The raw
softmax of a fine-tuned CNN is badly over-confident; since this app puts a
confidence percentage in front of a user, calibrating it is not optional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

import config as C
from dataset import ScanDataset, eval_transform, load_split
from model import AlzheimerNet


@torch.no_grad()
def collect_logits(model, records, device, amp: bool, tta: bool = True):
    ds = ScanDataset(records, transform=eval_transform())
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE * 2, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=True)
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x).float()
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3])).float()) / 2
        all_logits.append(logits.cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Single-parameter temperature scaling (Guo et al., 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray,
                               bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def plot_confusion(cm: np.ndarray, out: Path) -> None:
    norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=140)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.NUM_CLASSES), C.CLASS_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(C.NUM_CLASSES), C.CLASS_LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — held-out original MRI test set")
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.55 else "#123")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_roc(roc: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=140)
    for i, label in enumerate(C.CLASS_LABELS):
        r = roc[str(i)]
        ax.plot(r["fpr"], r["tpr"], lw=2,
                label=f"{label}  (AUC {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#999")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC — held-out test set")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def downsample_curve(x: np.ndarray, y: np.ndarray, n: int = 120):
    if len(x) <= n:
        return x.tolist(), y.tolist()
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[idx].tolist(), y[idx].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(C.CHECKPOINT_DIR / "best.pt"))
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = C.AMP and device.type == "cuda"
    tta = not args.no_tta

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = AlzheimerNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    val_recs, test_recs = load_split("val"), load_split("test")

    print("scoring validation split (for temperature calibration)...")
    val_logits, val_y = collect_logits(model, val_recs, device, amp, tta)
    temperature = fit_temperature(val_logits, val_y)
    print(f"fitted temperature T = {temperature:.4f}")

    print("scoring held-out test split...")
    test_logits, test_y = collect_logits(model, test_recs, device, amp, tta)

    y = test_y.numpy()
    raw_probs = torch.softmax(test_logits, dim=1).numpy()
    probs = torch.softmax(test_logits / temperature, dim=1).numpy()
    pred = probs.argmax(axis=1)

    cm = confusion_matrix(y, pred, labels=list(range(C.NUM_CLASSES)))
    report = classification_report(
        y, pred, labels=list(range(C.NUM_CLASSES)),
        target_names=C.CLASS_LABELS, output_dict=True, zero_division=0,
    )

    roc: dict = {}
    for i in range(C.NUM_CLASSES):
        binary = (y == i).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            roc[str(i)] = {"fpr": [0, 1], "tpr": [0, 1], "auc": float("nan")}
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        f, t = downsample_curve(fpr, tpr)
        roc[str(i)] = {"fpr": f, "tpr": t,
                       "auc": float(roc_auc_score(binary, probs[:, i]))}

    try:
        macro_auc = float(roc_auc_score(y, probs, multi_class="ovr",
                                        average="macro"))
    except ValueError:
        macro_auc = float("nan")

    # Free-energy OOD score, E(x) = -logsumexp(logits). In-distribution inputs
    # score low. The serving layer flags anything above the p99 of this
    # held-out distribution, which is what turns "the model is 94% sure" on a
    # photograph of a cat into a visible warning instead of a confident answer.
    lg = test_logits.numpy()
    energy = -(np.log(np.exp(lg - lg.max(axis=1, keepdims=True)).sum(axis=1))
               + lg.max(axis=1))

    onehot = np.eye(C.NUM_CLASSES)[y]
    metrics = {
        "model": C.MODEL_NAME,
        "architecture": "EfficientNet-B0 (ImageNet) + GAP + Linear head",
        "classes": C.CLASS_LABELS,
        "class_dirs": C.CLASS_DIRS,
        "img_size": C.IMG_SIZE,
        "temperature": temperature,
        "tta_horizontal_flip": tta,
        "test_set": {
            "n": int(len(y)),
            "source": "OriginalDataset held-out split (never augmented, "
                      "descendants removed from training)",
            "per_class_n": np.bincount(y, minlength=C.NUM_CLASSES).tolist(),
        },
        "headline": {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y, pred)),
            "macro_auc_ovr": macro_auc,
            "brier_score": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
            "mse": float(np.mean((probs - onehot) ** 2)),
            "ece_calibrated": expected_calibration_error(probs, y),
            "ece_uncalibrated": expected_calibration_error(raw_probs, y),
            "energy_mean": float(energy.mean()),
            "energy_p95": float(np.percentile(energy, 95)),
            "energy_p99": float(np.percentile(energy, 99)),
        },
        "per_class": {
            C.CLASS_LABELS[i]: {
                "precision": report[C.CLASS_LABELS[i]]["precision"],
                "recall": report[C.CLASS_LABELS[i]]["recall"],
                "f1": report[C.CLASS_LABELS[i]]["f1-score"],
                "support": report[C.CLASS_LABELS[i]]["support"],
                "auc": roc[str(i)]["auc"],
            }
            for i in range(C.NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "roc": roc,
        "caveats": [
            "Trained and evaluated on the Kaggle augmented-alzheimer-mri-dataset, "
            "which is brain MRI — not CT.",
            "The dataset carries no subject IDs, so slices from the same brain may "
            "appear in both training and test. Real-world performance on an unseen "
            "patient will be lower than the numbers above.",
            "Research and educational use only. Not a medical device and not a "
            "diagnostic tool.",
        ],
    }

    (C.REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion(cm, C.REPORTS_DIR / "confusion_matrix.png")
    plot_roc(roc, C.REPORTS_DIR / "roc_curves.png")

    h = metrics["headline"]
    print("\n=== Held-out test performance ===")
    print(f"  n                  {len(y)}")
    print(f"  accuracy           {h['accuracy']:.4f}")
    print(f"  balanced accuracy  {h['balanced_accuracy']:.4f}")
    print(f"  macro F1           {h['macro_f1']:.4f}")
    print(f"  macro AUC (OvR)    {h['macro_auc_ovr']:.4f}")
    print(f"  Cohen's kappa      {h['cohen_kappa']:.4f}")
    print(f"  ECE  raw -> cal    {h['ece_uncalibrated']:.4f} -> {h['ece_calibrated']:.4f}")
    print("\nPer class:")
    for name, m in metrics["per_class"].items():
        print(f"  {name:<20} P {m['precision']:.3f}  R {m['recall']:.3f}  "
              f"F1 {m['f1']:.3f}  AUC {m['auc']:.3f}  n={int(m['support'])}")
    print(f"\nwrote {C.REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
