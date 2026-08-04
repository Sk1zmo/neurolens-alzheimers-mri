"""Statistics for the paper.

Scores the held-out test set through the **deployed ONNX serving path**, not
the PyTorch checkpoint, so every number reported in the paper is the number the
production endpoint actually produces. Saves raw per-image probabilities so all
downstream figures derive from one scoring pass.

Outputs
  artifacts/paper/test_predictions.npz   probabilities, labels, energies, CAM
                                         region attribution
  artifacts/paper/statistics.json        headline metrics with bootstrap CIs,
                                         per-class metrics, morphometry ANOVA,
                                         attention statistics
  artifacts/paper/tables/*.tex|csv       ready-to-paste tables
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
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

import config as C

sys.path.insert(0, str(C.PROJECT_ROOT / "web" / "api"))
import _anatomy as A  # noqa: E402
import _inference as inf  # noqa: E402

PAPER = C.ARTIFACTS_DIR / "paper"
TABLES = PAPER / "tables"


# ---------------------------------------------------------------- inference
def score_test_set(limit: int | None = None) -> dict:
    records = json.loads((C.SPLITS_DIR / "test.json").read_text(encoding="utf-8"))
    if limit:
        records = records[:limit]

    n_regions = len(A.ATLAS)
    region_keys = [r.key for r in A.ATLAS]

    probs = np.zeros((len(records), C.NUM_CLASSES), dtype=np.float32)
    logits = np.zeros_like(probs)
    labels = np.zeros(len(records), dtype=np.int64)
    energies = np.zeros(len(records), dtype=np.float32)
    attention = np.zeros((len(records), n_regions), dtype=np.float32)
    morph_keys = list(A.METRIC_INFO.keys())
    morph = np.full((len(records), len(morph_keys)), np.nan, dtype=np.float32)

    print(f"scoring {len(records)} held-out images through the ONNX path...")
    for i, rec in enumerate(records):
        out = inf.predict(Path(rec["path"]).read_bytes(),
                          want_overlay=False, want_anatomy=True)
        probs[i] = out["probabilities"]
        logits[i] = out["logits"]
        labels[i] = int(rec["label"])
        energies[i] = out["energy"]

        anat = out.get("anatomy") or {}
        if "metrics" in anat:
            for j, k in enumerate(morph_keys):
                v = anat["metrics"].get(k)
                if v is not None and np.isfinite(v):
                    morph[i, j] = v
        for row in anat.get("attribution", []) or []:
            if row["key"] in region_keys:
                attention[i, region_keys.index(row["key"])] = row["attention_density"]

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)}")

    return {
        "probs": probs, "logits": logits, "labels": labels,
        "energies": energies, "attention": attention,
        "region_keys": region_keys, "morph": morph, "morph_keys": morph_keys,
    }


# ---------------------------------------------------------------- statistics
def bootstrap_ci(y: np.ndarray, pred: np.ndarray, fn, n: int = 2000,
                 seed: int = 5) -> tuple[float, float, float]:
    """Percentile bootstrap over test images."""
    rng = np.random.default_rng(seed)
    point = float(fn(y, pred))
    stats = np.empty(n, dtype=np.float64)
    idx_pool = np.arange(len(y))
    for b in range(n):
        idx = rng.choice(idx_pool, size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            stats[b] = np.nan
            continue
        stats[b] = fn(y[idx], pred[idx])
    stats = stats[np.isfinite(stats)]
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def one_way_anova(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """F statistic, p (via survival function), and eta-squared effect size."""
    from scipy import stats as st
    groups = [g[np.isfinite(g)] for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan"), float("nan"), float("nan")
    f, p = st.f_oneway(*groups)
    grand = np.concatenate(groups)
    ss_between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
    ss_total = ((grand - grand.mean()) ** 2).sum()
    eta2 = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    return float(f), float(p), eta2


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy import stats as st
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    r, p = st.spearmanr(x[m], y[m])
    return float(r), float(p)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size: P(a>b) - P(a<b). Robust to non-normality."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # Rank-based computation avoids the O(n*m) pairwise matrix.
    combined = np.concatenate([a, b])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    ra = ranks[:len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    PAPER.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    scored = score_test_set(args.limit)
    np.savez_compressed(PAPER / "test_predictions.npz", **{
        k: v for k, v in scored.items() if isinstance(v, np.ndarray)
    })
    (PAPER / "test_predictions_keys.json").write_text(json.dumps({
        "region_keys": scored["region_keys"], "morph_keys": scored["morph_keys"],
        "classes": C.CLASS_LABELS,
    }, indent=2), encoding="utf-8")

    y = scored["labels"]
    probs = scored["probs"]
    pred = probs.argmax(1)

    print("\ncomputing bootstrap confidence intervals...")
    stats: dict = {"n_test": int(len(y)), "n_bootstrap": args.bootstrap}

    headline = {}
    for name, fn in (
        ("accuracy", accuracy_score),
        ("balanced_accuracy", balanced_accuracy_score),
        ("macro_f1", lambda a, b: f1_score(a, b, average="macro", zero_division=0)),
        ("weighted_f1", lambda a, b: f1_score(a, b, average="weighted", zero_division=0)),
        ("cohen_kappa", cohen_kappa_score),
    ):
        point, lo, hi = bootstrap_ci(y, pred, fn, args.bootstrap)
        headline[name] = {"value": point, "ci95_low": lo, "ci95_high": hi}
        print(f"  {name:<20} {point:.4f}  [{lo:.4f}, {hi:.4f}]")
    stats["headline"] = headline

    # ---- per class -------------------------------------------------------
    prec, rec, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(C.NUM_CLASSES)), zero_division=0)
    per_class = {}
    for i, label in enumerate(C.CLASS_LABELS):
        binary = (y == i).astype(int)
        auc = float(roc_auc_score(binary, probs[:, i])) if binary.sum() else float("nan")
        ap_score = (float(average_precision_score(binary, probs[:, i]))
                    if binary.sum() else float("nan"))
        # Wilson interval for recall — correct for small n, unlike a normal
        # approximation, and ModerateDemented has n=13.
        per_class[label] = {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "f1": float(f1[i]), "support": int(support[i]),
            "auc": auc, "average_precision": ap_score,
            "recall_wilson_ci95": wilson(int(rec[i] * support[i]), int(support[i])),
        }
    stats["per_class"] = per_class
    stats["confusion_matrix"] = confusion_matrix(
        y, pred, labels=list(range(C.NUM_CLASSES))).tolist()

    # ---- adjacent-stage error analysis -----------------------------------
    cm = np.array(stats["confusion_matrix"])
    off = cm.sum() - np.trace(cm)
    adjacent = sum(cm[i, j] for i in range(C.NUM_CLASSES)
                   for j in range(C.NUM_CLASSES) if abs(i - j) == 1)
    stats["error_analysis"] = {
        "total_errors": int(off),
        "adjacent_stage_errors": int(adjacent),
        "adjacent_fraction": float(adjacent / off) if off else float("nan"),
        "note": "Misclassifications that land on a neighbouring severity stage "
                "are clinically less costly than distant ones.",
    }

    # ---- morphometry across stages ---------------------------------------
    print("\nmorphometry across stages...")
    rows = list(csv.DictReader(
        open(PAPER / "morphometry.csv", encoding="utf-8")))
    morph_stats = {}
    for key in A.METRIC_INFO:
        groups, means = [], {}
        for cls in C.CLASS_DIRS:
            vals = np.array([float(r[key]) for r in rows if r["class"] == cls])
            groups.append(vals)
            means[cls] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                          "n": int(len(vals))}
        f, p, eta2 = one_way_anova(groups)
        severity = np.concatenate(
            [np.full(len(g), C.DIR_TO_INDEX[c]) for g, c in zip(groups, C.CLASS_DIRS)])
        allvals = np.concatenate(groups)
        rho, rho_p = spearman(allvals, severity)
        morph_stats[key] = {
            "label": A.METRIC_INFO[key]["label"],
            "by_class": means,
            "anova_F": f, "anova_p": p, "eta_squared": eta2,
            "spearman_rho_vs_severity": rho, "spearman_p": rho_p,
            "cliffs_delta_moderate_vs_none": cliffs_delta(groups[3], groups[0]),
        }
        print(f"  {key:<26} F={f:8.2f}  p={p:.3e}  eta2={eta2:.3f}  rho={rho:+.3f}")
    stats["morphometry"] = morph_stats

    # ---- attention by region and class ------------------------------------
    att = scored["attention"]
    region_stats = {}
    for j, key in enumerate(scored["region_keys"]):
        by_class = {}
        for i, cls in enumerate(C.CLASS_DIRS):
            v = att[y == i, j]
            by_class[cls] = {"mean_density": float(v.mean()) if len(v) else float("nan"),
                             "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        f, p, eta2 = one_way_anova([att[y == i, j] for i in range(C.NUM_CLASSES)])
        region_stats[key] = {
            "name": A.ATLAS_BY_KEY[key].name,
            "lobe": A.ATLAS_BY_KEY[key].lobe,
            "by_class": by_class,
            "anova_F": f, "anova_p": p, "eta_squared": eta2,
        }
    stats["attention_by_region"] = region_stats

    # ---- OOD separation ---------------------------------------------------
    stats["energy"] = {
        "mean": float(scored["energies"].mean()),
        "p95": float(np.percentile(scored["energies"], 95)),
        "p99": float(np.percentile(scored["energies"], 99)),
    }

    (PAPER / "statistics.json").write_text(json.dumps(stats, indent=2),
                                           encoding="utf-8")
    print(f"\nwrote {PAPER / 'statistics.json'}")
    write_tables(stats)


def wilson(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def write_tables(stats: dict) -> None:
    # Table 1 — headline metrics
    with open(TABLES / "table1_headline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value", "95% CI low", "95% CI high"])
        for k, v in stats["headline"].items():
            w.writerow([k.replace("_", " ").title(), f"{v['value']:.4f}",
                        f"{v['ci95_low']:.4f}", f"{v['ci95_high']:.4f}"])

    lines = [r"\begin{tabular}{lccc}", r"\hline",
             r"Metric & Value & \multicolumn{2}{c}{95\% CI} \\", r"\hline"]
    for k, v in stats["headline"].items():
        lines.append(f"{k.replace('_', ' ').title()} & {v['value']:.4f} & "
                     f"{v['ci95_low']:.4f} & {v['ci95_high']:.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (TABLES / "table1_headline.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table 2 — per class
    with open(TABLES / "table2_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Stage", "Precision", "Recall", "F1", "AUC", "AP", "n"])
        for label, m in stats["per_class"].items():
            w.writerow([label, f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                        f"{m['f1']:.3f}", f"{m['auc']:.3f}",
                        f"{m['average_precision']:.3f}", m["support"]])

    # Table 3 — morphometry
    with open(TABLES / "table3_morphometry.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Index"] + [f"{c} mean (SD)" for c in C.CLASS_DIRS]
                   + ["ANOVA F", "p", "eta^2", "Spearman rho"])
        for key, m in stats["morphometry"].items():
            row = [m["label"]]
            for cls in C.CLASS_DIRS:
                s = m["by_class"][cls]
                row.append(f"{s['mean']:.4f} ({s['sd']:.4f})")
            row += [f"{m['anova_F']:.1f}",
                    f"{m['anova_p']:.2e}", f"{m['eta_squared']:.3f}",
                    f"{m['spearman_rho_vs_severity']:+.3f}"]
            w.writerow(row)

    print(f"wrote tables -> {TABLES}")


if __name__ == "__main__":
    main()
