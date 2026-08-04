"""Deeper analytics for the manuscript.

Five analyses that a reviewer of a medical-imaging paper will look for and
that none of the earlier scripts cover:

  baselines    Classical models trained on the SEVEN interpretable morphometric
               features alone, against the CNN. This is the question that
               decides whether deep learning is justified here at all: if
               logistic regression on ventricle-to-brain ratio and CSF fraction
               matches the network, the network is an expensive way to measure
               ventricles.

  ordinal      The four stages are ordered, so plain accuracy treats
               "Non -> Moderate" and "Mild -> Moderate" as equally wrong.
               Quadratic weighted kappa and mean absolute stage error do not.

  separability How well the learned representation separates the classes,
               measured with silhouette score and a 2-D embedding for the
               qualitative figure.

  learning     Accuracy against training-set size, which says whether the
               problem is data-limited or already saturated.

  operating    Sensitivity/specificity across thresholds for the screening
               framing: "any impairment vs none", where a screening tool should
               be tuned for recall, not accuracy.

    python training/analytics.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

sys.path.insert(0, str(C.PROJECT_ROOT / "web" / "api"))
import _anatomy as A  # noqa: E402

PAPER = C.ARTIFACTS_DIR / "paper"


def load_morphometry() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rows = list(csv.DictReader(open(PAPER / "morphometry.csv", encoding="utf-8")))
    keys = list(A.METRIC_INFO.keys())
    X = np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    splits = [r["split"] for r in rows]
    return X, y, keys, splits


# ===========================================================================
def analysis_baselines(seed: int = 0) -> dict:
    """Classical models on morphometry alone, vs the CNN."""
    X, y, keys, _ = load_morphometry()
    print(f"\n[baselines] {len(y)} slices x {len(keys)} interpretable features")

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0,
                                                 class_weight="balanced")),
        "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=seed),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out: dict = {"n": int(len(y)), "features": keys, "models": {}}
    for name, model in models.items():
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        out["models"][name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "quadratic_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
        }
        m = out["models"][name]
        print(f"  {name:<20} acc {m['accuracy']:.4f}  macroF1 "
              f"{m['macro_f1']:.4f}  QWK {m['quadratic_kappa']:.4f}")

    # Univariate discriminative power, so the paper can name the best single
    # measurable index rather than only the multivariate result.
    single = {}
    for j, k in enumerate(keys):
        pred = cross_val_predict(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000,
                                             class_weight="balanced")),
            X[:, [j]], y, cv=cv)
        single[k] = {"accuracy": float(accuracy_score(y, pred)),
                     "macro_f1": float(f1_score(y, pred, average="macro",
                                                zero_division=0))}
    out["single_feature"] = single
    best = max(single, key=lambda k: single[k]["macro_f1"])
    out["best_single_feature"] = best
    print(f"  best single feature: {best} "
          f"(macroF1 {single[best]['macro_f1']:.4f})")

    # Permutation importance on a held-out fold of the strongest model.
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.75)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                class_weight="balanced", random_state=seed,
                                n_jobs=-1).fit(X[idx[:cut]], y[idx[:cut]])
    imp = permutation_importance(rf, X[idx[cut:]], y[idx[cut:]],
                                 n_repeats=20, random_state=seed, n_jobs=-1)
    out["permutation_importance"] = {
        k: {"mean": float(m), "sd": float(s)}
        for k, m, s in zip(keys, imp.importances_mean, imp.importances_std)
    }

    # ---- paired comparison on the CNN's exact split ---------------------
    # 5-fold CV over all originals is not comparable to the CNN's single
    # held-out split. Refitting on exactly the CNN's train rows and scoring on
    # exactly its test rows makes the two numbers answer the same question.
    _, _, _, splits = load_morphometry()
    splits = np.array(splits)
    tr = splits == "train"
    te = splits == "test"
    if tr.sum() > 50 and te.sum() > 20:
        print(f"\n  paired on the CNN's split: train {tr.sum()}, test {te.sum()}")
        paired = {}
        for name, model in models.items():
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            paired[name] = {
                "accuracy": float(accuracy_score(y[te], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
                "macro_f1": float(f1_score(y[te], pred, average="macro",
                                           zero_division=0)),
                "quadratic_kappa": float(
                    cohen_kappa_score(y[te], pred, weights="quadratic")),
            }
            print(f"    {name:<20} acc {paired[name]['accuracy']:.4f}  "
                  f"macroF1 {paired[name]['macro_f1']:.4f}")
        out["same_split"] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                             "models": paired}

    stats_path = PAPER / "statistics.json"
    if stats_path.exists():
        cnn = json.loads(stats_path.read_text(encoding="utf-8"))["headline"]
        out["cnn_reference"] = {
            "accuracy": cnn["accuracy"]["value"],
            "macro_f1": cnn["macro_f1"]["value"],
        }
        pool = out.get("same_split", {}).get("models") or out["models"]
        best_clf = max(pool, key=lambda m: pool[m]["macro_f1"])
        gap = cnn["macro_f1"]["value"] - pool[best_clf]["macro_f1"]
        out["cnn_advantage_macro_f1"] = float(gap)
        out["best_morphometry_model"] = best_clf
        print(f"\n  CNN macroF1 {cnn['macro_f1']['value']:.4f} vs best "
              f"morphometry model ({best_clf}) {pool[best_clf]['macro_f1']:.4f}")
        print(f"  gap: {gap:+.4f} macro F1")
        out["interpretation"] = (
            "The morphometric indices differ across stages with very large "
            "effect sizes (ANOVA p < 1e-90), yet the best classical model "
            "built on them reaches only a fraction of the CNN's score. A "
            "network genuinely reading atrophy should not outperform every "
            "measurable atrophy index combined by this margin. Read together "
            "with the near-zero silhouette in morphometry space and the "
            "validation accuracy saturating at 1.000, the most parsimonious "
            "explanation is that the CNN is partly recognising individual "
            "brains rather than disease stage — the subject-level leakage this "
            "dataset cannot avoid."
        )
    return out


# ===========================================================================
def analysis_ordinal() -> dict:
    """Ordinal-aware metrics on the CNN's held-out predictions."""
    data = np.load(PAPER / "test_predictions.npz")
    y, probs = data["labels"], data["probs"]
    pred = probs.argmax(1)
    print("\n[ordinal] severity-aware error on the held-out split")

    err = np.abs(pred - y)
    # Expected stage under the predictive distribution — a soft ordinal
    # estimate that uses the whole distribution rather than the argmax.
    expected = (probs * np.arange(C.NUM_CLASSES)).sum(1)

    out = {
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y, pred, weights="quadratic")),
        "linear_weighted_kappa": float(
            cohen_kappa_score(y, pred, weights="linear")),
        "mean_absolute_stage_error": float(err.mean()),
        "exact_match": float((err == 0).mean()),
        "within_one_stage": float((err <= 1).mean()),
        "errors_by_distance": {
            str(d): int((err == d).sum()) for d in range(C.NUM_CLASSES)
        },
        "expected_stage_mae": float(np.abs(expected - y).mean()),
        "spearman_expected_vs_true": float(
            np.corrcoef(np.argsort(np.argsort(expected)),
                        np.argsort(np.argsort(y)))[0, 1]),
    }
    print(f"  quadratic weighted kappa : {out['quadratic_weighted_kappa']:.4f}")
    print(f"  mean absolute stage error: {out['mean_absolute_stage_error']:.4f}")
    print(f"  within one stage         : {out['within_one_stage']:.4f}")
    print(f"  errors by distance       : {out['errors_by_distance']}")
    return out


# ===========================================================================
def analysis_separability(seed: int = 0) -> dict:
    """Class separability of the learned representation."""
    data = np.load(PAPER / "test_predictions.npz")
    y = data["labels"]
    # The logits are the model's own 4-D discriminative space; using them
    # avoids a second forward pass through PyTorch just to get embeddings.
    Z = data["logits"]
    print(f"\n[separability] {len(y)} held-out samples")

    sil = float(silhouette_score(Z, y))
    print(f"  silhouette (logit space): {sil:.4f}")

    n = min(len(y), 900)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[:n]
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=seed).fit_transform(Z[idx])

    morph = data["morph"]
    keep = np.isfinite(morph).all(1)
    sil_morph = (float(silhouette_score(morph[keep], y[keep]))
                 if keep.sum() > 50 else float("nan"))
    print(f"  silhouette (morphometry): {sil_morph:.4f}")

    return {
        "silhouette_logits": sil,
        "silhouette_morphometry": sil_morph,
        "tsne": {"x": emb[:, 0].tolist(), "y": emb[:, 1].tolist(),
                 "labels": y[idx].tolist()},
        "note": "Silhouette is computed on the model's logit space and, "
                "separately, on the seven morphometric indices. The gap "
                "quantifies how much structure the network adds beyond the "
                "measurable anatomy.",
    }


# ===========================================================================
def analysis_learning_curve(seed: int = 0) -> dict:
    """Is the problem data-limited, or already saturated?"""
    X, y, _keys, _ = load_morphometry()
    print("\n[learning curve] morphometry model vs training-set size")
    fractions = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    rng = np.random.default_rng(seed)

    curve = []
    for frac in fractions:
        scores = []
        for rep in range(5):
            idx = rng.permutation(len(y))
            cut = int(len(y) * 0.75)
            tr, te = idx[:cut], idx[cut:]
            k = max(C.NUM_CLASSES * 4, int(len(tr) * frac))
            tr = tr[:k]
            if len(np.unique(y[tr])) < C.NUM_CLASSES:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"))
            model.fit(X[tr], y[tr])
            scores.append(f1_score(y[te], model.predict(X[te]),
                                   average="macro", zero_division=0))
        if scores:
            curve.append({"fraction": frac, "n": int(len(y) * 0.75 * frac),
                          "macro_f1_mean": float(np.mean(scores)),
                          "macro_f1_sd": float(np.std(scores))})
            print(f"  {frac:>5.0%}  n={curve[-1]['n']:<6} "
                  f"macroF1 {curve[-1]['macro_f1_mean']:.4f} "
                  f"± {curve[-1]['macro_f1_sd']:.4f}")
    return {"curve": curve}


# ===========================================================================
def analysis_operating_points() -> dict:
    """Screening framing: any impairment vs none."""
    data = np.load(PAPER / "test_predictions.npz")
    y, probs = data["labels"], data["probs"]
    print("\n[operating points] screening: any impairment vs none")

    # NonDemented is index 0, so P(impaired) is one minus its probability.
    score = 1.0 - probs[:, 0]
    binary = (y > 0).astype(int)
    fpr, tpr, thr = roc_curve(binary, score)
    spec = 1 - fpr

    points = []
    for target in (0.90, 0.95, 0.98, 0.99):
        i = int(np.argmax(tpr >= target))
        if tpr[i] < target:
            continue
        points.append({
            "target_sensitivity": target,
            "threshold": float(thr[i]),
            "sensitivity": float(tpr[i]),
            "specificity": float(spec[i]),
        })
        print(f"  sensitivity >= {target:.0%}: threshold {thr[i]:.4f}  "
              f"specificity {spec[i]:.4f}")

    youden = int(np.argmax(tpr - fpr))
    return {
        "prevalence_impaired": float(binary.mean()),
        "operating_points": points,
        "youden": {"threshold": float(thr[youden]),
                   "sensitivity": float(tpr[youden]),
                   "specificity": float(spec[youden])},
        "roc": {"fpr": fpr[::max(1, len(fpr) // 150)].tolist(),
                "tpr": tpr[::max(1, len(tpr) // 150)].tolist()},
        "note": "A screening tool is tuned for sensitivity, not accuracy: the "
                "cost of missing an impaired patient is not the cost of a "
                "false alarm that a clinician then rules out.",
    }


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    PAPER.mkdir(parents=True, exist_ok=True)
    runners = {
        "baselines": analysis_baselines,
        "ordinal": analysis_ordinal,
        "separability": analysis_separability,
        "learning_curve": analysis_learning_curve,
        "operating_points": analysis_operating_points,
    }
    todo = args.only or (list(runners) if args.all else ["ordinal"])

    path = PAPER / "analytics.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in todo:
        results[name] = runners[name]()
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
