"""Four experiments a reviewer will ask for, all on the deployed ONNX model.

  uncertainty  Selective prediction. If the system is to save clinician time it
               must know when to defer, so this measures accuracy on the cases
               retained as the most uncertain fraction is handed back.

  saliency     Deletion / insertion curves. Citing Zhou 2016 for CAM obliges us
               to show the maps are faithful: masking the highest-attention
               regions must collapse the predicted probability faster than
               masking random regions, or the map is decoration.

  robustness   Accuracy under rotation, noise, contrast shift, blur and
               downsampling — a proxy for the scanner and protocol variation
               the model will meet outside this cohort.

  throughput   Measured latency and cost per thousand scans, which is the only
               evidence behind any "reduces time and cost" claim.

    python training/experiments.py --all
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import config as C

sys.path.insert(0, str(C.PROJECT_ROOT / "web" / "api"))
import _inference as inf  # noqa: E402

PAPER = C.ARTIFACTS_DIR / "paper"


def load_test(limit: int | None) -> list[dict]:
    recs = json.loads((C.SPLITS_DIR / "test.json").read_text(encoding="utf-8"))
    if limit:
        rng = np.random.default_rng(3)
        idx = rng.permutation(len(recs))[:limit]
        recs = [recs[int(i)] for i in idx]
    return recs


def raw_logits(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    session = inf.get_session()
    logits, cam = session.run(None, {"input": inf.preprocess(img)})
    return np.asarray(logits, np.float32)[0], np.asarray(cam, np.float32)[0]


def probs_from(logits: np.ndarray) -> np.ndarray:
    t = float(inf.get_meta().get("temperature", 1.0)) or 1.0
    return inf.softmax(logits / t)


# ===========================================================================
def experiment_uncertainty(recs: list[dict], corrupt: float = 0.0,
                           tag: str = "clean") -> dict:
    """Selective prediction: accuracy vs the fraction of cases deferred.

    On the clean test split this is degenerate — the model makes almost no
    errors, so every deferral rule scores a perfect AURC and the experiment
    proves nothing. Selective prediction only means something where the model
    *does* fail, so this is also run over a corrupted copy of the same images
    (`corrupt` = Gaussian noise sigma), which is the regime the deferral
    mechanism actually exists for.
    """
    print(f"\n[uncertainty:{tag}] scoring {len(recs)} images"
          f"{f' (noise sigma {corrupt})' if corrupt else ''}")
    rng = np.random.default_rng(41)
    P, Y, E = [], [], []
    for i, r in enumerate(recs):
        img = inf.load_image(Path(r["path"]).read_bytes())
        if corrupt > 0:
            a = np.asarray(img, np.float32)
            img = Image.fromarray(
                np.clip(a + rng.normal(0, corrupt * 255, a.shape), 0, 255)
                .astype(np.uint8))
        lg, _ = raw_logits(img)
        P.append(probs_from(lg))
        Y.append(int(r["label"]))
        E.append(float(-np.log(np.exp(lg - lg.max()).sum()) - lg.max()))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(recs)}")

    P, Y, E = np.array(P), np.array(Y), np.array(E)
    pred = P.argmax(1)
    correct = (pred == Y).astype(float)

    conf = P.max(1)
    order = np.sort(P, axis=1)
    margin = order[:, -1] - order[:, -2]
    entropy = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)

    curves = {}
    for name, score in (("confidence", conf), ("margin", margin),
                        ("neg_entropy", -entropy), ("neg_energy", -E)):
        # Defer the lowest-scoring cases first.
        idx = np.argsort(-score)
        acc, cov = [], []
        for frac in np.linspace(0.1, 1.0, 19):
            k = max(1, int(round(frac * len(idx))))
            acc.append(float(correct[idx[:k]].mean()))
            cov.append(float(k / len(idx)))
        # Area under the risk-coverage curve; lower risk is better.
        risk = 1 - np.array(acc)
        curves[name] = {
            "coverage": cov, "accuracy": acc,
            "aurc": float(np.trapezoid(risk, cov) / (cov[-1] - cov[0])),
        }
        print(f"  {name:<12} AURC {curves[name]['aurc']:.5f}  "
              f"acc@50% coverage {acc[len(acc)//2 - 4]:.4f}")

    n_err = int((1 - correct).sum())
    return {
        "n": int(len(Y)),
        "corruption_sigma": corrupt,
        "base_accuracy": float(correct.mean()),
        "n_errors": n_err,
        "curves": curves,
        "degenerate": n_err < 5,
        "note": "Cases are deferred lowest-score-first. AURC is the area under "
                "the risk-coverage curve; lower is better. With fewer than ~5 "
                "errors the curves carry no information and are marked "
                "degenerate.",
    }


# ===========================================================================
def experiment_saliency(recs: list[dict], n: int = 120,
                        steps: int = 12) -> dict:
    """Deletion / insertion faithfulness of the class activation map."""
    recs = recs[:n]
    print(f"\n[saliency] deletion/insertion on {len(recs)} images")
    size = int(inf.get_meta()["img_size"])
    rng = np.random.default_rng(17)

    del_cam, ins_cam, del_rand, ins_rand = [], [], [], []
    for i, r in enumerate(recs):
        img = inf.load_image(Path(r["path"]).read_bytes())
        lg, cam = raw_logits(img)
        k = int(lg.argmax())
        base = float(probs_from(lg)[k])

        heat = inf.bilinear_resize(np.maximum(cam[k], 0), size, size)
        arr = np.asarray(img.resize((size, size), Image.BILINEAR),
                         dtype=np.float32)
        blur = np.asarray(
            img.resize((size, size), Image.BILINEAR).filter(
                ImageFilter.GaussianBlur(9)), dtype=np.float32)

        rank = np.argsort(heat.ravel())[::-1]
        rand = rng.permutation(heat.size)

        blurred_p = float(probs_from(raw_logits(
            Image.fromarray(blur.astype(np.uint8)))[0])[k])
        d_c, i_c = [base], [blurred_p]
        d_r, i_r = [base], [blurred_p]
        for s in range(1, steps + 1):
            m = int(heat.size * s / steps)

            mask = np.zeros(heat.size, bool); mask[rank[:m]] = True
            mask = mask.reshape(heat.shape)[..., None]
            d_c.append(float(probs_from(raw_logits(Image.fromarray(
                np.where(mask, blur, arr).astype(np.uint8)))[0])[k]))
            i_c.append(float(probs_from(raw_logits(Image.fromarray(
                np.where(mask, arr, blur).astype(np.uint8)))[0])[k]))

            mask = np.zeros(heat.size, bool); mask[rand[:m]] = True
            mask = mask.reshape(heat.shape)[..., None]
            d_r.append(float(probs_from(raw_logits(Image.fromarray(
                np.where(mask, blur, arr).astype(np.uint8)))[0])[k]))
            # Random insertion is the control deletion lacks: without it a high
            # insertion AUC could just mean "any pixels restore the answer".
            i_r.append(float(probs_from(raw_logits(Image.fromarray(
                np.where(mask, arr, blur).astype(np.uint8)))[0])[k]))

        del_cam.append(d_c); ins_cam.append(i_c)
        del_rand.append(d_r); ins_rand.append(i_r)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(recs)}")

    x = np.linspace(0, 1, steps + 1)
    dc = np.array(del_cam).mean(0)
    ic = np.array(ins_cam).mean(0)
    dr = np.array(del_rand).mean(0)
    ir = np.array(ins_rand).mean(0)
    auc = lambda y: float(np.trapezoid(y, x))  # noqa: E731

    out = {
        "fraction_masked": x.tolist(),
        "deletion_cam": dc.tolist(),
        "deletion_random": dr.tolist(),
        "insertion_cam": ic.tolist(),
        "insertion_random": ir.tolist(),
        "auc_deletion_cam": auc(dc),
        "auc_deletion_random": auc(dr),
        "auc_insertion_cam": auc(ic),
        "auc_insertion_random": auc(ir),
        "n": len(recs),
    }
    out["deletion_passes"] = bool(
        out["auc_deletion_cam"] < out["auc_deletion_random"])
    out["insertion_passes"] = bool(
        out["auc_insertion_cam"] > out["auc_insertion_random"])
    out["interpretation"] = (
        "Deletion masks the highest-attention pixels with blur; insertion "
        "reveals them from a blurred baseline. Deletion is confounded here: "
        "the CAM concentrates on the ventricles, which are homogeneous, so "
        "blurring them removes little information regardless of how important "
        "they are. Insertion against a random control is the more informative "
        "test on this data."
    )
    print(f"  deletion   CAM {out['auc_deletion_cam']:.4f} vs random "
          f"{out['auc_deletion_random']:.4f}  -> "
          f"{'pass' if out['deletion_passes'] else 'FAIL'}")
    print(f"  insertion  CAM {out['auc_insertion_cam']:.4f} vs random "
          f"{out['auc_insertion_random']:.4f}  -> "
          f"{'pass' if out['insertion_passes'] else 'FAIL'}")
    return out


# ===========================================================================
def experiment_robustness(recs: list[dict], n: int = 300) -> dict:
    """Accuracy under acquisition-style perturbations."""
    recs = recs[:n]
    print(f"\n[robustness] {len(recs)} images across perturbations")
    rng = np.random.default_rng(23)

    def rotate(im, deg):
        return im.rotate(deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    def noise(im, sigma):
        a = np.asarray(im, np.float32)
        return Image.fromarray(
            np.clip(a + rng.normal(0, sigma * 255, a.shape), 0, 255)
            .astype(np.uint8))

    def contrast(im, f):
        return ImageEnhance.Contrast(im).enhance(f)

    def blur(im, r):
        return im.filter(ImageFilter.GaussianBlur(r))

    def downsample(im, f):
        w, h = im.size
        small = im.resize((max(8, int(w * f)), max(8, int(h * f))),
                          Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    sweeps = {
        "rotation_deg": (rotate, [0, 3, 6, 10, 15, 25, 40]),
        "gaussian_noise_sigma": (noise, [0.0, 0.02, 0.05, 0.10, 0.15, 0.25]),
        "contrast_factor": (contrast, [1.0, 0.85, 0.7, 0.5, 1.3, 1.6]),
        "blur_radius_px": (blur, [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]),
        "downsample_factor": (downsample, [1.0, 0.75, 0.5, 0.35, 0.25]),
    }

    images = [(inf.load_image(Path(r["path"]).read_bytes()), int(r["label"]))
              for r in recs]

    results: dict = {}
    for name, (fn, levels) in sweeps.items():
        accs = []
        for level in levels:
            ok = 0
            for im, y in images:
                mod = im if level in (0, 0.0, 1.0) and name != "contrast_factor" \
                    else fn(im, level)
                lg, _ = raw_logits(mod)
                ok += int(int(lg.argmax()) == y)
            accs.append(ok / len(images))
        results[name] = {"levels": levels, "accuracy": accs}
        print(f"  {name:<22} " + "  ".join(f"{a:.3f}" for a in accs))

    return {"n": len(images), "sweeps": results}


# ===========================================================================
def experiment_throughput(recs: list[dict], n: int = 200) -> dict:
    """Measured latency and the cost arithmetic behind any efficiency claim."""
    recs = recs[:n]
    print(f"\n[throughput] timing {len(recs)} end-to-end predictions")

    payloads = [Path(r["path"]).read_bytes() for r in recs]
    inf.predict(payloads[0], want_overlay=False, want_anatomy=False)  # warm

    def timed(**kw) -> list[float]:
        out = []
        for b in payloads:
            t0 = time.perf_counter()
            inf.predict(b, **kw)
            out.append((time.perf_counter() - t0) * 1000)
        return out

    classify = timed(want_overlay=False, want_anatomy=False)
    full = timed(want_overlay=True, want_anatomy=True)

    def stats(v: list[float]) -> dict:
        a = np.array(v)
        return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                "p95_ms": float(np.percentile(a, 95)),
                "throughput_per_min": float(60_000 / a.mean())}

    s_cls, s_full = stats(classify), stats(full)

    # Vercel Fluid compute bills GB-seconds. 1.7 GB at the published on-demand
    # rate; recompute if the price changes.
    gb, usd_per_gb_s = 1.769, 0.0000180
    cost_1k = 1000 * (s_full["mean_ms"] / 1000) * gb * usd_per_gb_s

    out = {
        "n": len(payloads),
        "classification_only": s_cls,
        "full_pipeline": s_full,
        "cost_model": {
            "memory_gb": gb,
            "usd_per_gb_second": usd_per_gb_s,
            "usd_per_1000_scans": cost_1k,
            "assumption": "Vercel Fluid on-demand GB-second pricing, warm "
                          "container, CPU-only ONNX Runtime. Excludes cold "
                          "starts and storage.",
        },
        "hardware": "CPU-only inference — no GPU is required to serve this.",
    }
    print(f"  classify only : {s_cls['median_ms']:.1f} ms median, "
          f"{s_cls['throughput_per_min']:.0f}/min")
    print(f"  full pipeline : {s_full['median_ms']:.1f} ms median, "
          f"{s_full['throughput_per_min']:.0f}/min")
    print(f"  cost / 1000 scans: ${cost_1k:.4f}")
    return out


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    PAPER.mkdir(parents=True, exist_ok=True)
    recs = load_test(args.limit)

    todo = args.only or (["uncertainty", "saliency", "robustness", "throughput"]
                         if args.all else ["throughput"])
    path = PAPER / "experiments.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    runners = {
        "uncertainty": lambda: {
            "clean": experiment_uncertainty(recs, 0.0, "clean"),
            "corrupted": experiment_uncertainty(recs, 0.12, "noise-0.12"),
        },
        "saliency": lambda: experiment_saliency(recs),
        "robustness": lambda: experiment_robustness(recs),
        "throughput": lambda: experiment_throughput(recs),
    }
    for name in todo:
        results[name] = runners[name]()
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  -> saved {name}")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
