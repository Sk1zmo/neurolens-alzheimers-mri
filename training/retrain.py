"""Second pass over the user-upload database: pull it down and retrain.

Every scan a visitor uploads is written to Supabase twice, for two different
consumers:

  scans          -> what the *user* sees (their own history, in the app)
  training_queue -> what the *model* sees (this script)

A queue row only becomes training data once a reviewer has attached a
`verified_label` in the app's Review console. Model predictions are never fed
back as ground truth — that just amplifies whatever the model already believes.

The held-out test split never changes across retrains, so the before/after
numbers printed at the end are directly comparable. The new checkpoint is only
promoted if it actually beats the incumbent on macro-F1.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import config as C
from supabase_client import Supabase, SupabaseError

BUCKET = "scans"
HERE = Path(__file__).resolve().parent


def fetch_pool(limit: int | None = None) -> list[dict]:
    sb = Supabase()
    params = {
        "verified_label": "not.is.null",
        "used_in_training": "eq.false",
        "status": "eq.approved",
        "order": "created_at.asc",
    }
    if limit:
        params["limit"] = str(limit)

    rows = list(sb.select("training_queue", params))
    print(f"{len(rows)} reviewer-labelled scans waiting in the queue")
    if not rows:
        return []

    if C.RETRAIN_DIR.exists():
        shutil.rmtree(C.RETRAIN_DIR)
    for cls in C.CLASS_DIRS:
        (C.RETRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)

    saved: list[dict] = []
    for i, row in enumerate(rows, 1):
        cls = row["verified_label"]
        if cls not in C.DIR_TO_INDEX:
            print(f"  ! skipping {row['id']}: unknown label {cls!r}")
            continue
        try:
            blob = sb.download(BUCKET, row["storage_path"])
        except SupabaseError as e:
            print(f"  ! skipping {row['id']}: {e}")
            continue
        ext = Path(row["storage_path"]).suffix or ".jpg"
        dest = C.RETRAIN_DIR / cls / f"{row['id']}{ext}"
        dest.write_bytes(blob)
        saved.append({"path": str(dest), "label": C.DIR_TO_INDEX[cls],
                      "class": cls, "source": "user_upload", "id": row["id"]})
        if i % 25 == 0:
            print(f"  downloaded {i}/{len(rows)}")

    print(f"downloaded {len(saved)} scans -> {C.RETRAIN_DIR}")
    return saved


def mark_used(ids: list[str]) -> None:
    sb = Supabase()
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        in_list = ",".join(chunk)
        sb.update("training_queue", {"id": f"in.({in_list})"},
                  {"used_in_training": True})
    print(f"marked {len(ids)} queue rows as used_in_training")


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=HERE)
    if res.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(cmd)}")


def read_macro_f1(path: Path) -> float:
    if not path.exists():
        return -1.0
    return json.loads(path.read_text(encoding="utf-8"))["headline"]["macro_f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap how many queue rows to pull.")
    ap.add_argument("--min-new", type=int, default=25,
                    help="Refuse to retrain on fewer than this many new scans.")
    ap.add_argument("--epochs", type=int, default=8,
                    help="Fine-tune epochs for the incremental run.")
    ap.add_argument("--offline", action="store_true",
                    help="Reuse whatever is already in artifacts/retrain_pool.")
    ap.add_argument("--force", action="store_true",
                    help="Promote the new model even if macro-F1 drops.")
    args = ap.parse_args()

    C.ensure_dirs()
    baseline_metrics = C.REPORTS_DIR / "metrics.json"
    baseline_f1 = read_macro_f1(baseline_metrics)
    print(f"incumbent macro-F1: {baseline_f1:.4f}" if baseline_f1 >= 0
          else "no incumbent metrics found")

    if args.offline:
        pool = []
        for cls in C.CLASS_DIRS:
            d = C.RETRAIN_DIR / cls
            if d.is_dir():
                pool += [{"path": str(p), "label": C.DIR_TO_INDEX[cls],
                          "class": cls, "source": "user_upload", "id": p.stem}
                         for p in d.iterdir() if p.is_file()]
        print(f"offline mode: {len(pool)} scans on disk")
    else:
        pool = fetch_pool(args.limit)

    if len(pool) < args.min_new:
        print(f"only {len(pool)} new scans (< --min-new {args.min_new}); "
              "nothing to do.")
        return

    # Merge the user pool into the training manifest. val/test are untouched so
    # the comparison stays apples-to-apples.
    train_path = C.SPLITS_DIR / "train.json"
    base_train = json.loads(train_path.read_text(encoding="utf-8"))
    base_train = [r for r in base_train if r.get("source") != "user_upload"]

    backup = C.SPLITS_DIR / "train_base.json"
    if not backup.exists():
        backup.write_text(json.dumps(base_train), encoding="utf-8")

    merged = base_train + [{k: v for k, v in r.items() if k != "id"} for r in pool]
    train_path.write_text(json.dumps(merged), encoding="utf-8")
    print(f"training manifest: {len(base_train)} base + {len(pool)} user = {len(merged)}")

    py = sys.executable
    prev_ckpt = C.CHECKPOINT_DIR / "best.pt"
    cmd = [py, "train.py", "--epochs-head", "1",
           "--epochs-finetune", str(args.epochs), "--out", "retrained.pt"]
    if prev_ckpt.exists():
        cmd += ["--resume-from", str(prev_ckpt)]
    run(cmd)

    if baseline_metrics.exists():
        shutil.copy2(baseline_metrics, C.REPORTS_DIR / "metrics_previous.json")
    run([py, "evaluate.py", "--checkpoint",
         str(C.CHECKPOINT_DIR / "retrained.pt")])

    new_f1 = read_macro_f1(baseline_metrics)
    print(f"\nmacro-F1  {baseline_f1:.4f} -> {new_f1:.4f}  "
          f"({new_f1 - baseline_f1:+.4f})")

    if new_f1 < baseline_f1 and not args.force:
        print("regression — keeping the incumbent model. "
              "Re-run with --force to promote anyway.")
        prev = C.REPORTS_DIR / "metrics_previous.json"
        if prev.exists():
            shutil.copy2(prev, baseline_metrics)
        return

    shutil.copy2(C.CHECKPOINT_DIR / "retrained.pt", prev_ckpt)
    run([py, "export_onnx.py"])
    ids = [r["id"] for r in pool if r.get("id")]
    if ids and not args.offline:
        mark_used(ids)

    print("\nPromoted. Redeploy the web app to ship the new model:")
    print("  cd ../web && vercel --prod")


if __name__ == "__main__":
    main()
