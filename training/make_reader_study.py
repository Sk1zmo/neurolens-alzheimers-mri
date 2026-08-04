"""Build the blinded case set for the reader study.

Cases are drawn from the held-out test split and copied under opaque filenames,
so a reader cannot infer the label from the path. The manifest keeps the
mapping; the app only reveals it after a rating is committed.

Sampling is stratified but deliberately *not* proportional: the natural
distribution is ~50% NonDemented, and a reader who saw that distribution could
score well by answering "Non" throughout. Equalising the arms makes both the
reader's and the model's kappa informative.

    python training/make_reader_study.py --per-class 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

import config as C

OUT = C.PROJECT_ROOT / "web" / "public" / "reader-study"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--seed", type=int, default=90210)
    args = ap.parse_args()

    records = json.loads((C.SPLITS_DIR / "test.json").read_text(encoding="utf-8"))
    by_class: dict[str, list[dict]] = {c: [] for c in C.CLASS_DIRS}
    for r in records:
        by_class[r["class"]].append(r)

    rng = np.random.default_rng(args.seed)
    chosen: list[dict] = []
    for cls, pool in by_class.items():
        if not pool:
            continue
        take = min(args.per_class, len(pool))
        idx = rng.permutation(len(pool))[:take]
        chosen += [pool[int(i)] for i in idx]
        if take < args.per_class:
            print(f"  ! {cls}: only {take} available")

    # Shuffle so consecutive cases are not the same class.
    order = rng.permutation(len(chosen))
    chosen = [chosen[int(i)] for i in order]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    cases = []
    for i, rec in enumerate(chosen):
        src = Path(rec["path"])
        # Opaque, stable name — derived from the path so reruns with the same
        # seed produce the same set, but revealing nothing about the class.
        digest = hashlib.sha256(str(src).encode()).hexdigest()[:16]
        name = f"case_{i:03d}_{digest}{src.suffix.lower()}"
        shutil.copy2(src, OUT / name)
        cases.append({"file": name, "truth": rec["class"]})

    (OUT / "manifest.json").write_text(json.dumps({
        "n": len(cases),
        "per_class": args.per_class,
        "seed": args.seed,
        "source": "held-out test split",
        "note": "Classes are equalised, not proportional: at the natural ~50% "
                "NonDemented rate a reader could score well by answering 'Non' "
                "throughout, which would make kappa uninformative.",
        "cases": cases,
    }, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for c in cases:
        counts[c["truth"]] = counts.get(c["truth"], 0) + 1
    print(f"wrote {len(cases)} blinded cases -> {OUT}")
    for cls in C.CLASS_DIRS:
        print(f"  {cls:<20} {counts.get(cls, 0)}")


if __name__ == "__main__":
    main()
