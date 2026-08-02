"""Copy one held-out test image per class into web/public/samples/.

Sampled from the *test* split specifically, so the demo buttons on the analyze
page show the model handling images it genuinely never trained on — including
any augmented derivative of them, which prepare_split.py removed.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import config as C

OUT = C.PROJECT_ROOT / "web" / "public" / "samples"
FILENAMES = {
    "NonDemented": "non-demented.jpg",
    "VeryMildDemented": "very-mild-demented.jpg",
    "MildDemented": "mild-demented.jpg",
    "ModerateDemented": "moderate-demented.jpg",
}


def main() -> None:
    test_path = C.SPLITS_DIR / "test.json"
    if not test_path.exists():
        raise SystemExit(f"{test_path} not found — run prepare_split.py first.")

    records = json.loads(test_path.read_text(encoding="utf-8"))
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cls, name in FILENAMES.items():
        pool = by_class.get(cls, [])
        if not pool:
            print(f"! no test images for {cls}")
            continue
        # Deterministic pick so the samples don't churn between runs.
        chosen = sorted(pool, key=lambda r: r["path"])[len(pool) // 2]
        shutil.copy2(chosen["path"], OUT / name)
        manifest.append({"class": cls, "file": name,
                         "source": Path(chosen["path"]).name})
        print(f"  {cls:<20} {Path(chosen['path']).name} -> {name}")

    (OUT / "manifest.json").write_text(
        json.dumps({"note": "Held-out test-split images, never seen in training.",
                    "samples": manifest}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {len(manifest)} samples -> {OUT}")


if __name__ == "__main__":
    main()
