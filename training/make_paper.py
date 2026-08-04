"""Regenerate every research artifact in one command.

    python training/make_paper.py            # stats + figures + publish
    python training/make_paper.py --full     # also rebuild anatomy norms
    python training/make_paper.py --deploy   # ...then push to Vercel

Order matters: norms feed the morphometry CSV, which feeds the statistics,
which feed the figures. Each stage reads the previous stage's artifact rather
than recomputing it, so what is plotted is always what is tabulated.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Recompute anatomy norms and the morphometry CSV.")
    ap.add_argument("--deploy", action="store_true",
                    help="Build and deploy the web app afterwards.")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    py = sys.executable

    if args.full:
        run([py, "build_anatomy_norms.py", "--per-class", "500"], HERE)
        run([py, "qc_anatomy.py"], HERE)

    run([py, "paper_stats.py", "--bootstrap", str(args.bootstrap)], HERE)
    run([py, "paper_figures.py"], HERE)

    print("\nartifacts -> artifacts/paper/  and  web/public/paper/")

    if args.deploy:
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        vercel = "vercel.cmd" if sys.platform == "win32" else "vercel"
        run([npm, "run", "build"], WEB)
        run([vercel, "--prod", "--yes"], WEB)


if __name__ == "__main__":
    main()
