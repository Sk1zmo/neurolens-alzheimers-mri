"""Live CLI progress view for a running train.py.

Run this in its own terminal:

    python training/watch_training.py

Why it infers rather than reads: when train.py's stdout is piped rather than
attached to a terminal, Python block-buffers it, so per-epoch lines do not
appear until the buffer fills. This watcher reconstructs progress from
observable state instead — the checkpoint's mtime, the process start time and
live GPU telemetry — so it works against an already-running job that was
launched without `-u`.

Once training finishes it prints the real numbers from training_history.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

import psutil

import config as C

BAR_W = 34


def gpu_stats() -> tuple[int, int, int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        util, used, total = (int(v.strip()) for v in out.stdout.split(",")[:3])
        return util, used, total
    except Exception:
        return None


def training_process() -> tuple[int, float] | None:
    """(pid, start_epoch_seconds) of a live train.py, or None.

    The start time comes from the OS, not from when this watcher launched, so
    attaching to an already-running job still reports a truthful percentage.
    The earliest matching process wins — DataLoader workers are also `python`
    with train.py on their command line, but they spawn later.
    """
    best: tuple[int, float] | None = None
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            name = (proc.info["name"] or "").lower()
            if not name.startswith("python"):
                continue
            cmdline = proc.info["cmdline"] or []
            if not any("train.py" in part for part in cmdline):
                continue
            started = float(proc.info["create_time"])
            if best is None or started < best[1]:
                best = (int(proc.info["pid"]), started)
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            continue
    return best


def fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def bar(frac: float) -> str:
    # ASCII only: Windows consoles running a legacy code page mangle block
    # glyphs into visual noise, which defeats the point of a progress bar.
    frac = min(1.0, max(0.0, frac))
    filled = int(round(frac * BAR_W))
    return "=" * filled + "-" * (BAR_W - filled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int,
                    default=C.EPOCHS_HEAD + C.EPOCHS_FINETUNE,
                    help="Total epochs the run was launched with.")
    ap.add_argument("--epoch-seconds", type=float, default=None,
                    help="Override the measured epoch duration.")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    ckpt = C.CHECKPOINT_DIR / "best.pt"
    history = C.REPORTS_DIR / "training_history.json"
    first_seen_ckpt = ckpt.stat().st_mtime if ckpt.exists() else None
    saves: list[float] = []

    print("Watching training. Ctrl-C to stop watching "
          "(this does not stop training).\n")

    while True:
        proc = training_process()
        if proc is None:
            print("\n")
            if history.exists():
                h = json.loads(history.read_text(encoding="utf-8"))
                best = h.get("best", {})
                print("Training finished.")
                print(f"  best val macro-F1 : {best.get('macro_f1', float('nan')):.4f}")
                print(f"  best val accuracy : {best.get('acc', float('nan')):.4f}")
                print(f"  at epoch          : {best.get('epoch')}")
                print(f"  wall clock        : {fmt(h.get('elapsed_sec', 0))}")
                rows = h.get("history", [])
                if rows:
                    print("\n  last epochs:")
                    for r in rows[-6:]:
                        print(f"    {r['stage']:<9} ep {r['epoch']:>2}  "
                              f"val acc {r.get('val_acc', 0):.4f}  "
                              f"macroF1 {r.get('val_macro_f1', 0):.4f}")
            else:
                print("No train.py process found and no history written — "
                      "the run may have crashed. Check its log.")
            return

        pid, proc_started = proc
        elapsed = max(1.0, time.time() - proc_started)
        mtime = ckpt.stat().st_mtime if ckpt.exists() else None

        # Track every distinct checkpoint save. The gaps between saves are
        # whole multiples of the epoch time, so their greatest common divisor
        # is a far better rate estimate than a single interval.
        if mtime and (not saves or mtime > saves[-1] + 1):
            saves.append(mtime)

        if args.epoch_seconds:
            per_epoch = args.epoch_seconds
        elif len(saves) >= 2:
            gaps = [b - a for a, b in zip(saves, saves[1:])]
            smallest = min(gaps)
            # Assume the tightest observed gap is one epoch unless it is
            # implausibly short, in which case fall back to the mean.
            per_epoch = smallest if smallest > 30 else (sum(gaps) / len(gaps))
        elif first_seen_ckpt:
            # One save seen: the run reached it after some whole number of
            # epochs, so divide the time to it by a conservative guess of one.
            per_epoch = max(30.0, first_seen_ckpt - proc_started)
        else:
            per_epoch = 330.0  # measured default for this dataset on a 4060

        done = elapsed / per_epoch
        frac = min(0.99, done / max(1, args.epochs))

        g = gpu_stats()
        gpu_txt = (f"GPU {g[0]:>3}%  {g[1]:>5}/{g[2]} MiB" if g else "GPU n/a")
        last_save = (datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
                     if mtime else "--")
        eta = max(0.0, (args.epochs - done) * per_epoch)

        line = (f"\r[{bar(frac)}] ~{frac*100:5.1f}%  "
                f"ep ~{min(done, args.epochs):4.1f}/{args.epochs}  "
                f"elapsed {fmt(elapsed):>7}  eta ~{fmt(eta):>7}  "
                f"{gpu_txt}  best {last_save}  pid {pid}   ")
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped watching (training continues)")
