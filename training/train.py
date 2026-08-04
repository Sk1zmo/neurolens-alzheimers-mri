"""Train the Alzheimer's MRI classifier.

Two-stage schedule:
  1. Warm-up  — backbone frozen, only the linear head learns. Stops the
     randomly-initialised head from wrecking pretrained features on step one.
  2. Fine-tune — everything unfrozen, discriminative LRs (backbone 10x lower
     than the head), cosine decay, early stopping on validation macro-F1.

Selection metric is macro-F1, not accuracy: ModerateDemented is ~1% of the
original data, and plain accuracy would happily ignore it.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Progress has to reach the terminal even when stdout is piped to a file or a
# supervising process, where Python would otherwise block-buffer it and show
# nothing for an hour.
print = functools.partial(builtins.print, flush=True)  # noqa: A001

import config as C
from dataset import ScanDataset, class_weights, eval_transform, load_split, train_transform
from model import AlzheimerNet


def build_loaders(balanced_sampler: bool = True):
    train_recs = load_split("train")
    val_recs = load_split("val")

    train_ds = ScanDataset(train_recs, transform=train_transform())
    val_ds = ScanDataset(val_recs, transform=eval_transform())

    if balanced_sampler:
        labels = np.array(train_ds.labels)
        counts = np.bincount(labels, minlength=C.NUM_CLASSES).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = (1.0 / counts)[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(labels), replacement=True,
        )
        shuffle = False
    else:
        sampler, shuffle = None, True

    common = dict(num_workers=C.NUM_WORKERS, pin_memory=True,
                  persistent_workers=C.NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, sampler=sampler,
                              shuffle=shuffle, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE * 2, shuffle=False,
                            **common)
    return train_ds, val_ds, train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool):
    model.eval()
    total_loss, n = 0.0, 0
    preds, targets = [], []
    loader = tqdm(loader, desc="val", unit="batch", leave=False,
                  dynamic_ncols=True, mininterval=0.5)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "loss": total_loss / max(n, 1),
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def run_epoch(model, loader, criterion, optimizer, scaler, device, amp,
              scheduler=None, desc="train"):
    model.train()
    total_loss, n, correct = 0.0, 0, 0
    bar = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True,
               mininterval=0.5)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
        bar.set_postfix(loss=f"{total_loss / max(n, 1):.4f}",
                        acc=f"{correct / max(n, 1):.4f}", refresh=False)
    bar.close()
    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=C.EPOCHS_HEAD)
    ap.add_argument("--epochs-finetune", type=int, default=C.EPOCHS_FINETUNE)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Checkpoint to warm-start from (used by retrain.py).")
    ap.add_argument("--out", type=str, default="best.pt")
    args = ap.parse_args()

    C.BATCH_SIZE = args.batch_size
    C.NUM_WORKERS = args.workers
    C.ensure_dirs()
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = C.AMP and device.type == "cuda"
    print(f"Device: {device} | AMP: {amp}")

    train_ds, val_ds, train_loader, val_loader = build_loaders()
    print(f"train {len(train_ds)} | val {len(val_ds)}")
    print("train class counts:",
          np.bincount(np.array(train_ds.labels), minlength=C.NUM_CLASSES).tolist())

    model = AlzheimerNet(pretrained=True).to(device)
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"warm-started from {args.resume_from}")
    model = model.to(memory_format=torch.channels_last)

    weights = class_weights(train_ds.labels, C.NUM_CLASSES).to(device)
    print("class weights:", [round(w, 3) for w in weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=C.LABEL_SMOOTHING)

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict] = []
    best = {"macro_f1": -1.0, "epoch": -1}
    ckpt_path = C.CHECKPOINT_DIR / args.out
    started = time.time()
    history_path = C.REPORTS_DIR / (
        "training_history.json" if args.out == "best.pt"
        else f"training_history_{Path(args.out).stem}.json")

    def flush_history(done: bool = False) -> None:
        """Write after every epoch, not just at the end.

        A run killed at epoch 19 of 25 otherwise leaves no curve at all, which
        is exactly when you most want to see what it was doing.
        """
        history_path.write_text(json.dumps({
            "history": history, "best": best, "complete": done,
            "elapsed_sec": time.time() - started,
            "train_size": len(train_ds), "val_size": len(val_ds),
            "config": {
                "batch_size": args.batch_size, "workers": args.workers,
                "epochs_head": args.epochs_head,
                "epochs_finetune": args.epochs_finetune,
                "lr_head": C.LR_HEAD, "lr_backbone": C.LR_BACKBONE,
                "weight_decay": C.WEIGHT_DECAY,
                "label_smoothing": C.LABEL_SMOOTHING,
                "img_size": C.IMG_SIZE, "seed": C.SEED,
            },
        }, indent=2), encoding="utf-8")

    # ---------------- stage 1: frozen backbone ----------------
    if args.epochs_head > 0:
        print("\n=== Stage 1: head warm-up (backbone frozen) ===")
        model.set_backbone_trainable(False)
        opt = torch.optim.AdamW(model.classifier.parameters(), lr=C.LR_HEAD,
                                weight_decay=C.WEIGHT_DECAY)
        for ep in range(args.epochs_head):
            tr = run_epoch(model, train_loader, criterion, opt, scaler, device,
                           amp, desc=f"head {ep+1}/{args.epochs_head}")
            va = evaluate(model, val_loader, criterion, device, amp)
            history.append({"stage": "head", "epoch": ep + 1, **
                            {f"train_{k}": v for k, v in tr.items()},
                            **{f"val_{k}": v for k, v in va.items()}})
            print(f"[head {ep+1}/{args.epochs_head}] "
                  f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['acc']:.4f} "
                  f"macroF1 {va['macro_f1']:.4f}")
            flush_history()

    # ---------------- stage 2: full fine-tune ----------------
    print("\n=== Stage 2: full fine-tune ===")
    model.set_backbone_trainable(True)
    opt = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": C.LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": C.LR_HEAD * 0.3},
    ], weight_decay=C.WEIGHT_DECAY)

    steps = max(1, len(train_loader)) * max(1, args.epochs_finetune)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[C.LR_BACKBONE, C.LR_HEAD * 0.3], total_steps=steps,
        pct_start=0.15, div_factor=10.0, final_div_factor=100.0,
    )

    stale = 0
    for ep in range(args.epochs_finetune):
        tr = run_epoch(model, train_loader, criterion, opt, scaler, device, amp,
                       sched, desc=f"ft {ep+1}/{args.epochs_finetune}")
        va = evaluate(model, val_loader, criterion, device, amp)
        history.append({"stage": "finetune", "epoch": ep + 1,
                        "lr": sched.get_last_lr()[0],
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        flag = ""
        if va["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": va["macro_f1"], "acc": va["acc"], "epoch": ep + 1}
            torch.save({"model": model.state_dict(), "classes": C.CLASS_DIRS,
                        "labels": C.CLASS_LABELS, "img_size": C.IMG_SIZE,
                        "val": va, "epoch": ep + 1}, ckpt_path)
            stale = 0
            flag = "  <- best"
        else:
            stale += 1

        print(f"[ft {ep+1}/{args.epochs_finetune}] "
              f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} "
              f"macroF1 {va['macro_f1']:.4f}{flag}")
        flush_history()

        if stale >= C.EARLY_STOP_PATIENCE:
            print(f"early stop: no val macro-F1 gain in {stale} epochs")
            break

    elapsed = time.time() - started
    print(f"\nBest val macro-F1 {best['macro_f1']:.4f} @ epoch {best['epoch']} "
          f"({elapsed/60:.1f} min)")
    print(f"checkpoint -> {ckpt_path}")
    flush_history(done=True)
    print(f"history    -> {history_path}")


if __name__ == "__main__":
    main()
