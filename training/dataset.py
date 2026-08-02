"""Dataset + transform helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SPLITS_DIR


def eval_transform():
    """Deterministic path. Mirrored exactly in web/api/_inference.py."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform():
    """Mild, anatomy-preserving augmentation.

    No vertical flip and no large rotation: a brain MRI slice has a canonical
    orientation, and destroying it would teach the model to ignore
    left/right-asymmetric atrophy patterns that actually matter.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.10)),
    ])


class ScanDataset(Dataset):
    """Reads a list of (path, label) records produced by prepare_split.py."""

    def __init__(self, records: Sequence[dict], transform=None):
        self.records = list(records)
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        return self.transform(img), int(rec["label"])

    @property
    def labels(self) -> list[int]:
        return [int(r["label"]) for r in self.records]


class ImagePathDataset(Dataset):
    """Bare image loader used by the leak-detection embedding pass."""

    def __init__(self, paths: Sequence[Path]):
        self.paths = [str(p) for p in paths]
        self.transform = eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


def load_split(name: str) -> list[dict]:
    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split '{name}' not found at {path}. Run prepare_split.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for lab in labels:
        counts[int(lab)] += 1
    counts = counts.clamp(min=1.0)
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()
