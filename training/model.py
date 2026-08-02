"""CAM-friendly EfficientNet-B0 classifier.

The head is deliberately kept as `features -> global average pool -> single
Linear`. That structure makes the *classic* Class Activation Map exact:

    CAM_c(x, y) = sum_k  W[c, k] * A_k(x, y)

i.e. it needs only the final feature map and the classifier weight matrix — no
backward pass. That is what lets the deployed serverless function produce a
real explanation heatmap with ONNX Runtime alone, without shipping PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from config import NUM_CLASSES

FEATURE_DIM = 1280
FEATURE_GRID = 7  # 224 / 32


class AlzheimerNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features          # -> (B, 1280, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.classifier.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.features(x)
        v = torch.flatten(self.pool(fmap), 1)
        return self.classifier(self.dropout(v))

    def forward_with_features(self, x: torch.Tensor):
        """Returns (logits, feature_map) — used by the ONNX export."""
        fmap = self.features(x)
        v = torch.flatten(self.pool(fmap), 1)
        return self.classifier(v), fmap

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised pooled embedding, used for near-duplicate detection."""
        v = torch.flatten(self.pool(self.features(x)), 1)
        return torch.nn.functional.normalize(v, dim=1)

    def head_parameters(self):
        return self.classifier.parameters()

    def backbone_parameters(self):
        return self.features.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = trainable


class ExportWrapper(nn.Module):
    """Traced for ONNX. Emits logits *and* the raw feature map so the
    serverless runtime can build the CAM itself."""

    def __init__(self, model: AlzheimerNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        logits, fmap = self.model.forward_with_features(x)
        return logits, fmap
