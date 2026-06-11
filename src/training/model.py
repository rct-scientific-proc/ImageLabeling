"""ResNet18 model construction and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a ResNet18 with its final FC layer replaced for *num_classes*."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    num_classes: int,
    epoch: int,
    classes: list[str],
) -> None:
    """Persist model weights + metadata to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "epoch": epoch,
            "classes": classes,
        },
        path,
    )


def load_checkpoint(path: Path, num_classes: int, device: str = "cpu") -> nn.Module | None:
    """Load a checkpoint into a fresh model.

    Returns ``None`` if the checkpoint is missing or its class count no longer
    matches (e.g. the user added/removed labels since the last run).
    """
    path = Path(path)
    if not path.exists():
        return None

    ckpt = torch.load(path, map_location=device)
    if ckpt.get("num_classes") != num_classes:
        return None

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model
