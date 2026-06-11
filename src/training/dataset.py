"""PyTorch Dataset backed by the project's H5 file.

Only entries with ``gt == True`` and a valid (non-UNLABELED) label are exposed,
so the dataset contains genuine, labeled training examples.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.h5io import UNLABELED

# ImageNet normalisation (matches the pre-trained ResNet18 weights)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class H5LabeledDataset(Dataset):
    """Reads genuine labeled samples (``gt=True``) from an H5 file.

    Optionally restrict to a particular split (0=train, 1=val, 2=test).
    """

    def __init__(self, h5_path: Path, split: int | None = None) -> None:
        self.h5_path = Path(h5_path)

        with h5py.File(self.h5_path, "r") as f:
            labels = f["labels"][:]
            gt = f["gt"][:]
            split_arr = f["split"][:]

        mask = (labels != UNLABELED) & (gt == True)  # noqa: E712
        if split is not None:
            mask &= split_arr == split

        self._indices: np.ndarray = np.where(mask)[0]
        self._labels: np.ndarray = labels[self._indices].astype(np.int64)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        ds_idx = int(self._indices[i])
        # Open per-access so the dataset is safe across DataLoader workers
        with h5py.File(self.h5_path, "r") as f:
            img = f["images"][ds_idx]   # (H, W, 3) uint8

        x = img.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = np.transpose(x, (2, 0, 1))   # → (3, H, W)
        tensor = torch.from_numpy(np.ascontiguousarray(x))
        label = int(self._labels[i])
        return tensor, label

    @property
    def labels(self) -> np.ndarray:
        return self._labels
