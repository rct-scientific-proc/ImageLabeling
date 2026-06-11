"""QThread-based batch inference worker.

Runs the latest checkpoint over all *unlabeled* images and emits per-image
softmax probability vectors.  The calling code stores them in memory for
use by the sort-by-confidence grid control.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from torch.utils.data import DataLoader, Dataset

from src.h5io import UNLABELED
from src.training.model import load_checkpoint

# ImageNet normalisation — must match training
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Unlabeled dataset (index-aware, no label needed)
# ---------------------------------------------------------------------------

class _UnlabeledDataset(Dataset):
    def __init__(self, h5_path: Path) -> None:
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            labels = f["labels"][:]
        self._indices: np.ndarray = np.where(labels == UNLABELED)[0]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        ds_idx = int(self._indices[i])
        with h5py.File(self.h5_path, "r") as f:
            img = f["images"][ds_idx]
        x = img.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(x)), ds_idx


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class InferenceWorker(QThread):
    """Runs softmax inference over every unlabeled image.

    Signals
    -------
    sig_progress(int current, int total)
    sig_finished(dict)
        ``{dataset_index: np.ndarray(num_classes)}`` — softmax probabilities.
    sig_error(str)
    """

    sig_progress = pyqtSignal(int, int)
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(
        self,
        h5_path: Path,
        num_classes: int,
        checkpoint_path: Path,
        batch_size: int = 256,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.h5_path = Path(h5_path)
        self.num_classes = num_classes
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size

    def run(self) -> None:
        try:
            self._infer()
        except Exception as exc:  # noqa: BLE001
            self.sig_error.emit(str(exc))

    def _infer(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_checkpoint(self.checkpoint_path, self.num_classes, device=device)
        if model is None:
            self.sig_error.emit(
                "No valid checkpoint found. Train the model first."
            )
            return

        model.to(device)
        model.eval()

        dataset = _UnlabeledDataset(self.h5_path)
        if len(dataset) == 0:
            self.sig_finished.emit({})
            return

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        scores: dict[int, np.ndarray] = {}
        processed = 0

        with torch.no_grad():
            for imgs, ds_indices in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for i, ds_idx in enumerate(ds_indices.tolist()):
                    scores[int(ds_idx)] = probs[i]
                processed += len(ds_indices)
                self.sig_progress.emit(processed, len(dataset))

        self.sig_finished.emit(scores)
