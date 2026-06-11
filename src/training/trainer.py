"""QThread-based training worker so the UI stays responsive during training."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PyQt5.QtCore import QThread, pyqtSignal
from torch.utils.data import DataLoader, random_split

from src.h5io import get_classes
from src.training.dataset import H5LabeledDataset
from src.training.model import build_model, load_checkpoint, save_checkpoint

DEFAULT_CHECKPOINT = Path("assets/checkpoint.pt")


class TrainingWorker(QThread):
    """Runs ResNet18 fine-tuning in a background thread.

    Signals
    -------
    sig_progress(int current, int total, str status)
        Emitted after every epoch (and on key milestones).
    sig_finished(str message)
        Emitted once when training stops (completed, target reached, or aborted).
    sig_error(str message)
        Emitted if an exception occurs.
    """

    sig_progress = pyqtSignal(int, int, str)
    sig_finished = pyqtSignal(str)
    sig_error = pyqtSignal(str)

    def __init__(
        self,
        h5_path: Path,
        epochs: int,
        batch_size: int,
        target_metric: float,
        val_split: float = 0.20,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        learning_rate: float = 1e-4,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.h5_path = Path(h5_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_metric = target_metric
        self.val_split = max(0.0, min(val_split, 0.9))
        self.checkpoint_path = Path(checkpoint_path)
        self.learning_rate = learning_rate
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        self._stop_requested = True

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 - training loop
        try:
            self._train()
        except Exception as exc:  # noqa: BLE001
            self.sig_error.emit(str(exc))

    def _train(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classes = get_classes(self.h5_path)
        num_classes = len(classes)

        dataset = H5LabeledDataset(self.h5_path)
        if len(dataset) == 0:
            self.sig_finished.emit("No labeled images available to train on.")
            return

        # Train / val split
        val_count = int(len(dataset) * self.val_split)
        train_count = len(dataset) - val_count
        if val_count == 0:
            train_ds, val_ds = dataset, None
        else:
            train_ds, val_ds = random_split(
                dataset,
                [train_count, val_count],
                generator=torch.Generator().manual_seed(42),
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = (
            DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
            if val_ds is not None
            else None
        )

        # Resume from checkpoint when class count still matches, else start fresh
        model = load_checkpoint(self.checkpoint_path, num_classes, device=device)
        resumed = model is not None
        if model is None:
            model = build_model(num_classes, pretrained=True)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        start_msg = "Resumed from checkpoint" if resumed else "Started fresh"
        split_info = f"{train_count} train / {val_count} val" if val_count else f"{len(dataset)} train (no val)"
        self.sig_progress.emit(0, self.epochs, f"{start_msg} — {split_info}")

        for epoch in range(1, self.epochs + 1):
            if self._stop_requested:
                self.sig_finished.emit(f"Training stopped at epoch {epoch - 1}.")
                return

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                if self._stop_requested:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            train_acc = correct / total if total else 0.0
            train_loss = running_loss / total if total else 0.0

            # Validation pass
            if val_loader is not None:
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        preds = model(inputs).argmax(dim=1)
                        val_correct += (preds == targets).sum().item()
                        val_total += targets.size(0)
                val_acc = val_correct / val_total if val_total else 0.0
                status = (
                    f"Epoch {epoch}/{self.epochs} — "
                    f"loss {train_loss:.3f}, train {train_acc:.2%}, val {val_acc:.2%}"
                )
                metric = val_acc
            else:
                val_acc = None
                status = (
                    f"Epoch {epoch}/{self.epochs} — "
                    f"loss {train_loss:.3f}, acc {train_acc:.2%}"
                )
                metric = train_acc

            # Save a checkpoint after every epoch
            save_checkpoint(
                self.checkpoint_path, model, num_classes, epoch, classes
            )

            self.sig_progress.emit(epoch, self.epochs, status)

            if metric >= self.target_metric:
                self.sig_finished.emit(
                    f"Target val acc {self.target_metric:.2%} reached at epoch {epoch}."
                )
                return

        self.sig_finished.emit(f"Training complete ({self.epochs} epochs).")
