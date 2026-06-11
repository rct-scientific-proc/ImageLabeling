"""QThread-based training worker so the UI stays responsive during training."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PyQt5.QtCore import QThread, pyqtSignal
from torch.utils.data import DataLoader

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
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        learning_rate: float = 1e-4,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.h5_path = Path(h5_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_metric = target_metric
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

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,   # H5 + workers can be fragile on Windows; keep simple
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
        self.sig_progress.emit(0, self.epochs, f"{start_msg} — {len(dataset)} samples")

        for epoch in range(1, self.epochs + 1):
            if self._stop_requested:
                self.sig_finished.emit(f"Training stopped at epoch {epoch - 1}.")
                return

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in loader:
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

            epoch_acc = correct / total if total else 0.0
            epoch_loss = running_loss / total if total else 0.0

            # Save a checkpoint after every epoch
            save_checkpoint(
                self.checkpoint_path, model, num_classes, epoch, classes
            )

            self.sig_progress.emit(
                epoch,
                self.epochs,
                f"Epoch {epoch}/{self.epochs} — loss {epoch_loss:.3f}, acc {epoch_acc:.2%}",
            )

            if epoch_acc >= self.target_metric:
                self.sig_finished.emit(
                    f"Target accuracy {self.target_metric:.2%} reached at epoch {epoch}."
                )
                return

        self.sig_finished.emit(f"Training complete ({self.epochs} epochs).")
