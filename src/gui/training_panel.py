"""Collapsible training configuration + status panel."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrainingPanel(QWidget):
    sig_start_training = pyqtSignal(dict)   # emits config dict
    sig_stop_training = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return {
            "epochs": self._epochs.value(),
            "batch_size": self._batch_size.value(),
            "target_metric": self._target_metric.value(),
            "val_split": self._val_split.value(),
            "inference_batch_size": self._inference_batch_size.value(),
        }

    def set_progress(self, value: int, maximum: int, status: str = "") -> None:
        self._progress.setMaximum(maximum)
        self._progress.setValue(value)
        if status:
            self._status_label.setText(status)

    def set_training_active(self, active: bool) -> None:
        self._btn_train.setEnabled(not active)
        self._btn_stop.setEnabled(active)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # --- config group (collapsible via toggle) ---
        self._config_box = QGroupBox("Training Configuration")
        self._config_box.setCheckable(True)
        self._config_box.setChecked(False)   # collapsed by default
        form = QFormLayout(self._config_box)

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 1000)
        self._epochs.setValue(10)
        form.addRow("Epochs:", self._epochs)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 1024)
        self._batch_size.setValue(32)
        form.addRow("Batch size:", self._batch_size)

        self._target_metric = QDoubleSpinBox()
        self._target_metric.setRange(0.0, 1.0)
        self._target_metric.setSingleStep(0.01)
        self._target_metric.setValue(0.90)
        self._target_metric.setDecimals(2)
        form.addRow("Target val acc:", self._target_metric)

        self._val_split = QDoubleSpinBox()
        self._val_split.setRange(0.0, 0.9)
        self._val_split.setSingleStep(0.05)
        self._val_split.setValue(0.20)
        self._val_split.setDecimals(2)
        form.addRow("Val split:", self._val_split)

        self._inference_batch_size = QSpinBox()
        self._inference_batch_size.setRange(1, 1024)
        self._inference_batch_size.setValue(64)
        form.addRow("Inference batch:", self._inference_batch_size)

        self._btn_train = QPushButton("Start Training")
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_train.clicked.connect(
            lambda: self.sig_start_training.emit(self.get_config())
        )
        self._btn_stop.clicked.connect(self.sig_stop_training)
        form.addRow(self._btn_train, self._btn_stop)

        outer.addWidget(self._config_box)

        # --- status group ---
        status_box = QGroupBox("Training Status")
        status_layout = QVBoxLayout(status_box)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        status_layout.addWidget(self._progress)

        self._status_label = QLabel("Idle")
        status_layout.addWidget(self._status_label)

        outer.addWidget(status_box)
