"""Label statistics panel.

Displays a live count of labeled images per class, plus total unlabeled.
Call ``refresh(h5_path)`` whenever the H5 file changes.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Palette cycling for class rows (background tints) — also used by image grid overlays
CLASS_COLOURS: list[str] = [
    "#d4e8ff",  # blue
    "#d4f5d4",  # green
    "#fff4cc",  # yellow
    "#ffdcd4",  # red/salmon
    "#ead4ff",  # purple
    "#d4f5f5",  # cyan
    "#ffe8cc",  # orange
]

HARD_NEGATIVE_COLOUR: str = "#e8e8e8"   # neutral grey for hard_negative

# Keep old private names as aliases for backward compat
_ROW_COLOURS = CLASS_COLOURS
_HN_COLOUR = HARD_NEGATIVE_COLOUR


class _StatRow(QWidget):
    """A single class row: coloured bar + label name + count badge."""

    def __init__(self, colour: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colour = colour
        self.setAutoFillBackground(True)
        self._apply_colour(colour)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        self._name_lbl = QLabel()
        self._name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._count_lbl = QLabel()
        self._count_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._count_lbl.setMinimumWidth(36)
        self._count_lbl.setStyleSheet(
            "background: rgba(0,0,0,0.12); border-radius: 8px;"
            "padding: 1px 6px; font-weight: bold;"
        )

        layout.addWidget(self._name_lbl)
        layout.addWidget(self._count_lbl)

    def update_data(self, name: str, count: int) -> None:
        self._name_lbl.setText(name)
        self._count_lbl.setText(str(count))

    def _apply_colour(self, hex_colour: str) -> None:
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(hex_colour))
        self.setPalette(palette)


class LabelStatsPanel(QWidget):
    """Shows per-class labeled image counts, plus an unlabeled total."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, h5_path: Path | None) -> None:
        """Read counts from *h5_path* and update the display."""
        # Clear existing rows
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if h5_path is None:
            self._total_lbl.setText("")
            return

        import h5py
        import numpy as np
        from src.h5io import UNLABELED, get_classes

        with h5py.File(h5_path, "r") as f:
            labels = f["labels"][:]

        classes = get_classes(h5_path)
        total = len(labels)
        unlabeled_count = int(np.sum(labels == UNLABELED))

        for class_idx, name in enumerate(classes):
            count = int(np.sum(labels == class_idx))
            if name == "hard_negative":
                colour = HARD_NEGATIVE_COLOUR
            else:
                colour = CLASS_COLOURS[class_idx % len(CLASS_COLOURS)]

            row = _StatRow(colour)
            row.update_data(name, count)
            self._rows_layout.addWidget(row)

        self._total_lbl.setText(
            f"Unlabeled: {unlabeled_count}  /  Total: {total}"
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        box = QGroupBox("Label Counts")
        inner = QVBoxLayout(box)
        inner.setSpacing(2)
        inner.setContentsMargins(4, 6, 4, 6)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(2)
        inner.addLayout(self._rows_layout)

        self._total_lbl = QLabel()
        self._total_lbl.setAlignment(Qt.AlignRight)
        self._total_lbl.setStyleSheet("color: #555; font-size: 11px; padding-top: 4px;")
        inner.addWidget(self._total_lbl)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)
