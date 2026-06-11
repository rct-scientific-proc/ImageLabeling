"""Right-side label management panel.

Exposes:
    active_label_index  – currently selected class index (or None)
    sig_add_label       – emitted with (str) when user adds a new label
    sig_remove_label    – emitted with (int) index when user requests removal
    sig_selection_changed – emitted with (int | None)
"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LabelPanel(QWidget):
    sig_add_label = pyqtSignal(str)
    sig_remove_label = pyqtSignal(int)
    sig_selection_changed = pyqtSignal(object)   # int | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._classes: list[str] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_label_index(self) -> int | None:
        row = self._list.currentRow()
        return row if row >= 0 else None

    def set_classes(self, classes: list[str]) -> None:
        """Replace the displayed class list (preserves selection if possible)."""
        prev_row = self._list.currentRow()
        self._classes = list(classes)
        self._list.blockSignals(True)
        self._list.clear()
        for name in self._classes:
            self._list.addItem(QListWidgetItem(name))
        if 0 <= prev_row < self._list.count():
            self._list.setCurrentRow(prev_row)
        self._list.blockSignals(False)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        box = QGroupBox("Labels")
        inner = QVBoxLayout(box)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selection_changed)
        inner.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add")
        self._btn_remove = QPushButton("Remove")
        self._btn_add.clicked.connect(self._on_add)
        self._btn_remove.clicked.connect(self._on_remove)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_remove)
        inner.addLayout(btn_row)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

    def _on_selection_changed(self, row: int) -> None:
        self.sig_selection_changed.emit(row if row >= 0 else None)

    def _on_add(self) -> None:
        text, ok = QInputDialog.getText(self, "Add Label", "Label name:")
        if not ok:
            return
        text = text.strip()
        if not text:
            return
        if text == "hard_negative":
            QMessageBox.warning(
                self, "Reserved", '"hard_negative" is reserved and cannot be added.'
            )
            return
        if text in self._classes:
            QMessageBox.warning(self, "Duplicate", f'Label "{text}" already exists.')
            return
        self.sig_add_label.emit(text)

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            QMessageBox.information(self, "Remove Label", "Select a label to remove.")
            return
        name = self._classes[row]
        if name == "hard_negative":
            QMessageBox.warning(
                self, "Reserved", '"hard_negative" cannot be removed.'
            )
            return
        reply = QMessageBox.question(
            self,
            "Remove Label",
            f'Remove "{name}"? This will fail if any images are assigned to it.',
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.sig_remove_label.emit(row)
