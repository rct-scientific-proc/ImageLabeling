"""Main application window."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gui.label_panel import LabelPanel
from src.gui.training_panel import TrainingPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Labeling Tool")
        self.resize(1280, 800)

        self._h5_path: Path | None = None
        self._show_labeled: bool = False   # False = unlabeled, True = labeled

        self._build_menu()
        self._build_central()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Directory…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_directory)
        file_menu.addAction(open_action)

        open_h5_action = QAction("Open H5 File…", self)
        open_h5_action.setShortcut("Ctrl+Shift+O")
        open_h5_action.triggered.connect(self._on_open_h5)
        file_menu.addAction(open_h5_action)

        # View menu
        view_menu = menubar.addMenu("View")

        self._toggle_view_action = QAction("Show Labeled Images", self)
        self._toggle_view_action.setCheckable(True)
        self._toggle_view_action.setChecked(False)
        self._toggle_view_action.triggered.connect(self._on_toggle_view)
        view_menu.addAction(self._toggle_view_action)

    # ------------------------------------------------------------------
    # Central widget layout
    # ------------------------------------------------------------------

    def _build_central(self) -> None:
        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # ---- Left / centre column --------------------------------
        centre_col = QVBoxLayout()
        centre_col.setSpacing(6)

        # Image grid placeholder (replaced in Phase 4)
        self._grid_placeholder = QLabel("Image grid — Phase 4")
        self._grid_placeholder.setAlignment(Qt.AlignCenter)
        self._grid_placeholder.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self._grid_placeholder.setStyleSheet(
            "border: 1px dashed #888; color: #888; font-size: 14px;"
        )
        centre_col.addWidget(self._grid_placeholder, stretch=1)

        # Action buttons below the grid
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self._btn_add_labels = QPushButton("Add Labels")
        self._btn_add_labels.setEnabled(False)
        self._btn_add_labels.clicked.connect(self._on_add_labels)

        self._btn_hard_negative = QPushButton("Label All as Hard Negative")
        self._btn_hard_negative.setEnabled(False)
        self._btn_hard_negative.clicked.connect(self._on_label_hard_negative)

        action_row.addWidget(self._btn_add_labels)
        action_row.addWidget(self._btn_hard_negative)
        action_row.addStretch()
        centre_col.addLayout(action_row)

        root_layout.addLayout(centre_col, stretch=1)

        # ---- Right column ----------------------------------------
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        self._label_panel = LabelPanel()
        self._label_panel.setFixedWidth(220)
        self._label_panel.sig_add_label.connect(self._on_label_added)
        self._label_panel.sig_remove_label.connect(self._on_label_removed)
        self._label_panel.sig_selection_changed.connect(self._on_label_selection_changed)
        right_col.addWidget(self._label_panel)

        self._training_panel = TrainingPanel()
        self._training_panel.setFixedWidth(220)
        self._training_panel.sig_start_training.connect(self._on_start_training)
        self._training_panel.sig_stop_training.connect(self._on_stop_training)
        right_col.addWidget(self._training_panel)

        right_col.addStretch()
        root_layout.addLayout(right_col)

        self.setCentralWidget(central)

        # Status bar
        self.statusBar().showMessage("No file open")

    # ------------------------------------------------------------------
    # Slots — File menu
    # ------------------------------------------------------------------

    def _on_open_directory(self) -> None:
        dir_str = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not dir_str:
            return

        h5_path, _ = QFileDialog.getSaveFileName(
            self, "Save H5 Dataset As", dir_str, "HDF5 files (*.h5)"
        )
        if not h5_path:
            return

        size_str, ok = QInputDialog.getText(
            self,
            "Image Size",
            "Target image size (H W), e.g. 64 64:",
            text="64 64",
        )
        if not ok:
            return
        try:
            parts = size_str.strip().split()
            image_size = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Invalid Input", "Enter two integers, e.g. 64 64")
            return

        self._run_ingest(Path(dir_str), Path(h5_path), image_size)

    def _on_open_h5(self) -> None:
        h5_str, _ = QFileDialog.getOpenFileName(
            self, "Open H5 Dataset", "", "HDF5 files (*.h5)"
        )
        if not h5_str:
            return
        self._load_h5(Path(h5_str))

    def _run_ingest(
        self, dir_path: Path, h5_path: Path, image_size: tuple[int, int]
    ) -> None:
        from src.ingest import ingest_directory
        from src.scanner import scan_directory

        try:
            paths = scan_directory(dir_path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        if not paths:
            QMessageBox.warning(self, "No Images", "No supported images found.")
            return

        progress = QProgressDialog(
            "Ingesting images…", "Cancel", 0, len(paths), self
        )
        progress.setWindowTitle("Ingesting")
        progress.setModal(True)
        progress.show()

        def _cb(current: int, total: int) -> None:
            progress.setValue(current)

        try:
            n = ingest_directory(dir_path, h5_path, image_size, progress_callback=_cb)
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Ingest Error", str(exc))
            return

        progress.close()
        self._load_h5(h5_path)
        self.statusBar().showMessage(f"Ingested {n} images → {h5_path.name}")

    def _load_h5(self, h5_path: Path) -> None:
        from src.h5io import get_classes

        self._h5_path = h5_path
        classes = get_classes(h5_path)
        self._label_panel.set_classes(classes)
        self._btn_hard_negative.setEnabled(True)
        self.setWindowTitle(f"Image Labeling Tool — {h5_path.name}")
        self.statusBar().showMessage(f"Opened: {h5_path}")

    # ------------------------------------------------------------------
    # Slots — View menu
    # ------------------------------------------------------------------

    def _on_toggle_view(self, checked: bool) -> None:
        self._show_labeled = checked
        label = "Show Unlabeled Images" if checked else "Show Labeled Images"
        self._toggle_view_action.setText(label)
        # Grid will be refreshed in Phase 4
        self.statusBar().showMessage(
            "Showing labeled images" if checked else "Showing unlabeled images"
        )

    # ------------------------------------------------------------------
    # Slots — Label panel
    # ------------------------------------------------------------------

    def _on_label_added(self, name: str) -> None:
        if self._h5_path is None:
            return
        from src.h5io import get_classes, update_classes

        classes = get_classes(self._h5_path)
        # Insert before hard_negative (always last)
        classes.insert(len(classes) - 1, name)
        update_classes(self._h5_path, classes)
        self._label_panel.set_classes(classes)

    def _on_label_removed(self, row: int) -> None:
        if self._h5_path is None:
            return
        from src.h5io import get_classes, get_n_images, update_classes
        import h5py
        import numpy as np
        from src.h5io import UNLABELED

        classes = get_classes(self._h5_path)
        target_name = classes[row]

        # Block removal if any image is assigned to this class
        with h5py.File(self._h5_path, "r") as f:
            labels = f["labels"][:]
        if np.any(labels == row):
            QMessageBox.warning(
                self,
                "Cannot Remove",
                f'"{target_name}" is assigned to one or more images.',
            )
            return

        classes.pop(row)
        update_classes(self._h5_path, classes)
        self._label_panel.set_classes(classes)

    def _on_label_selection_changed(self, index: object) -> None:
        has_selection = index is not None
        self._btn_add_labels.setEnabled(has_selection and self._h5_path is not None)

    # ------------------------------------------------------------------
    # Slots — Action buttons (stubs, fully wired in Phase 6 / 7)
    # ------------------------------------------------------------------

    def _on_add_labels(self) -> None:
        # Phase 6 will implement pending-label commit logic
        pass

    def _on_label_hard_negative(self) -> None:
        # Phase 7 will implement bulk hard-negative assignment
        pass

    # ------------------------------------------------------------------
    # Slots — Training panel (stubs, wired in Phase 8)
    # ------------------------------------------------------------------

    def _on_start_training(self, config: dict) -> None:
        pass

    def _on_stop_training(self) -> None:
        pass
