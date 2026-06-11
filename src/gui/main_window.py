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

from src.gui.image_grid import ImageGridWidget
from src.gui.label_panel import LabelPanel
from src.gui.label_stats_panel import LabelStatsPanel
from src.gui.training_panel import TrainingPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Labeling Tool")
        self.resize(1280, 800)

        self._h5_path: Path | None = None
        self._show_labeled: bool = False   # False = unlabeled, True = labeled
        # Maps dataset index → class index for images clicked but not yet committed
        self._pending_assignments: dict[int, int] = {}

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

        self._image_grid = ImageGridWidget()
        self._image_grid.sig_cell_clicked.connect(self._on_cell_clicked)
        self._image_grid.sig_cells_range_selected.connect(self._on_cells_range_selected)
        centre_col.addWidget(self._image_grid, stretch=1)

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

        self._stats_panel = LabelStatsPanel()
        self._stats_panel.setFixedWidth(220)
        right_col.addWidget(self._stats_panel)

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
        self._refresh_grid()

    def _refresh_grid(self) -> None:
        """Load the current view (labeled / unlabeled) into the image grid."""
        if self._h5_path is None:
            return

        import h5py
        import numpy as np
        from PyQt5.QtGui import QColor
        from src.h5io import UNLABELED, get_classes
        from src.gui.label_stats_panel import CLASS_COLOURS, HARD_NEGATIVE_COLOUR

        with h5py.File(self._h5_path, "r") as f:
            images = f["images"][:]
            labels = f["labels"][:]
            filenames = list(f["filenames"].asstr()[:])

        classes = get_classes(self._h5_path)

        # Build a QColor for each class index (used by the image grid overlays)
        def _class_colour(class_idx: int) -> QColor:
            name = classes[class_idx]
            hex_col = HARD_NEGATIVE_COLOUR if name == "hard_negative" else CLASS_COLOURS[class_idx % len(CLASS_COLOURS)]
            c = QColor(hex_col)
            c.setAlpha(130)
            return c

        if self._show_labeled:
            mask = labels != UNLABELED
            label_names = {
                int(i): classes[int(labels[i])]
                for i in np.where(mask)[0]
            }
            label_colours = {
                int(i): _class_colour(int(labels[i]))
                for i in np.where(mask)[0]
            }
        else:
            mask = labels == UNLABELED
            label_names = {}
            label_colours = {}

        indices = list(np.where(mask)[0].astype(int))
        self._image_grid.load_images(
            indices=indices,
            images=images[mask],
            filenames=[filenames[i] for i in indices],
            pending=set(self._pending_assignments.keys()),
            label_names=label_names,
            label_colours=label_colours,
        )
        n = len(indices)
        kind = "labeled" if self._show_labeled else "unlabeled"
        self.statusBar().showMessage(
            f"{self._h5_path.name} — {n} {kind} image(s)"
        )
        self._stats_panel.refresh(self._h5_path)

    # ------------------------------------------------------------------
    # Slots — View menu
    # ------------------------------------------------------------------

    def _on_toggle_view(self, checked: bool) -> None:
        self._show_labeled = checked
        self._toggle_view_action.setText(
            "Show Unlabeled Images" if checked else "Show Labeled Images"
        )
        self._refresh_grid()

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
        from src.h5io import (
            get_classes,
            remap_labels_after_removal,
            update_classes,
            UNLABELED,
        )
        import h5py
        import numpy as np

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
        # Fix up any label indices that shifted down due to the removal
        remap_labels_after_removal(self._h5_path, row)
        self._label_panel.set_classes(classes)

    def _on_label_selection_changed(self, index: object) -> None:
        self._update_add_labels_button()

    # ------------------------------------------------------------------
    # Slots — Cell click
    # ------------------------------------------------------------------

    def _on_cell_clicked(self, dataset_index: int) -> None:
        """Assign the active label to the clicked image as a pending change."""
        if self._show_labeled:
            return

        active = self._label_panel.active_label_index
        if active is None:
            self.statusBar().showMessage("Select a label from the panel first.")
            return

        if dataset_index in self._pending_assignments:
            # Second click on a pending image cancels the assignment
            del self._pending_assignments[dataset_index]
            self._image_grid.mark_pending(dataset_index, False)
        else:
            self._pending_assignments[dataset_index] = active
            self._image_grid.mark_pending(dataset_index, True)

        self._update_add_labels_button()

    def _on_cells_range_selected(self, dataset_indices: list) -> None:
        """Assign the active label to all images in a shift-click range."""
        if self._show_labeled:
            return

        active = self._label_panel.active_label_index
        if active is None:
            self.statusBar().showMessage("Select a label from the panel first.")
            return

        for dataset_index in dataset_indices:
            self._pending_assignments[dataset_index] = active
            self._image_grid.mark_pending(dataset_index, True)

        self._update_add_labels_button()
        self.statusBar().showMessage(
            f"{len(dataset_indices)} image(s) marked as pending."
        )

    # ------------------------------------------------------------------
    # Slots — Action buttons (stubs, fully wired in Phase 6 / 7)
    # ------------------------------------------------------------------

    def _on_add_labels(self) -> None:
        if not self._pending_assignments or self._h5_path is None:
            return
        from src.h5io import update_labels, update_gt

        indices = list(self._pending_assignments.keys())
        label_values = list(self._pending_assignments.values())
        update_labels(self._h5_path, indices, label_values)
        update_gt(self._h5_path, indices, [True] * len(indices))

        committed = len(indices)
        self._pending_assignments.clear()
        self._refresh_grid()
        self._update_add_labels_button()
        self.statusBar().showMessage(
            f"Committed {committed} label(s) to {self._h5_path.name}"
        )

    def _on_label_hard_negative(self) -> None:
        """Assign hard_negative + gt=False to every unlabeled image in the current grid view."""
        if self._h5_path is None:
            return

        from src.h5io import get_classes, update_gt, update_labels

        # Only operate on the images visible on the current page, not the whole dataset
        unlabeled_indices = list(self._image_grid.current_page_indices)
        if not unlabeled_indices:
            self.statusBar().showMessage("No unlabeled images to assign.")
            return

        classes = get_classes(self._h5_path)
        hard_neg_idx = classes.index("hard_negative")

        update_labels(self._h5_path, unlabeled_indices, [hard_neg_idx] * len(unlabeled_indices))
        update_gt(self._h5_path, unlabeled_indices, [False] * len(unlabeled_indices))

        # Clear any pending assignments that were just committed
        for idx in unlabeled_indices:
            self._pending_assignments.pop(idx, None)

        self._refresh_grid()
        self._update_add_labels_button()
        self.statusBar().showMessage(
            f"Labeled {len(unlabeled_indices)} image(s) as hard_negative."
        )

    def _update_add_labels_button(self) -> None:
        enabled = (
            self._h5_path is not None
            and not self._show_labeled
            and bool(self._pending_assignments)
            and self._label_panel.active_label_index is not None
        )
        self._btn_add_labels.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Slots — Training panel (stubs, wired in Phase 8)
    # ------------------------------------------------------------------

    def _on_start_training(self, config: dict) -> None:
        pass

    def _on_stop_training(self) -> None:
        pass
