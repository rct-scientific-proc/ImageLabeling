"""Scrollable, paginated image grid widget.

ImageCell
---------
A single grid cell. Displays a thumbnail with:
  - filename tooltip on hover
  - semi-transparent "pending" overlay when clicked but not yet committed
  - class-name badge overlay for already-labeled images

ImageGridWidget
---------------
Manages a grid of ImageCell widgets.  Emits:
  sig_cell_clicked(index)  – logical dataset index of the clicked cell
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# ImageCell
# ---------------------------------------------------------------------------

_PENDING_COLOUR = QColor(255, 200, 0, 120)       # amber, semi-transparent
_BADGE_FONT = QFont("Arial", 7, QFont.Bold)


def _hex_to_overlay(hex_colour: str, alpha: int = 110) -> QColor:
    """Convert a hex colour string to a semi-transparent QColor for overlays."""
    c = QColor(hex_colour)
    c.setAlpha(alpha)
    return c


class ImageCell(QFrame):
    """A single thumbnail cell in the image grid."""

    clicked = pyqtSignal(int)        # normal left-click — emits dataset index
    shift_clicked = pyqtSignal(int)  # shift+left-click — emits dataset index

    def __init__(self, cell_size: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cell_size = cell_size
        self._index: int = -1
        self._pending: bool = False
        self._label_name: str | None = None
        self._overlay_colour: QColor | None = None

        self.setFixedSize(cell_size, cell_size)
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self.setCursor(Qt.PointingHandCursor)

        self._pixmap: QPixmap | None = None
        self._canvas = QLabel(self)
        self._canvas.setFixedSize(cell_size, cell_size)
        self._canvas.setAlignment(Qt.AlignCenter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(
        self,
        index: int,
        pixmap: QPixmap,
        filename: str,
        pending: bool = False,
        label_name: str | None = None,
        overlay_colour: QColor | None = None,
    ) -> None:
        self._index = index
        self._pixmap = pixmap
        self._pending = pending
        self._label_name = label_name
        self._overlay_colour = overlay_colour
        self.setToolTip(filename)
        self._render()

    def set_pending(self, pending: bool) -> None:
        self._pending = pending
        self._render()

    def set_empty(self) -> None:
        self._index = -1
        self._pixmap = None
        self._pending = False
        self._label_name = None
        self.setToolTip("")
        self._canvas.setPixmap(QPixmap())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self._pixmap is None:
            return

        base = self._pixmap.scaled(
            self._cell_size,
            self._cell_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # Paint overlays onto a copy
        result = QPixmap(self._cell_size, self._cell_size)
        result.fill(Qt.black)

        painter = QPainter(result)
        # Centre the thumbnail
        x = (self._cell_size - base.width()) // 2
        y = (self._cell_size - base.height()) // 2
        painter.drawPixmap(x, y, base)

        # Pending overlay
        if self._pending:
            painter.fillRect(0, 0, self._cell_size, self._cell_size, _PENDING_COLOUR)

        # Labeled overlay + badge
        if self._label_name is not None:
            colour = self._overlay_colour if self._overlay_colour is not None else QColor(0, 180, 0, 110)
            painter.fillRect(0, 0, self._cell_size, self._cell_size, colour)
            painter.setFont(_BADGE_FONT)
            painter.setPen(Qt.black)
            badge_rect = result.rect().adjusted(2, 2, -2, -2)
            painter.drawText(
                badge_rect,
                Qt.AlignBottom | Qt.AlignHCenter | Qt.TextWordWrap,
                self._label_name,
            )

        painter.end()
        self._canvas.setPixmap(result)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._index >= 0:
            if event.modifiers() & Qt.ShiftModifier:
                self.shift_clicked.emit(self._index)
            else:
                self.clicked.emit(self._index)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# ImageGridWidget
# ---------------------------------------------------------------------------


class ImageGridWidget(QWidget):
    """Paginated, scrollable grid of ImageCell widgets.

    Call ``load_images()`` to populate the grid from numpy arrays.
    """

    sig_cell_clicked = pyqtSignal(int)              # single click — dataset index
    sig_cells_range_selected = pyqtSignal(list)     # shift-click range — list of dataset indices

    # Default grid dimensions
    DEFAULT_ROWS = 3
    DEFAULT_COLS = 3
    DEFAULT_CELL_SIZE = 160

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows = self.DEFAULT_ROWS
        self._cols = self.DEFAULT_COLS
        self._cell_size = self.DEFAULT_CELL_SIZE
        self._page = 0

        # Data held by the widget
        self._indices: list[int] = []               # dataset indices on current view
        self._pixmaps: list[QPixmap] = []           # pre-converted pixmaps
        self._filenames: list[str] = []
        self._pending: set[int] = set()             # dataset indices pending commit
        self._label_names: dict[int, str] = {}      # dataset index → class name
        self._label_colours: dict[int, QColor] = {} # dataset index → overlay colour

        self._cells: list[ImageCell] = []
        self._anchor_ds_index: int | None = None   # anchor for shift-click range
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_images(
        self,
        indices: list[int],
        images,                       # np.ndarray (N, H, W, 3) uint8
        filenames: list[str],
        pending: set[int] | None = None,
        label_names: dict[int, str] | None = None,
        label_colours: dict[int, QColor] | None = None,
    ) -> None:
        """Replace the current image set and redraw from page 0."""
        self._indices = list(indices)
        self._filenames = list(filenames)
        self._pending = set(pending) if pending else set()
        self._label_names = dict(label_names) if label_names else {}
        self._label_colours = dict(label_colours) if label_colours else {}
        self._anchor_ds_index = None
        self._page = 0

        # Convert numpy arrays → QPixmap (done once, cached)
        self._pixmaps = []
        for img in images:
            h, w, c = img.shape
            qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format_RGB888)
            self._pixmaps.append(QPixmap.fromImage(qimg))

        self._update_page_label()
        self._render_page()

    def mark_pending(self, dataset_index: int, pending: bool) -> None:
        if pending:
            self._pending.add(dataset_index)
        else:
            self._pending.discard(dataset_index)
        self._refresh_cell(dataset_index)

    def set_grid_size(self, rows: int, cols: int) -> None:
        self._rows = max(1, rows)
        self._cols = max(1, cols)
        self._rebuild_cells()
        self._page = 0
        self._update_page_label()
        self._render_page()

    @property
    def page_size(self) -> int:
        return self._rows * self._cols

    @property
    def total_pages(self) -> int:
        if not self._indices:
            return 1
        return max(1, (len(self._indices) + self.page_size - 1) // self.page_size)

    # ------------------------------------------------------------------
    # Private — UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # --- grid config row ---
        config_row = QHBoxLayout()
        config_row.setSpacing(6)

        config_row.addWidget(QLabel("Grid:"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(1, 10)
        self._rows_spin.setValue(self._rows)
        self._rows_spin.setFixedWidth(48)
        self._rows_spin.valueChanged.connect(
            lambda: self.set_grid_size(self._rows_spin.value(), self._cols_spin.value())
        )
        config_row.addWidget(self._rows_spin)

        config_row.addWidget(QLabel("×"))
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(1, 10)
        self._cols_spin.setValue(self._cols)
        self._cols_spin.setFixedWidth(48)
        self._cols_spin.valueChanged.connect(
            lambda: self.set_grid_size(self._rows_spin.value(), self._cols_spin.value())
        )
        config_row.addWidget(self._cols_spin)
        config_row.addStretch()
        outer.addLayout(config_row)

        # --- scroll area containing the grid ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(4)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self._grid_container)
        outer.addWidget(scroll, stretch=1)

        # --- navigation row ---
        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)

        self._btn_prev = QPushButton("◀  Prev")
        self._btn_prev.clicked.connect(self._prev_page)
        self._btn_next = QPushButton("Next  ▶")
        self._btn_next.clicked.connect(self._next_page)
        self._page_label = QLabel("Page 1 / 1")
        self._page_label.setAlignment(Qt.AlignCenter)

        nav_row.addWidget(self._btn_prev)
        nav_row.addStretch()
        nav_row.addWidget(self._page_label)
        nav_row.addStretch()
        nav_row.addWidget(self._btn_next)
        outer.addLayout(nav_row)

        self._rebuild_cells()

    def _rebuild_cells(self) -> None:
        """Tear down existing cells and recreate for the current rows×cols."""
        # Remove old cells from layout
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cells = []

        for r in range(self._rows):
            for c in range(self._cols):
                cell = ImageCell(self._cell_size)
                cell.clicked.connect(self._on_cell_normal_click)
                cell.shift_clicked.connect(self._on_cell_shift_click)
                self._grid_layout.addWidget(cell, r, c)
                self._cells.append(cell)

    # ------------------------------------------------------------------
    # Private — rendering
    # ------------------------------------------------------------------

    def _render_page(self) -> None:
        page_start = self._page * self.page_size
        for slot, cell in enumerate(self._cells):
            data_pos = page_start + slot
            if data_pos >= len(self._indices):
                cell.set_empty()
                continue

            ds_idx = self._indices[data_pos]
            pixmap = self._pixmaps[data_pos]
            fname = self._filenames[data_pos]
            pending = ds_idx in self._pending
            label_name = self._label_names.get(ds_idx)
            overlay_colour = self._label_colours.get(ds_idx)
            cell.set_image(ds_idx, pixmap, fname, pending=pending, label_name=label_name, overlay_colour=overlay_colour)

        self._btn_prev.setEnabled(self._page > 0)
        self._btn_next.setEnabled(self._page < self.total_pages - 1)

    def _refresh_cell(self, dataset_index: int) -> None:
        """Re-render a single cell identified by dataset index."""
        page_start = self._page * self.page_size
        for slot, cell in enumerate(self._cells):
            data_pos = page_start + slot
            if data_pos < len(self._indices) and self._indices[data_pos] == dataset_index:
                cell.set_pending(dataset_index in self._pending)
                break

    def _update_page_label(self) -> None:
        self._page_label.setText(f"Page {self._page + 1} / {self.total_pages}")

    # ------------------------------------------------------------------
    # Private — navigation
    # ------------------------------------------------------------------

    def _prev_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._update_page_label()
            self._render_page()

    def _next_page(self) -> None:
        if self._page < self.total_pages - 1:
            self._page += 1
            self._update_page_label()
            self._render_page()

    # ------------------------------------------------------------------
    # Private — cell click
    # ------------------------------------------------------------------

    def _on_cell_normal_click(self, dataset_index: int) -> None:
        self._anchor_ds_index = dataset_index
        self.sig_cell_clicked.emit(dataset_index)

    def _on_cell_shift_click(self, dataset_index: int) -> None:
        # If no anchor yet, or anchor is no longer in the current view, treat as normal
        if (
            self._anchor_ds_index is None
            or self._anchor_ds_index not in self._indices
        ):
            self._anchor_ds_index = dataset_index
            self.sig_cell_clicked.emit(dataset_index)
            return

        anchor_pos = self._indices.index(self._anchor_ds_index)
        end_pos = self._indices.index(dataset_index)
        start = min(anchor_pos, end_pos)
        end = max(anchor_pos, end_pos)
        selected = self._indices[start : end + 1]
        # Anchor stays on the first click, not the shift-click (file-explorer behaviour)
        self.sig_cells_range_selected.emit(selected)
