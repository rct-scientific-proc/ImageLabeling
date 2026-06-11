"""Application log panel.

A read-only QTextEdit that displays timestamped log records from Python's
standard ``logging`` module.  Wire it up once via ``install()``:

    from src.gui.log_panel import LogPanel
    panel = LogPanel()
    panel.install()   # registers a logging.Handler for the root logger

All ``logging.debug/info/warning/error/critical`` calls anywhere in the
codebase will then appear in the panel automatically.
"""

from __future__ import annotations

import logging
from datetime import datetime

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Colours per log level
_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG:    "#888888",
    logging.INFO:     "#e0e0e0",
    logging.WARNING:  "#ffcc44",
    logging.ERROR:    "#ff6655",
    logging.CRITICAL: "#ff2222",
}
_DEFAULT_COLOUR = "#e0e0e0"


# ---------------------------------------------------------------------------
# Qt-signal bridge — allows the logging handler to emit into the GUI thread
# ---------------------------------------------------------------------------

class _LogSignalEmitter(QObject):
    record_received = pyqtSignal(int, str)   # (levelno, formatted_message)


# ---------------------------------------------------------------------------
# Logging handler
# ---------------------------------------------------------------------------

class _QtLogHandler(logging.Handler):
    def __init__(self, emitter: _LogSignalEmitter) -> None:
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._emitter.record_received.emit(record.levelno, msg)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# LogPanel widget
# ---------------------------------------------------------------------------

class LogPanel(QWidget):
    """Scrollable, colour-coded log viewer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._emitter = _LogSignalEmitter()
        self._handler: _QtLogHandler | None = None
        self._build_ui()
        self._emitter.record_received.connect(self._append_record)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(self, level: int = logging.DEBUG) -> None:
        """Attach a logging handler to the root logger (idempotent)."""
        if self._handler is not None:
            return

        formatter = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
        handler = _QtLogHandler(self._emitter)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(min(logging.getLogger().level or logging.DEBUG, level))
        self._handler = handler

    def uninstall(self) -> None:
        """Remove the logging handler (called on close)."""
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.Monospace)
        self._text.setFont(font)
        self._text.setStyleSheet("background: #1e1e1e; color: #e0e0e0;")
        outer.addWidget(self._text, stretch=1)

        btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(60)
        btn_clear.clicked.connect(self._text.clear)
        btn_row.addStretch()
        btn_row.addWidget(btn_clear)
        outer.addLayout(btn_row)

    def _append_record(self, levelno: int, message: str) -> None:
        colour = _LEVEL_COLOURS.get(levelno, _DEFAULT_COLOUR)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message + "\n", fmt)

        # Auto-scroll to bottom
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()
