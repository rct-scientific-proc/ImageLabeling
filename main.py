import sys

# torch / torchvision must be imported before any Qt DLLs are loaded on Windows
# to avoid a DLL initialization failure (WinError 1114).
import torch  # noqa: F401
import torchvision  # noqa: F401

from PyQt5.QtWidgets import QApplication
from src.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
