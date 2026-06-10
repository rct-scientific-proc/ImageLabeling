from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeling Tool")
        self.resize(1280, 800)

        placeholder = QLabel("Image Labeling Tool — Phase 1 scaffold")
        placeholder.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(placeholder)
