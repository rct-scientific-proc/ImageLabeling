"""Image ingestion pipeline.

Ties together the directory scanner, image preprocessing, and H5 I/O to
produce a fully-initialized H5 dataset from a folder of images.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from src.scanner import scan_directory
from src.h5io import create_h5, append_images, get_n_images

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_SIZE: tuple[int, int] = (64, 64)
_BATCH_SIZE: int = 256   # images processed per H5 write


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_image(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    """Load an image file, resize, and return a uint8 RGB array (H, W, 3).

    Returns ``None`` and emits a warning if the file cannot be read.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((size[1], size[0]), Image.LANCZOS)  # resize(W, H)
            return np.asarray(img, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Skipping {path.name}: {exc}", stacklevel=2)
        return None


# ---------------------------------------------------------------------------
# Top-level ingest
# ---------------------------------------------------------------------------


def ingest_directory(
    dir_path: Path,
    h5_path: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    progress_callback=None,
) -> int:
    """Scan *dir_path*, preprocess every image, and write to *h5_path*.

    Creates a new H5 file at *h5_path*. Raises ``FileExistsError`` if it
    already exists.

    Parameters
    ----------
    dir_path:
        Root directory to scan recursively.
    h5_path:
        Destination H5 file path (must not exist).
    image_size:
        Target ``(height, width)`` for all stored images.
    progress_callback:
        Optional ``callable(current: int, total: int)`` called after each
        batch is written — useful for driving a GUI progress bar.

    Returns
    -------
    int
        Number of images successfully ingested.
    """
    paths = scan_directory(dir_path)
    if not paths:
        raise ValueError(f"No supported images found in: {dir_path}")

    create_h5(h5_path, image_size)

    total = len(paths)
    ingested = 0
    batch_imgs: list[np.ndarray] = []
    batch_names: list[str] = []

    for i, p in enumerate(paths):
        img = preprocess_image(p, image_size)
        if img is None:
            continue

        batch_imgs.append(img)
        batch_names.append(p.name)

        if len(batch_imgs) >= _BATCH_SIZE:
            append_images(h5_path, np.stack(batch_imgs), batch_names)
            ingested += len(batch_imgs)
            batch_imgs = []
            batch_names = []
            if progress_callback is not None:
                progress_callback(i + 1, total)

    # Flush remaining
    if batch_imgs:
        append_images(h5_path, np.stack(batch_imgs), batch_names)
        ingested += len(batch_imgs)
        if progress_callback is not None:
            progress_callback(total, total)

    return ingested
