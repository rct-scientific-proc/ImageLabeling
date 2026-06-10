"""H5 file creation and helper utilities.

All datasets share the same first-axis length N and have 1-to-1 index mapping.

Datasets
--------
images    uint8  (N, H, W, 3)   RGB pixel values
labels    uint16 (N,)            class index; UNLABELED sentinel = 0xFFFF
gt        bool   (N,)            True = genuine example; False = hard negative
split     uint8  (N,)            0=train 1=val 2=test
classes   str    (K,)            class names; last entry is always "hard_negative"
filenames str    (N,)            original filenames
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNLABELED: int = 0xFFFF          # sentinel — never a valid class index
HARD_NEGATIVE: str = "hard_negative"
_STR_DT = h5py.string_dtype(encoding="utf-8")

# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------


def create_h5(path: Path, image_size: tuple[int, int]) -> None:
    """Create a new, empty H5 file with all required resizable datasets.

    Parameters
    ----------
    path:
        Destination file path.  Must not already exist.
    image_size:
        (height, width) that every image will be stored as.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"H5 file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    H, W = image_size
    with h5py.File(path, "w") as f:
        # Images — chunked by single sample for efficient random access
        f.create_dataset(
            "images",
            shape=(0, H, W, 3),
            maxshape=(None, H, W, 3),
            dtype=np.uint8,
            chunks=(1, H, W, 3),
        )
        f.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint16,
        )
        f.create_dataset(
            "gt",
            shape=(0,),
            maxshape=(None,),
            dtype=bool,
        )
        f.create_dataset(
            "split",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
        )
        f.create_dataset(
            "filenames",
            shape=(0,),
            maxshape=(None,),
            dtype=_STR_DT,
        )
        # classes always starts with the hard_negative sentinel
        f.create_dataset(
            "classes",
            shape=(1,),
            maxshape=(None,),
            dtype=_STR_DT,
        )
        f["classes"][0] = HARD_NEGATIVE

        # Store image size as attributes for convenience
        f.attrs["image_height"] = H
        f.attrs["image_width"] = W


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


def append_images(
    h5_path: Path,
    images: np.ndarray,
    filenames: Sequence[str],
) -> None:
    """Append *images* to the H5 file, initialising metadata to defaults.

    New entries receive:
        labels  = UNLABELED (0xFFFF)
        gt      = True
        split   = 0  (train)

    Parameters
    ----------
    images:
        Array of shape (n, H, W, 3) dtype uint8.
    filenames:
        Sequence of n filename strings.
    """
    images = np.asarray(images, dtype=np.uint8)
    n = len(images)
    if n == 0:
        return
    if len(filenames) != n:
        raise ValueError("len(filenames) must equal len(images)")

    with h5py.File(h5_path, "a") as f:
        old_n = f["images"].shape[0]
        new_n = old_n + n

        f["images"].resize(new_n, axis=0)
        f["images"][old_n:] = images

        f["labels"].resize(new_n, axis=0)
        f["labels"][old_n:] = np.full(n, UNLABELED, dtype=np.uint16)

        f["gt"].resize(new_n, axis=0)
        f["gt"][old_n:] = np.ones(n, dtype=bool)

        f["split"].resize(new_n, axis=0)
        f["split"][old_n:] = np.zeros(n, dtype=np.uint8)

        f["filenames"].resize(new_n, axis=0)
        f["filenames"][old_n:] = np.array(list(filenames), dtype=object)


# ---------------------------------------------------------------------------
# Update helpers
# ---------------------------------------------------------------------------


def update_labels(
    h5_path: Path,
    indices: Sequence[int],
    label_indices: Sequence[int],
) -> None:
    """Write *label_indices* for the given *indices*.

    Raises ValueError if any label_index equals UNLABELED (0xFFFF).
    """
    label_indices = list(label_indices)
    if any(v == UNLABELED for v in label_indices):
        raise ValueError(
            f"Cannot assign UNLABELED sentinel (0x{UNLABELED:04X}) as a label. "
            "Use it only as the 'not yet labeled' state."
        )
    with h5py.File(h5_path, "a") as f:
        for idx, lbl in zip(indices, label_indices):
            f["labels"][idx] = np.uint16(lbl)


def update_gt(
    h5_path: Path,
    indices: Sequence[int],
    gt_values: Sequence[bool],
) -> None:
    """Write ground-truth flags for the given *indices*."""
    with h5py.File(h5_path, "a") as f:
        for idx, val in zip(indices, gt_values):
            f["gt"][idx] = bool(val)


def update_split(
    h5_path: Path,
    indices: Sequence[int],
    split_values: Sequence[int],
) -> None:
    """Write split assignments (0=train, 1=val, 2=test) for the given *indices*."""
    valid = {0, 1, 2}
    if any(v not in valid for v in split_values):
        raise ValueError("split values must be 0 (train), 1 (val), or 2 (test)")
    with h5py.File(h5_path, "a") as f:
        for idx, val in zip(indices, split_values):
            f["split"][idx] = np.uint8(val)


def update_classes(h5_path: Path, classes: list[str]) -> None:
    """Replace the entire *classes* array.

    The caller is responsible for ensuring ``"hard_negative"`` is the last entry.
    """
    if not classes or classes[-1] != HARD_NEGATIVE:
        raise ValueError(
            f'Last entry of classes must always be "{HARD_NEGATIVE}". '
            f"Got: {classes[-1]!r}"
        )
    with h5py.File(h5_path, "a") as f:
        f["classes"].resize((len(classes),))
        f["classes"][:] = np.array(classes, dtype=object)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def get_classes(h5_path: Path) -> list[str]:
    """Return the current class list as Python strings."""
    with h5py.File(h5_path, "r") as f:
        return list(f["classes"].asstr()[:])


def get_image_size(h5_path: Path) -> tuple[int, int]:
    """Return (height, width) stored in the file attributes."""
    with h5py.File(h5_path, "r") as f:
        return int(f.attrs["image_height"]), int(f.attrs["image_width"])


def get_n_images(h5_path: Path) -> int:
    """Return total number of images stored."""
    with h5py.File(h5_path, "r") as f:
        return int(f["images"].shape[0])
