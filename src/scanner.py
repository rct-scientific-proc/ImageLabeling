from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)


def scan_directory(path: Path) -> list[Path]:
    """Recursively find all supported image files under *path*.

    Returns a sorted list of absolute paths.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    return sorted(
        p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
