"""Unicode-safe OpenCV image I/O adapted from Training _Evaluation utilities."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Iterable

import cv2
import numpy as np

from .errors import ImageIOError


READABLE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
WRITABLE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def read_image(path: str | Path, *, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Decode an image without cv2.imread, including Unicode Windows paths."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ImageIOError(f"Image file not found: {source}")
    try:
        encoded = np.fromfile(str(source), dtype=np.uint8)
        image = cv2.imdecode(encoded, flags) if encoded.size else None
    except (OSError, cv2.error) as exc:
        raise ImageIOError(f"Could not read image {source}: {exc}") from exc
    if image is None:
        raise ImageIOError(f"OpenCV could not decode image: {source}")
    return image


def write_image(
    path: str | Path,
    image: np.ndarray,
    *,
    overwrite: bool = False,
    encode_parameters: Iterable[int] = (),
) -> Path:
    """Encode and atomically replace an image without cv2.imwrite."""
    target = Path(path).expanduser().resolve()
    extension = target.suffix.lower()
    if extension not in WRITABLE_EXTENSIONS:
        raise ImageIOError(
            f"Unsupported output image extension {extension!r}: {target}"
        )
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ImageIOError("image must be a non-empty NumPy array.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing image: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        success, encoded = cv2.imencode(extension, image, list(encode_parameters))
    except cv2.error as exc:
        raise ImageIOError(f"Could not encode image for {target}: {exc}") from exc
    if not success:
        raise ImageIOError(f"OpenCV could not encode image for: {target}")

    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temp = Path(raw_temp)
    try:
        encoded.tofile(str(temp))
        with temp.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing image: {target}")
        os.replace(temp, target)
    except OSError as exc:
        raise ImageIOError(f"Could not write image {target}: {exc}") from exc
    finally:
        temp.unlink(missing_ok=True)
    return target


def discover_images(folder: str | Path, *, recursive: bool = False) -> tuple[Path, ...]:
    """Return deterministic resolved still-image paths without reading them."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ImageIOError(f"Image folder not found: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    images = [
        item.resolve()
        for item in candidates
        if item.is_file() and item.suffix.lower() in READABLE_EXTENSIONS
    ]
    return tuple(
        sorted(images, key=lambda item: (item.as_posix().casefold(), item.as_posix()))
    )
