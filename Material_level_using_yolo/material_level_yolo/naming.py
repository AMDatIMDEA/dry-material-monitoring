"""Deterministic, portable names for derived still-image artifacts."""

from __future__ import annotations

from pathlib import Path
import re

from experiment_records import sha256_file


KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
EXTENSION_PATTERN = re.compile(r"^\.[a-z0-9]{1,8}$")


def artifact_name(
    source_image: str | Path,
    *,
    index: int,
    kind: str,
    extension: str = ".png",
) -> str:
    """Name an artifact from its stable ordinal and source-content hash."""
    source = Path(source_image).expanduser().resolve()
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError("index must be an integer greater than or equal to 1.")
    if KIND_PATTERN.fullmatch(kind) is None:
        raise ValueError("kind must be a portable lowercase artifact label.")
    normalized_extension = extension.lower()
    if not normalized_extension.startswith("."):
        normalized_extension = "." + normalized_extension
    if EXTENSION_PATTERN.fullmatch(normalized_extension) is None:
        raise ValueError("extension must be a short portable file extension.")
    digest = sha256_file(source)[:12]
    return f"image_{index:06d}_{digest}_{kind}{normalized_extension}"
