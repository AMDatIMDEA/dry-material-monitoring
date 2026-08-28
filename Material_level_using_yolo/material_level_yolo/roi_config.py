"""Operator workflow for freezing a calibrated internal-tube image ROI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import cv2
import yaml

from experiment_records import format_utc, sha256_file

from .errors import ConfigurationError
from .image_io import read_image


@dataclass(frozen=True, slots=True)
class PixelRoi:
    left: int
    top: int
    right: int
    bottom: int


def freeze_tube_roi(
    config_path: str | Path,
    image_path: str | Path,
    bounds: PixelRoi,
    *,
    axis: str | None = None,
    now_utc: Callable[[], datetime] | None = None,
) -> Path:
    """Freeze one normalized shared ROI and write its immutable reference JSON."""
    config = Path(config_path).expanduser().resolve()
    source = Path(image_path).expanduser().resolve()
    if not config.is_file():
        raise ConfigurationError(f"Configuration file not found: {config}")
    image = read_image(source)
    height, width = image.shape[:2]
    _validate_bounds(bounds, width, height)

    raw = yaml.safe_load(config.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ConfigurationError("ROI freezing requires configuration schema_version 2.")
    defaults = raw.get("defaults")
    if not isinstance(defaults, dict) or not isinstance(defaults.get("tube_roi"), dict):
        raise ConfigurationError("Configuration defaults.tube_roi is missing.")
    profiles = raw.get("model_profiles")
    if isinstance(profiles, dict):
        for name, profile in profiles.items():
            overrides = profile.get("overrides") if isinstance(profile, dict) else None
            if isinstance(overrides, dict) and "tube_roi" in overrides:
                raise ConfigurationError(
                    f"Profile {name!r} overrides tube_roi; remove it before freezing the shared ROI."
                )

    roi = defaults["tube_roi"]
    selected_axis = axis or roi.get("axis", "top_to_bottom")
    if selected_axis not in {"top_to_bottom", "bottom_to_top"}:
        raise ConfigurationError("axis must be top_to_bottom or bottom_to_top.")
    normalized = {
        "left": bounds.left / width,
        "top": bounds.top / height,
        "right": bounds.right / width,
        "bottom": bounds.bottom / height,
    }
    reference = config.parent / "calibration" / "tube_roi.json"
    reference_payload = {
        "roi_reference_schema_version": 1,
        "created_utc": format_utc((now_utc or (lambda: datetime.now(timezone.utc)))()),
        "source_image": str(source),
        "source_image_sha256": sha256_file(source),
        "source_width_px": width,
        "source_height_px": height,
        "pixel_roi": asdict(bounds),
        "normalized_roi": normalized,
        "axis": selected_axis,
        "scope": "shared_all_yolo_material_profiles",
    }
    _atomic_text(
        reference,
        json.dumps(reference_payload, indent=2, ensure_ascii=False) + "\n",
    )
    roi.update(
        {
            "frozen": True,
            **normalized,
            "axis": selected_axis,
            "calibration_reference": reference.relative_to(config.parent).as_posix(),
        }
    )
    _atomic_text(
        config,
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
    )
    return reference


def select_pixel_roi(image_path: str | Path) -> PixelRoi:
    """Open a simple OpenCV rectangle selector for a representative still."""
    image = read_image(image_path)
    window = "Select useful internal tube ROI; Enter confirms, Esc cancels"
    try:
        x, y, width, height = cv2.selectROI(window, image, showCrosshair=True)
    finally:
        cv2.destroyWindow(window)
    if width <= 0 or height <= 0:
        raise ConfigurationError("ROI selection was cancelled or empty; config was not changed.")
    return PixelRoi(int(x), int(y), int(x + width), int(y + height))


def _validate_bounds(bounds: PixelRoi, width: int, height: int) -> None:
    if not isinstance(bounds, PixelRoi):
        raise ConfigurationError("bounds must be PixelRoi.")
    if not (
        0 <= bounds.left < bounds.right <= width
        and 0 <= bounds.top < bounds.bottom <= height
    ):
        raise ConfigurationError(
            f"ROI must lie inside the {width}x{height} image and have positive size."
        )
    if bounds.right - bounds.left < 2 or bounds.bottom - bounds.top < 2:
        raise ConfigurationError("ROI must be at least 2x2 pixels.")
    if (
        bounds.left == 0
        and bounds.top == 0
        and bounds.right == width
        and bounds.bottom == height
    ):
        raise ConfigurationError(
            "The whole image is the unsafe placeholder, not a calibrated internal-tube ROI."
        )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
