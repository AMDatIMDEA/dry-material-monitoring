"""Validated, comment-preserving persistence for a manual centre selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import yaml

from material_volume.config import load_config


@dataclass(slots=True, frozen=True)
class CenterSelection:
    center_x_px: float
    center_y_px: float
    diameter_mm: float
    free_image_margin_percent: float
    frame_width_px: int
    frame_height_px: int
    fps: int
    depth_width: int | None = None
    depth_height: int | None = None
    stream: str = "depth"
    format: str = "z16"
    camera_serial_number: str | None = None
    selected_utc: str | None = None


@dataclass(slots=True, frozen=True)
class ConfigSaveResult:
    config_path: Path
    backup_path: Path


def render_updated_config(original_text: str, selection: CenterSelection) -> str:
    """Render only centre-related scalar changes, retaining all other YAML text."""
    selected_utc = selection.selected_utc or datetime.now(timezone.utc).isoformat()
    updates: dict[tuple[str, str], Any] = {
        ("tube", "inner_diameter_mm"): selection.diameter_mm,
        ("camera", "depth_width"): selection.depth_width or selection.frame_width_px,
        ("camera", "depth_height"): selection.depth_height or selection.frame_height_px,
        ("camera", "fps"): selection.fps,
        # Keep the old fields synchronized for older scripts/config consumers.
        ("calibration", "center_x_px"): selection.center_x_px,
        ("calibration", "center_y_px"): selection.center_y_px,
        ("detection_roi", "center_x_px"): selection.center_x_px,
        ("detection_roi", "center_y_px"): selection.center_y_px,
        ("detection_roi", "diameter_mm"): selection.diameter_mm,
        ("detection_roi", "free_image_margin_percent"): (
            selection.free_image_margin_percent
        ),
        ("detection_roi", "frame_width_px"): selection.frame_width_px,
        ("detection_roi", "frame_height_px"): selection.frame_height_px,
        ("detection_roi", "fps"): selection.fps,
        ("detection_roi", "stream"): selection.stream,
        ("detection_roi", "format"): selection.format,
        ("detection_roi", "camera_serial_number"): selection.camera_serial_number,
        ("detection_roi", "selected_utc"): selected_utc,
    }
    updated = original_text
    for (section, key), value in updates.items():
        updated = _upsert_scalar(updated, section, key, value)
    if not updated.endswith(("\n", "\r")):
        updated += "\n"
    return updated


def save_center_config(
    config_path: str | Path,
    selection: CenterSelection,
) -> ConfigSaveResult:
    """Validate, back up, and atomically replace config.yaml."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise OSError(f"Configuration file not found: {path}")
    original = path.read_text(encoding="utf-8")
    updated = render_updated_config(original, selection)
    try:
        parsed = yaml.safe_load(updated)
    except yaml.YAMLError as exc:
        raise OSError(f"Updated configuration would not be valid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OSError("Updated configuration would not have a YAML mapping at its root.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = path.with_name(f"{path.stem}.backup-{timestamp}{path.suffix}")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = Path(stream.name)
        load_config(temp_path)
        shutil.copy2(path, backup)
        os.replace(temp_path, path)
        temp_path = None
    except (OSError, ValueError) as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise OSError(f"Could not save the centre setup: {exc}") from exc
    return ConfigSaveResult(config_path=path, backup_path=backup)


def _upsert_scalar(text: str, section: str, key: str, value: Any) -> str:
    lines = text.splitlines(keepends=True)
    section_pattern = re.compile(rf"^{re.escape(section)}:\s*(?:#.*)?(?:\r?\n)?$")
    key_pattern = re.compile(
        rf"^(?P<indent>[ \t]+){re.escape(key)}\s*:\s*"
        r"(?P<value>[^#\r\n]*?)(?P<comment>\s+#.*)?(?P<newline>\r?\n)?$"
    )
    section_index: int | None = None
    end = len(lines)
    for index, line in enumerate(lines):
        if section_pattern.match(line):
            section_index = index
            continue
        if section_index is not None:
            raw = line.rstrip("\r\n")
            if raw and not raw[0].isspace() and not raw.lstrip().startswith("#"):
                end = index
                break
            match = key_pattern.match(line)
            if match:
                newline = match.group("newline") or ""
                comment = match.group("comment") or ""
                lines[index] = (
                    f"{match.group('indent')}{key}: {_yaml_scalar(value)}{comment}{newline}"
                )
                return "".join(lines)

    new_line = f"  {key}: {_yaml_scalar(value)}\n"
    if section_index is not None:
        lines.insert(end, new_line)
        return "".join(lines)

    separator = "" if not text or text.endswith(("\n\n", "\r\n\r\n")) else "\n"
    return f"{text}{separator}{section}:\n{new_line}"


def _yaml_scalar(value: Any) -> str:
    rendered = yaml.safe_dump(
        value,
        default_flow_style=True,
        allow_unicode=True,
        width=10_000,
    ).strip()
    # PyYAML appends an explicit document terminator for plain scalars.
    return rendered.removesuffix("\n...").removesuffix("...").rstrip()
