"""Comment-preserving config.yaml updates for accepted optimizer results."""

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
from set_up.setup_optimizer import ProfileDiscovery, SetupEvaluation


@dataclass(slots=True, frozen=True)
class ConfigSaveResult:
    config_path: Path
    backup_path: Path


def render_updated_config(
    original_text: str,
    evaluation: SetupEvaluation,
    discovery: ProfileDiscovery,
    generated_utc: str | None = None,
) -> str:
    """Return updated YAML while retaining unrelated text and comments."""
    replacements = {
        ("tube", "outer_diameter_mm"): evaluation.tube.outer_diameter_mm,
        ("tube", "wall_thickness_mm"): (
            evaluation.tube.outer_diameter_mm - evaluation.tube.inner_diameter_mm
        )
        / 2.0,
        ("tube", "inner_diameter_mm"): evaluation.tube.inner_diameter_mm,
        ("tube", "usable_height_mm"): evaluation.tube.usable_height_mm,
        ("camera", "distance_to_rim_mm"): evaluation.camera_to_rim_mm,
        ("camera", "depth_width"): evaluation.profile.width,
        ("camera", "depth_height"): evaluation.profile.height,
        ("camera", "fps"): evaluation.profile.fps,
    }
    updated = original_text
    for (section, key), value in replacements.items():
        updated = _replace_existing_scalar(updated, section, key, value)
    if re.search(r"(?m)^detection_roi:\s*(?:#.*)?$", updated):
        updated = _replace_or_insert_scalar(
            updated,
            "detection_roi",
            "diameter_mm",
            evaluation.tube.inner_diameter_mm,
        )

    generated = generated_utc or datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "generated_utc": generated,
        "tube_outer_diameter_mm": evaluation.tube.outer_diameter_mm,
        "tube_wall_thickness_mm": (
            evaluation.tube.outer_diameter_mm - evaluation.tube.inner_diameter_mm
        )
        / 2.0,
        "tube_inner_diameter_mm": evaluation.tube.inner_diameter_mm,
        "tube_usable_height_mm": evaluation.tube.usable_height_mm,
        "maximum_fill_height_mm": evaluation.tube.maximum_fill_height_mm,
        "camera_to_rim_mm": evaluation.camera_to_rim_mm,
        "camera_to_bottom_mm": evaluation.camera_to_bottom_mm,
        "nearest_surface_distance_mm": evaluation.nearest_surface_distance_mm,
        "farthest_surface_distance_mm": evaluation.farthest_surface_distance_mm,
        "nearest_tilt_envelope_distance_mm": (
            evaluation.nearest_tilt_envelope_distance_mm
        ),
        "farthest_tilt_envelope_distance_mm": (
            evaluation.farthest_tilt_envelope_distance_mm
        ),
        "depth_width": evaluation.profile.width,
        "depth_height": evaluation.profile.height,
        "fps": evaluation.profile.fps,
        "fx_px": evaluation.profile.intrinsics.fx,
        "fy_px": evaluation.profile.intrinsics.fy,
        "cx_px": evaluation.profile.intrinsics.ppx,
        "cy_px": evaluation.profile.intrinsics.ppy,
        "horizontal_fov_degrees": evaluation.profile.horizontal_fov_degrees,
        "vertical_fov_degrees": evaluation.profile.vertical_fov_degrees,
        "intrinsics_source": evaluation.profile.intrinsics_source,
        "free_image_margin_percent": evaluation.safety.free_margin_fraction * 100.0,
        "centring_allowance_mm": evaluation.safety.centring_error_mm,
        "tilt_allowance_degrees": evaluation.safety.maximum_tilt_degrees,
        "profile_minimum_usable_depth_mm": evaluation.effective_minimum_depth_mm,
        "maximum_reliable_depth_mm": evaluation.effective_maximum_depth_mm,
        "setup_status": evaluation.status.value,
        "stereo_baseline_mm": evaluation.profile.stereo_baseline_mm,
        "rim_stereo_overlap_width_mm": evaluation.rim.stereo_overlap_width_mm,
        "physical_measurable_fill_min_mm": (
            evaluation.physical_measurable_fill_min_mm
        ),
        "physical_measurable_fill_max_mm": (
            evaluation.physical_measurable_fill_max_mm
        ),
        "safe_measurable_fill_min_mm": evaluation.safe_measurable_fill_min_mm,
        "safe_measurable_fill_max_mm": evaluation.safe_measurable_fill_max_mm,
        "camera_data_source": discovery.source,
        "connected_camera": not discovery.approximate,
        "camera_model": discovery.device_model,
        "camera_serial_number": discovery.serial_number,
        "values_are_approximate": discovery.approximate,
    }
    metadata_yaml = yaml.safe_dump(
        {"setup_optimizer": metadata},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).rstrip()
    updated = _replace_top_level_block(updated, "setup_optimizer", metadata_yaml)
    if not updated.endswith(("\n", "\r")):
        updated += "\n"
    return updated


def save_setup_to_config(
    config_path: str | Path,
    evaluation: SetupEvaluation,
    discovery: ProfileDiscovery,
    generated_utc: str | None = None,
) -> ConfigSaveResult:
    """Back up and atomically update config after explicit caller confirmation."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise OSError(f"Configuration file not found: {path}")
    original = path.read_text(encoding="utf-8")
    updated = render_updated_config(
        original,
        evaluation=evaluation,
        discovery=discovery,
        generated_utc=generated_utc,
    )
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
        # Typed loading catches interactions with existing settings (for example,
        # a wall exclusion that is too large for a newly entered small tube).
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
        raise OSError(f"Could not save the accepted setup: {exc}") from exc
    return ConfigSaveResult(config_path=path, backup_path=backup)


def _replace_existing_scalar(
    text: str, section: str, key: str, value: int | float
) -> str:
    lines = text.splitlines(keepends=True)
    section_pattern = re.compile(rf"^{re.escape(section)}:\s*(?:#.*)?(?:\r?\n)?$")
    key_pattern = re.compile(
        rf"^(?P<indent>[ \t]+){re.escape(key)}\s*:\s*"
        r"(?P<value>[^#\r\n]*?)(?P<comment>\s+#.*)?(?P<newline>\r?\n)?$"
    )
    inside = False
    for index, line in enumerate(lines):
        raw = line.rstrip("\r\n")
        if section_pattern.match(line):
            inside = True
            continue
        if inside and raw and not raw[0].isspace() and not raw.lstrip().startswith("#"):
            break
        if inside:
            match = key_pattern.match(line)
            if match:
                rendered = str(int(value)) if isinstance(value, int) else repr(float(value))
                comment = match.group("comment") or ""
                newline = match.group("newline") or ""
                lines[index] = (
                    f"{match.group('indent')}{key}: {rendered}{comment}{newline}"
                )
                return "".join(lines)
    raise OSError(f"Existing configuration key '{section}.{key}' was not found.")


def _replace_top_level_block(text: str, key: str, new_block: str) -> str:
    lines = text.splitlines(keepends=True)
    start: int | None = None
    end = len(lines)
    key_pattern = re.compile(rf"^{re.escape(key)}:\s*(?:#.*)?(?:\r?\n)?$")
    top_level_pattern = re.compile(r"^[A-Za-z0-9_.-]+:")
    for index, line in enumerate(lines):
        if key_pattern.match(line):
            start = index
            break
    if start is not None:
        for index in range(start + 1, len(lines)):
            raw = lines[index].rstrip("\r\n")
            if raw and not raw[0].isspace() and top_level_pattern.match(raw):
                end = index
                break
        replacement = new_block + "\n\n"
        return "".join(lines[:start]) + replacement + "".join(lines[end:])
    separator = "" if not text or text.endswith(("\n\n", "\r\n\r\n")) else "\n"
    return text + separator + new_block + "\n"


def _replace_or_insert_scalar(
    text: str,
    section: str,
    key: str,
    value: int | float,
) -> str:
    """Replace a scalar or append it to an existing top-level mapping."""
    try:
        return _replace_existing_scalar(text, section, key, value)
    except OSError:
        lines = text.splitlines(keepends=True)
        section_pattern = re.compile(
            rf"^{re.escape(section)}:\s*(?:#.*)?(?:\r?\n)?$"
        )
        start: int | None = None
        end = len(lines)
        for index, line in enumerate(lines):
            if section_pattern.match(line):
                start = index
                continue
            if start is not None:
                raw = line.rstrip("\r\n")
                if raw and not raw[0].isspace() and not raw.lstrip().startswith("#"):
                    end = index
                    break
        if start is None:
            raise OSError(f"Existing configuration section '{section}' was not found.")
        rendered = str(int(value)) if isinstance(value, int) else repr(float(value))
        lines.insert(end, f"  {key}: {rendered}\n")
        return "".join(lines)
