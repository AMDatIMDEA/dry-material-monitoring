"""Strict YAML configuration for synchronized acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from material_level_yolo.domain import CameraSettings

from .errors import ExperimentConfigurationError
from .models import AcquisitionSettings, ExperimentConfig


BACKENDS = frozenset({"auto", "dshow", "msmf", "v4l2"})
STORAGE_PROFILES = frozenset({"compact", "research", "full_raw"})


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ExperimentConfigurationError(f"Configuration file not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentConfigurationError(f"Could not read {config_path}: {exc}") from exc
    root = _mapping(data, "configuration root")
    _keys(root, {"schema_version", "depth", "c920", "acquisition", "output"}, "root")
    version = _integer(root.get("schema_version"), "schema_version", minimum=1)
    if version != 1:
        raise ExperimentConfigurationError(
            f"Unsupported schema_version {version}; expected 1."
        )
    base = config_path.parent
    depth = _mapping(root.get("depth"), "depth")
    _keys(depth, {"config", "calibration"}, "depth")
    camera = _load_camera(root.get("c920"))
    acquisition = _load_acquisition(root.get("acquisition"), camera)
    output = _mapping(root.get("output"), "output")
    _keys(output, {"root", "storage_profile"}, "output")
    storage = _choice(output.get("storage_profile"), "output.storage_profile", STORAGE_PROFILES)
    return validate_config(ExperimentConfig(
        schema_version=version,
        config_path=config_path,
        depth_config_path=_path(depth.get("config"), "depth.config", base),
        depth_calibration_path=_path(depth.get("calibration"), "depth.calibration", base),
        c920=camera,
        acquisition=acquisition,
        output_root=_path(output.get("root"), "output.root", base),
        storage_profile=storage,
    ))


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Validate programmatic/CLI-overridden settings with the YAML rules."""
    camera = config.c920
    acquisition = config.acquisition
    if camera.backend not in BACKENDS:
        raise ExperimentConfigurationError("c920.backend is unsupported.")
    for field, value in (
        ("c920.width", camera.width),
        ("c920.height", camera.height),
        ("c920.warmup_frames", camera.warmup_frames),
        ("c920.preview_wait_ms", camera.preview_wait_ms),
    ):
        minimum = 0 if field == "c920.warmup_frames" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ExperimentConfigurationError(f"{field} must be an integer >= {minimum}.")
    if not math_is_finite_positive(camera.fps):
        raise ExperimentConfigurationError("c920.fps must be finite and greater than zero.")
    if acquisition.frame_count < 2:
        raise ExperimentConfigurationError("acquisition.frame_count must be >= 2.")
    if acquisition.worker_count != 2:
        raise ExperimentConfigurationError("acquisition.worker_count must be exactly 2.")
    if not math_is_finite_positive(acquisition.span_seconds):
        raise ExperimentConfigurationError(
            "acquisition.span_seconds must be finite and greater than zero."
        )
    if not math_is_finite_positive(acquisition.scheduling_poll_seconds):
        raise ExperimentConfigurationError(
            "acquisition.scheduling_poll_seconds must be finite and greater than zero."
        )
    if camera.fps * acquisition.span_seconds < acquisition.frame_count - 1:
        raise ExperimentConfigurationError(
            "Requested C920 FPS/span cannot distribute the configured frame count."
        )
    if config.storage_profile not in STORAGE_PROFILES:
        raise ExperimentConfigurationError("output.storage_profile is unsupported.")
    return config


def math_is_finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number > 0.0 and number not in {float("inf"), float("-inf")} and number == number


def _load_camera(value: Any) -> CameraSettings:
    raw = _mapping(value, "c920")
    expected = {
        "device_index",
        "backend",
        "width",
        "height",
        "fps",
        "warmup_frames",
        "capture_key",
        "quit_key",
        "window_name",
        "preview_wait_ms",
        "mirror_preview",
    }
    _keys(raw, expected, "c920", optional={"device_name_contains"})
    capture_key = _character(raw.get("capture_key"), "c920.capture_key")
    quit_key = _character(raw.get("quit_key"), "c920.quit_key")
    if capture_key.casefold() == quit_key.casefold():
        raise ExperimentConfigurationError("C920 capture_key and quit_key must differ.")
    return CameraSettings(
        device_index=_integer(raw.get("device_index"), "c920.device_index", minimum=0),
        backend=_choice(raw.get("backend"), "c920.backend", BACKENDS),
        width=_integer(raw.get("width"), "c920.width", minimum=1),
        height=_integer(raw.get("height"), "c920.height", minimum=1),
        fps=_number(raw.get("fps"), "c920.fps", minimum=0.000001),
        warmup_frames=_integer(raw.get("warmup_frames"), "c920.warmup_frames", minimum=0),
        capture_count=6,
        capture_interval_seconds=0.2,
        capture_duration_seconds=None,
        capture_key=capture_key,
        quit_key=quit_key,
        window_name=_text(raw.get("window_name"), "c920.window_name"),
        preview_wait_ms=_integer(raw.get("preview_wait_ms"), "c920.preview_wait_ms", minimum=1),
        mirror_preview=_boolean(raw.get("mirror_preview"), "c920.mirror_preview"),
        close_on_capture=False,
        freeze_after_capture_ms=0,
        device_name_contains=(
            None
            if raw.get("device_name_contains") is None
            else _text(raw.get("device_name_contains"), "c920.device_name_contains")
        ),
    )


def _load_acquisition(value: Any, camera: CameraSettings) -> AcquisitionSettings:
    raw = _mapping(value, "acquisition")
    expected = {"frame_count", "span_seconds", "worker_count", "scheduling_poll_seconds"}
    _keys(raw, expected, "acquisition")
    count = _integer(raw.get("frame_count"), "acquisition.frame_count", minimum=2)
    span = _number(raw.get("span_seconds"), "acquisition.span_seconds", minimum=0.000001)
    workers = _integer(raw.get("worker_count"), "acquisition.worker_count", minimum=2)
    if workers != 2:
        raise ExperimentConfigurationError(
            "acquisition.worker_count must be exactly 2: one bounded owner per camera."
        )
    if camera.fps * span < count - 1:
        raise ExperimentConfigurationError(
            "Requested C920 FPS/span cannot distribute the configured frame count."
        )
    return AcquisitionSettings(
        frame_count=count,
        span_seconds=span,
        worker_count=workers,
        scheduling_poll_seconds=_number(
            raw.get("scheduling_poll_seconds"),
            "acquisition.scheduling_poll_seconds",
            minimum=0.000001,
        ),
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentConfigurationError(f"{field} must be a YAML mapping.")
    return value


def _keys(
    value: Mapping[str, Any],
    expected: set[str],
    field: str,
    *,
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected - optional)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ExperimentConfigurationError(
            f"Invalid keys in {field} ({'; '.join(details)})."
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigurationError(f"{field} must be non-blank text.")
    return value.strip()


def _character(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 1 or ord(text) > 127:
        raise ExperimentConfigurationError(f"{field} must be one ASCII character.")
    return text


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExperimentConfigurationError(f"{field} must be an integer >= {minimum}.")
    return value


def _number(value: Any, field: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentConfigurationError(f"{field} must be numeric.")
    number = float(value)
    if not number >= minimum or number == float("inf"):
        raise ExperimentConfigurationError(f"{field} must be finite and >= {minimum}.")
    return number


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentConfigurationError(f"{field} must be true or false.")
    return value


def _choice(value: Any, field: str, choices: frozenset[str]) -> str:
    text = _text(value, field).lower()
    if text not in choices:
        raise ExperimentConfigurationError(
            f"{field} must be one of: {', '.join(sorted(choices))}."
        )
    return text


def _path(value: Any, field: str, base: Path) -> Path:
    text = _text(value, field)
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
