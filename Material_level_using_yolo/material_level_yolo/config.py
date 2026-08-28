"""Strict YAML loading with paths resolved relative to the selected YAML file."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import yaml

from .domain import (
    AggregationThresholds,
    CameraSettings,
    EstimationThresholds,
    InferenceSettings,
    ModelProfile,
    NmsSettings,
    OutputSettings,
    ProjectConfig,
    SemanticRole,
    TubeRoi,
)
from .errors import ConfigurationError


SUPPORTED_TASKS = frozenset({"segment", "detect"})
TASK_MODES = {
    "segment": "segmentation_interface",
    "detect": "bbox_vertical_extent",
}
STORAGE_PROFILES = frozenset({"compact", "research", "full_raw"})
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
CAMERA_BACKENDS = frozenset({"auto", "dshow", "msmf", "v4l2"})
IMAGE_FORMATS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
COMMON_PROFILE_KEYS = frozenset(
    {
        "expected_task",
        "estimation_mode",
        "inference",
        "tube_roi",
        "estimation",
        "aggregation",
        "camera",
        "output",
    }
)


def load_config(path: str | Path) -> ProjectConfig:
    """Load, resolve, and validate one project configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read YAML configuration {config_path}: {exc}") from exc
    root = _mapping(raw, "configuration root")
    _keys(
        root,
        {"schema_version", "default_profile", "model_profiles"},
        "configuration root",
        optional={"defaults"},
    )
    schema_version = _integer(root.get("schema_version"), "schema_version", minimum=1)
    if schema_version not in {1, 2}:
        raise ConfigurationError(
            f"Unsupported configuration schema_version {schema_version}; expected 1 or 2."
        )
    default_profile = _text(root.get("default_profile"), "default_profile")
    profile_values = _mapping(root.get("model_profiles"), "model_profiles")
    if not profile_values:
        raise ConfigurationError("model_profiles must define at least one named profile.")

    defaults: Mapping[str, Any] | None = None
    if schema_version == 2:
        defaults = _mapping(root.get("defaults"), "defaults")
        _keys(defaults, set(COMMON_PROFILE_KEYS), "defaults")
    elif "defaults" in root:
        raise ConfigurationError("defaults requires configuration schema_version 2.")

    profiles: dict[str, ModelProfile] = {}
    for name, value in profile_values.items():
        profile_name = _profile_name(name)
        profile_value = _mapping(value, f"model_profiles.{profile_name}")
        if defaults is not None:
            _keys(
                profile_value,
                {"weights", "semantic_roles"},
                f"model_profiles.{profile_name}",
                optional={"recorded_material_names", "overrides"},
            )
            overrides = _mapping(
                profile_value.get("overrides", {}),
                f"model_profiles.{profile_name}.overrides",
            )
            _keys(
                overrides,
                set(),
                f"model_profiles.{profile_name}.overrides",
                optional=set(COMMON_PROFILE_KEYS),
            )
            profile_value = {
                "weights": profile_value.get("weights"),
                "semantic_roles": profile_value.get("semantic_roles"),
                "recorded_material_names": profile_value.get("recorded_material_names"),
                **_deep_merge(defaults, overrides),
            }
        profiles[profile_name] = _load_profile(
            profile_name,
            profile_value,
            config_path.parent,
        )
    if default_profile not in profiles:
        raise ConfigurationError(
            f"default_profile {default_profile!r} is not present in model_profiles."
        )
    aliases: dict[str, str] = {}
    for profile in profiles.values():
        for alias in profile.recorded_material_names:
            normalized = alias.casefold()
            previous = aliases.get(normalized)
            if previous is not None and previous != profile.name:
                raise ConfigurationError(
                    f"Recorded material alias {alias!r} is shared by profiles "
                    f"{previous!r} and {profile.name!r}."
                )
            aliases[normalized] = profile.name
    return ProjectConfig(
        schema_version=schema_version,
        config_path=config_path,
        default_profile=default_profile,
        profiles=profiles,
    )


def _load_profile(name: str, value: Mapping[str, Any], base: Path) -> ModelProfile:
    expected = {
        "weights",
        "semantic_roles",
        "expected_task",
        "estimation_mode",
        "inference",
        "tube_roi",
        "aggregation",
        "camera",
        "output",
    }
    _keys(
        value,
        expected,
        f"model_profiles.{name}",
        optional={"estimation", "recorded_material_names"},
    )
    roles = _mapping(value.get("semantic_roles"), f"{name}.semantic_roles")
    _keys(roles, {"material", "empty"}, f"{name}.semantic_roles")
    material = _role(roles.get("material"), f"{name}.semantic_roles.material")
    empty = _role(roles.get("empty"), f"{name}.semantic_roles.empty")
    if material.class_name.casefold() == empty.class_name.casefold():
        raise ConfigurationError(f"Profile {name!r} uses the same name for material and empty.")
    if material.explicit_id is not None and material.explicit_id == empty.explicit_id:
        raise ConfigurationError(f"Profile {name!r} uses the same explicit ID for both roles.")

    expected_task = _choice(value.get("expected_task"), f"{name}.expected_task", SUPPORTED_TASKS)
    estimation_mode = _text(value.get("estimation_mode"), f"{name}.estimation_mode")
    required_mode = TASK_MODES[expected_task]
    if estimation_mode != required_mode:
        raise ConfigurationError(
            f"Profile {name!r} expected_task={expected_task!r} requires "
            f"estimation_mode={required_mode!r}; silent task/mode switching is forbidden."
        )

    inference_raw = _mapping(value.get("inference"), f"{name}.inference")
    _keys(
        inference_raw,
        {"confidence", "iou", "image_size", "device", "max_detections", "nms"},
        f"{name}.inference",
    )
    device = _text(inference_raw.get("device"), f"{name}.inference.device").lower()
    if re.fullmatch(r"(?:auto|cpu|cuda(?::\d+)?)", device) is None:
        raise ConfigurationError(
            f"Profile {name!r} inference.device must be 'auto', 'cpu', 'cuda', "
            "or 'cuda:<index>'."
        )
    nms_raw = _mapping(inference_raw.get("nms"), f"{name}.inference.nms")
    _keys(nms_raw, {"agnostic", "class_filter_only_semantic"}, f"{name}.inference.nms")
    inference = InferenceSettings(
        confidence=_fraction(inference_raw.get("confidence"), f"{name}.inference.confidence"),
        iou=_fraction(inference_raw.get("iou"), f"{name}.inference.iou"),
        image_size=_integer(inference_raw.get("image_size"), f"{name}.inference.image_size", minimum=32),
        device=device,
        max_detections=_integer(
            inference_raw.get("max_detections"), f"{name}.inference.max_detections", minimum=1
        ),
        nms=NmsSettings(
            agnostic=_boolean(nms_raw.get("agnostic"), f"{name}.inference.nms.agnostic"),
            class_filter_only_semantic=_boolean(
                nms_raw.get("class_filter_only_semantic"),
                f"{name}.inference.nms.class_filter_only_semantic",
            ),
        ),
    )
    roi = _load_roi(value.get("tube_roi"), name, base)
    estimation = _load_estimation(value.get("estimation"), name)
    aggregation = _load_aggregation(value.get("aggregation"), name)
    camera = _load_camera(value.get("camera"), name)
    if aggregation.minimum_valid_images > camera.capture_count:
        raise ConfigurationError(
            f"Profile {name!r} aggregation.minimum_valid_images cannot exceed "
            "camera.capture_count."
        )
    if aggregation.manual_minimum_valid_images > 1:
        raise ConfigurationError(
            f"Profile {name!r} aggregation.manual_minimum_valid_images cannot exceed "
            "the one still captured by manual_camera."
        )
    if camera.capture_key.casefold() == camera.quit_key.casefold():
        raise ConfigurationError(
            f"Profile {name!r} must use different capture_key and quit_key values."
        )
    output = _load_output(value.get("output"), name, base)
    aliases = _material_aliases(
        value.get("recorded_material_names"),
        name,
        material.class_name,
    )
    return ModelProfile(
        name=name,
        recorded_material_names=aliases,
        weights_path=_path(value.get("weights"), f"{name}.weights", base),
        material_role=material,
        empty_role=empty,
        expected_task=expected_task,
        estimation_mode=estimation_mode,
        inference=inference,
        tube_roi=roi,
        estimation=estimation,
        aggregation=aggregation,
        camera=camera,
        output=output,
    )


def _material_aliases(value: Any, profile: str, semantic_name: str) -> tuple[str, ...]:
    if value is None:
        raw_values: list[Any] = [profile, semantic_name]
    elif not isinstance(value, list) or not value:
        raise ConfigurationError(
            f"{profile}.recorded_material_names must be a non-empty YAML list."
        )
    else:
        raw_values = value
    aliases: list[str] = []
    seen: set[str] = set()
    for index, value_item in enumerate(raw_values):
        alias = _text(value_item, f"{profile}.recorded_material_names[{index}]")
        normalized = alias.casefold()
        if normalized not in seen:
            aliases.append(alias)
            seen.add(normalized)
    return tuple(aliases)


def _role(value: Any, field: str) -> SemanticRole:
    raw = _mapping(value, field)
    _keys(raw, {"class_name", "explicit_id"}, field)
    explicit = raw.get("explicit_id")
    return SemanticRole(
        class_name=_text(raw.get("class_name"), f"{field}.class_name"),
        explicit_id=(None if explicit is None else _integer(explicit, f"{field}.explicit_id", minimum=0)),
    )


def _load_roi(value: Any, profile: str, base: Path) -> TubeRoi:
    field = f"{profile}.tube_roi"
    raw = _mapping(value, field)
    _keys(
        raw,
        {"left", "top", "right", "bottom", "axis", "calibration_reference"},
        field,
        optional={"frozen"},
    )
    left = _fraction(raw.get("left"), f"{field}.left", include_one=False)
    top = _fraction(raw.get("top"), f"{field}.top", include_one=False)
    right = _fraction(raw.get("right"), f"{field}.right")
    bottom = _fraction(raw.get("bottom"), f"{field}.bottom")
    if left >= right or top >= bottom:
        raise ConfigurationError(f"{field} must satisfy left < right and top < bottom.")
    axis = _choice(raw.get("axis"), f"{field}.axis", {"top_to_bottom", "bottom_to_top"})
    calibration = raw.get("calibration_reference")
    whole_image = left == 0.0 and top == 0.0 and right == 1.0 and bottom == 1.0
    frozen = _boolean(
        raw.get("frozen", calibration is not None and not whole_image),
        f"{field}.frozen",
    )
    if frozen and whole_image:
        raise ConfigurationError(
            f"{field} cannot freeze the placeholder whole-image ROI; define the tube interior."
        )
    if frozen and calibration is None:
        raise ConfigurationError(
            f"{field}.calibration_reference is required when the ROI is frozen."
        )
    return TubeRoi(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        axis=axis,
        calibration_reference=(
            None if calibration is None else _path(calibration, f"{field}.calibration_reference", base)
        ),
        frozen=frozen,
    )


def _load_aggregation(value: Any, profile: str) -> AggregationThresholds:
    field = f"{profile}.aggregation"
    raw = _mapping(value, field)
    expected = {
        "minimum_valid_images",
        "maximum_spread_percentage_points",
        "maximum_overlap_fraction",
        "maximum_unclassified_fraction",
        "minimum_material_confidence",
        "minimum_empty_confidence",
    }
    _keys(raw, expected, field, optional={"manual_minimum_valid_images"})
    return AggregationThresholds(
        minimum_valid_images=_integer(raw.get("minimum_valid_images"), f"{field}.minimum_valid_images", minimum=1),
        manual_minimum_valid_images=_integer(
            raw.get("manual_minimum_valid_images", 1),
            f"{field}.manual_minimum_valid_images",
            minimum=1,
        ),
        maximum_spread_percentage_points=_number(
            raw.get("maximum_spread_percentage_points"),
            f"{field}.maximum_spread_percentage_points",
            minimum=0.0,
            maximum=100.0,
        ),
        maximum_overlap_fraction=_fraction(raw.get("maximum_overlap_fraction"), f"{field}.maximum_overlap_fraction"),
        maximum_unclassified_fraction=_fraction(
            raw.get("maximum_unclassified_fraction"), f"{field}.maximum_unclassified_fraction"
        ),
        minimum_material_confidence=_fraction(
            raw.get("minimum_material_confidence"), f"{field}.minimum_material_confidence"
        ),
        minimum_empty_confidence=_fraction(
            raw.get("minimum_empty_confidence"), f"{field}.minimum_empty_confidence"
        ),
    )


def _load_estimation(value: Any, profile: str) -> EstimationThresholds:
    field = f"{profile}.estimation"
    raw = (
        {
            "mask_threshold": 0.5,
            "row_occupancy_threshold": 0.5,
            "row_smoothing_window": 5,
            "minimum_dominant_run_rows": 2,
            "maximum_instances_per_role": 1,
            "minimum_column_classified_fraction": 0.5,
            "minimum_single_class_lateral_fraction": 0.25,
            "minimum_single_class_axial_fraction": 0.02,
            "single_class_end_occupancy_threshold": 0.25,
            "minimum_whole_image_detection_lateral_fraction": 0.01,
            "minimum_whole_image_detection_axial_fraction": 0.01,
            "minimum_whole_image_detection_area_fraction": 0.0005,
        }
        if value is None
        else _mapping(value, field)
    )
    expected = {
        "mask_threshold",
        "row_occupancy_threshold",
        "row_smoothing_window",
        "minimum_dominant_run_rows",
        "maximum_instances_per_role",
        "minimum_column_classified_fraction",
    }
    _keys(
        raw,
        expected,
        field,
        optional={
            "minimum_single_class_lateral_fraction",
            "minimum_single_class_axial_fraction",
            "single_class_end_occupancy_threshold",
            "minimum_whole_image_detection_lateral_fraction",
            "minimum_whole_image_detection_axial_fraction",
            "minimum_whole_image_detection_area_fraction",
        },
    )
    smoothing = _integer(raw.get("row_smoothing_window"), f"{field}.row_smoothing_window", minimum=1)
    if smoothing % 2 == 0:
        raise ConfigurationError(f"{field}.row_smoothing_window must be odd.")
    return EstimationThresholds(
        mask_threshold=_fraction(raw.get("mask_threshold"), f"{field}.mask_threshold"),
        row_occupancy_threshold=_fraction(
            raw.get("row_occupancy_threshold"), f"{field}.row_occupancy_threshold"
        ),
        row_smoothing_window=smoothing,
        minimum_dominant_run_rows=_integer(
            raw.get("minimum_dominant_run_rows"), f"{field}.minimum_dominant_run_rows", minimum=1
        ),
        maximum_instances_per_role=_integer(
            raw.get("maximum_instances_per_role"), f"{field}.maximum_instances_per_role", minimum=1
        ),
        minimum_column_classified_fraction=_fraction(
            raw.get("minimum_column_classified_fraction"),
            f"{field}.minimum_column_classified_fraction",
        ),
        minimum_single_class_lateral_fraction=_fraction(
            raw.get("minimum_single_class_lateral_fraction", 0.25),
            f"{field}.minimum_single_class_lateral_fraction",
        ),
        minimum_single_class_axial_fraction=_fraction(
            raw.get("minimum_single_class_axial_fraction", 0.02),
            f"{field}.minimum_single_class_axial_fraction",
        ),
        single_class_end_occupancy_threshold=_fraction(
            raw.get("single_class_end_occupancy_threshold", 0.25),
            f"{field}.single_class_end_occupancy_threshold",
        ),
        minimum_whole_image_detection_lateral_fraction=_fraction(
            raw.get("minimum_whole_image_detection_lateral_fraction", 0.01),
            f"{field}.minimum_whole_image_detection_lateral_fraction",
        ),
        minimum_whole_image_detection_axial_fraction=_fraction(
            raw.get("minimum_whole_image_detection_axial_fraction", 0.01),
            f"{field}.minimum_whole_image_detection_axial_fraction",
        ),
        minimum_whole_image_detection_area_fraction=_fraction(
            raw.get("minimum_whole_image_detection_area_fraction", 0.0005),
            f"{field}.minimum_whole_image_detection_area_fraction",
        ),
    )


def _deep_merge(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, Mapping):
            override = overrides.get(key, {})
            if override is not None and not isinstance(override, Mapping):
                raise ConfigurationError(f"Override {key!r} must be a YAML mapping.")
            merged[key] = _deep_merge(value, override or {})
        else:
            merged[key] = value
    for key, value in overrides.items():
        if key not in base:
            raise ConfigurationError(f"Unknown profile override: {key}")
        if not isinstance(base[key], Mapping):
            merged[key] = value
    return merged


def _load_camera(value: Any, profile: str) -> CameraSettings:
    field = f"{profile}.camera"
    raw = _mapping(value, field)
    expected = {
        "device_index",
        "backend",
        "width",
        "height",
        "fps",
        "warmup_frames",
        "capture_count",
        "capture_interval_seconds",
        "capture_key",
        "quit_key",
        "window_name",
    }
    _keys(
        raw,
        expected,
        field,
        optional={
            "device_name_contains",
            "capture_duration_seconds",
            "preview_wait_ms",
            "mirror_preview",
            "close_on_capture",
            "freeze_after_capture_ms",
        },
    )
    return CameraSettings(
        device_index=_integer(raw.get("device_index"), f"{field}.device_index", minimum=0),
        backend=_choice(raw.get("backend"), f"{field}.backend", CAMERA_BACKENDS),
        width=_integer(raw.get("width"), f"{field}.width", minimum=1),
        height=_integer(raw.get("height"), f"{field}.height", minimum=1),
        fps=_number(raw.get("fps"), f"{field}.fps", minimum=0.000001),
        warmup_frames=_integer(raw.get("warmup_frames"), f"{field}.warmup_frames", minimum=0),
        capture_count=_integer(raw.get("capture_count"), f"{field}.capture_count", minimum=1),
        capture_interval_seconds=_number(
            raw.get("capture_interval_seconds"), f"{field}.capture_interval_seconds", minimum=0.000001
        ),
        capture_duration_seconds=(
            None
            if raw.get("capture_duration_seconds") is None
            else _number(
                raw.get("capture_duration_seconds"),
                f"{field}.capture_duration_seconds",
                minimum=0.000001,
            )
        ),
        capture_key=_single_character(raw.get("capture_key"), f"{field}.capture_key"),
        quit_key=_single_character(raw.get("quit_key"), f"{field}.quit_key"),
        window_name=_text(raw.get("window_name"), f"{field}.window_name"),
        preview_wait_ms=_integer(raw.get("preview_wait_ms", 10), f"{field}.preview_wait_ms", minimum=1),
        mirror_preview=_boolean(raw.get("mirror_preview", False), f"{field}.mirror_preview"),
        close_on_capture=_boolean(raw.get("close_on_capture", True), f"{field}.close_on_capture"),
        freeze_after_capture_ms=_integer(
            raw.get("freeze_after_capture_ms", 500),
            f"{field}.freeze_after_capture_ms",
            minimum=0,
        ),
        device_name_contains=(
            None
            if raw.get("device_name_contains") is None
            else _text(raw.get("device_name_contains"), f"{field}.device_name_contains")
        ),
    )


def _load_output(value: Any, profile: str, base: Path) -> OutputSettings:
    field = f"{profile}.output"
    raw = _mapping(value, field)
    expected = {
        "root",
        "storage_profile",
        "image_format",
        "save_overlays",
        "save_masks",
        "overwrite",
        "log_level",
    }
    _keys(raw, expected, field)
    image_format = _text(raw.get("image_format"), f"{field}.image_format").lower()
    if not image_format.startswith("."):
        image_format = "." + image_format
    if image_format not in IMAGE_FORMATS:
        raise ConfigurationError(
            f"{field}.image_format must be one of: {', '.join(sorted(IMAGE_FORMATS))}."
        )
    log_level = _text(raw.get("log_level"), f"{field}.log_level").upper()
    if log_level not in LOG_LEVELS:
        raise ConfigurationError(f"{field}.log_level is not a standard logging level.")
    return OutputSettings(
        root=_path(raw.get("root"), f"{field}.root", base),
        storage_profile=_choice(raw.get("storage_profile"), f"{field}.storage_profile", STORAGE_PROFILES),
        image_format=image_format,
        save_overlays=_boolean(raw.get("save_overlays"), f"{field}.save_overlays"),
        save_masks=_boolean(raw.get("save_masks"), f"{field}.save_masks"),
        overwrite=_boolean(raw.get("overwrite"), f"{field}.overwrite"),
        log_level=log_level,
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} must be a YAML mapping.")
    return value


def _keys(
    value: Mapping[str, Any],
    expected: set[str],
    field: str,
    *,
    optional: set[str] = frozenset(),
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected - optional)
    if missing or unknown:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if unknown:
            parts.append("unknown: " + ", ".join(unknown))
        raise ConfigurationError(f"Invalid keys in {field} ({'; '.join(parts)}).")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be non-blank text.")
    return value.strip()


def _profile_name(value: Any) -> str:
    name = _text(value, "model profile name")
    if not all(character.isascii() and (character.isalnum() or character in "._-") for character in name):
        raise ConfigurationError(
            f"Model profile name {name!r} must use only ASCII letters, digits, '.', '_', or '-'."
        )
    return name


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{field} must be an integer >= {minimum}.")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field} must be a finite number.")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ConfigurationError(f"{field} must be finite.")
    if minimum is not None and result < minimum:
        raise ConfigurationError(f"{field} must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise ConfigurationError(f"{field} must be <= {maximum}.")
    return result


def _fraction(value: Any, field: str, *, include_one: bool = True) -> float:
    result = _number(value, field, minimum=0.0, maximum=1.0)
    if not include_one and result == 1.0:
        raise ConfigurationError(f"{field} must be less than 1.")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field} must be true or false.")
    return value


def _choice(value: Any, field: str, choices: set[str] | frozenset[str]) -> str:
    result = _text(value, field).lower()
    if result not in choices:
        raise ConfigurationError(f"{field} must be one of: {', '.join(sorted(choices))}.")
    return result


def _single_character(value: Any, field: str) -> str:
    result = _text(value, field)
    if len(result) != 1:
        raise ConfigurationError(f"{field} must be exactly one character.")
    return result


def _path(value: Any, field: str, base: Path) -> Path:
    raw = _text(value, field)
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
