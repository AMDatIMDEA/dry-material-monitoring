"""Operator workflow for folder input and C920 still capture without live inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping

import numpy as np

from experiment_records import (
    format_utc,
    parse_measurement_id,
    validate_experiment_id,
    validate_measurement_identity,
    volume_tolerance,
)

from .config import ProjectConfig
from .camera import (
    CameraPreviewProtocol,
    OpenCVCameraPreview,
    PreviewEvent,
    warm_up_preview,
)
from .domain import CameraSettings, ModelProfile
from .errors import AcquisitionError, OperatorInputError
from .image_io import write_image
from .inference import UltralyticsInferenceAdapter
from .workflow import (
    InferenceAdapterProtocol,
    OfflineMeasurementOutcome,
    OfflineMeasurementRequest,
    process_saved_images,
)


SUPPORTED_ACQUISITION_MODES = frozenset({"manual_camera", "timed_camera", "folder"})
SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


@dataclass(frozen=True, slots=True)
class OperatorInputs:
    profile_name: str | None
    experiment_id: str
    purpose: str
    total_capacity_ml: float
    acquisition_mode: str
    output_root: Path | None = None
    folder_path: Path | None = None
    recursive: bool = False
    bulk_density_g_per_ml: float | None = None
    total_possible_weight_g: float | None = None
    remaining_weight_g: float | None = None
    manual_material_level: float | None = None
    manual_material_level_unit: str | None = None
    usable_internal_height_mm: float | None = None
    material_name: str | None = None
    operator_notes: str | None = None
    measurement_index: int | None = None
    measurement_id: str | None = None
    trigger_time_utc: datetime | str | None = None
    timed_capture_count: int | None = None
    capture_interval_seconds: float | None = None
    capture_duration_seconds: float | None = None
    software_repository: Path | None = None


@dataclass(frozen=True, slots=True)
class ValidatedOperatorInputs:
    profile: ModelProfile
    experiment_id: str
    purpose: str
    total_capacity_ml: float
    acquisition_mode: str
    output_root: Path
    folder_images: tuple[Path, ...]
    recursive: bool
    bulk_density_g_per_ml: float | None
    total_possible_weight_g: float | None
    remaining_weight_g: float | None
    manual_material_level_mm: float | None
    manual_material_level_unit: str | None
    usable_internal_height_mm: float | None
    material_name: str | None
    operator_notes: str | None
    measurement_index: int | None
    measurement_id: str | None
    trigger_time_utc: datetime | str | None
    timed_capture_count: int | None
    capture_interval_seconds: float | None
    capture_duration_seconds: float | None
    software_repository: Path | None

    def confirmation_summary(self) -> dict[str, Any]:
        camera = self.profile.camera
        schedule: dict[str, Any] | None = None
        if self.acquisition_mode == "timed_camera":
            schedule = {
                "capture_count": self.timed_capture_count,
                "duration_seconds": self.capture_duration_seconds,
                "interval_seconds": self.capture_interval_seconds,
            }
        return {
            "model_profile": self.profile.name,
            "configured_material_role": self.profile.material_role.class_name,
            "experiment_id": self.experiment_id,
            "purpose": self.purpose,
            "material_name": self.material_name,
            "total_capacity_ml": self.total_capacity_ml,
            "bulk_density_g_per_ml": self.bulk_density_g_per_ml,
            "total_possible_weight_g": self.total_possible_weight_g,
            "independently_measured_remaining_weight_g": self.remaining_weight_g,
            "manual_material_level": (
                None
                if self.manual_material_level_mm is None
                else {
                    "entered_unit": self.manual_material_level_unit,
                    "converted_mm": self.manual_material_level_mm,
                }
            ),
            "operator_notes": self.operator_notes,
            "acquisition_mode": self.acquisition_mode,
            "output_root": str(self.output_root),
            "folder_image_count": len(self.folder_images),
            "folder_recursion": self.recursive if self.acquisition_mode == "folder" else None,
            "timed_schedule": schedule,
            "minimum_valid_images_required": (
                self.profile.aggregation.manual_minimum_valid_images
                if self.acquisition_mode == "manual_camera"
                else self.profile.aggregation.minimum_valid_images
            ),
            "roi_mode": self.profile.tube_roi.usage_mode,
            "roi_bounds": {
                "left": self.profile.tube_roi.left,
                "top": self.profile.tube_roi.top,
                "right": self.profile.tube_roi.right,
                "bottom": self.profile.tube_roi.bottom,
            },
            "camera": (
                None
                if self.acquisition_mode == "folder"
                else {
                    "device_index": camera.device_index,
                    "device_name_contains": camera.device_name_contains,
                    "backend": camera.backend,
                    "requested_resolution": [camera.width, camera.height],
                    "requested_fps": camera.fps,
                    "warmup_frames": camera.warmup_frames,
                    "capture_key": camera.capture_key,
                    "quit_key": camera.quit_key,
                    "left_click_captures": self.acquisition_mode == "manual_camera",
                    "mirror_preview": camera.mirror_preview,
                    "close_on_capture": camera.close_on_capture,
                    "freeze_after_capture_ms": camera.freeze_after_capture_ms,
                }
            ),
            "continuous_or_preview_inference": False,
        }


@dataclass(frozen=True, slots=True)
class OperatorWorkflowResult:
    outcome: OfflineMeasurementOutcome | None
    cancelled: bool
    cancellation_reason: str | None
    captured_still_count: int


@dataclass(frozen=True, slots=True)
class _CapturedStills:
    paths: tuple[Path, ...]
    capture_start_utc: datetime
    capture_end_utc: datetime


def discover_images(source: str | Path, *, recursive: bool = False) -> tuple[Path, ...]:
    """Discover supported saved stills deterministically; never decode or mutate them."""
    resolved = Path(source).expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
            raise OperatorInputError(
                f"Unsupported image extension: {resolved.suffix or '<none>'}"
            )
        return (resolved,)
    if not resolved.is_dir():
        raise OperatorInputError(f"Input image or folder was not found: {resolved}")
    iterator = resolved.rglob("*") if recursive else resolved.iterdir()
    images = tuple(
        sorted(
            (
                path.resolve()
                for path in iterator
                if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: (path.as_posix().casefold(), path.as_posix()),
        )
    )
    if not images:
        scope = "recursively " if recursive else ""
        raise OperatorInputError(
            f"No supported saved images were found {scope}in: {resolved}"
        )
    return images


def validate_operator_inputs(
    project: ProjectConfig,
    inputs: OperatorInputs,
) -> ValidatedOperatorInputs:
    """Validate and canonicalize all operator values before confirmation/acquisition."""
    material_name = _optional_text("material_name", inputs.material_name)
    profile = (
        project.select_profile_for_material(material_name, inputs.profile_name)
        if material_name is not None
        else project.select_profile(inputs.profile_name)
    )
    profile.require_frozen_roi()
    try:
        experiment_id = validate_experiment_id(inputs.experiment_id)
    except (TypeError, ValueError) as exc:
        raise OperatorInputError(str(exc)) from exc
    purpose = _required_text("purpose", inputs.purpose)
    mode = _required_text("acquisition_mode", inputs.acquisition_mode)
    if mode not in SUPPORTED_ACQUISITION_MODES:
        raise OperatorInputError(
            "acquisition_mode must be exactly one of: folder, manual_camera, timed_camera."
        )
    capacity = _positive("total_capacity_ml", inputs.total_capacity_ml)
    density = _optional_positive("bulk_density_g_per_ml", inputs.bulk_density_g_per_ml)
    total_weight = _optional_nonnegative(
        "total_possible_weight_g", inputs.total_possible_weight_g
    )
    remaining_weight = _optional_nonnegative("remaining_weight_g", inputs.remaining_weight_g)
    usable_height = _optional_positive(
        "usable_internal_height_mm", inputs.usable_internal_height_mm
    )
    if density is not None and total_weight is not None:
        expected = capacity * density
        if not math.isclose(
            total_weight,
            expected,
            rel_tol=0.0,
            abs_tol=volume_tolerance(expected),
        ):
            raise OperatorInputError(
                "total_possible_weight_g conflicts with total_capacity_ml * bulk_density_g_per_ml."
            )
    manual_mm, manual_unit = _manual_level(
        inputs.manual_material_level,
        inputs.manual_material_level_unit,
    )
    if manual_mm is not None:
        if usable_height is None:
            raise OperatorInputError(
                "usable_internal_height_mm is required with a manual material level."
            )
        if manual_mm > usable_height:
            raise OperatorInputError(
                "manual material level cannot exceed usable_internal_height_mm."
            )
    output_root = profile.output.root if inputs.output_root is None else Path(inputs.output_root).expanduser()
    if not output_root.is_absolute():
        raise OperatorInputError("output_root must be an absolute path.")
    output_root = output_root.resolve()
    manifest_path = output_root / experiment_id / "session_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OperatorInputError(
                f"Existing session manifest cannot be validated: {manifest_path}"
            ) from exc
        if manifest.get("experiment_id") != experiment_id:
            raise OperatorInputError("Existing session manifest has a different experiment_id.")
        if manifest.get("purpose") not in {None, purpose}:
            raise OperatorInputError("Existing session manifest has a different purpose.")

    images: tuple[Path, ...] = ()
    if mode == "folder":
        if inputs.folder_path is None:
            raise OperatorInputError("folder mode requires an input image or folder path.")
        images = discover_images(inputs.folder_path, recursive=inputs.recursive)
    elif inputs.folder_path is not None:
        raise OperatorInputError("Camera modes do not accept a folder_path.")

    count = inputs.timed_capture_count
    interval = inputs.capture_interval_seconds
    duration = inputs.capture_duration_seconds
    if mode == "timed_camera":
        interval = _positive(
            "capture_interval_seconds",
            profile.camera.capture_interval_seconds if interval is None else interval,
        )
        if count is not None and duration is not None:
            raise OperatorInputError(
                "Choose timed capture_count or capture_duration_seconds, not both."
            )
        if count is None and duration is None:
            if profile.camera.capture_duration_seconds is not None:
                duration = profile.camera.capture_duration_seconds
            else:
                count = profile.camera.capture_count
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise OperatorInputError("timed_capture_count must be an integer >= 1.")
        duration = _optional_positive("capture_duration_seconds", duration)
    elif any(value is not None for value in (count, interval, duration)):
        raise OperatorInputError("Timed scheduling inputs are valid only for timed_camera.")

    if inputs.measurement_index is not None and (
        isinstance(inputs.measurement_index, bool)
        or not isinstance(inputs.measurement_index, int)
        or inputs.measurement_index < 1
    ):
        raise OperatorInputError("measurement_index must be an integer >= 1.")
    trigger_time = inputs.trigger_time_utc
    try:
        if trigger_time is not None:
            trigger_time = format_utc(trigger_time)
        if inputs.measurement_id is not None:
            parsed = parse_measurement_id(inputs.measurement_id)
            identity_index = parsed.index if inputs.measurement_index is None else inputs.measurement_index
            identity_trigger = parsed.trigger_time_utc if trigger_time is None else trigger_time
            validate_measurement_identity(
                inputs.measurement_id,
                identity_index,
                identity_trigger,
            )
    except (TypeError, ValueError) as exc:
        raise OperatorInputError(str(exc)) from exc
    notes = _optional_text("operator_notes", inputs.operator_notes)
    return ValidatedOperatorInputs(
        profile=profile,
        experiment_id=experiment_id,
        purpose=purpose,
        total_capacity_ml=capacity,
        acquisition_mode=mode,
        output_root=output_root,
        folder_images=images,
        recursive=bool(inputs.recursive),
        bulk_density_g_per_ml=density,
        total_possible_weight_g=total_weight,
        remaining_weight_g=remaining_weight,
        manual_material_level_mm=manual_mm,
        manual_material_level_unit=manual_unit,
        usable_internal_height_mm=usable_height,
        material_name=material_name,
        operator_notes=notes,
        measurement_index=inputs.measurement_index,
        measurement_id=inputs.measurement_id,
        trigger_time_utc=trigger_time,
        timed_capture_count=count,
        capture_interval_seconds=interval,
        capture_duration_seconds=duration,
        software_repository=inputs.software_repository,
    )


def run_operator_workflow(
    project: ProjectConfig,
    inputs: OperatorInputs,
    *,
    confirm: Callable[[Mapping[str, Any]], bool],
    adapter: InferenceAdapterProtocol | None = None,
    camera: CameraPreviewProtocol | None = None,
    utc_now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> OperatorWorkflowResult:
    """Confirm, acquire deliberate stills, close acquisition, then run inference."""
    validated = validate_operator_inputs(project, inputs)
    if not confirm(validated.confirmation_summary()):
        return OperatorWorkflowResult(None, True, "confirmation_declined", 0)
    utc_clock = utc_now or (lambda: datetime.now(timezone.utc))
    monotonic_clock = monotonic or time.monotonic

    if validated.acquisition_mode == "folder":
        outcome = process_saved_images(
            project,
            validated.profile.name,
            validated.folder_images,
            _offline_request(validated, capture=None),
            adapter=adapter,
            now_utc=utc_clock,
        )
        return OperatorWorkflowResult(outcome, False, None, len(validated.folder_images))

    inference = adapter or UltralyticsInferenceAdapter(validated.profile)
    print(
        "STAGE MODEL_LOADING - loading and validating YOLO weights before "
        "the camera opens. Large models can take one or more minutes."
    )
    inference.load()  # Fail model compatibility before the camera is opened.
    print("STAGE MODEL_READY - model task and semantic classes are valid.")
    preview = camera or OpenCVCameraPreview()
    with tempfile.TemporaryDirectory(prefix="material_level_yolo_capture_") as raw_spool:
        spool = Path(raw_spool)
        captured: _CapturedStills | None = None
        cancellation: str | None = None
        print("STAGE CAMERA_OPENING - resolving and opening the configured camera.")
        preview.open(validated.profile.camera)
        print("STAGE CAMERA_WARMUP - camera opened; warming preview frames.")
        try:
            cancellation = warm_up_preview(preview, validated.profile.camera)
            if cancellation is None:
                if validated.acquisition_mode == "manual_camera":
                    captured, cancellation = _capture_manual(
                        preview,
                        validated.profile.camera,
                        spool,
                        utc_clock,
                    )
                else:
                    captured, cancellation = _capture_timed(
                        preview,
                        validated,
                        spool,
                        utc_clock,
                        monotonic_clock,
                    )
        finally:
            preview.close()
        if cancellation is not None or captured is None:
            count = 0 if captured is None else len(captured.paths)
            return OperatorWorkflowResult(None, True, cancellation or "capture_cancelled", count)
        outcome = process_saved_images(
            project,
            validated.profile.name,
            captured.paths,
            _offline_request(validated, capture=captured),
            adapter=inference,
            now_utc=utc_clock,
        )
        return OperatorWorkflowResult(outcome, False, None, len(captured.paths))


def _capture_manual(
    preview: CameraPreviewProtocol,
    settings: CameraSettings,
    spool: Path,
    utc_now: Callable[[], datetime],
) -> tuple[_CapturedStills | None, str | None]:
    while True:
        frame = preview.read()
        if frame is None:
            raise AcquisitionError("Camera read failed before a manual still was captured.")
        preview.show(
            frame,
            f"{settings.capture_key}/left click: capture | {settings.quit_key}/Esc: cancel",
        )
        event = preview.poll_event(settings.preview_wait_ms)
        if event in {PreviewEvent.CANCEL, PreviewEvent.CLOSED}:
            return None, "window_closed" if event is PreviewEvent.CLOSED else "operator_cancelled"
        if event is PreviewEvent.CAPTURE:
            stamp = _aware_utc(utc_now())
            path = write_image(spool / f"captured_000001{settings_image_suffix(settings)}", frame)
            if not settings.close_on_capture and settings.freeze_after_capture_ms > 0:
                preview.show(frame, "Still captured - acquisition frozen")
                preview.poll_event(settings.freeze_after_capture_ms)
            return _CapturedStills((path,), stamp, stamp), None


def _capture_timed(
    preview: CameraPreviewProtocol,
    inputs: ValidatedOperatorInputs,
    spool: Path,
    utc_now: Callable[[], datetime],
    monotonic: Callable[[], float],
) -> tuple[_CapturedStills | None, str | None]:
    settings = inputs.profile.camera
    assert inputs.capture_interval_seconds is not None
    target_count = inputs.timed_capture_count
    if target_count is None:
        assert inputs.capture_duration_seconds is not None
        target_count = math.floor(
            inputs.capture_duration_seconds / inputs.capture_interval_seconds + 1e-12
        ) + 1
    start = monotonic()
    next_target = start
    paths: list[Path] = []
    first_stamp: datetime | None = None
    last_stamp: datetime | None = None
    while len(paths) < target_count:
        frame = preview.read()
        if frame is None:
            raise AcquisitionError(
                f"Camera read failed after {len(paths)} of {target_count} timed stills."
            )
        preview.show(
            frame,
            f"Timed capture {len(paths)}/{target_count} | {settings.quit_key}/Esc: cancel",
        )
        now = monotonic()
        if now >= next_target:
            stamp = _aware_utc(utc_now())
            path = write_image(
                spool / f"captured_{len(paths) + 1:06d}{settings_image_suffix(settings)}",
                frame,
            )
            paths.append(path)
            first_stamp = first_stamp or stamp
            last_stamp = stamp
            next_target = start + len(paths) * inputs.capture_interval_seconds
        event = preview.poll_event(settings.preview_wait_ms)
        if event in {PreviewEvent.CANCEL, PreviewEvent.CLOSED}:
            reason = "window_closed" if event is PreviewEvent.CLOSED else "operator_cancelled"
            captured = (
                None
                if not paths
                else _CapturedStills(tuple(paths), first_stamp, last_stamp)  # type: ignore[arg-type]
            )
            return captured, reason
    assert first_stamp is not None and last_stamp is not None
    return _CapturedStills(tuple(paths), first_stamp, last_stamp), None


def _offline_request(
    inputs: ValidatedOperatorInputs,
    *,
    capture: _CapturedStills | None,
) -> OfflineMeasurementRequest:
    trigger = inputs.trigger_time_utc
    if trigger is None and capture is not None:
        trigger = capture.capture_start_utc
    return OfflineMeasurementRequest(
        experiment_id=inputs.experiment_id,
        purpose=inputs.purpose,
        total_capacity_ml=inputs.total_capacity_ml,
        output_root=inputs.output_root,
        measurement_index=inputs.measurement_index,
        measurement_id=inputs.measurement_id,
        trigger_time_utc=trigger,
        material_name=inputs.material_name,
        bulk_density_g_per_ml=inputs.bulk_density_g_per_ml,
        total_possible_weight_g=inputs.total_possible_weight_g,
        reference_material_weight_g=inputs.remaining_weight_g,
        manual_material_level_mm=inputs.manual_material_level_mm,
        usable_internal_height_mm=inputs.usable_internal_height_mm,
        notes=inputs.operator_notes,
        acquisition_mode=inputs.acquisition_mode,
        capture_start_utc=(None if capture is None else capture.capture_start_utc),
        capture_end_utc=(None if capture is None else capture.capture_end_utc),
        minimum_valid_images=(
            inputs.profile.aggregation.manual_minimum_valid_images
            if inputs.acquisition_mode == "manual_camera"
            else None
        ),
        software_repository=inputs.software_repository,
    )


def settings_image_suffix(settings: CameraSettings) -> str:
    del settings
    # Camera source evidence uses lossless PNG regardless of overlay output format.
    return ".png"


def _manual_level(value: Any, unit: Any) -> tuple[float | None, str | None]:
    if value is None and unit is None:
        return None, None
    if value is None or unit is None:
        raise OperatorInputError("Manual material level requires both a value and explicit unit.")
    number = _optional_nonnegative("manual_material_level", value)
    assert number is not None
    if not isinstance(unit, str) or unit.strip().casefold() not in {"mm", "cm"}:
        raise OperatorInputError("manual_material_level_unit must be 'mm' or 'cm'.")
    normalized = unit.strip().casefold()
    return number * (10.0 if normalized == "cm" else 1.0), normalized


def _required_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OperatorInputError(f"{name} must be non-blank text.")
    return value.strip()


def _optional_text(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _required_text(name, value)


def _positive(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise OperatorInputError(f"{name} must be greater than zero.")
    return result


def _optional_positive(name: str, value: Any) -> float | None:
    return None if value is None else _positive(name, value)


def _optional_nonnegative(name: str, value: Any) -> float | None:
    if value is None:
        return None
    result = _finite(name, value)
    if result < 0.0:
        raise OperatorInputError(f"{name} must be nonnegative.")
    return result


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorInputError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise OperatorInputError(f"{name} must be finite.")
    return result


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AcquisitionError("The acquisition UTC clock returned a naive timestamp.")
    return value.astimezone(timezone.utc)
