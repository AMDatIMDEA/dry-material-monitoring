"""Normalization and validation for the common measurement contract."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import math
import re
from typing import Any

from .errors import RecordValidationError
from .identifiers import format_utc, parse_utc, validate_experiment_id, validate_measurement_identity
from .records import CommonMeasurementRecord, derive_manual_references, derive_weight_reference_ml
from .schema import (
    ABSOLUTE_VOLUME_TOLERANCE_ML,
    COMMON_COLUMNS,
    METHOD_ACQUISITION_MODES,
    PERCENT_TOLERANCE,
    RELATIVE_FORMULA_TOLERANCE,
    SCHEMA_VERSION,
    AcquisitionMode,
    Method,
    RecordStatus,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOFTWARE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}(?:\+dirty)?$")
ESTIMATE_FIELDS = (
    "estimated_material_percent",
    "estimated_material_volume_ml",
    "estimated_empty_percent",
    "estimated_empty_volume_ml",
)
TIMESTAMP_FIELDS = (
    "trigger_time_utc",
    "capture_start_utc",
    "capture_end_utc",
    "processing_time_utc",
)
NONNEGATIVE_FIELDS = (
    "total_possible_weight_g",
    "reference_material_weight_g",
    "reference_volume_from_weight_ml",
    "manual_material_level_mm",
    "reference_volume_from_manual_ml",
    "estimated_material_volume_ml",
    "estimated_empty_volume_ml",
)
PERCENT_FIELDS = (
    "manual_material_percent",
    "estimated_material_percent",
    "estimated_empty_percent",
)
ALIGNMENT_FIELDS = (
    "schema_version",
    "experiment_id",
    "measurement_id",
    "measurement_index",
    "trigger_time_utc",
    "material_name",
    "total_capacity_ml",
    "bulk_density_g_per_ml",
    "total_possible_weight_g",
    "reference_material_weight_g",
    "reference_volume_from_weight_ml",
    "manual_material_level_mm",
    "manual_material_percent",
    "reference_volume_from_manual_ml",
    "notes",
)


def volume_tolerance(capacity_ml: float) -> float:
    return max(ABSOLUTE_VOLUME_TOLERANCE_ML, abs(capacity_ml) * RELATIVE_FORMULA_TOLERANCE)


def formula_tolerance(expected: float) -> float:
    return max(ABSOLUTE_VOLUME_TOLERANCE_ML, abs(expected) * RELATIVE_FORMULA_TOLERANCE)


def _finite_number(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordValidationError(f"{name} must be a finite number or blank.")
    number = float(value)
    if not math.isfinite(number):
        raise RecordValidationError(f"{name} must be finite.")
    return number


def _required_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{name} must be non-blank text.")
    return value


def _require_blank_estimates(record: CommonMeasurementRecord) -> None:
    present = [name for name in ESTIMATE_FIELDS if getattr(record, name) is not None]
    if present:
        raise RecordValidationError(
            f"Estimate fields must be blank for status={record.status}: {', '.join(present)}"
        )


def _require_hash(name: str, value: str | None) -> None:
    if value is None or SHA256_PATTERN.fullmatch(value) is None:
        raise RecordValidationError(f"{name} must be a 64-character lowercase SHA-256 hash.")


def prepare_record(
    record: CommonMeasurementRecord,
    *,
    usable_internal_height_mm: float | None = None,
) -> CommonMeasurementRecord:
    """Derive permitted reference values, canonicalize timestamps, and validate."""
    if not isinstance(record, CommonMeasurementRecord):
        raise TypeError("record must be a CommonMeasurementRecord.")

    timestamp_updates: dict[str, str | None] = {}
    for name in TIMESTAMP_FIELDS:
        value = getattr(record, name)
        timestamp_updates[name] = None if value is None else format_utc(value)
    try:
        method = Method(record.method).value
        mode = AcquisitionMode(record.acquisition_mode).value
        status = RecordStatus(record.status).value
    except (TypeError, ValueError):
        # The detailed validation below supplies the field-specific error.
        method = record.method
        mode = record.acquisition_mode
        status = record.status
    prepared = replace(
        record,
        method=method,
        acquisition_mode=mode,
        status=status,
        **timestamp_updates,
    )

    try:
        expected_weight = derive_weight_reference_ml(
            prepared.reference_material_weight_g, prepared.bulk_density_g_per_ml
        )
    except (TypeError, ValueError, ZeroDivisionError):
        expected_weight = None
    if expected_weight is not None and prepared.reference_volume_from_weight_ml is None:
        prepared = replace(prepared, reference_volume_from_weight_ml=expected_weight)

    try:
        manual_percent, manual_volume = derive_manual_references(
            prepared.manual_material_level_mm,
            usable_internal_height_mm,
            prepared.total_capacity_ml,
        )
    except (TypeError, ValueError, ZeroDivisionError):
        manual_percent, manual_volume = None, None
    if manual_percent is not None:
        prepared = replace(
            prepared,
            manual_material_percent=(
                manual_percent
                if prepared.manual_material_percent is None
                else prepared.manual_material_percent
            ),
            reference_volume_from_manual_ml=(
                manual_volume
                if prepared.reference_volume_from_manual_ml is None
                else prepared.reference_volume_from_manual_ml
            ),
        )

    validate_record(prepared, usable_internal_height_mm=usable_internal_height_mm)
    return prepared


def validate_record(
    record: CommonMeasurementRecord,
    *,
    usable_internal_height_mm: float | None = None,
) -> None:
    """Validate one already-derived or caller-supplied common record."""
    if record.schema_version != SCHEMA_VERSION:
        raise RecordValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}, found {record.schema_version!r}."
        )
    experiment_id = validate_experiment_id(_required_text("experiment_id", record.experiment_id))
    del experiment_id
    if record.measurement_index is None or isinstance(record.measurement_index, bool):
        raise RecordValidationError("measurement_index must be an integer greater than or equal to 1.")
    if not isinstance(record.measurement_index, int) or record.measurement_index < 1:
        raise RecordValidationError("measurement_index must be an integer greater than or equal to 1.")
    trigger_text = _required_text("trigger_time_utc", record.trigger_time_utc)
    validate_measurement_identity(
        _required_text("measurement_id", record.measurement_id),
        record.measurement_index,
        trigger_text,
    )

    try:
        method = Method(_required_text("method", record.method))
    except ValueError as exc:
        raise RecordValidationError("method must be 'depth' or 'yolo'.") from exc
    try:
        mode = AcquisitionMode(_required_text("acquisition_mode", record.acquisition_mode))
    except ValueError as exc:
        raise RecordValidationError("acquisition_mode is not allowed by schema 1.0.0.") from exc
    if mode not in METHOD_ACQUISITION_MODES[method]:
        raise RecordValidationError(f"acquisition_mode={mode.value!r} is not valid for method={method.value!r}.")
    try:
        status = RecordStatus(_required_text("status", record.status))
    except ValueError as exc:
        raise RecordValidationError("status is not allowed by schema 1.0.0.") from exc

    capacity = _finite_number("total_capacity_ml", record.total_capacity_ml)
    if capacity is None or capacity <= 0.0:
        raise RecordValidationError("total_capacity_ml must be greater than zero.")
    density = _finite_number("bulk_density_g_per_ml", record.bulk_density_g_per_ml)
    if density is not None and density <= 0.0:
        raise RecordValidationError("bulk_density_g_per_ml must be greater than zero when supplied.")

    numeric_values: dict[str, float | None] = {}
    for name in (*NONNEGATIVE_FIELDS, *PERCENT_FIELDS, "quality_score"):
        numeric_values[name] = _finite_number(name, getattr(record, name))
    for name in NONNEGATIVE_FIELDS:
        value = numeric_values[name]
        if value is not None and value < 0.0:
            raise RecordValidationError(f"{name} must be nonnegative.")
    for name in PERCENT_FIELDS:
        value = numeric_values[name]
        if value is not None and not 0.0 <= value <= 100.0:
            raise RecordValidationError(f"{name} must be in [0, 100].")
    quality = numeric_values["quality_score"]
    if quality is not None and not 0.0 <= quality <= 1.0:
        raise RecordValidationError("quality_score must be in [0, 1].")

    timestamps: list[tuple[str, datetime]] = []
    for name in TIMESTAMP_FIELDS:
        value = getattr(record, name)
        if value is not None:
            timestamps.append((name, parse_utc(value, field_name=name)))
    for (left_name, left), (right_name, right) in zip(timestamps, timestamps[1:]):
        if left > right:
            raise RecordValidationError(f"Timestamps are not chronological: {left_name} > {right_name}.")

    if not isinstance(record.valid, (bool, type(None))):
        raise RecordValidationError("valid must be true, false, or blank.")
    _required_text("artifact_directory", record.artifact_directory)
    for name in ("material_name", "notes", "source_artifact"):
        value = getattr(record, name)
        if value is not None and not isinstance(value, str):
            raise RecordValidationError(f"{name} must be text or blank.")

    if (record.reference_material_weight_g is None) != (record.bulk_density_g_per_ml is None):
        if record.reference_volume_from_weight_ml is not None:
            raise RecordValidationError(
                "reference_volume_from_weight_ml must be blank unless weight and density are both supplied."
            )
    if record.reference_material_weight_g is None and record.bulk_density_g_per_ml is None:
        if record.reference_volume_from_weight_ml is not None:
            raise RecordValidationError(
                "reference_volume_from_weight_ml must be blank without explicit weight and density."
            )
    if record.reference_material_weight_g is not None and density is not None:
        expected = float(record.reference_material_weight_g) / density
        actual = record.reference_volume_from_weight_ml
        if actual is None or not math.isclose(actual, expected, abs_tol=formula_tolerance(expected), rel_tol=0.0):
            raise RecordValidationError("reference_volume_from_weight_ml does not match weight / density.")

    if record.manual_material_level_mm is not None:
        height = _finite_number("usable_internal_height_mm", usable_internal_height_mm)
        if height is None:
            if record.manual_material_percent is not None or record.reference_volume_from_manual_ml is not None:
                raise RecordValidationError(
                    "Manual derived fields require explicit usable_internal_height_mm geometry."
                )
        else:
            if height <= 0.0:
                raise RecordValidationError("usable_internal_height_mm must be greater than zero.")
            if float(record.manual_material_level_mm) > height:
                raise RecordValidationError("manual_material_level_mm exceeds usable_internal_height_mm.")
            expected_percent = 100.0 * float(record.manual_material_level_mm) / height
            expected_volume = capacity * expected_percent / 100.0
            if record.manual_material_percent is None or not math.isclose(
                record.manual_material_percent, expected_percent, abs_tol=PERCENT_TOLERANCE, rel_tol=0.0
            ):
                raise RecordValidationError("manual_material_percent does not match the documented geometry.")
            if record.reference_volume_from_manual_ml is None or not math.isclose(
                record.reference_volume_from_manual_ml,
                expected_volume,
                abs_tol=volume_tolerance(capacity),
                rel_tol=0.0,
            ):
                raise RecordValidationError("reference_volume_from_manual_ml does not match manual percent * capacity.")
    elif record.manual_material_percent is not None or record.reference_volume_from_manual_ml is not None:
        raise RecordValidationError("Manual derived fields require manual_material_level_mm.")

    if record.total_possible_weight_g is not None and density is not None:
        expected_weight = capacity * density
        if not math.isclose(
            record.total_possible_weight_g,
            expected_weight,
            abs_tol=formula_tolerance(expected_weight),
            rel_tol=0.0,
        ):
            raise RecordValidationError("total_possible_weight_g conflicts with capacity * bulk density.")

    folder_without_capture_times = mode is AcquisitionMode.FOLDER
    completed = status in {RecordStatus.COMPLETE_VALID, RecordStatus.COMPLETE_INVALID}
    if completed:
        if record.processing_time_utc is None:
            raise RecordValidationError("Completed records require processing_time_utc.")
        if not folder_without_capture_times and (
            record.capture_start_utc is None or record.capture_end_utc is None
        ):
            raise RecordValidationError("Completed camera records require capture start/end timestamps.")
        _required_text("source_artifact", record.source_artifact)
        _require_hash("config_sha256", record.config_sha256)
        _require_hash("calibration_or_model_sha256", record.calibration_or_model_sha256)
    elif record.processing_time_utc is not None:
        raise RecordValidationError(
            f"processing_time_utc must be blank when status={status.value}; no method result was produced."
        )

    if status is RecordStatus.COMPLETE_VALID:
        if record.valid is not True:
            raise RecordValidationError("complete_valid requires valid=true.")
        if any(getattr(record, name) is None for name in ESTIMATE_FIELDS):
            raise RecordValidationError("complete_valid requires all four estimate fields.")
        material_percent = float(record.estimated_material_percent)
        empty_percent = float(record.estimated_empty_percent)
        material_volume = float(record.estimated_material_volume_ml)
        empty_volume = float(record.estimated_empty_volume_ml)
        if abs(material_percent + empty_percent - 100.0) > PERCENT_TOLERANCE:
            raise RecordValidationError("Material and empty percentages do not conserve 100%.")
        tolerance = volume_tolerance(capacity)
        expected_material = capacity * material_percent / 100.0
        if abs(material_volume - expected_material) > tolerance:
            raise RecordValidationError("estimated_material_volume_ml does not match percent * capacity.")
        if abs(material_volume + empty_volume - capacity) > tolerance:
            raise RecordValidationError("Material and empty volumes do not conserve capacity.")
        if abs(empty_volume - (capacity - material_volume)) > tolerance:
            raise RecordValidationError("estimated_empty_volume_ml does not match capacity - material volume.")
    elif status in {RecordStatus.COMPLETE_INVALID, RecordStatus.ACQUISITION_FAILED, RecordStatus.PROCESSING_FAILED}:
        if record.valid is not False:
            raise RecordValidationError(f"{status.value} requires valid=false.")
        _require_blank_estimates(record)
    else:
        if record.valid is not None:
            raise RecordValidationError(f"{status.value} requires valid to be blank.")
        _require_blank_estimates(record)

    if status is RecordStatus.PENDING_OFFLINE_INFERENCE:
        if method is not Method.YOLO:
            raise RecordValidationError("pending_offline_inference is a YOLO record state.")
        if record.capture_start_utc is None or record.capture_end_utc is None:
            raise RecordValidationError("pending_offline_inference requires capture start/end timestamps.")
        _required_text("source_artifact", record.source_artifact)
    elif status is RecordStatus.PARTIAL_CAPTURE:
        if record.capture_start_utc is None:
            raise RecordValidationError("partial_capture requires capture_start_utc.")
        _required_text("source_artifact", record.source_artifact)
    elif status is RecordStatus.PROCESSING_FAILED:
        if record.capture_start_utc is None or record.capture_end_utc is None:
            raise RecordValidationError("processing_failed requires capture start/end timestamps.")
        _required_text("source_artifact", record.source_artifact)

    for name in ("config_sha256", "calibration_or_model_sha256"):
        value = getattr(record, name)
        if value is not None and SHA256_PATTERN.fullmatch(value) is None:
            raise RecordValidationError(f"{name} must be a 64-character lowercase SHA-256 hash.")
    if record.software_commit is not None and SOFTWARE_COMMIT_PATTERN.fullmatch(record.software_commit) is None:
        raise RecordValidationError("software_commit must be a full lowercase Git hash, optionally suffixed '+dirty'.")

    if tuple(record.to_mapping()) != COMMON_COLUMNS:
        raise RecordValidationError("Record serialization order no longer matches the common schema.")


def validate_aligned_records(
    depth_record: CommonMeasurementRecord,
    yolo_record: CommonMeasurementRecord,
    *,
    usable_internal_height_mm: float | None = None,
) -> tuple[CommonMeasurementRecord, CommonMeasurementRecord]:
    """Validate shared capture/reference fields without comparing method estimates."""
    depth = prepare_record(
        depth_record, usable_internal_height_mm=usable_internal_height_mm
    )
    yolo = prepare_record(
        yolo_record, usable_internal_height_mm=usable_internal_height_mm
    )
    if depth.method != Method.DEPTH.value or yolo.method != Method.YOLO.value:
        raise RecordValidationError(
            "validate_aligned_records requires a depth record followed by a YOLO record."
        )
    mismatches = [name for name in ALIGNMENT_FIELDS if getattr(depth, name) != getattr(yolo, name)]
    if mismatches:
        raise RecordValidationError(
            "Aligned method records have different shared fields: " + ", ".join(mismatches)
        )
    return depth, yolo
