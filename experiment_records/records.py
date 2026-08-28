"""Typed common measurement records and contract-level calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
import math
from typing import Any, Mapping

from .schema import COMMON_COLUMNS, SCHEMA_VERSION


TimestampValue = datetime | str


@dataclass(frozen=True, slots=True)
class CommonMeasurementRecord:
    """One method-neutral row, declared in the exact common-column order."""

    schema_version: str = SCHEMA_VERSION
    experiment_id: str | None = None
    measurement_id: str | None = None
    measurement_index: int | None = None
    method: str | None = None
    acquisition_mode: str | None = None
    trigger_time_utc: TimestampValue | None = None
    capture_start_utc: TimestampValue | None = None
    capture_end_utc: TimestampValue | None = None
    processing_time_utc: TimestampValue | None = None
    material_name: str | None = None
    total_capacity_ml: float | None = None
    bulk_density_g_per_ml: float | None = None
    total_possible_weight_g: float | None = None
    reference_material_weight_g: float | None = None
    reference_volume_from_weight_ml: float | None = None
    manual_material_level_mm: float | None = None
    manual_material_percent: float | None = None
    reference_volume_from_manual_ml: float | None = None
    estimated_material_percent: float | None = None
    estimated_material_volume_ml: float | None = None
    estimated_empty_percent: float | None = None
    estimated_empty_volume_ml: float | None = None
    valid: bool | None = None
    quality_score: float | None = None
    status: str | None = None
    notes: str | None = None
    artifact_directory: str | None = None
    source_artifact: str | None = None
    config_sha256: str | None = None
    calibration_or_model_sha256: str | None = None
    software_commit: str | None = None

    def key(self) -> tuple[str | None, str | None, str | None]:
        return self.experiment_id, self.measurement_id, self.method

    def to_mapping(self) -> dict[str, Any]:
        values = asdict(self)
        return {column: values[column] for column in COMMON_COLUMNS}

    def to_row(self) -> tuple[Any, ...]:
        values = self.to_mapping()
        return tuple(values[column] for column in COMMON_COLUMNS)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CommonMeasurementRecord":
        return cls(**{column: values.get(column) for column in COMMON_COLUMNS})


if tuple(field.name for field in fields(CommonMeasurementRecord)) != COMMON_COLUMNS:
    raise RuntimeError("CommonMeasurementRecord fields do not match COMMON_COLUMNS.")


def derive_weight_reference_ml(
    reference_material_weight_g: float | None,
    bulk_density_g_per_ml: float | None,
) -> float | None:
    """Calculate a weight reference only when both explicit inputs exist."""
    if reference_material_weight_g is None or bulk_density_g_per_ml is None:
        return None
    weight = float(reference_material_weight_g)
    density = float(bulk_density_g_per_ml)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("reference_material_weight_g must be finite and nonnegative.")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("bulk_density_g_per_ml must be finite and greater than zero.")
    return weight / density


def derive_manual_references(
    manual_material_level_mm: float | None,
    usable_internal_height_mm: float | None,
    total_capacity_ml: float | None,
) -> tuple[float | None, float | None]:
    """Calculate straight-tube manual percent and volume from explicit geometry."""
    if (
        manual_material_level_mm is None
        or usable_internal_height_mm is None
        or total_capacity_ml is None
    ):
        return None, None
    level = float(manual_material_level_mm)
    height = float(usable_internal_height_mm)
    capacity = float(total_capacity_ml)
    if not math.isfinite(level) or level < 0.0:
        raise ValueError("manual_material_level_mm must be finite and nonnegative.")
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("usable_internal_height_mm must be finite and greater than zero.")
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("total_capacity_ml must be finite and greater than zero.")
    percent = 100.0 * level / height
    volume = capacity * percent / 100.0
    return percent, volume


def derive_estimate(
    estimated_material_percent: float,
    total_capacity_ml: float,
) -> tuple[float, float, float, float]:
    """Return material percent/volume and their conserving empty complements."""
    material_percent = float(estimated_material_percent)
    capacity = float(total_capacity_ml)
    if not math.isfinite(material_percent) or not 0.0 <= material_percent <= 100.0:
        raise ValueError("estimated_material_percent must be finite and in [0, 100].")
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("total_capacity_ml must be finite and greater than zero.")
    empty_percent = 100.0 - material_percent
    material_volume = capacity * material_percent / 100.0
    empty_volume = capacity - material_volume
    return material_percent, material_volume, empty_percent, empty_volume


def with_estimate(
    record: CommonMeasurementRecord,
    estimated_material_percent: float,
) -> CommonMeasurementRecord:
    """Fill all four estimate fields from one explicit accepted percentage."""
    if record.total_capacity_ml is None:
        raise ValueError("total_capacity_ml is required before deriving an estimate.")
    material_percent, material_volume, empty_percent, empty_volume = derive_estimate(
        estimated_material_percent, record.total_capacity_ml
    )
    return replace(
        record,
        estimated_material_percent=material_percent,
        estimated_material_volume_ml=material_volume,
        estimated_empty_percent=empty_percent,
        estimated_empty_volume_ml=empty_volume,
    )
