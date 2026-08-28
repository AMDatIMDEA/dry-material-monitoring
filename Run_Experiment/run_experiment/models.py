"""Typed configuration, state, timing, and outcome models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from material_level_yolo.domain import CameraSettings


class ExperimentState(StrEnum):
    INITIALIZING = "INITIALIZING"
    ARMED = "ARMED"
    CAPTURING = "CAPTURING"
    PROCESSING = "PROCESSING"
    SAVED = "SAVED"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class AcquisitionSettings:
    frame_count: int
    span_seconds: float
    worker_count: int
    scheduling_poll_seconds: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: int
    config_path: Path
    depth_config_path: Path
    depth_calibration_path: Path
    c920: CameraSettings
    acquisition: AcquisitionSettings
    output_root: Path
    storage_profile: str


@dataclass(frozen=True, slots=True)
class ExperimentInputs:
    experiment_id: str
    purpose: str
    material_name: str | None = None
    total_capacity_ml: float | None = None
    bulk_density_g_per_ml: float | None = None
    total_possible_weight_g: float | None = None
    # Retained for API compatibility as first-trigger-only inputs. New callers
    # should pass MeasurementReferenceInputs to SynchronizedExperiment.trigger().
    reference_material_weight_g: float | None = None
    manual_material_level_mm: float | None = None
    operator_notes: str | None = None
    output_root: Path | None = None


@dataclass(frozen=True, slots=True)
class ValidatedExperimentInputs:
    experiment_id: str
    purpose: str
    material_name: str | None
    total_capacity_ml: float | None
    bulk_density_g_per_ml: float | None
    total_possible_weight_g: float | None
    reference_material_weight_g: float | None
    manual_material_level_mm: float | None
    operator_notes: str | None
    output_root: Path


@dataclass(frozen=True, slots=True)
class MeasurementReferenceInputs:
    """Independent reference observations entered for one trigger only."""

    reference_material_weight_g: float | None = None
    manual_material_level_mm: float | None = None
    measurement_notes: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatedMeasurementReferenceInputs:
    reference_material_weight_g: float | None
    manual_material_level_mm: float | None
    measurement_notes: str | None


@dataclass(frozen=True, slots=True)
class C920FrameTiming:
    index: int
    path: Path
    sha256: str
    target_offset_seconds: float
    target_monotonic: float
    target_utc: str
    actual_monotonic: float
    timing_error_seconds: float
    actual_utc: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class C920CaptureResult:
    frames: tuple[C920FrameTiming, ...]
    planned_count: int
    start_monotonic: float | None
    end_monotonic: float | None
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class DepthCaptureResult:
    captured: Any | None
    start_monotonic: float
    end_monotonic: float
    capture_start_utc: str | None
    capture_end_utc: str | None
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    measurement_id: str
    measurement_index: int
    session_directory: Path
    measurement_directory: Path
    capture_manifest: Path
    c920: C920CaptureResult
    depth_outcome: Any | None
    depth_record: Any
    yolo_record: Any
    final_state: ExperimentState
    overlap_duration_seconds: float
    state_history: tuple[ExperimentState, ...]
