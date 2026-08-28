"""Session-aware D405 acquisition, provenance, evidence, and common records."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Iterator

import numpy as np
import yaml

from experiment_records import (
    CommonMeasurementRecord,
    MeasurementWorkbookStore,
    RecordStatus,
    SCHEMA_VERSION,
    format_utc,
    git_revision,
    make_measurement_id,
    next_measurement_identity,
    parse_measurement_id,
    prepare_record,
    sha256_file,
    validate_experiment_id,
    validate_measurement_identity,
    volume_tolerance,
    with_estimate,
)

from .config import AppConfig
from .errors import MeasurementQualityError
from .export import ArtifactSelection, save_measurement_artifacts
from .models import CalibrationData, DepthBurst, VolumeEstimate
from .pipeline import MaterialVolumePipeline, RefillWarningLatch


class StorageProfile(StrEnum):
    COMPACT = "compact"
    RESEARCH = "research"
    FULL_RAW = "full_raw"


@dataclass(slots=True, frozen=True)
class DepthMeasurementRequest:
    """Noninteractive inputs for one session-aware depth measurement."""

    experiment_id: str
    purpose: str | None = None
    operator_notes: str | None = None
    material_name: str | None = None
    total_capacity_ml: float | None = None
    bulk_density_g_per_ml: float | None = None
    total_possible_weight_g: float | None = None
    reference_material_weight_g: float | None = None
    manual_material_level_mm: float | None = None
    output_root: Path | None = None
    storage_profile: StorageProfile | str | None = None
    measurement_index: int | None = None
    measurement_id: str | None = None
    trigger_time_utc: datetime | str | None = None
    capture_start_utc: datetime | str | None = None
    capture_end_utc: datetime | str | None = None
    processing_time_utc: datetime | str | None = None
    acquisition_mode: str = "standalone_camera"
    calibration_path: Path | None = None
    software_repository: Path | None = None


@dataclass(slots=True, frozen=True)
class CapturedDepthBurst:
    burst: DepthBurst
    trigger_time_utc: str
    capture_start_utc: str
    capture_end_utc: str


@dataclass(slots=True, frozen=True)
class DepthMeasurementOutcome:
    estimate: VolumeEstimate | None
    record: CommonMeasurementRecord
    session_directory: Path
    measurement_directory: Path
    artifact_directory: Path
    capture_manifest: Path
    artifact_manifest: Path


@dataclass(slots=True, frozen=True)
class DepthMeasurementReservation:
    """Durable pre-acquisition identity and paths for an external orchestrator."""

    experiment_id: str
    measurement_id: str
    measurement_index: int
    trigger_time_utc: str
    session_directory: Path
    measurement_directory: Path
    artifact_directory: Path
    config_sha256: str
    calibration_sha256: str
    software_commit: str | None
    _resolved: Any


@dataclass(slots=True, frozen=True)
class _ResolvedSession:
    request: DepthMeasurementRequest
    profile: StorageProfile
    trigger_time_utc: str
    measurement_index: int
    measurement_id: str
    output_root: Path
    session_directory: Path
    measurement_directory: Path
    artifact_directory: Path
    capture_manifest: Path
    artifact_manifest: Path
    config_snapshot: Path
    config_sha256: str
    calibration_reference: str
    calibration_sha256: str
    software_commit: str | None


class PreparedDepthMethod:
    """One already-open source plus the existing calibrated numerical pipeline.

    The caller owns source startup/warm-up and cleanup. Reusing this object keeps
    the RealSense pipeline warm across measurements. Capture and processing are
    separate methods so a later orchestrator can overlap camera acquisition only.
    """

    def __init__(
        self,
        config: AppConfig,
        calibration: CalibrationData,
        source: Any,
        *,
        latch: RefillWarningLatch | None = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.source = source
        self.pipeline = MaterialVolumePipeline(config, calibration)
        self.latch = latch if latch is not None else RefillWarningLatch(config)

    def capture(
        self,
        request: DepthMeasurementRequest,
        *,
        now_utc: Any = None,
    ) -> CapturedDepthBurst:
        """Capture one normal configured burst and retain raw Z16 only when requested."""
        clock = now_utc if now_utc is not None else _utc_now
        trigger = request.trigger_time_utc
        if trigger is None and request.measurement_id is not None:
            trigger = parse_measurement_id(request.measurement_id).trigger_time_utc
        trigger = trigger or clock()
        capture_start = request.capture_start_utc or clock()
        profile = _storage_profile(request, self.config)
        if profile is StorageProfile.FULL_RAW:
            burst = self.source.capture_burst(
                self.config.fusion.burst_frames,
                retain_raw=True,
            )
        else:
            # Preserve compatibility with existing DepthSource implementations
            # whose capture_burst API accepts only the frame count.
            burst = self.source.capture_burst(self.config.fusion.burst_frames)
        capture_end = request.capture_end_utc or clock()
        return CapturedDepthBurst(
            burst=burst,
            trigger_time_utc=format_utc(trigger),
            capture_start_utc=format_utc(capture_start),
            capture_end_utc=format_utc(capture_end),
        )

    def reserve(self, request: DepthMeasurementRequest) -> DepthMeasurementReservation:
        """Allocate durable session identity/evidence before synchronized acquisition."""
        if request.trigger_time_utc is None:
            raise ValueError("A synchronized reservation requires trigger_time_utc.")
        resolved = _prepare_session(self.config, self.calibration, request, captured=None)
        return DepthMeasurementReservation(
            experiment_id=request.experiment_id,
            measurement_id=resolved.measurement_id,
            measurement_index=resolved.measurement_index,
            trigger_time_utc=resolved.trigger_time_utc,
            session_directory=resolved.session_directory,
            measurement_directory=resolved.measurement_directory,
            artifact_directory=resolved.artifact_directory,
            config_sha256=resolved.config_sha256,
            calibration_sha256=resolved.calibration_sha256,
            software_commit=resolved.software_commit,
            _resolved=resolved,
        )

    def process(
        self,
        captured: CapturedDepthBurst,
        request: DepthMeasurementRequest,
        *,
        now_utc: Any = None,
        reservation: DepthMeasurementReservation | None = None,
        manage_capture_manifest: bool = True,
    ) -> DepthMeasurementOutcome:
        """Process a captured burst with the unchanged pipeline and persist one session row."""
        clock = now_utc if now_utc is not None else _utc_now
        if reservation is None:
            resolved = _prepare_session(self.config, self.calibration, request, captured)
        else:
            resolved = reservation._resolved
            if (
                request.experiment_id != reservation.experiment_id
                or request.measurement_id not in {None, reservation.measurement_id}
                or request.measurement_index not in {None, reservation.measurement_index}
            ):
                raise ValueError("Depth request does not match its preallocated reservation.")
        if manage_capture_manifest:
            _write_capture_manifest(resolved, captured, state="CAPTURED")
        try:
            estimate = self.pipeline.measure(captured.burst)
            self.latch.update(estimate)
        except MeasurementQualityError as exc:
            return _save_processing_failure(
                config=self.config,
                resolved=resolved,
                captured=captured,
                error=exc,
                manage_capture_manifest=manage_capture_manifest,
            )

        processing_time = format_utc(request.processing_time_utc or clock())
        return _save_completed_measurement(
            config=self.config,
            resolved=resolved,
            captured=captured,
            estimate=estimate,
            processing_time_utc=processing_time,
            manage_capture_manifest=manage_capture_manifest,
        )

    def run_one(
        self,
        request: DepthMeasurementRequest,
        *,
        now_utc: Any = None,
    ) -> DepthMeasurementOutcome:
        captured = self.capture(request, now_utc=now_utc)
        return self.process(captured, request, now_utc=now_utc)


def run_one_depth_measurement(
    config: AppConfig,
    calibration: CalibrationData,
    source: Any,
    request: DepthMeasurementRequest,
    *,
    latch: RefillWarningLatch | None = None,
) -> DepthMeasurementOutcome:
    """Clean one-measurement API for standalone callers and later orchestration."""
    return PreparedDepthMethod(
        config,
        calibration,
        source,
        latch=latch,
    ).run_one(request)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _storage_profile(request: DepthMeasurementRequest, config: AppConfig) -> StorageProfile:
    return StorageProfile(request.storage_profile or config.research.storage_profile)


def _prepare_session(
    config: AppConfig,
    calibration: CalibrationData,
    request: DepthMeasurementRequest,
    captured: CapturedDepthBurst | None,
) -> _ResolvedSession:
    experiment_id = validate_experiment_id(request.experiment_id)
    profile = _storage_profile(request, config)
    if request.output_root is None:
        output_root = config.research_output_path
    else:
        candidate = Path(request.output_root).expanduser()
        if not candidate.is_absolute():
            raise ValueError("An operator-selected output_root must be an absolute path.")
        output_root = candidate.resolve()
    session_directory = output_root / experiment_id
    session_directory.mkdir(parents=True, exist_ok=True)
    lock_path = session_directory / ".session.lock"

    with _exclusive_lock(lock_path):
        trigger_value = request.trigger_time_utc or (
            None if captured is None else captured.trigger_time_utc
        )
        if trigger_value is None:
            raise ValueError("Depth session allocation requires a trigger timestamp.")
        trigger = format_utc(trigger_value)
        index, measurement_id = _resolve_identity(
            session_directory,
            request.measurement_index,
            request.measurement_id,
            trigger,
        )
        config_snapshot, config_hash = _ensure_effective_config(session_directory, config)
        calibration_reference, calibration_hash = _ensure_calibration_provenance(
            session_directory,
            calibration,
            request.calibration_path,
        )
        repository = (
            Path(request.software_repository).resolve()
            if request.software_repository is not None
            else Path(__file__).resolve().parents[2]
        )
        software_commit = git_revision(repository)
        _ensure_environment_manifest(session_directory)
        _ensure_session_manifest(
            session_directory=session_directory,
            experiment_id=experiment_id,
            purpose=request.purpose,
            profile=profile,
            config_snapshot=config_snapshot,
            config_hash=config_hash,
            calibration_reference=calibration_reference,
            calibration_hash=calibration_hash,
            software_commit=software_commit,
        )
        measurement_directory = session_directory / "measurements" / measurement_id
        artifact_directory = measurement_directory / "depth"
        if measurement_directory.exists():
            raise FileExistsError(
                f"Measurement evidence already exists; refusing to overwrite: {measurement_directory}"
            )
        artifact_directory.mkdir(parents=True, exist_ok=False)

    return _ResolvedSession(
        request=request,
        profile=profile,
        trigger_time_utc=trigger,
        measurement_index=index,
        measurement_id=measurement_id,
        output_root=output_root,
        session_directory=session_directory,
        measurement_directory=measurement_directory,
        artifact_directory=artifact_directory,
        capture_manifest=measurement_directory / "capture_manifest.json",
        artifact_manifest=artifact_directory / "artifact_manifest.json",
        config_snapshot=config_snapshot,
        config_sha256=config_hash,
        calibration_reference=calibration_reference,
        calibration_sha256=calibration_hash,
        software_commit=software_commit,
    )


def _resolve_identity(
    session_directory: Path,
    requested_index: int | None,
    requested_id: str | None,
    trigger_time_utc: str,
) -> tuple[int, str]:
    existing_ids = _durable_measurement_ids(session_directory)
    existing_by_index = {
        parse_measurement_id(value).index: value for value in existing_ids
    }
    if requested_id is not None:
        parsed = parse_measurement_id(requested_id)
        index = parsed.index if requested_index is None else requested_index
        validate_measurement_identity(requested_id, index, trigger_time_utc)
        occupied = existing_by_index.get(index)
        if occupied is not None and occupied != requested_id:
            raise ValueError(
                f"measurement_index {index} is already assigned to {occupied}."
            )
        return index, requested_id
    if requested_index is not None:
        if requested_index in existing_by_index:
            raise ValueError(
                f"measurement_index {requested_index} is already assigned to "
                f"{existing_by_index[requested_index]}."
            )
        measurement_id = make_measurement_id(requested_index, trigger_time_utc)
        return requested_index, measurement_id
    return next_measurement_identity(existing_ids, trigger_time_utc)


def _durable_measurement_ids(session_directory: Path) -> set[str]:
    values: set[str] = set()
    measurements = session_directory / "measurements"
    if measurements.is_dir():
        for child in measurements.iterdir():
            if child.is_dir():
                try:
                    parse_measurement_id(child.name)
                except ValueError:
                    continue
                values.add(child.name)
    workbook = session_directory / "depth_measurements.xlsx"
    if workbook.is_file():
        store = MeasurementWorkbookStore(workbook, method="depth")
        values.update(
            item.measurement_id for item in store.load_records() if item.measurement_id is not None
        )
    return values


def _save_completed_measurement(
    *,
    config: AppConfig,
    resolved: _ResolvedSession,
    captured: CapturedDepthBurst,
    estimate: VolumeEstimate,
    processing_time_utc: str,
    manage_capture_manifest: bool,
) -> DepthMeasurementOutcome:
    _validate_capacity(resolved.request.total_capacity_ml, estimate.result.capacity_ml)
    record = _completed_record(
        config=config,
        resolved=resolved,
        captured=captured,
        estimate=estimate,
        processing_time_utc=processing_time_utc,
    )
    selection = _artifact_selection(resolved.profile)
    save_measurement_artifacts(
        estimate,
        config,
        resolved.artifact_directory,
        selection,
    )
    raw_artifact = _save_raw_burst(resolved, captured)
    diagnostics_path = resolved.artifact_directory / "depth_diagnostics.json"
    _atomic_json(
        diagnostics_path,
        _depth_diagnostics(config, captured, estimate, raw_artifact),
    )
    _write_artifact_manifest(resolved)

    store = MeasurementWorkbookStore(
        resolved.session_directory / "depth_measurements.xlsx",
        method="depth",
    )
    store.upsert(record, usable_internal_height_mm=config.tube.usable_height_mm)
    _finalize_manifests(
        resolved,
        record.status,
        record.valid,
        manage_capture_manifest=manage_capture_manifest,
    )
    return DepthMeasurementOutcome(
        estimate=estimate,
        record=record,
        session_directory=resolved.session_directory,
        measurement_directory=resolved.measurement_directory,
        artifact_directory=resolved.artifact_directory,
        capture_manifest=resolved.capture_manifest,
        artifact_manifest=resolved.artifact_manifest,
    )


def _completed_record(
    *,
    config: AppConfig,
    resolved: _ResolvedSession,
    captured: CapturedDepthBurst,
    estimate: VolumeEstimate,
    processing_time_utc: str,
) -> CommonMeasurementRecord:
    valid = bool(estimate.result.quality.valid)
    record = CommonMeasurementRecord(
        experiment_id=resolved.request.experiment_id,
        measurement_id=resolved.measurement_id,
        measurement_index=resolved.measurement_index,
        method="depth",
        acquisition_mode=resolved.request.acquisition_mode,
        trigger_time_utc=resolved.trigger_time_utc,
        capture_start_utc=(resolved.request.capture_start_utc or captured.capture_start_utc),
        capture_end_utc=(resolved.request.capture_end_utc or captured.capture_end_utc),
        processing_time_utc=processing_time_utc,
        material_name=resolved.request.material_name,
        total_capacity_ml=estimate.result.capacity_ml,
        bulk_density_g_per_ml=resolved.request.bulk_density_g_per_ml,
        total_possible_weight_g=resolved.request.total_possible_weight_g,
        reference_material_weight_g=resolved.request.reference_material_weight_g,
        manual_material_level_mm=resolved.request.manual_material_level_mm,
        valid=valid,
        quality_score=estimate.result.quality.score,
        status=(
            RecordStatus.COMPLETE_VALID.value
            if valid
            else RecordStatus.COMPLETE_INVALID.value
        ),
        notes=resolved.request.operator_notes,
        artifact_directory=_session_relative(resolved.session_directory, resolved.artifact_directory),
        source_artifact=_session_relative(
            resolved.session_directory,
            resolved.artifact_directory / "result.json",
        ),
        config_sha256=resolved.config_sha256,
        calibration_or_model_sha256=resolved.calibration_sha256,
        software_commit=resolved.software_commit,
    )
    if valid:
        record = with_estimate(record, estimate.result.fill_percent)
    record = prepare_record(
        record,
        usable_internal_height_mm=config.tube.usable_height_mm,
    )
    return record


def _save_processing_failure(
    *,
    config: AppConfig,
    resolved: _ResolvedSession,
    captured: CapturedDepthBurst,
    error: MeasurementQualityError,
    manage_capture_manifest: bool,
) -> DepthMeasurementOutcome:
    capacity = config.tube.capacity_ml
    _validate_capacity(resolved.request.total_capacity_ml, capacity)
    record = _failure_record(
        config=config,
        resolved=resolved,
        captured=captured,
        capacity=capacity,
        failure_path=resolved.artifact_directory / "failure.json",
    )
    failure_path = resolved.artifact_directory / "failure.json"
    _atomic_json(
        failure_path,
        {
            "schema_version": 1,
            "status": "processing_failed",
            "error_type": type(error).__name__,
            "message": str(error),
            "failure_time_utc": format_utc(_utc_now()),
        },
    )
    raw_artifact = _save_raw_burst(resolved, captured)
    _atomic_json(
        resolved.artifact_directory / "depth_diagnostics.json",
        {
            "schema_version": 1,
            "quality": None,
            "processing_failure": str(error),
            "raw_artifact": raw_artifact,
            "capture": _capture_metadata(captured),
        },
    )
    _write_artifact_manifest(resolved)
    store = MeasurementWorkbookStore(
        resolved.session_directory / "depth_measurements.xlsx", method="depth"
    )
    store.upsert(record, usable_internal_height_mm=config.tube.usable_height_mm)
    _finalize_manifests(
        resolved,
        record.status,
        record.valid,
        manage_capture_manifest=manage_capture_manifest,
    )
    return DepthMeasurementOutcome(
        estimate=None,
        record=record,
        session_directory=resolved.session_directory,
        measurement_directory=resolved.measurement_directory,
        artifact_directory=resolved.artifact_directory,
        capture_manifest=resolved.capture_manifest,
        artifact_manifest=resolved.artifact_manifest,
    )


def _failure_record(
    *,
    config: AppConfig,
    resolved: _ResolvedSession,
    captured: CapturedDepthBurst,
    capacity: float,
    failure_path: Path,
) -> CommonMeasurementRecord:
    return prepare_record(
        CommonMeasurementRecord(
            experiment_id=resolved.request.experiment_id,
            measurement_id=resolved.measurement_id,
            measurement_index=resolved.measurement_index,
            method="depth",
            acquisition_mode=resolved.request.acquisition_mode,
            trigger_time_utc=resolved.trigger_time_utc,
            capture_start_utc=(resolved.request.capture_start_utc or captured.capture_start_utc),
            capture_end_utc=(resolved.request.capture_end_utc or captured.capture_end_utc),
            material_name=resolved.request.material_name,
            total_capacity_ml=capacity,
            bulk_density_g_per_ml=resolved.request.bulk_density_g_per_ml,
            total_possible_weight_g=resolved.request.total_possible_weight_g,
            reference_material_weight_g=resolved.request.reference_material_weight_g,
            manual_material_level_mm=resolved.request.manual_material_level_mm,
            valid=False,
            status=RecordStatus.PROCESSING_FAILED.value,
            notes=resolved.request.operator_notes,
            artifact_directory=_session_relative(
                resolved.session_directory, resolved.artifact_directory
            ),
            source_artifact=_session_relative(resolved.session_directory, failure_path),
            config_sha256=resolved.config_sha256,
            calibration_or_model_sha256=resolved.calibration_sha256,
            software_commit=resolved.software_commit,
        ),
        usable_internal_height_mm=config.tube.usable_height_mm,
    )


def _validate_capacity(requested: float | None, measured: float) -> None:
    if requested is None:
        return
    value = float(requested)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("total_capacity_ml must be finite and greater than zero.")
    if abs(value - measured) > volume_tolerance(measured):
        raise ValueError(
            "The supplied total_capacity_ml does not match the capacity from the frozen "
            f"D405 tube geometry ({value:.12g} vs {measured:.12g} mL)."
        )


def _artifact_selection(profile: StorageProfile) -> ArtifactSelection:
    if profile is StorageProfile.COMPACT:
        return ArtifactSelection(
            save_json=True,
            save_height_map_npy=False,
            save_height_map_png=True,
            save_surface_ply=False,
            save_color_snapshot=False,
        )
    return ArtifactSelection()


def _save_raw_burst(resolved: _ResolvedSession, captured: CapturedDepthBurst) -> str | None:
    if resolved.profile is not StorageProfile.FULL_RAW:
        return None
    raw_frames = captured.burst.raw_frames_z16
    if raw_frames is None or len(raw_frames) != len(captured.burst.frames_m):
        raise ValueError(
            "full_raw requires one retained lossless source Z16 frame per processed frame; "
            "the source did not supply them."
        )
    raw_directory = resolved.artifact_directory / "raw_depth"
    raw_directory.mkdir(parents=True, exist_ok=True)
    path = raw_directory / "raw_depth_burst_z16.npz"
    temp = _temporary_path(path, ".tmp.npz")
    try:
        np.savez_compressed(
            temp,
            frames_z16=np.stack(raw_frames),
            depth_scale_m=np.array(captured.burst.depth_scale_m, dtype=np.float64),
            frame_timestamps_ms=np.asarray(
                captured.burst.frame_timestamps_ms, dtype=np.float64
            ),
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return _session_relative(resolved.session_directory, path)


def _depth_diagnostics(
    config: AppConfig,
    captured: CapturedDepthBurst,
    estimate: VolumeEstimate,
    raw_artifact: str | None,
) -> dict[str, Any]:
    result = estimate.result
    quality = result.quality
    return {
        "schema_version": 1,
        "quality": {
            "valid": quality.valid,
            "score": quality.score,
            "surface_coverage": quality.surface_coverage,
            "median_temporal_valid_fraction": quality.median_temporal_valid_fraction,
            "temporal_mad_mm": quality.temporal_mad_mm,
            "maximum_internal_hole_radius_mm": quality.maximum_hole_radius_mm,
            "calibration_plane_rms_mm": quality.calibration_plane_rms_mm,
            "spatial_outliers_replaced": quality.spatial_outliers_replaced,
            "heuristic_variability_percent": quality.uncertainty_percent,
            "heuristic_variability_is_gum_uncertainty": False,
            "reasons": quality.reasons,
        },
        "level_mm": {
            "minimum": result.minimum_level_mm,
            "mean": result.mean_level_mm,
            "maximum": result.maximum_level_mm,
        },
        "refill": {
            "required": result.refill_required,
            "warning_state": result.warning_state,
            "threshold_percent": result.refill_threshold_percent,
        },
        "capture": _capture_metadata(captured),
        "fusion": {
            "burst_frames": config.fusion.burst_frames,
            "mad_sigma": config.fusion.mad_sigma,
            "minimum_inlier_band_mm": config.fusion.minimum_inlier_band_mm,
            "per_pixel_min_valid_fraction": config.fusion.per_pixel_min_valid_fraction,
        },
        "raw_artifact": raw_artifact,
    }


def _capture_metadata(captured: CapturedDepthBurst) -> dict[str, Any]:
    burst = captured.burst
    stream_profile = None
    if burst.active_stream_profile is not None:
        width, height, fps = burst.active_stream_profile
        stream_profile = {
            "stream": "depth",
            "format": "Z16",
            "width_px": width,
            "height_px": height,
            "fps": fps,
        }
    return {
        "trigger_time_utc": captured.trigger_time_utc,
        "capture_start_utc": captured.capture_start_utc,
        "capture_end_utc": captured.capture_end_utc,
        "source_name": burst.source_name,
        "camera_serial_number": burst.camera_serial_number,
        "active_stream_profile": stream_profile,
        "depth_scale_m": burst.depth_scale_m,
        "intrinsics": asdict(burst.intrinsics),
        "frame_count": len(burst.frames_m),
        "frame_timestamps_ms": burst.frame_timestamps_ms,
        "sensor_settings": burst.sensor_settings,
        "raw_z16_retained": burst.raw_frames_z16 is not None,
    }


def _write_capture_manifest(
    resolved: _ResolvedSession,
    captured: CapturedDepthBurst,
    *,
    state: str,
) -> None:
    _atomic_json(
        resolved.capture_manifest,
        {
            "schema_version": 1,
            "experiment_id": resolved.request.experiment_id,
            "measurement_id": resolved.measurement_id,
            "measurement_index": resolved.measurement_index,
            "method": "depth",
            "state": state,
            "storage_profile": resolved.profile.value,
            "capture": _capture_metadata(captured),
            "updated_utc": format_utc(_utc_now()),
        },
    )


def _write_artifact_manifest(resolved: _ResolvedSession) -> None:
    artifacts = []
    for path in sorted(resolved.artifact_directory.rglob("*")):
        if not path.is_file() or path == resolved.artifact_manifest:
            continue
        artifacts.append(
            {
                "path": path.relative_to(resolved.artifact_directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "type": _artifact_type(path),
            }
        )
    _atomic_json(
        resolved.artifact_manifest,
        {
            "schema_version": 1,
            "experiment_id": resolved.request.experiment_id,
            "measurement_id": resolved.measurement_id,
            "method": "depth",
            "storage_profile": resolved.profile.value,
            "created_utc": format_utc(_utc_now()),
            "artifacts": artifacts,
        },
    )


def _artifact_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".npy": "application/x-npy",
        ".npz": "application/x-npz",
        ".png": "image/png",
        ".ply": "model/ply",
    }.get(path.suffix.lower(), "application/octet-stream")


def _ensure_effective_config(
    session_directory: Path,
    config: AppConfig,
) -> tuple[Path, str]:
    data = _serializable(asdict(config))
    yaml_bytes = yaml.safe_dump(
        data,
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    combined = session_directory / "effective_config.yaml"
    depth_snapshot = session_directory / "provenance" / "depth_effective_config.yaml"
    for path in (combined, depth_snapshot):
        if path.exists():
            if path.read_bytes() != yaml_bytes:
                raise ValueError(
                    f"Session effective configuration is frozen and differs from: {path}"
                )
        else:
            _atomic_bytes(path, yaml_bytes)
    return depth_snapshot, sha256_file(depth_snapshot)


def _ensure_calibration_provenance(
    session_directory: Path,
    calibration: CalibrationData,
    calibration_path: Path | None,
) -> tuple[str, str]:
    provenance_directory = session_directory / "provenance" / "calibration"
    provenance_directory.mkdir(parents=True, exist_ok=True)
    source = Path(calibration_path).expanduser().resolve() if calibration_path else None
    if source is not None and not source.is_file():
        raise FileNotFoundError(f"Calibration provenance file not found: {source}")

    if source is not None:
        if not _calibrations_equal(CalibrationData.load(source), calibration):
            raise ValueError(
                "The supplied calibration_path does not contain the CalibrationData "
                "used by this measurement."
            )
        source_hash = sha256_file(source)
        target = provenance_directory / source.name
        if target.exists():
            if sha256_file(target) != source_hash:
                raise ValueError(f"Frozen session calibration differs: {target}")
        else:
            _atomic_copy(source, target)
        return target.relative_to(session_directory).as_posix(), source_hash

    target = provenance_directory / "generated_calibration.npz"
    if target.exists():
        frozen = CalibrationData.load(target)
        if not _calibrations_equal(frozen, calibration):
            raise ValueError(f"Frozen session calibration differs: {target}")
    else:
        temp = _temporary_path(target, ".tmp.npz")
        try:
            calibration.save(temp)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    return target.relative_to(session_directory).as_posix(), sha256_file(target)


def _calibrations_equal(left: CalibrationData, right: CalibrationData) -> bool:
    scalar_fields = (
        "inner_diameter_mm",
        "usable_height_mm",
        "center_x_px",
        "center_y_px",
        "plane_rms_mm",
        "plane_coverage",
        "measured_rim_distance_mm",
        "calibration_method",
        "created_utc",
    )
    if left.intrinsics != right.intrinsics:
        return False
    if any(getattr(left, name) != getattr(right, name) for name in scalar_fields):
        return False
    return all(
        np.array_equal(getattr(left, name), getattr(right, name), equal_nan=True)
        for name in (
            "bottom_plane_point_m",
            "tube_axis_toward_camera",
            "basis_x",
            "basis_y",
            "bottom_center_m",
        )
    )


def _ensure_environment_manifest(session_directory: Path) -> None:
    path = session_directory / "provenance" / "environment.json"
    if path.exists():
        return
    packages = {}
    for name in (
        "d405-material-volume",
        "polymer-experiment-records",
        "numpy",
        "scipy",
        "PyYAML",
        "Pillow",
        "opencv-python",
        "pyrealsense2",
        "openpyxl",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "created_utc": format_utc(_utc_now()),
        },
    )


def _ensure_session_manifest(
    *,
    session_directory: Path,
    experiment_id: str,
    purpose: str | None,
    profile: StorageProfile,
    config_snapshot: Path,
    config_hash: str,
    calibration_reference: str,
    calibration_hash: str,
    software_commit: str | None,
) -> None:
    path = session_directory / "session_manifest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        frozen = {
            "experiment_id": experiment_id,
            "storage_profile": profile.value,
            "config_sha256": config_hash,
            "calibration_sha256": calibration_hash,
        }
        mismatches = [name for name, value in frozen.items() if data.get(name) != value]
        if purpose is not None and data.get("purpose") not in {None, purpose}:
            mismatches.append("purpose")
        if mismatches:
            raise ValueError(
                "Existing session has different frozen metadata: " + ", ".join(mismatches)
            )
        return
    now = format_utc(_utc_now())
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "common_record_schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "purpose": purpose,
            "storage_profile": profile.value,
            "created_utc": now,
            "updated_utc": now,
            "effective_config": config_snapshot.relative_to(session_directory).as_posix(),
            "config_sha256": config_hash,
            "calibration_reference": calibration_reference,
            "calibration_sha256": calibration_hash,
            "software_commit": software_commit,
            "measurements": [],
            "scientifically_validated": False,
        },
    )


def _finalize_manifests(
    resolved: _ResolvedSession,
    status: str | None,
    valid: bool | None,
    *,
    manage_capture_manifest: bool = True,
) -> None:
    if manage_capture_manifest:
        capture = json.loads(resolved.capture_manifest.read_text(encoding="utf-8"))
        capture.update(
            {
                "state": "SAVED",
                "record_status": status,
                "valid": valid,
                "artifact_manifest": _session_relative(
                    resolved.session_directory, resolved.artifact_manifest
                ),
                "updated_utc": format_utc(_utc_now()),
            }
        )
        _atomic_json(resolved.capture_manifest, capture)

    lock_path = resolved.session_directory / ".session.lock"
    with _exclusive_lock(lock_path):
        path = resolved.session_directory / "session_manifest.json"
        session = json.loads(path.read_text(encoding="utf-8"))
        measurements = [
            item
            for item in session.get("measurements", [])
            if item.get("measurement_id") != resolved.measurement_id
        ]
        measurements.append(
            {
                "measurement_id": resolved.measurement_id,
                "measurement_index": resolved.measurement_index,
                "method": "depth",
                "status": status,
                "valid": valid,
                "capture_manifest": _session_relative(
                    resolved.session_directory, resolved.capture_manifest
                ),
            }
        )
        session["measurements"] = sorted(
            measurements, key=lambda item: int(item["measurement_index"])
        )
        session["updated_utc"] = format_utc(_utc_now())
        _atomic_json(path, session)


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _session_relative(session_directory: Path, path: Path) -> str:
    resolved_session = session_directory.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_session).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes the session directory: {path}") from exc


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    payload = (json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, payload)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_path(path, ".tmp")
    try:
        with temp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_path(target, ".tmp")
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _temporary_path(target: Path, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=suffix, dir=target.parent)
    os.close(descriptor)
    return Path(raw)


@contextmanager
def _exclusive_lock(path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for session lock: {path}. Verify no writer is active "
                    "before removing a stale lock after a crash."
                ) from exc
            time.sleep(0.05)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)
