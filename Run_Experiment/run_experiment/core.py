"""Bounded synchronized C920/D405 acquisition with crash-safe evidence manifests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from experiment_records import (
    CommonMeasurementRecord,
    MeasurementWorkbookStore,
    RecordStatus,
    filesystem_utc,
    format_utc,
    git_revision,
    parse_measurement_id,
    prepare_record,
    sha256_file,
    validate_experiment_id,
    volume_tolerance,
)
from material_level_yolo.camera import (
    CameraPreviewProtocol,
    PreviewEvent,
    warm_up_preview,
)
from material_level_yolo.image_io import write_image
from material_volume.session import DepthMeasurementRequest

from .errors import BusyTriggerError, ExperimentConfigurationError, SynchronizedAcquisitionError
from .config import validate_config
from .models import (
    C920CaptureResult,
    C920FrameTiming,
    DepthCaptureResult,
    ExperimentConfig,
    ExperimentInputs,
    ExperimentOutcome,
    ExperimentState,
    MeasurementReferenceInputs,
    ValidatedExperimentInputs,
    ValidatedMeasurementReferenceInputs,
)


class DepthControllerProtocol(Protocol):
    config: Any

    def start(self) -> None: ...
    def reserve(self, request: DepthMeasurementRequest) -> Any: ...
    def capture(self, request: DepthMeasurementRequest, *, now_utc: Any = None) -> Any: ...
    def process(
        self,
        captured: Any,
        request: DepthMeasurementRequest,
        reservation: Any,
        *,
        now_utc: Any = None,
    ) -> Any: ...
    def close(self) -> None: ...


class _ManifestStore:
    def __init__(self, path: Path, initial: dict[str, Any]) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = initial
        _atomic_json(path, initial)

    def patch(self, **values: Any) -> None:
        with self._lock:
            self._data.update(values)
            self._data["updated_utc"] = format_utc(datetime.now(timezone.utc))
            _atomic_json(self.path, self._data)

    def append_frame(self, frame: C920FrameTiming, session: Path) -> None:
        with self._lock:
            frames = list(self._data["c920"]["frames"])
            item = asdict(frame)
            item["path"] = frame.path.relative_to(session).as_posix()
            frames.append(item)
            self._data["c920"]["frames"] = frames
            self._data["updated_utc"] = format_utc(datetime.now(timezone.utc))
            _atomic_json(self.path, self._data)

    def patch_depth(self, **values: Any) -> None:
        with self._lock:
            self._data["depth"].update(values)
            self._data["updated_utc"] = format_utc(datetime.now(timezone.utc))
            _atomic_json(self.path, self._data)


class SynchronizedExperiment:
    """Own two warmed camera controllers and serialize one trigger at a time."""

    def __init__(
        self,
        config: ExperimentConfig,
        inputs: ExperimentInputs,
        *,
        c920: CameraPreviewProtocol,
        depth: DepthControllerProtocol,
        utc_now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        state_callback: Callable[[ExperimentState], None] | None = None,
        software_repository: Path | None = None,
    ) -> None:
        self.config = validate_config(config)
        self.depth = depth
        self.c920 = c920
        self.inputs = validate_inputs(config, inputs, depth.config)
        self.utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic or time.monotonic
        self.sleep = sleep or time.sleep
        self.state_callback = state_callback
        self.software_repository = (
            Path(software_repository).resolve()
            if software_repository is not None
            else Path(__file__).resolve().parents[2]
        )
        self._state = ExperimentState.INITIALIZING
        self._history: list[ExperimentState] = [self._state]
        self._guard = threading.Lock()
        self._busy = False
        self._stop_requested = threading.Event()
        self._initialized = False
        self._closed = False
        self._last_c920_frame: np.ndarray | None = None
        self._accepted_trigger_count = 0

    @property
    def state(self) -> ExperimentState:
        return self._state

    @property
    def state_history(self) -> tuple[ExperimentState, ...]:
        return tuple(self._history)

    def confirmation_summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.inputs.experiment_id,
            "purpose": self.inputs.purpose,
            "material_name": self.inputs.material_name,
            "total_capacity_ml": self.inputs.total_capacity_ml,
            "bulk_density_g_per_ml": self.inputs.bulk_density_g_per_ml,
            "total_possible_weight_g": self.inputs.total_possible_weight_g,
            "operator_notes": self.inputs.operator_notes,
            "reference_inputs_collected_per_measurement": True,
            "output_root": str(self.inputs.output_root),
            "c920": {
                "device_index": self.config.c920.device_index,
                "device_name_contains": self.config.c920.device_name_contains,
                "backend": self.config.c920.backend,
                "resolution": [self.config.c920.width, self.config.c920.height],
                "fps": self.config.c920.fps,
                "warmup_frames": self.config.c920.warmup_frames,
                "frame_count": self.config.acquisition.frame_count,
                "span_seconds": self.config.acquisition.span_seconds,
            },
            "depth_config": str(self.config.depth_config_path),
            "depth_calibration": str(self.config.depth_calibration_path),
            "depth_burst_frames": self.depth.config.fusion.burst_frames,
            "storage_profile": self.config.storage_profile,
            "exact_hardware_synchronization_claimed": False,
            "yolo_loaded_or_run": False,
        }

    def initialize(self) -> None:
        if self._initialized:
            return
        # INITIALIZING is the construction state, so notify the UI explicitly
        # even though no history transition is needed here.
        if self.state_callback is not None:
            self.state_callback(ExperimentState.INITIALIZING)
        try:
            self.depth.start()
            self.c920.open(self.config.c920)
            cancelled = warm_up_preview(self.c920, self.config.c920)
            if cancelled is not None:
                raise SynchronizedAcquisitionError(
                    f"C920 initialization was cancelled: {cancelled}"
                )
            self._initialized = True
            self._set_state(ExperimentState.ARMED)
        except Exception:
            self._set_state(ExperimentState.ERROR)
            self.close()
            raise

    def next_measurement_index(self) -> int:
        """Return the next durable session index without allocating a measurement."""
        session = self.inputs.output_root / self.inputs.experiment_id
        durable_ids: set[str] = set()
        measurements = session / "measurements"
        if measurements.is_dir():
            for child in measurements.iterdir():
                if child.is_dir():
                    try:
                        parse_measurement_id(child.name)
                    except ValueError:
                        continue
                    durable_ids.add(child.name)
        for name, method in (
            ("depth_measurements.xlsx", "depth"),
            ("yolo_measurements.xlsx", "yolo"),
        ):
            path = session / name
            if path.is_file():
                for record in MeasurementWorkbookStore(path, method=method).load_records():
                    if (
                        record.experiment_id == self.inputs.experiment_id
                        and record.measurement_id is not None
                    ):
                        durable_ids.add(record.measurement_id)
        return max(
            (parse_measurement_id(value).index for value in durable_ids),
            default=0,
        ) + 1

    def trigger(
        self,
        reference_inputs: MeasurementReferenceInputs | None = None,
    ) -> ExperimentOutcome:
        if reference_inputs is None and self._accepted_trigger_count == 0:
            # Backward-compatible one-shot use of the former session fields. They
            # are never reused by a later measurement.
            reference_inputs = MeasurementReferenceInputs(
                reference_material_weight_g=self.inputs.reference_material_weight_g,
                manual_material_level_mm=self.inputs.manual_material_level_mm,
            )
        references = validate_measurement_references(
            reference_inputs,
            self.depth.config,
        )
        with self._guard:
            if not self._initialized or self._state is not ExperimentState.ARMED or self._busy:
                raise BusyTriggerError(
                    f"Trigger rejected while state={self._state.value}; system must be ARMED."
                )
            self._busy = True
            self._accepted_trigger_count += 1
            self._set_state(ExperimentState.CAPTURING)
        try:
            return self._run_trigger(references)
        except Exception:
            self._set_state(ExperimentState.ERROR)
            raise
        finally:
            with self._guard:
                self._busy = False
                if self._stop_requested.is_set():
                    self.close()
                elif self._initialized and self._state in {
                    ExperimentState.SAVED,
                    ExperimentState.PARTIAL,
                    ExperimentState.ERROR,
                }:
                    self._set_state(ExperimentState.ARMED)

    def preview_once(self) -> PreviewEvent:
        if self._state is not ExperimentState.ARMED:
            return PreviewEvent.NONE
        frame = self.c920.read()
        if frame is None:
            raise SynchronizedAcquisitionError("C920 preview read failed while ARMED.")
        self.c920.show(
            frame,
            f"ARMED - {self.config.c920.capture_key}/left click: trigger | "
            f"{self.config.c920.quit_key}/Esc: stop",
        )
        self._last_c920_frame = frame.copy()
        return self.c920.poll_event(self.config.c920.preview_wait_ms)

    def request_stop(self) -> None:
        self._stop_requested.set()
        with self._guard:
            busy = self._busy
        if not busy:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.c920.close()
        finally:
            self.depth.close()
            self._initialized = False
            self._set_state(ExperimentState.STOPPED)

    def _run_trigger(
        self,
        references: ValidatedMeasurementReferenceInputs,
    ) -> ExperimentOutcome:
        trigger_utc = _aware_utc(self.utc_now())
        trigger_monotonic = self.monotonic()
        depth_request = self._depth_request(trigger_utc, references)
        reservation = self.depth.reserve(depth_request)
        depth_request = replace(
            depth_request,
            measurement_index=reservation.measurement_index,
            measurement_id=reservation.measurement_id,
        )
        session = reservation.session_directory
        measurement = reservation.measurement_directory
        raw_directory = measurement / "c920" / "raw"
        raw_directory.mkdir(parents=True, exist_ok=False)
        capture_manifest = measurement / "capture_manifest.json"
        manifest = _ManifestStore(
            capture_manifest,
            self._initial_manifest(
                reservation,
                trigger_utc,
                trigger_monotonic,
                depth_request,
                references,
            ),
        )

        ready = threading.Barrier(self.config.acquisition.worker_count + 1)
        start_event = threading.Event()
        start_holder: dict[str, Any] = {}
        with ThreadPoolExecutor(
            max_workers=self.config.acquisition.worker_count,
            thread_name_prefix="synchronized-capture",
        ) as executor:
            c920_future = executor.submit(
                self._capture_c920,
                ready,
                start_event,
                start_holder,
                raw_directory,
                session,
                manifest,
            )
            depth_future = executor.submit(
                self._capture_depth,
                ready,
                start_event,
                depth_request,
                manifest,
            )
            ready.wait()
            acquisition_start = self.monotonic()
            acquisition_start_utc = _aware_utc(self.utc_now())
            start_holder["value"] = acquisition_start
            start_holder["utc"] = acquisition_start_utc
            manifest.patch(
                state="CAPTURING",
                acquisition_barrier_monotonic=acquisition_start,
                acquisition_barrier_utc=format_utc(acquisition_start_utc),
                trigger_to_barrier_seconds=acquisition_start - trigger_monotonic,
            )
            start_event.set()
            c920_result = c920_future.result()
            depth_result = depth_future.result()

        self._set_state(ExperimentState.PROCESSING)
        manifest.patch(state="PROCESSING")
        depth_outcome = None
        if depth_result.captured is not None:
            try:
                depth_outcome = self.depth.process(
                    depth_result.captured,
                    depth_request,
                    reservation,
                    now_utc=self.utc_now,
                )
                depth_record = depth_outcome.record
            except Exception as exc:
                depth_record = self._write_depth_failure_record(
                    reservation,
                    depth_request,
                    depth_result,
                    "processing_failed",
                    exc,
                )
        else:
            depth_record = self._write_depth_failure_record(
                reservation,
                depth_request,
                depth_result,
                "acquisition_failed",
                RuntimeError(depth_result.failure or "D405 acquisition failed"),
            )

        yolo_record = self._write_pending_yolo_record(
            reservation,
            depth_request,
            c920_result,
            capture_manifest,
        )
        overlap = _overlap_duration(c920_result, depth_result)
        acquisition_complete = (
            len(c920_result.frames) == self.config.acquisition.frame_count
            and c920_result.failure is None
            and depth_result.captured is not None
        )
        final_state = (
            ExperimentState.SAVED if acquisition_complete else ExperimentState.PARTIAL
        )
        manifest.patch(
            state=final_state.value,
            overlap_duration_seconds=overlap,
            exact_hardware_synchronization_claimed=False,
            depth_record_status=depth_record.status,
            yolo_record_status=yolo_record.status,
            workbook_paths={
                "depth": "depth_measurements.xlsx",
                "yolo": "yolo_measurements.xlsx",
            },
            resumable=(len(c920_result.frames) > 0),
            resume_policy=(
                "offline_yolo_same_identity"
                if len(c920_result.frames) > 0
                else "repeat_synchronized_acquisition_with_new_identity"
            ),
        )
        self._update_session_manifest(
            session,
            reservation,
            yolo_record,
            capture_manifest,
            final_state,
        )
        self._set_state(final_state)
        return ExperimentOutcome(
            measurement_id=reservation.measurement_id,
            measurement_index=reservation.measurement_index,
            session_directory=session,
            measurement_directory=measurement,
            capture_manifest=capture_manifest,
            c920=c920_result,
            depth_outcome=depth_outcome,
            depth_record=depth_record,
            yolo_record=yolo_record,
            final_state=final_state,
            overlap_duration_seconds=overlap,
            state_history=tuple(self._history),
        )

    def _capture_c920(
        self,
        ready: threading.Barrier,
        start_event: threading.Event,
        start_holder: dict[str, Any],
        raw_directory: Path,
        session: Path,
        manifest: _ManifestStore,
    ) -> C920CaptureResult:
        ready.wait()
        start_event.wait()
        base = start_holder["value"]
        base_utc = start_holder["utc"]
        count = self.config.acquisition.frame_count
        span = self.config.acquisition.span_seconds
        offsets = tuple(index * span / (count - 1) for index in range(count))
        frames: list[C920FrameTiming] = []
        failure: str | None = None
        for index, offset in enumerate(offsets, start=1):
            target = base + offset
            latest: np.ndarray | None = None
            try:
                while self.monotonic() < target:
                    if self._stop_requested.is_set():
                        failure = "graceful_stop_requested"
                        break
                    latest = self.c920.read()
                    if latest is None:
                        failure = f"C920 read failed before frame {index}/{count}"
                        break
                    self.c920.show(latest, f"CAPTURING {len(frames)}/{count}")
                    self._last_c920_frame = latest.copy()
                    remaining = target - self.monotonic()
                    if remaining > 0:
                        self.sleep(
                            min(self.config.acquisition.scheduling_poll_seconds, remaining)
                        )
                if failure is not None:
                    break
                frame = self.c920.read()
                if frame is None:
                    frame = latest
                if frame is None:
                    failure = f"C920 did not provide frame {index}/{count}"
                    break
                self._last_c920_frame = frame.copy()
                actual_monotonic = self.monotonic()
                actual_utc = _aware_utc(self.utc_now())
                name = f"c920_{index:06d}_{filesystem_utc(actual_utc)}.png"
                path = write_image(raw_directory / name, frame)
                timing = C920FrameTiming(
                    index=index,
                    path=path,
                    sha256=sha256_file(path),
                    target_offset_seconds=offset,
                    target_monotonic=target,
                    target_utc=format_utc(base_utc + timedelta(seconds=offset)),
                    actual_monotonic=actual_monotonic,
                    timing_error_seconds=actual_monotonic - target,
                    actual_utc=format_utc(actual_utc),
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                )
                frames.append(timing)
                manifest.append_frame(timing, session)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                break
        start = frames[0].actual_monotonic if frames else None
        end = frames[-1].actual_monotonic if frames else None
        manifest.patch(
            c920_capture_status=("complete" if len(frames) == count else "partial"),
            c920_failure=failure,
        )
        return C920CaptureResult(tuple(frames), count, start, end, failure)

    def _capture_depth(
        self,
        ready: threading.Barrier,
        start_event: threading.Event,
        request: DepthMeasurementRequest,
        manifest: _ManifestStore,
    ) -> DepthCaptureResult:
        ready.wait()
        start_event.wait()
        start = self.monotonic()
        manifest.patch_depth(status="capturing", start_monotonic=start)
        captured = None
        failure = None
        try:
            captured = self.depth.capture(request, now_utc=self.utc_now)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        end = self.monotonic()
        start_utc = None if captured is None else captured.capture_start_utc
        end_utc = None if captured is None else captured.capture_end_utc
        manifest.patch_depth(
            status=("captured" if captured is not None else "failed"),
            end_monotonic=end,
            capture_start_utc=start_utc,
            capture_end_utc=end_utc,
            failure=failure,
        )
        return DepthCaptureResult(captured, start, end, start_utc, end_utc, failure)

    def _depth_request(
        self,
        trigger: datetime,
        references: ValidatedMeasurementReferenceInputs,
    ) -> DepthMeasurementRequest:
        return DepthMeasurementRequest(
            experiment_id=self.inputs.experiment_id,
            purpose=self.inputs.purpose,
            operator_notes=_combined_notes(
                self.inputs.operator_notes,
                references.measurement_notes,
            ),
            material_name=self.inputs.material_name,
            total_capacity_ml=self.inputs.total_capacity_ml,
            bulk_density_g_per_ml=self.inputs.bulk_density_g_per_ml,
            total_possible_weight_g=self.inputs.total_possible_weight_g,
            reference_material_weight_g=references.reference_material_weight_g,
            manual_material_level_mm=references.manual_material_level_mm,
            output_root=self.inputs.output_root,
            storage_profile=self.config.storage_profile,
            trigger_time_utc=trigger,
            acquisition_mode="synchronized",
            calibration_path=self.config.depth_calibration_path,
            software_repository=self.software_repository,
        )

    def _initial_manifest(
        self,
        reservation: Any,
        trigger_utc: datetime,
        trigger_monotonic: float,
        request: DepthMeasurementRequest,
        references: ValidatedMeasurementReferenceInputs,
    ) -> dict[str, Any]:
        now = format_utc(self.utc_now())
        count = self.config.acquisition.frame_count
        span = self.config.acquisition.span_seconds
        return {
            "capture_manifest_schema_version": 1,
            "experiment_id": self.inputs.experiment_id,
            "measurement_id": reservation.measurement_id,
            "measurement_index": reservation.measurement_index,
            "state": "ALLOCATED",
            "created_utc": now,
            "updated_utc": now,
            "trigger_time_utc": format_utc(trigger_utc),
            "trigger_monotonic": trigger_monotonic,
            "acquisition_barrier_monotonic": None,
            "acquisition_barrier_utc": None,
            "trigger_to_barrier_seconds": None,
            "exact_hardware_synchronization_claimed": False,
            "reference_inputs": {
                "reference_material_weight_g": request.reference_material_weight_g,
                "manual_material_level_mm": request.manual_material_level_mm,
                "measurement_notes": references.measurement_notes,
                "general_experiment_notes": self.inputs.operator_notes,
            },
            "c920": {
                "planned_count": count,
                "span_seconds": span,
                "target_offsets_seconds": [
                    index * span / (count - 1) for index in range(count)
                ],
                "lossless_format": "png",
                "frames": [],
            },
            "depth": {
                "configured_burst_frames": self.depth.config.fusion.burst_frames,
                "status": "allocated",
            },
            "yolo_policy": "pending_row_after_complete_c920_capture",
            "yolo_loaded_or_run": False,
            "failures": [],
            "resumable": True,
        }

    def _write_pending_yolo_record(
        self,
        reservation: Any,
        request: DepthMeasurementRequest,
        c920: C920CaptureResult,
        manifest_path: Path,
    ) -> CommonMeasurementRecord:
        count = len(c920.frames)
        if count == self.config.acquisition.frame_count and c920.failure is None:
            status = RecordStatus.PENDING_OFFLINE_INFERENCE.value
            valid = None
        elif count > 0:
            status = RecordStatus.PARTIAL_CAPTURE.value
            valid = None
        else:
            status = RecordStatus.ACQUISITION_FAILED.value
            valid = False
        record = prepare_record(
            CommonMeasurementRecord(
                experiment_id=self.inputs.experiment_id,
                measurement_id=reservation.measurement_id,
                measurement_index=reservation.measurement_index,
                method="yolo",
                acquisition_mode="synchronized",
                trigger_time_utc=request.trigger_time_utc,
                capture_start_utc=(None if not c920.frames else c920.frames[0].actual_utc),
                capture_end_utc=(None if not c920.frames else c920.frames[-1].actual_utc),
                material_name=request.material_name,
                total_capacity_ml=request.total_capacity_ml,
                bulk_density_g_per_ml=request.bulk_density_g_per_ml,
                total_possible_weight_g=request.total_possible_weight_g,
                reference_material_weight_g=request.reference_material_weight_g,
                manual_material_level_mm=request.manual_material_level_mm,
                valid=valid,
                status=status,
                notes=request.operator_notes,
                artifact_directory=(
                    reservation.measurement_directory.relative_to(
                        reservation.session_directory
                    ).as_posix()
                    + "/c920"
                ),
                source_artifact=manifest_path.relative_to(
                    reservation.session_directory
                ).as_posix(),
                software_commit=git_revision(self.software_repository),
            ),
            usable_internal_height_mm=self.depth.config.tube.usable_height_mm,
        )
        MeasurementWorkbookStore(
            reservation.session_directory / "yolo_measurements.xlsx",
            method="yolo",
        ).upsert(
            record,
            usable_internal_height_mm=self.depth.config.tube.usable_height_mm,
        )
        return record

    def _write_depth_failure_record(
        self,
        reservation: Any,
        request: DepthMeasurementRequest,
        capture: DepthCaptureResult,
        status: str,
        error: Exception,
    ) -> CommonMeasurementRecord:
        failure_path = reservation.artifact_directory / "failure.json"
        _atomic_json(
            failure_path,
            {
                "schema_version": 1,
                "status": status,
                "error_type": type(error).__name__,
                "message": str(error),
                "failure_time_utc": format_utc(self.utc_now()),
            },
        )
        record = prepare_record(
            CommonMeasurementRecord(
                experiment_id=self.inputs.experiment_id,
                measurement_id=reservation.measurement_id,
                measurement_index=reservation.measurement_index,
                method="depth",
                acquisition_mode="synchronized",
                trigger_time_utc=request.trigger_time_utc,
                capture_start_utc=capture.capture_start_utc,
                capture_end_utc=capture.capture_end_utc,
                material_name=request.material_name,
                total_capacity_ml=request.total_capacity_ml,
                bulk_density_g_per_ml=request.bulk_density_g_per_ml,
                total_possible_weight_g=request.total_possible_weight_g,
                reference_material_weight_g=request.reference_material_weight_g,
                manual_material_level_mm=request.manual_material_level_mm,
                valid=False,
                status=status,
                notes=request.operator_notes,
                artifact_directory=reservation.artifact_directory.relative_to(
                    reservation.session_directory
                ).as_posix(),
                source_artifact=failure_path.relative_to(
                    reservation.session_directory
                ).as_posix(),
                config_sha256=reservation.config_sha256,
                calibration_or_model_sha256=reservation.calibration_sha256,
                software_commit=reservation.software_commit,
            ),
            usable_internal_height_mm=self.depth.config.tube.usable_height_mm,
        )
        MeasurementWorkbookStore(
            reservation.session_directory / "depth_measurements.xlsx",
            method="depth",
        ).upsert(
            record,
            usable_internal_height_mm=self.depth.config.tube.usable_height_mm,
        )
        return record

    def _update_session_manifest(
        self,
        session: Path,
        reservation: Any,
        yolo_record: CommonMeasurementRecord,
        capture_manifest: Path,
        final_state: ExperimentState,
    ) -> None:
        path = session / "session_manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        measurements = [
            item
            for item in data.get("measurements", [])
            if not (
                item.get("measurement_id") == reservation.measurement_id
                and item.get("method") == "yolo"
            )
        ]
        measurements.append(
            {
                "measurement_id": reservation.measurement_id,
                "measurement_index": reservation.measurement_index,
                "method": "yolo",
                "status": yolo_record.status,
                "valid": yolo_record.valid,
                "capture_manifest": capture_manifest.relative_to(session).as_posix(),
            }
        )
        data["measurements"] = sorted(
            measurements,
            key=lambda item: (item.get("measurement_index") or 0, item.get("method") or ""),
        )
        synchronized = [
            item
            for item in data.get("synchronized_acquisitions", [])
            if item.get("measurement_id") != reservation.measurement_id
        ]
        synchronized.append(
            {
                "measurement_id": reservation.measurement_id,
                "state": final_state.value,
                "capture_manifest": capture_manifest.relative_to(session).as_posix(),
                "yolo_inference_deferred": True,
            }
        )
        data["synchronized_acquisitions"] = synchronized
        data["updated_utc"] = format_utc(self.utc_now())
        _atomic_json(path, data)

    def _set_state(self, state: ExperimentState) -> None:
        if self._state is state and self._history:
            return
        self._state = state
        self._history.append(state)
        if (
            self._initialized
            and self._last_c920_frame is not None
            and state
            in {
                ExperimentState.PROCESSING,
                ExperimentState.SAVED,
                ExperimentState.PARTIAL,
                ExperimentState.ERROR,
            }
        ):
            self.c920.show(self._last_c920_frame, state.value)
        if self.state_callback is not None:
            self.state_callback(state)


def validate_inputs(
    config: ExperimentConfig,
    inputs: ExperimentInputs,
    depth_config: Any,
) -> ValidatedExperimentInputs:
    try:
        experiment_id = validate_experiment_id(inputs.experiment_id)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigurationError(str(exc)) from exc
    purpose = _text("purpose", inputs.purpose)
    material = _optional_text("material_name", inputs.material_name)
    notes = _optional_text("operator_notes", inputs.operator_notes)
    capacity = _positive("total_capacity_ml", inputs.total_capacity_ml)
    if not math.isclose(
        capacity,
        float(depth_config.tube.capacity_ml),
        rel_tol=0.0,
        abs_tol=volume_tolerance(float(depth_config.tube.capacity_ml)),
    ):
        raise ExperimentConfigurationError(
            "total_capacity_ml must match the frozen depth tube geometry."
        )
    density = _optional_positive("bulk_density_g_per_ml", inputs.bulk_density_g_per_ml)
    total_weight = _optional_nonnegative(
        "total_possible_weight_g", inputs.total_possible_weight_g
    )
    reference_weight = _optional_nonnegative(
        "reference_material_weight_g", inputs.reference_material_weight_g
    )
    manual = _optional_nonnegative(
        "manual_material_level_mm", inputs.manual_material_level_mm
    )
    if manual is not None and manual > depth_config.tube.usable_height_mm:
        raise ExperimentConfigurationError(
            "manual_material_level_mm cannot exceed the configured usable tube height."
        )
    if density is not None and total_weight is not None:
        expected = capacity * density
        if not math.isclose(
            total_weight,
            expected,
            rel_tol=0.0,
            abs_tol=volume_tolerance(expected),
        ):
            raise ExperimentConfigurationError(
                "total_possible_weight_g conflicts with capacity * bulk density."
            )
    root = config.output_root if inputs.output_root is None else Path(inputs.output_root).expanduser()
    if not root.is_absolute():
        raise ExperimentConfigurationError("output_root must be absolute.")
    return ValidatedExperimentInputs(
        experiment_id=experiment_id,
        purpose=purpose,
        material_name=material,
        total_capacity_ml=capacity,
        bulk_density_g_per_ml=density,
        total_possible_weight_g=total_weight,
        reference_material_weight_g=reference_weight,
        manual_material_level_mm=manual,
        operator_notes=notes,
        output_root=root.resolve(),
    )


def validate_measurement_references(
    inputs: MeasurementReferenceInputs | None,
    depth_config: Any,
) -> ValidatedMeasurementReferenceInputs:
    """Validate independent inputs for one measurement without allocating it."""
    values = inputs or MeasurementReferenceInputs()
    if not isinstance(values, MeasurementReferenceInputs):
        raise ExperimentConfigurationError(
            "reference_inputs must be MeasurementReferenceInputs or None."
        )
    weight = _optional_nonnegative(
        "reference_material_weight_g",
        values.reference_material_weight_g,
    )
    manual = _optional_nonnegative(
        "manual_material_level_mm",
        values.manual_material_level_mm,
    )
    if manual is not None and manual > depth_config.tube.usable_height_mm:
        raise ExperimentConfigurationError(
            "manual_material_level_mm cannot exceed the configured usable tube height."
        )
    return ValidatedMeasurementReferenceInputs(
        reference_material_weight_g=weight,
        manual_material_level_mm=manual,
        measurement_notes=_optional_text(
            "measurement_notes",
            values.measurement_notes,
        ),
    )


def _combined_notes(
    general_experiment_notes: str | None,
    measurement_notes: str | None,
) -> str | None:
    if general_experiment_notes is not None and measurement_notes is not None:
        return (
            f"Experiment notes: {general_experiment_notes}\n"
            f"Measurement notes: {measurement_notes}"
        )
    return measurement_notes or general_experiment_notes


def _overlap_duration(c920: C920CaptureResult, depth: DepthCaptureResult) -> float:
    if c920.start_monotonic is None or c920.end_monotonic is None:
        return 0.0
    return max(
        0.0,
        min(c920.end_monotonic, depth.end_monotonic)
        - max(c920.start_monotonic, depth.start_monotonic),
    )


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentConfigurationError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ExperimentConfigurationError(f"{name} must be finite.")
    return result


def _positive(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result <= 0:
        raise ExperimentConfigurationError(f"{name} must be greater than zero.")
    return result


def _optional_positive(name: str, value: Any) -> float | None:
    return None if value is None else _positive(name, value)


def _optional_nonnegative(name: str, value: Any) -> float | None:
    if value is None:
        return None
    result = _finite(name, value)
    if result < 0:
        raise ExperimentConfigurationError(f"{name} must be nonnegative.")
    return result


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigurationError(f"{name} must be non-blank text.")
    return value.strip()


def _optional_text(name: str, value: Any) -> str | None:
    return None if value is None else _text(name, value)


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SynchronizedAcquisitionError("UTC clock returned a naive timestamp.")
    return value.astimezone(timezone.utc)


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
