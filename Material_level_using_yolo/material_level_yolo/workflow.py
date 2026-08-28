"""Offline saved-still processing, evidence export, and common YOLO records."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterator, Protocol, Sequence

import cv2
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
    sha256_bytes,
    sha256_file,
    validate_experiment_id,
    validate_measurement_identity,
    with_estimate,
)

from .config import ProjectConfig
from .domain import InferenceOutput, ModelProfile, SemanticClassMap
from .errors import InferenceError
from .estimation import (
    GroupLevelEstimate,
    ImageLevelEstimate,
    SemanticObservation,
    aggregate_image_estimates,
    estimate_image_level,
)
from .extraction import observation_from_inference
from .image_io import read_image, write_image
from .inference import UltralyticsInferenceAdapter, resolve_semantic_classes
from .naming import artifact_name


YOLO_IMAGE_DETAIL_COLUMNS = (
    "diagnostic_schema_version",
    "experiment_id",
    "measurement_id",
    "image_index",
    "source_path",
    "source_sha256",
    "estimation_mode",
    "class_pattern",
    "majority_class_pattern",
    "retained_for_aggregation",
    "pattern_filter_reasons",
    "material_class_id",
    "material_class_name",
    "empty_class_id",
    "empty_class_name",
    "material_confidences",
    "empty_confidences",
    "material_instance_count",
    "empty_instance_count",
    "material_coverage_fraction",
    "empty_coverage_fraction",
    "overlap_fraction",
    "unclassified_fraction",
    "interface_position_px",
    "interface_fraction_from_full",
    "interface_spread_fraction",
    "candidate_material_percent",
    "accepted_material_percent",
    "valid",
    "rejection_reasons",
    "roi_left_px",
    "roi_top_px",
    "roi_right_px",
    "roi_bottom_px",
    "roi_mode",
    "analysis_bounds_source",
    "level_calibration_mode",
    "quality_warnings",
    "axis",
    "source_copy",
    "overlay_artifact",
    "material_mask_artifact",
    "empty_mask_artifact",
    "diagnostics_artifact",
)


class InferenceAdapterProtocol(Protocol):
    @property
    def weights_sha256(self) -> str | None: ...

    @property
    def resolved_device(self) -> str: ...

    def load(self) -> SemanticClassMap: ...

    def predict(self, image: np.ndarray) -> InferenceOutput: ...


@dataclass(frozen=True, slots=True)
class OfflineMeasurementRequest:
    experiment_id: str
    total_capacity_ml: float
    purpose: str | None = None
    output_root: Path | None = None
    measurement_index: int | None = None
    measurement_id: str | None = None
    trigger_time_utc: datetime | str | None = None
    material_name: str | None = None
    bulk_density_g_per_ml: float | None = None
    total_possible_weight_g: float | None = None
    reference_material_weight_g: float | None = None
    manual_material_level_mm: float | None = None
    usable_internal_height_mm: float | None = None
    notes: str | None = None
    acquisition_mode: str = "folder"
    capture_start_utc: datetime | str | None = None
    capture_end_utc: datetime | str | None = None
    minimum_valid_images: int | None = None
    resume_pending_synchronized: bool = False
    allow_provenance_revision: bool = False
    processing_revision: int | None = None
    source_artifact: str | None = None
    software_repository: Path | None = None


@dataclass(frozen=True, slots=True)
class OfflineMeasurementOutcome:
    record: CommonMeasurementRecord
    group: GroupLevelEstimate
    session_directory: Path
    artifact_directory: Path
    result_path: Path
    image_details_path: Path
    artifact_manifest_path: Path


@dataclass(slots=True)
class _ProcessedImage:
    index: int
    source: Path
    source_sha256: str
    observation: SemanticObservation
    estimate: ImageLevelEstimate
    source_copy: str | None = None
    overlay: str | None = None
    material_mask: str | None = None
    empty_mask: str | None = None
    diagnostics: str | None = None


def process_saved_images(
    project: ProjectConfig,
    profile_name: str | None,
    image_paths: Sequence[str | Path],
    request: OfflineMeasurementRequest,
    *,
    adapter: InferenceAdapterProtocol | None = None,
    now_utc: Any = None,
) -> OfflineMeasurementOutcome:
    """Process one or more saved stills; no camera acquisition occurs here."""
    profile = project.select_profile(profile_name)
    profile.require_frozen_roi()
    if request.resume_pending_synchronized and (
        request.acquisition_mode != "synchronized"
        or request.measurement_id is None
        or request.measurement_index is None
        or request.capture_start_utc is None
        or request.capture_end_utc is None
    ):
        raise ValueError(
            "Resuming synchronized evidence requires synchronized mode, supplied "
            "measurement ID/index, and capture start/end timestamps."
        )
    if request.allow_provenance_revision and (
        request.acquisition_mode != "synchronized"
        or not request.resume_pending_synchronized
        or request.processing_revision is None
        or isinstance(request.processing_revision, bool)
        or request.processing_revision < 1
    ):
        raise ValueError(
            "A provenance revision requires synchronized resume mode and a positive "
            "processing_revision."
        )
    sources = tuple(Path(path).expanduser().resolve() for path in image_paths)
    if not sources:
        raise ValueError("At least one saved still image is required.")
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Saved still image not found: {missing[0]}")
    inference = adapter or UltralyticsInferenceAdapter(profile)
    resolved_device = getattr(inference, "resolved_device", "unreported")
    class_map = inference.load()
    expected_class_map = resolve_semantic_classes(
        class_map.model_names,
        profile.material_role,
        profile.empty_role,
    )
    if class_map != expected_class_map:
        raise InferenceError("Inference adapter returned an unvalidated semantic class map.")
    weights_sha256 = inference.weights_sha256
    if weights_sha256 is None:
        raise ValueError("The validated inference adapter did not provide a weights SHA-256.")
    clock = now_utc if now_utc is not None else _utc_now
    session, artifact, measurement_id, index, trigger, config_hash = _prepare_output(
        project,
        profile,
        request,
        clock,
    )
    _ensure_session_manifest(
        session=session,
        request=request,
        profile=profile,
        config_hash=config_hash,
        weights_hash=weights_sha256,
        allow_provenance_revision=request.allow_provenance_revision,
        processing_revision=request.processing_revision,
    )

    processed: list[_ProcessedImage] = []
    for image_index, source in enumerate(sources, start=1):
        source_hash = sha256_file(source)
        image = read_image(source)
        try:
            output = inference.predict(image)
            if output.class_map != class_map:
                raise InferenceError("Inference adapter changed semantic mapping after validation.")
            if output.model_task != profile.expected_task:
                raise InferenceError("Inference adapter changed the model task after validation.")
            if output.weights_sha256 != weights_sha256:
                raise InferenceError("Inference adapter changed the model weights after validation.")
            if output.inference_device != "unreported":
                if resolved_device not in {"unreported", output.inference_device}:
                    raise InferenceError("Inference adapter changed devices during processing.")
                resolved_device = output.inference_device
            observation = observation_from_inference(
                output,
                profile,
                image.shape,
                source_path=source,
                source_sha256=source_hash,
            )
        except InferenceError as exc:
            blank = np.zeros(image.shape[:2], dtype=np.float32)
            observation = SemanticObservation(
                material_mask=blank,
                empty_mask=blank.copy(),
                material_confidences=(),
                empty_confidences=(),
                material_instance_count=0,
                empty_instance_count=0,
                material_present=False,
                empty_present=False,
                estimation_mode=profile.estimation_mode,
                source_path=source,
                source_sha256=source_hash,
                extraction_reasons=(f"inference_failed:{type(exc).__name__}",),
            )
        estimate = estimate_image_level(observation, profile)
        if sha256_file(source) != source_hash:
            raise OSError(f"Source image changed during offline processing: {source}")
        item = _ProcessedImage(
            index=image_index,
            source=source,
            source_sha256=source_hash,
            observation=observation,
            estimate=estimate,
        )
        processed.append(item)

    group = aggregate_image_estimates(
        (item.estimate for item in processed),
        profile,
        minimum_valid_images=request.minimum_valid_images,
    )
    for position, item in enumerate(processed):
        image = read_image(item.source)
        _save_image_evidence(
            item,
            image,
            profile,
            session,
            artifact,
            group,
            position,
        )
    result_path = artifact / "result.json"
    details_path = artifact / "image_details.csv"
    _atomic_json(
        result_path,
        _group_result_mapping(
            group,
            profile,
            class_map,
            weights_sha256,
            request.total_capacity_ml,
            resolved_device,
        ),
    )
    detail_rows = tuple(
        _detail_row(
            item,
            request.experiment_id,
            measurement_id,
            class_map,
            request.acquisition_mode,
            group,
        )
        for item in processed
    )
    _atomic_csv(details_path, YOLO_IMAGE_DETAIL_COLUMNS, detail_rows)
    manifest_path = artifact / "artifact_manifest.json"
    _write_artifact_manifest(artifact, manifest_path, request.experiment_id, measurement_id)

    processing_time = format_utc(clock())
    record = _common_record(
        request=request,
        profile=profile,
        group=group,
        experiment_id=request.experiment_id,
        measurement_id=measurement_id,
        measurement_index=index,
        trigger_time_utc=trigger,
        processing_time_utc=processing_time,
        session_directory=session,
        artifact_directory=artifact,
        config_sha256=config_hash,
        weights_sha256=weights_sha256,
        source_artifact=(
            request.source_artifact
            or (
                processed[0].source_copy or str(processed[0].source)
                if len(processed) == 1
                else _relative(session, details_path)
            )
        ),
    )
    store = MeasurementWorkbookStore(session / "yolo_measurements.xlsx", method="yolo")
    store.upsert_with_diagnostic_rows(
        record,
        sheet_name="yolo_image_details",
        columns=YOLO_IMAGE_DETAIL_COLUMNS,
        rows=detail_rows,
        replace_where={
            "experiment_id": request.experiment_id,
            "measurement_id": measurement_id,
        },
        usable_internal_height_mm=request.usable_internal_height_mm,
    )
    _update_session_manifest(
        session=session,
        request=request,
        profile=profile,
        config_hash=config_hash,
        weights_hash=weights_sha256,
        record=record,
        artifact=artifact,
        allow_provenance_revision=request.allow_provenance_revision,
        processing_revision=request.processing_revision,
    )
    return OfflineMeasurementOutcome(
        record=record,
        group=group,
        session_directory=session,
        artifact_directory=artifact,
        result_path=result_path,
        image_details_path=details_path,
        artifact_manifest_path=manifest_path,
    )


def _prepare_output(
    project: ProjectConfig,
    profile: ModelProfile,
    request: OfflineMeasurementRequest,
    clock: Any,
) -> tuple[Path, Path, str, int, str, str]:
    experiment_id = validate_experiment_id(request.experiment_id)
    if request.output_root is None:
        output_root = profile.output.root
    else:
        output_root = Path(request.output_root).expanduser()
        if not output_root.is_absolute():
            raise ValueError("output_root must be absolute when explicitly supplied.")
        output_root = output_root.resolve()
    session = output_root / experiment_id
    session.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(session / ".yolo_session.lock"):
        trigger_value = request.trigger_time_utc
        if trigger_value is None and request.measurement_id is not None:
            trigger_value = parse_measurement_id(request.measurement_id).trigger_time_utc
        trigger = format_utc(trigger_value or clock())
        existing = _existing_measurement_ids(session)
        if request.measurement_id is not None:
            parsed = parse_measurement_id(request.measurement_id)
            index = parsed.index if request.measurement_index is None else request.measurement_index
            validate_measurement_identity(request.measurement_id, index, trigger)
            measurement_id = request.measurement_id
        elif request.measurement_index is not None:
            index = request.measurement_index
            measurement_id = make_measurement_id(index, trigger)
        else:
            index, measurement_id = next_measurement_identity(existing, trigger)
        occupied = {
            parse_measurement_id(item).index: item for item in existing
        }.get(index)
        if occupied is not None and not (
            request.resume_pending_synchronized
            and request.measurement_id == occupied
            and request.acquisition_mode == "synchronized"
        ):
            raise FileExistsError(f"measurement_index {index} is already assigned to {occupied}.")
        artifact = session / "measurements" / measurement_id / "yolo"
        if artifact.exists():
            if not (
                request.allow_provenance_revision
                and artifact.is_dir()
                and not any(artifact.iterdir())
            ):
                raise FileExistsError(f"YOLO measurement evidence already exists: {artifact}")
        else:
            artifact.mkdir(parents=True, exist_ok=False)
        config_hash = _ensure_effective_config(
            session,
            project,
            profile,
            allow_revision=request.allow_provenance_revision,
        )
    return session, artifact, measurement_id, index, trigger, config_hash


def _existing_measurement_ids(session: Path) -> set[str]:
    values: set[str] = set()
    measurements = session / "measurements"
    if measurements.is_dir():
        for child in measurements.iterdir():
            if not child.is_dir():
                continue
            try:
                parse_measurement_id(child.name)
            except ValueError:
                continue
            values.add(child.name)
    workbook = session / "yolo_measurements.xlsx"
    if workbook.is_file():
        values.update(
            record.measurement_id
            for record in MeasurementWorkbookStore(workbook, method="yolo").load_records()
            if record.measurement_id is not None
        )
    return values


def _ensure_effective_config(
    session: Path,
    project: ProjectConfig,
    profile: ModelProfile,
    *,
    allow_revision: bool = False,
) -> str:
    target = session / "provenance" / f"yolo_effective_config_{profile.name}.yaml"
    payload = effective_profile_config_bytes(project, profile)
    if target.exists() and target.read_bytes() != payload:
        if not allow_revision:
            raise ValueError(
                f"Session YOLO effective configuration is frozen and differs: {target}"
            )
        _atomic_bytes(target, payload)
    elif not target.exists():
        _atomic_bytes(target, payload)
    return sha256_file(target)


def effective_profile_config_bytes(
    project: ProjectConfig,
    profile: ModelProfile,
) -> bytes:
    """Return the exact deterministic effective-profile snapshot bytes."""
    return yaml.safe_dump(
        {
            "configuration_schema_version": project.schema_version,
            "selected_profile": profile.name,
            "profile": _serializable(asdict(profile)),
        },
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")


def effective_profile_config_sha256(
    project: ProjectConfig,
    profile: ModelProfile,
) -> str:
    """Hash the exact snapshot that process_saved_images will persist."""
    return sha256_bytes(effective_profile_config_bytes(project, profile))


def _save_image_evidence(
    item: _ProcessedImage,
    image: np.ndarray,
    profile: ModelProfile,
    session: Path,
    artifact: Path,
    group: GroupLevelEstimate,
    group_position: int,
) -> None:
    source_directory = artifact / "source_images"
    source_name = artifact_name(
        item.source,
        index=item.index,
        kind="source",
        extension=item.source.suffix,
    )
    source_copy = source_directory / source_name
    _atomic_copy(item.source, source_copy)
    if sha256_file(source_copy) != item.source_sha256:
        raise OSError(f"Copied source hash differs from original: {item.source}")
    item.source_copy = _relative(session, source_copy)

    diagnostic_name = artifact_name(
        item.source, index=item.index, kind="diagnostics", extension=".json"
    )
    diagnostic_path = artifact / "per_image" / diagnostic_name
    _atomic_json(
        diagnostic_path,
        _image_diagnostic_mapping(
            item.estimate,
            group,
            group_position,
        ),
    )
    item.diagnostics = _relative(session, diagnostic_path)

    if profile.output.save_masks:
        material_path = artifact / "masks" / artifact_name(
            item.source, index=item.index, kind="material_mask", extension=".png"
        )
        empty_path = artifact / "masks" / artifact_name(
            item.source, index=item.index, kind="empty_mask", extension=".png"
        )
        write_image(
            material_path,
            np.where(item.observation.material_mask >= profile.estimation.mask_threshold, 255, 0).astype(np.uint8),
        )
        write_image(
            empty_path,
            np.where(item.observation.empty_mask >= profile.estimation.mask_threshold, 255, 0).astype(np.uint8),
        )
        item.material_mask = _relative(session, material_path)
        item.empty_mask = _relative(session, empty_path)
    if profile.output.save_overlays:
        overlay_path = artifact / "overlays" / artifact_name(
            item.source,
            index=item.index,
            kind="overlay",
            extension=profile.output.image_format,
        )
        write_image(
            overlay_path,
            _render_overlay(
                image,
                item.observation,
                item.estimate,
                profile,
                group,
                group_position,
            ),
        )
        item.overlay = _relative(session, overlay_path)


def _render_overlay(
    image: np.ndarray,
    observation: SemanticObservation,
    estimate: ImageLevelEstimate,
    profile: ModelProfile,
    group: GroupLevelEstimate,
    group_position: int,
) -> np.ndarray:
    overlay = np.asarray(image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    material = observation.material_mask >= profile.estimation.mask_threshold
    empty = observation.empty_mask >= profile.estimation.mask_threshold
    tint = overlay.copy()
    tint[material] = (40, 180, 40)
    tint[empty] = (220, 120, 40)
    tint[material & empty] = (180, 40, 180)
    overlay = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0.0)
    roi = estimate.roi
    cv2.rectangle(overlay, (roi.left, roi.top), (roi.right - 1, roi.bottom - 1), (255, 255, 0), 2)
    retained = group.retained_flags[group_position]
    color = (0, 255, 0) if retained else (0, 180, 255)
    if estimate.interface_position_px is not None:
        interface_y = int(round(estimate.interface_position_px))
        interface_y = max(roi.top, min(roi.bottom - 1, interface_y))
        cv2.line(overlay, (roi.left, interface_y), (roi.right - 1, interface_y), color, 2)

    class_labels: list[str] = []
    if observation.material_present:
        confidence = max(observation.material_confidences, default=0.0)
        class_labels.append(f"{profile.material_role.class_name} {confidence:.2f}")
    if observation.empty_present:
        confidence = max(observation.empty_confidences, default=0.0)
        class_labels.append(f"{profile.empty_role.class_name} {confidence:.2f}")
    labels = [" + ".join(class_labels) if class_labels else "No semantic class"]
    if group.valid and retained and group.material_percent is not None:
        if estimate.candidate_material_percent is None:
            # A unanimous six-frame single-class endpoint can be scientifically
            # explicit at group level without an individual interface candidate.
            labels.append(f"Final level {group.material_percent:.2f}%")
        else:
            labels.append(
                f"Final level {group.material_percent:.2f}% | "
                f"image {estimate.candidate_material_percent:.2f}%"
            )
    elif estimate.candidate_material_percent is not None:
        labels.append(f"Candidate level {estimate.candidate_material_percent:.2f}%")
    for line_index, label in enumerate(labels):
        cv2.putText(
            overlay,
            label,
            (roi.left + 4, max(18, roi.top + 18) + line_index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _image_diagnostic_mapping(
    estimate: ImageLevelEstimate,
    group: GroupLevelEstimate,
    group_position: int,
) -> dict[str, Any]:
    filter_reasons = group.pattern_filter_reasons[group_position]
    return {
        "diagnostic_schema_version": 1,
        "valid": estimate.valid,
        "class_pattern": estimate.class_pattern,
        "majority_class_pattern": group.majority_pattern,
        "retained_for_aggregation": group.retained_flags[group_position],
        "pattern_filter_reasons": filter_reasons,
        "accepted_material_percent": (
            group.accepted_material_percentages[group_position]
        ),
        "candidate_material_percent": estimate.candidate_material_percent,
        "interface_fraction_from_full": estimate.interface_fraction_from_full,
        "interface_position_px": estimate.interface_position_px,
        "interface_spread_fraction": estimate.interface_spread_fraction,
        "material_coverage_fraction": estimate.material_coverage_fraction,
        "empty_coverage_fraction": estimate.empty_coverage_fraction,
        "overlap_fraction": estimate.overlap_fraction,
        "unclassified_fraction": estimate.unclassified_fraction,
        "material_confidences": estimate.material_confidences,
        "empty_confidences": estimate.empty_confidences,
        "material_instance_count": estimate.material_instance_count,
        "empty_instance_count": estimate.empty_instance_count,
        "roi": asdict(estimate.roi),
        "roi_mode": estimate.roi_mode,
        "analysis_bounds_source": estimate.analysis_bounds_source,
        "level_calibration_mode": estimate.level_calibration_mode,
        "axis": estimate.axis,
        "estimation_mode": estimate.estimation_mode,
        "row_material_occupancy": estimate.row_material_occupancy,
        "row_empty_occupancy": estimate.row_empty_occupancy,
        "quality_warnings": estimate.quality_warnings,
        "rejection_reasons": estimate.rejection_reasons,
    }


def _group_result_mapping(
    group: GroupLevelEstimate,
    profile: ModelProfile,
    class_map: SemanticClassMap,
    weights_sha256: str,
    capacity_ml: float,
    resolved_device: str,
) -> dict[str, Any]:
    estimates = {
        "estimated_material_percent": None,
        "estimated_material_volume_ml": None,
        "estimated_empty_percent": None,
        "estimated_empty_volume_ml": None,
    }
    if group.valid and group.material_percent is not None:
        from experiment_records import derive_estimate

        material_percent, material_volume, empty_percent, empty_volume = derive_estimate(
            group.material_percent,
            capacity_ml,
        )
        estimates = {
            "estimated_material_percent": material_percent,
            "estimated_material_volume_ml": material_volume,
            "estimated_empty_percent": empty_percent,
            "estimated_empty_volume_ml": empty_volume,
        }
    return {
        "result_schema_version": 1,
        "method": "yolo",
        "valid": group.valid,
        **estimates,
        "candidate_material_percent": group.candidate_material_percent,
        "robust_spread_percentage_points": group.robust_spread_percentage_points,
        "accepted_image_count": group.accepted_count,
        "rejected_image_count": group.rejected_count,
        "total_image_count": group.total_count,
        "minimum_valid_images_required": group.minimum_valid_images_required,
        "majority_class_pattern": group.majority_pattern,
        "class_pattern_counts": dict(group.pattern_counts),
        "retained_image_count": group.accepted_count,
        "aggregation_method": group.aggregation_method,
        "rejection_reasons": group.rejection_reasons,
        "model_profile": profile.name,
        "roi_mode": profile.tube_roi.usage_mode,
        "level_calibration_modes": sorted(
            {image.level_calibration_mode for image in group.images}
        ),
        "quality_warnings": sorted(
            {
                warning
                for image in group.images
                for warning in image.quality_warnings
            }
        ),
        "expected_task": profile.expected_task,
        "estimation_mode": profile.estimation_mode,
        "semantic_class_map": {
            "material": {"id": class_map.material_id, "name": class_map.material_name},
            "empty": {"id": class_map.empty_id, "name": class_map.empty_name},
        },
        "weights_sha256": weights_sha256,
        "requested_inference_device": profile.inference.device,
        "resolved_inference_device": resolved_device,
        "scientifically_validated": False,
    }


def _detail_row(
    item: _ProcessedImage,
    experiment_id: str,
    measurement_id: str,
    class_map: SemanticClassMap,
    acquisition_mode: str,
    group: GroupLevelEstimate,
) -> tuple[Any, ...]:
    estimate = item.estimate
    group_position = item.index - 1
    retained = group.retained_flags[group_position]
    filter_reasons = group.pattern_filter_reasons[group_position]
    values = {
        "diagnostic_schema_version": 1,
        "experiment_id": experiment_id,
        "measurement_id": measurement_id,
        "image_index": item.index,
        "source_path": (
            str(item.source) if acquisition_mode == "folder" else item.source_copy
        ),
        "source_sha256": item.source_sha256,
        "estimation_mode": estimate.estimation_mode,
        "class_pattern": estimate.class_pattern,
        "majority_class_pattern": group.majority_pattern,
        "retained_for_aggregation": retained,
        "pattern_filter_reasons": ";".join(filter_reasons),
        "material_class_id": class_map.material_id,
        "material_class_name": class_map.material_name,
        "empty_class_id": class_map.empty_id,
        "empty_class_name": class_map.empty_name,
        "material_confidences": json.dumps(estimate.material_confidences),
        "empty_confidences": json.dumps(estimate.empty_confidences),
        "material_instance_count": estimate.material_instance_count,
        "empty_instance_count": estimate.empty_instance_count,
        "material_coverage_fraction": estimate.material_coverage_fraction,
        "empty_coverage_fraction": estimate.empty_coverage_fraction,
        "overlap_fraction": estimate.overlap_fraction,
        "unclassified_fraction": estimate.unclassified_fraction,
        "interface_position_px": estimate.interface_position_px,
        "interface_fraction_from_full": estimate.interface_fraction_from_full,
        "interface_spread_fraction": estimate.interface_spread_fraction,
        "candidate_material_percent": estimate.candidate_material_percent,
        "accepted_material_percent": group.accepted_material_percentages[group_position],
        "valid": estimate.valid,
        "rejection_reasons": ";".join((*estimate.rejection_reasons, *filter_reasons)),
        "roi_left_px": estimate.roi.left,
        "roi_top_px": estimate.roi.top,
        "roi_right_px": estimate.roi.right,
        "roi_bottom_px": estimate.roi.bottom,
        "roi_mode": estimate.roi_mode,
        "analysis_bounds_source": estimate.analysis_bounds_source,
        "level_calibration_mode": estimate.level_calibration_mode,
        "quality_warnings": ";".join(estimate.quality_warnings),
        "axis": estimate.axis,
        "source_copy": item.source_copy,
        "overlay_artifact": item.overlay,
        "material_mask_artifact": item.material_mask,
        "empty_mask_artifact": item.empty_mask,
        "diagnostics_artifact": item.diagnostics,
    }
    return tuple(values[column] for column in YOLO_IMAGE_DETAIL_COLUMNS)


def _common_record(
    *,
    request: OfflineMeasurementRequest,
    profile: ModelProfile,
    group: GroupLevelEstimate,
    experiment_id: str,
    measurement_id: str,
    measurement_index: int,
    trigger_time_utc: str,
    processing_time_utc: str,
    session_directory: Path,
    artifact_directory: Path,
    config_sha256: str,
    weights_sha256: str,
    source_artifact: str,
) -> CommonMeasurementRecord:
    repository = (
        Path(request.software_repository).resolve()
        if request.software_repository is not None
        else Path(__file__).resolve().parents[2]
    )
    record = CommonMeasurementRecord(
        experiment_id=experiment_id,
        measurement_id=measurement_id,
        measurement_index=measurement_index,
        method="yolo",
        acquisition_mode=request.acquisition_mode,
        trigger_time_utc=trigger_time_utc,
        capture_start_utc=request.capture_start_utc,
        capture_end_utc=request.capture_end_utc,
        processing_time_utc=processing_time_utc,
        material_name=request.material_name,
        total_capacity_ml=request.total_capacity_ml,
        bulk_density_g_per_ml=request.bulk_density_g_per_ml,
        total_possible_weight_g=request.total_possible_weight_g,
        reference_material_weight_g=request.reference_material_weight_g,
        manual_material_level_mm=request.manual_material_level_mm,
        valid=group.valid,
        status=(
            RecordStatus.COMPLETE_VALID.value
            if group.valid
            else RecordStatus.COMPLETE_INVALID.value
        ),
        notes=request.notes,
        artifact_directory=_relative(session_directory, artifact_directory),
        source_artifact=source_artifact,
        config_sha256=config_sha256,
        calibration_or_model_sha256=weights_sha256,
        software_commit=git_revision(repository),
    )
    if group.valid and group.material_percent is not None:
        record = with_estimate(record, group.material_percent)
    return prepare_record(
        record,
        usable_internal_height_mm=request.usable_internal_height_mm,
    )


def _update_session_manifest(
    *,
    session: Path,
    request: OfflineMeasurementRequest,
    profile: ModelProfile,
    config_hash: str,
    weights_hash: str,
    record: CommonMeasurementRecord,
    artifact: Path,
    allow_provenance_revision: bool = False,
    processing_revision: int | None = None,
) -> None:
    """Append the completed YOLO entry without replacing depth entries."""
    _ensure_session_manifest(
        session=session,
        request=request,
        profile=profile,
        config_hash=config_hash,
        weights_hash=weights_hash,
        allow_provenance_revision=allow_provenance_revision,
        processing_revision=processing_revision,
    )
    path = session / "session_manifest.json"
    with _exclusive_lock(session / ".yolo_session.lock"):
        data = json.loads(path.read_text(encoding="utf-8"))
        measurements = [
            item
            for item in data.get("measurements", [])
            if not (
                item.get("measurement_id") == record.measurement_id
                and item.get("method") == "yolo"
            )
        ]
        measurements.append(
            {
                "measurement_id": record.measurement_id,
                "measurement_index": record.measurement_index,
                "method": "yolo",
                "acquisition_mode": record.acquisition_mode,
                "status": record.status,
                "valid": record.valid,
                "artifact_directory": _relative(session, artifact),
            }
        )
        data["measurements"] = sorted(
            measurements,
            key=lambda item: (item.get("measurement_index") or 0, item.get("method") or ""),
        )
        data["updated_utc"] = format_utc(_utc_now())
        _atomic_json(path, data)


def _ensure_session_manifest(
    *,
    session: Path,
    request: OfflineMeasurementRequest,
    profile: ModelProfile,
    config_hash: str,
    weights_hash: str,
    allow_provenance_revision: bool = False,
    processing_revision: int | None = None,
) -> None:
    path = session / "session_manifest.json"
    with _exclusive_lock(session / ".yolo_session.lock"):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("experiment_id") != request.experiment_id:
                raise ValueError("Existing session manifest has a different experiment_id.")
            existing_purpose = data.get("purpose")
            if request.purpose is not None and existing_purpose not in {None, request.purpose}:
                raise ValueError("Existing session manifest has a different purpose.")
            if existing_purpose is None and request.purpose is not None:
                data["purpose"] = request.purpose
        else:
            now = format_utc(_utc_now())
            data = {
                "schema_version": 1,
                "common_record_schema_version": SCHEMA_VERSION,
                "experiment_id": request.experiment_id,
                "purpose": request.purpose,
                "created_utc": now,
                "scientifically_validated": False,
                "measurements": [],
            }
        profiles = data.setdefault("yolo_profiles", {})
        current_profile = {
            "effective_config": f"provenance/yolo_effective_config_{profile.name}.yaml",
            "config_sha256": config_hash,
            "model_sha256": weights_hash,
        }
        existing_profile = profiles.get(profile.name)
        if existing_profile is not None and existing_profile != current_profile:
            if not allow_provenance_revision:
                raise ValueError(
                    f"Existing session has different frozen YOLO provenance for profile {profile.name!r}."
                )
            revisions = data.setdefault("yolo_profile_revisions", {}).setdefault(
                profile.name,
                [],
            )
            revision_item = {
                "processing_revision": processing_revision,
                "superseded_utc": format_utc(_utc_now()),
                **existing_profile,
            }
            if revision_item not in revisions:
                revisions.append(revision_item)
        profiles[profile.name] = current_profile
        data["updated_utc"] = format_utc(_utc_now())
        _atomic_json(path, data)


def _write_artifact_manifest(
    artifact: Path,
    manifest: Path,
    experiment_id: str,
    measurement_id: str,
) -> None:
    entries = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path != manifest:
            entries.append(
                {
                    "path": path.relative_to(artifact).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    _atomic_json(
        manifest,
        {
            "artifact_manifest_schema_version": 1,
            "experiment_id": experiment_id,
            "measurement_id": measurement_id,
            "method": "yolo",
            "created_utc": format_utc(_utc_now()),
            "artifacts": entries,
        },
    )


def _atomic_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_path(path, ".tmp.csv")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
    )


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
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {target}")
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


def _relative(session: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(session.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes session directory: {path}") from exc


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def _exclusive_lock(path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for YOLO session lock: {path}") from exc
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
