"""Tube-axis material-level estimation from one or two semantic support masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .domain import ModelProfile


SUPPORTED_CLASS_PATTERNS = (
    "material_only",
    "empty_only",
    "material_and_empty",
)


@dataclass(frozen=True, slots=True)
class RoiBounds:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    """Semantic evidence for one still after adapter-specific extraction."""

    material_mask: np.ndarray
    empty_mask: np.ndarray
    material_confidences: tuple[float, ...] = (1.0,)
    empty_confidences: tuple[float, ...] = (1.0,)
    material_instance_count: int = 1
    empty_instance_count: int = 1
    material_present: bool = True
    empty_present: bool = True
    estimation_mode: str = "segmentation_interface"
    source_path: Path | None = None
    source_sha256: str | None = None
    extraction_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImageLevelEstimate:
    source_path: Path | None
    source_sha256: str | None
    valid: bool
    class_pattern: str
    material_percent: float | None
    candidate_material_percent: float | None
    interface_fraction_from_full: float | None
    interface_position_px: float | None
    interface_spread_fraction: float | None
    material_coverage_fraction: float
    empty_coverage_fraction: float
    overlap_fraction: float
    unclassified_fraction: float
    material_confidences: tuple[float, ...]
    empty_confidences: tuple[float, ...]
    material_instance_count: int
    empty_instance_count: int
    roi: RoiBounds
    roi_mode: str
    analysis_bounds_source: str
    level_calibration_mode: str
    axis: str
    estimation_mode: str
    row_material_occupancy: tuple[float, ...]
    row_empty_occupancy: tuple[float, ...]
    quality_warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupLevelEstimate:
    valid: bool
    material_percent: float | None
    candidate_material_percent: float | None
    robust_spread_percentage_points: float | None
    accepted_count: int
    rejected_count: int
    total_count: int
    minimum_valid_images_required: int
    majority_pattern: str | None
    pattern_counts: tuple[tuple[str, int], ...]
    retained_flags: tuple[bool, ...]
    accepted_material_percentages: tuple[float | None, ...]
    pattern_filter_reasons: tuple[tuple[str, ...], ...]
    aggregation_method: str
    rejection_reasons: tuple[str, ...]
    images: tuple[ImageLevelEstimate, ...]


def estimate_image_level(
    observation: SemanticObservation,
    profile: ModelProfile,
) -> ImageLevelEstimate:
    """Estimate fill from geometry inside the frozen calibrated tube ROI.

    Material-only, Empty-only, and dual-class observations are valid patterns.
    Coverage, gaps, and overlap are diagnostics; mask area is never equated with
    volume and gaps/overlap are not standalone rejection reasons.
    """
    material_raw = np.asarray(observation.material_mask)
    empty_raw = np.asarray(observation.empty_mask)
    if material_raw.ndim != 2 or empty_raw.ndim != 2:
        raise ValueError("material_mask and empty_mask must both be two-dimensional.")
    if material_raw.shape != empty_raw.shape or material_raw.size == 0:
        raise ValueError("material_mask and empty_mask must have one non-empty shared shape.")

    threshold = profile.estimation.mask_threshold
    material = np.asarray(material_raw >= threshold, dtype=bool)
    empty = np.asarray(empty_raw >= threshold, dtype=bool)
    pattern = _class_pattern(observation)
    whole_image_mode = profile.tube_roi.usage_mode == "whole_image"
    detected_support = material | empty
    roi = (
        _support_bounds(detected_support)
        if whole_image_mode and np.any(detected_support)
        else _roi_bounds(material.shape, profile)
    )
    analysis_bounds_source = (
        "detected_semantic_extent" if whole_image_mode else "configured_roi"
    )
    level_calibration_mode = (
        "detected_semantic_extent_uncalibrated"
        if whole_image_mode and pattern == "material_and_empty"
        else (
            "unavailable_without_vertical_calibration"
            if whole_image_mode
            else "configured_roi_limits"
        )
    )
    warnings: list[str] = []
    if whole_image_mode:
        warnings.append("calibrated_tube_roi_quality_gates_disabled")
        if pattern == "material_and_empty":
            warnings.extend(
                (
                    "full_empty_vertical_limits_not_calibrated",
                    "detected_semantic_extent_used_as_provisional_level_limits",
                )
            )
    material_roi = material[roi.top : roi.bottom, roi.left : roi.right]
    empty_roi = empty[roi.top : roi.bottom, roi.left : roi.right]
    if profile.tube_roi.axis == "bottom_to_top":
        material_axis = material_roi[::-1]
        empty_axis = empty_roi[::-1]
    else:
        material_axis = material_roi
        empty_axis = empty_roi

    overlap = material_axis & empty_axis
    classified = material_axis | empty_axis
    material_only_mask = material_axis & ~empty_axis
    empty_only_mask = empty_axis & ~material_axis
    material_coverage = float(np.mean(material_axis))
    empty_coverage = float(np.mean(empty_axis))
    overlap_fraction = float(np.mean(overlap))
    unclassified_fraction = float(np.mean(~classified))
    row_material = np.mean(material_only_mask, axis=1, dtype=np.float64)
    row_empty = np.mean(empty_only_mask, axis=1, dtype=np.float64)
    reasons = list(observation.extraction_reasons)
    if not profile.tube_roi.frozen and profile.tube_roi.usage_mode != "whole_image":
        reasons.append("tube_roi_not_frozen")
    _append_role_reasons(reasons, observation, profile, pattern)
    if whole_image_mode and np.any(detected_support):
        reasons.extend(_whole_image_detection_sanity(detected_support, roi, profile))

    boundary: float | None = None
    column_boundaries = np.asarray([], dtype=np.float64)
    compressed_states: tuple[int, ...] = ()
    states = np.zeros(roi.height, dtype=np.int8)

    if pattern == "material_and_empty":
        if not np.any(material_axis):
            reasons.append("missing_material_mask")
        if not np.any(empty_axis):
            reasons.append("missing_empty_mask")
        if np.any(material_axis) and np.any(empty_axis):
            smooth_material = _smooth_rows(
                row_material,
                profile.estimation.row_smoothing_window,
            )
            smooth_empty = _smooth_rows(
                row_empty,
                profile.estimation.row_smoothing_window,
            )
            states = _dominant_states(
                smooth_material,
                smooth_empty,
                (
                    0.0
                    if whole_image_mode
                    else profile.estimation.row_occupancy_threshold
                ),
            )
            compressed_states = _compress_nonzero(states)
            boundary = _minimum_cost_boundary(row_material, row_empty)
            column_boundaries = _column_boundaries(
                material_only_mask,
                empty_only_mask,
                classified,
            )
            if _has_implausible_orientation(compressed_states):
                reasons.append("implausible_material_empty_orientation")
            if len(compressed_states) > 2:
                reasons.append("multiple_interface_transitions")
            if not whole_image_mode:
                minimum_run = profile.estimation.minimum_dominant_run_rows
                if _longest_run(states, 1) < minimum_run:
                    reasons.append("insufficient_material_row_support")
                if _longest_run(states, -1) < minimum_run:
                    reasons.append("insufficient_empty_row_support")
    elif pattern == "material_only":
        if whole_image_mode:
            reasons.append("vertical_level_calibration_unavailable_for_single_class")
        else:
            boundary, column_boundaries, geometry_reasons = _single_class_boundary(
                material_axis,
                role="material",
                profile=profile,
            )
            reasons.extend(geometry_reasons)
    elif pattern == "empty_only":
        if whole_image_mode:
            reasons.append("vertical_level_calibration_unavailable_for_single_class")
        else:
            boundary, column_boundaries, geometry_reasons = _single_class_boundary(
                empty_axis,
                role="empty",
                profile=profile,
            )
            reasons.extend(geometry_reasons)
    else:
        reasons.append("missing_semantic_classes")

    interface_spread = (
        float(np.percentile(column_boundaries, 90) - np.percentile(column_boundaries, 10))
        if column_boundaries.size
        else None
    )
    if boundary is None:
        reasons.append("interface_geometry_unavailable")

    reasons = list(dict.fromkeys(reasons))
    interface_fraction = None if boundary is None else float(boundary / roi.height)
    candidate_percent = (
        None
        if interface_fraction is None
        else float(np.clip(100.0 * (1.0 - interface_fraction), 0.0, 100.0))
    )
    source_position = (
        None
        if boundary is None
        else (
            roi.top + boundary
            if profile.tube_roi.axis == "top_to_bottom"
            else roi.bottom - boundary
        )
    )
    valid = not reasons and candidate_percent is not None
    return ImageLevelEstimate(
        source_path=observation.source_path,
        source_sha256=observation.source_sha256,
        valid=valid,
        class_pattern=pattern,
        material_percent=(candidate_percent if valid else None),
        candidate_material_percent=candidate_percent,
        interface_fraction_from_full=interface_fraction,
        interface_position_px=(None if source_position is None else float(source_position)),
        interface_spread_fraction=interface_spread,
        material_coverage_fraction=material_coverage,
        empty_coverage_fraction=empty_coverage,
        overlap_fraction=overlap_fraction,
        unclassified_fraction=unclassified_fraction,
        material_confidences=tuple(float(value) for value in observation.material_confidences),
        empty_confidences=tuple(float(value) for value in observation.empty_confidences),
        material_instance_count=observation.material_instance_count,
        empty_instance_count=observation.empty_instance_count,
        roi=roi,
        roi_mode=profile.tube_roi.usage_mode,
        analysis_bounds_source=analysis_bounds_source,
        level_calibration_mode=level_calibration_mode,
        axis=profile.tube_roi.axis,
        estimation_mode=observation.estimation_mode,
        row_material_occupancy=tuple(float(value) for value in row_material),
        row_empty_occupancy=tuple(float(value) for value in row_empty),
        quality_warnings=tuple(dict.fromkeys(warnings)),
        rejection_reasons=tuple(reasons),
    )


def aggregate_image_estimates(
    images: Iterable[ImageLevelEstimate],
    profile: ModelProfile,
    *,
    minimum_valid_images: int | None = None,
) -> GroupLevelEstimate:
    """Filter by majority class pattern and average retained image values."""
    retained = tuple(images)
    if minimum_valid_images is not None and (
        isinstance(minimum_valid_images, bool)
        or not isinstance(minimum_valid_images, int)
        or minimum_valid_images < 1
    ):
        raise ValueError("minimum_valid_images must be an integer >= 1.")

    counts = {
        pattern: sum(item.class_pattern == pattern for item in retained)
        for pattern in SUPPORTED_CLASS_PATTERNS
    }
    nonzero = {pattern: count for pattern, count in counts.items() if count > 0}
    majority: str | None = None
    group_reasons: list[str] = []
    if not nonzero:
        group_reasons.append("no_supported_class_pattern")
    else:
        highest = max(nonzero.values())
        winners = [pattern for pattern, count in nonzero.items() if count == highest]
        if len(winners) == 1:
            majority = winners[0]
        elif (
            len(retained) == 6
            and highest == 3
            and "material_and_empty" in winners
        ):
            # An exact 3-vs-3 six-frame tie deliberately favors the richer
            # two-role evidence, as required by the acquisition protocol.
            majority = "material_and_empty"
        else:
            group_reasons.append("ambiguous_majority_class_pattern")

    unanimous_single_class_endpoint = (
        len(retained) == 6
        and majority in {"material_only", "empty_only"}
        and counts[majority] == 6
    )
    flags: list[bool] = []
    accepted_by_image: list[float | None] = []
    filter_reasons: list[tuple[str, ...]] = []
    accepted: list[float] = []
    for item in retained:
        reasons: list[str] = []
        if item.class_pattern not in SUPPORTED_CLASS_PATTERNS:
            reasons.append("unsupported_class_pattern")
        elif majority is None:
            reasons.append("majority_class_pattern_unavailable")
        elif item.class_pattern != majority:
            reasons.append("class_pattern_disagrees_with_majority")
        elif unanimous_single_class_endpoint and not _unanimous_endpoint_eligible(item):
            reasons.append("individual_image_invalid")
        elif (
            not unanimous_single_class_endpoint
            and (not item.valid or item.material_percent is None)
        ):
            reasons.append("individual_image_invalid")
        accepted_flag = not reasons
        flags.append(accepted_flag)
        filter_reasons.append(tuple(reasons))
        if accepted_flag:
            value = (
                100.0
                if unanimous_single_class_endpoint and majority == "material_only"
                else (
                    0.0
                    if unanimous_single_class_endpoint and majority == "empty_only"
                    else float(item.material_percent)
                )
            )
            accepted.append(value)
            accepted_by_image.append(value)
        else:
            accepted_by_image.append(None)

    candidate = float(np.mean(accepted)) if accepted else None
    spread = (
        float(np.percentile(accepted, 90) - np.percentile(accepted, 10))
        if accepted
        else None
    )
    if not accepted:
        group_reasons.append("no_valid_images_after_majority_filter")
    if (
        spread is not None
        and spread > profile.aggregation.maximum_spread_percentage_points
    ):
        group_reasons.append("excessive_group_spread")
    group_reasons = list(dict.fromkeys(group_reasons))
    valid = not group_reasons and candidate is not None
    return GroupLevelEstimate(
        valid=valid,
        material_percent=(candidate if valid else None),
        candidate_material_percent=candidate,
        robust_spread_percentage_points=spread,
        accepted_count=len(accepted),
        rejected_count=len(retained) - len(accepted),
        total_count=len(retained),
        minimum_valid_images_required=1,
        majority_pattern=majority,
        pattern_counts=tuple((pattern, counts[pattern]) for pattern in SUPPORTED_CLASS_PATTERNS),
        retained_flags=tuple(flags),
        accepted_material_percentages=tuple(accepted_by_image),
        pattern_filter_reasons=tuple(filter_reasons),
        aggregation_method="arithmetic_mean",
        rejection_reasons=tuple(group_reasons),
        images=retained,
    )


def _unanimous_endpoint_eligible(item: ImageLevelEstimate) -> bool:
    """Allow only confident, geometrically sane images into a 6/6 endpoint rule."""

    if item.valid and item.material_percent is not None:
        return True
    # Whole-image mode has no calibrated vertical endpoints, so a single class
    # cannot form an individual interface estimate. Six unanimous confident
    # frames may nevertheless invoke the explicit semantic endpoint rule. Tiny
    # detection, confidence, extraction, and other geometry failures remain.
    permitted = {
        "vertical_level_calibration_unavailable_for_single_class",
        "interface_geometry_unavailable",
    }
    return bool(item.rejection_reasons) and set(item.rejection_reasons) <= permitted


def _class_pattern(observation: SemanticObservation) -> str:
    if observation.material_present and observation.empty_present:
        return "material_and_empty"
    if observation.material_present:
        return "material_only"
    if observation.empty_present:
        return "empty_only"
    return "none"


def _single_class_boundary(
    support: np.ndarray,
    *,
    role: str,
    profile: ModelProfile,
) -> tuple[float | None, np.ndarray, tuple[str, ...]]:
    height, width = support.shape
    occupied_columns = np.flatnonzero(np.any(support, axis=0))
    reasons: list[str] = []
    if not occupied_columns.size:
        return None, np.asarray([], dtype=np.float64), (f"missing_{role}_mask",)

    lateral_fraction = float(occupied_columns.size / width)
    if lateral_fraction < profile.estimation.minimum_single_class_lateral_fraction:
        reasons.append(f"insufficient_{role}_lateral_support")

    positions: list[float] = []
    axial_extents: list[float] = []
    for column in occupied_columns:
        rows = np.flatnonzero(support[:, int(column)])
        if role == "material":
            boundary = float(rows[0])
            axial_extent = float((height - rows[0]) / height)
        else:
            boundary = float(rows[-1] + 1)
            axial_extent = float((rows[-1] + 1) / height)
        positions.append(boundary / height)
        axial_extents.append(axial_extent)

    median_extent = float(np.median(axial_extents))
    if median_extent < profile.estimation.minimum_single_class_axial_fraction:
        reasons.append(f"tiny_{role}_axial_extent")
    expected_end_occupancy = (
        float(np.mean(support[-1]))
        if role == "material"
        else float(np.mean(support[0]))
    )
    if expected_end_occupancy < profile.estimation.single_class_end_occupancy_threshold:
        reasons.append(
            "material_not_connected_to_tube_bottom"
            if role == "material"
            else "empty_not_connected_to_tube_top"
        )
    normalized = np.asarray(positions, dtype=np.float64)
    boundary_px = float(np.median(normalized) * height)
    return boundary_px, normalized, tuple(reasons)


def _roi_bounds(shape: tuple[int, int], profile: ModelProfile) -> RoiBounds:
    height, width = shape
    configured = profile.tube_roi
    left = max(0, min(width - 1, int(np.floor(configured.left * width))))
    top = max(0, min(height - 1, int(np.floor(configured.top * height))))
    right = max(left + 1, min(width, int(np.ceil(configured.right * width))))
    bottom = max(top + 1, min(height, int(np.ceil(configured.bottom * height))))
    return RoiBounds(left=left, top=top, right=right, bottom=bottom)


def _support_bounds(support: np.ndarray) -> RoiBounds:
    """Return the tight semantic extent used only for uncalibrated analysis."""
    rows, columns = np.nonzero(support)
    if not rows.size or not columns.size:
        height, width = support.shape
        return RoiBounds(left=0, top=0, right=width, bottom=height)
    return RoiBounds(
        left=int(np.min(columns)),
        top=int(np.min(rows)),
        right=int(np.max(columns)) + 1,
        bottom=int(np.max(rows)) + 1,
    )


def _whole_image_detection_sanity(
    support: np.ndarray,
    bounds: RoiBounds,
    profile: ModelProfile,
) -> tuple[str, ...]:
    """Reject only obviously tiny semantic evidence relative to the source image."""
    height, width = support.shape
    reasons: list[str] = []
    if bounds.width / width < profile.estimation.minimum_whole_image_detection_lateral_fraction:
        reasons.append("tiny_detection_lateral_extent")
    if bounds.height / height < profile.estimation.minimum_whole_image_detection_axial_fraction:
        reasons.append("tiny_detection_axial_extent")
    if float(np.mean(support)) < profile.estimation.minimum_whole_image_detection_area_fraction:
        reasons.append("tiny_detection_area_fraction")
    return tuple(reasons)


def _smooth_rows(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 1:
        return values.astype(np.float64, copy=True)
    usable = min(window, values.size if values.size % 2 else max(1, values.size - 1))
    if usable <= 1:
        return values.astype(np.float64, copy=True)
    radius = usable // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(usable) / usable, mode="valid")


def _dominant_states(material: np.ndarray, empty: np.ndarray, threshold: float) -> np.ndarray:
    states = np.zeros(material.shape, dtype=np.int8)
    states[(material >= threshold) & (material > empty)] = 1
    states[(empty >= threshold) & (empty > material)] = -1
    return states


def _compress_nonzero(states: np.ndarray) -> tuple[int, ...]:
    result: list[int] = []
    for value in states:
        state = int(value)
        if state and (not result or result[-1] != state):
            result.append(state)
    return tuple(result)


def _has_implausible_orientation(states: tuple[int, ...]) -> bool:
    return any(left == 1 and right == -1 for left, right in zip(states, states[1:]))


def _longest_run(states: np.ndarray, target: int) -> int:
    best = current = 0
    for value in states:
        if int(value) == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _minimum_cost_boundary(material: np.ndarray, empty: np.ndarray) -> float:
    prefix_material = np.concatenate(([0.0], np.cumsum(material, dtype=np.float64)))
    suffix_empty = np.concatenate((np.cumsum(empty[::-1], dtype=np.float64)[::-1], [0.0]))
    costs = prefix_material + suffix_empty
    minimum = float(np.min(costs))
    candidates = np.flatnonzero(np.isclose(costs, minimum, rtol=0.0, atol=1e-12))
    return float(np.median(candidates))


def _column_boundaries(
    material: np.ndarray,
    empty: np.ndarray,
    classified: np.ndarray,
) -> np.ndarray:
    height, width = material.shape
    positions: list[float] = []
    for column in range(width):
        if not np.any(classified[:, column]):
            continue
        boundary = _minimum_cost_boundary(
            material[:, column].astype(np.float64),
            empty[:, column].astype(np.float64),
        )
        positions.append(boundary / height)
    return np.asarray(positions, dtype=np.float64)


def _append_role_reasons(
    reasons: list[str],
    observation: SemanticObservation,
    profile: ModelProfile,
    pattern: str,
) -> None:
    if pattern in {"material_only", "material_and_empty"}:
        if (
            not observation.material_confidences
            or max(observation.material_confidences)
            < profile.aggregation.minimum_material_confidence
        ):
            reasons.append("low_or_missing_material_confidence")
    if pattern in {"empty_only", "material_and_empty"}:
        if (
            not observation.empty_confidences
            or max(observation.empty_confidences)
            < profile.aggregation.minimum_empty_confidence
        ):
            reasons.append("low_or_missing_empty_confidence")
