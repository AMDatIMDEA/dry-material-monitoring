from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import AppConfig
from .geometry import bbox_size, clean_instance, effective_areas, union_bbox
from .matching import match_instances
from .models import AnalysisResult, BBox, MaskInstance, PairCandidate, TubeMeasurement


class TubeAnalyzer:
    """Convert class masks into validated per-tube measurements."""

    def __init__(self, config: AppConfig):
        self.config = config

    def analyze(self, instances: Sequence[MaskInstance]) -> AnalysisResult:
        self._validate_shapes(instances)
        empty: List[MaskInstance] = []
        material: List[MaskInstance] = []
        too_small: List[Tuple[MaskInstance, str]] = []
        ignored = 0

        for raw_instance in instances:
            class_name = raw_instance.class_name.casefold()
            if class_name in self.config.empty_names:
                kind = "empty"
            elif class_name in self.config.material_names:
                kind = "material"
            else:
                ignored += 1
                continue
            instance = clean_instance(raw_instance, self.config.morphology)
            if instance.area < self.config.pairing.min_mask_area_px:
                too_small.append((instance, kind))
            elif kind == "empty":
                empty.append(instance)
            else:
                material.append(instance)

        matches, empty_candidate_counts, material_candidate_counts = match_instances(
            empty, material, self.config.pairing
        )
        matched_empty = {candidate.empty_index for candidate in matches}
        matched_material = {candidate.material_index for candidate in matches}

        pair_geometry = [
            self._geometry(empty[candidate.empty_index], material[candidate.material_index])
            for candidate in matches
        ]
        reference_height, reference_width = self._reference_geometry(pair_geometry)

        measurements: List[TubeMeasurement] = []
        for candidate in matches:
            measurements.append(
                self._measure_pair(
                    empty[candidate.empty_index],
                    material[candidate.material_index],
                    candidate,
                    reference_height,
                )
            )

        for index, instance in enumerate(empty):
            if index not in matched_empty:
                reason = (
                    "one_to_one_pairing_conflict"
                    if empty_candidate_counts.get(index, 0) > 0
                    else None
                )
                measurements.append(
                    self._measure_single(
                        instance, "empty", reference_height, reference_width, reason
                    )
                )
        for index, instance in enumerate(material):
            if index not in matched_material:
                reason = (
                    "one_to_one_pairing_conflict"
                    if material_candidate_counts.get(index, 0) > 0
                    else None
                )
                measurements.append(
                    self._measure_single(
                        instance, "material", reference_height, reference_width, reason
                    )
                )
        for instance, kind in too_small:
            measurements.append(self._small_mask_measurement(instance, kind))

        measurements.sort(key=lambda value: (value.bbox[0], value.bbox[1]))
        for number, measurement in enumerate(measurements, start=1):
            measurement.tube_id = f"tube_{number:03d}"
        return AnalysisResult(
            measurements=measurements,
            ignored_instances=ignored,
            reference_height_px=reference_height,
            reference_width_px=reference_width,
        )

    @staticmethod
    def _validate_shapes(instances: Sequence[MaskInstance]) -> None:
        shapes = {instance.mask.shape for instance in instances}
        if len(shapes) > 1:
            raise ValueError("All masks in one analysis must have the same image shape")

    @staticmethod
    def _geometry(empty: MaskInstance, material: MaskInstance) -> Tuple[float, float]:
        width, height = bbox_size(union_bbox(empty.mask, material.mask))
        return float(height), float(width)

    def _reference_geometry(
        self, pair_geometry: Sequence[Tuple[float, float]]
    ) -> Tuple[Optional[float], Optional[float]]:
        expected = self.config.completeness.expected_full_height_px
        reference_height = float(expected) if expected is not None else None
        reference_width: Optional[float] = None
        enough_pairs = (
            len(pair_geometry) >= self.config.completeness.min_reference_pairs
            and self.config.completeness.infer_from_valid_pairs
        )
        if enough_pairs:
            heights = [height for height, _ in pair_geometry]
            widths = [width for _, width in pair_geometry]
            if reference_height is None:
                reference_height = float(np.median(heights))
            reference_width = float(np.median(widths))
        return reference_height, reference_width

    def _measure_pair(
        self,
        empty: MaskInstance,
        material: MaskInstance,
        candidate: PairCandidate,
        reference_height: Optional[float],
    ) -> TubeMeasurement:
        bbox = union_bbox(empty.mask, material.mask)
        _, height = bbox_size(bbox)
        empty_area, material_area, overlap_fraction = effective_areas(
            empty.mask, material.mask
        )
        total_area = empty_area + material_area
        reason: Optional[str] = None
        completeness_score = 1.0
        if total_area < self.config.measurement.min_combined_area_px:
            reason = "combined_area_too_small"
        elif overlap_fraction > self.config.measurement.max_overlap_fraction:
            reason = "excessive_class_overlap"
        elif reference_height is not None:
            relative_error = abs(height - reference_height) / max(reference_height, 1.0)
            completeness_score = max(
                0.0,
                1.0
                - relative_error
                / max(self.config.completeness.height_tolerance_ratio, 1e-9),
            )
            if relative_error > self.config.completeness.height_tolerance_ratio:
                reason = "incomplete_combined_height"

        detector_confidence = float(np.sqrt(empty.confidence * material.confidence))
        confidence = float(
            np.clip(
                0.55 * candidate.score
                + 0.25 * detector_confidence
                + 0.20 * completeness_score,
                0.0,
                1.0,
            )
        )
        valid = reason is None and total_area > 0
        percentage = 100.0 * material_area / total_area if valid else None
        return TubeMeasurement(
            tube_id="",
            percentage=percentage,
            confidence=confidence,
            valid=valid,
            rejection_reason=reason,
            bbox=bbox,
            empty_area_px=empty_area,
            material_area_px=material_area,
            pair_score=candidate.score,
            source="paired",
            empty_mask=empty.mask,
            material_mask=material.mask,
        )

    def _measure_single(
        self,
        instance: MaskInstance,
        kind: str,
        reference_height: Optional[float],
        reference_width: Optional[float],
        pairing_reason: Optional[str],
    ) -> TubeMeasurement:
        bbox = instance.bbox
        width, height = bbox_size(bbox)
        reason = pairing_reason
        geometry_scores: List[float] = []
        if reason is None and reference_height is None:
            reason = "no_full_tube_height_reference"
        if reason is None and reference_height is not None:
            height_error = abs(height - reference_height) / max(reference_height, 1.0)
            geometry_scores.append(
                max(0.0, 1.0 - height_error / max(
                    self.config.completeness.height_tolerance_ratio, 1e-9
                ))
            )
            if height_error > self.config.completeness.height_tolerance_ratio:
                reason = "single_class_mask_is_incomplete"
        if reason is None and reference_width is not None:
            width_error = abs(width - reference_width) / max(reference_width, 1.0)
            geometry_scores.append(
                max(0.0, 1.0 - width_error / max(
                    self.config.completeness.width_tolerance_ratio, 1e-9
                ))
            )
            if width_error > self.config.completeness.width_tolerance_ratio:
                reason = "single_class_width_is_inconsistent"

        geometry_confidence = float(np.mean(geometry_scores)) if geometry_scores else 0.0
        confidence = float(np.clip(0.45 * instance.confidence + 0.45 * geometry_confidence, 0, 1))
        valid = reason is None and instance.area >= self.config.measurement.min_combined_area_px
        if reason is None and not valid:
            reason = "combined_area_too_small"
        percentage = (0.0 if kind == "empty" else 100.0) if valid else None
        return TubeMeasurement(
            tube_id="",
            percentage=percentage,
            confidence=confidence,
            valid=valid,
            rejection_reason=reason,
            bbox=bbox,
            empty_area_px=float(instance.area if kind == "empty" else 0),
            material_area_px=float(instance.area if kind == "material" else 0),
            pair_score=None,
            source=f"single_{kind}",
            empty_mask=instance.mask if kind == "empty" else None,
            material_mask=instance.mask if kind == "material" else None,
        )

    @staticmethod
    def _small_mask_measurement(instance: MaskInstance, kind: str) -> TubeMeasurement:
        return TubeMeasurement(
            tube_id="",
            percentage=None,
            confidence=instance.confidence,
            valid=False,
            rejection_reason="mask_area_below_pairing_minimum",
            bbox=instance.bbox,
            empty_area_px=float(instance.area if kind == "empty" else 0),
            material_area_px=float(instance.area if kind == "material" else 0),
            pair_score=None,
            source=f"single_{kind}",
            empty_mask=instance.mask if kind == "empty" else None,
            material_mask=instance.mask if kind == "material" else None,
        )
