"""Convert one Ultralytics result into method-neutral semantic support masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .domain import InferenceOutput, ModelProfile
from .estimation import SemanticObservation


def observation_from_inference(
    output: InferenceOutput,
    profile: ModelProfile,
    image_shape: tuple[int, ...],
    *,
    source_path: Path | None = None,
    source_sha256: str | None = None,
) -> SemanticObservation:
    """Extract the highest-confidence mask/box for each semantic role.

    All role confidences and instance counts remain in diagnostics so duplicate
    detections are visible even though only one instance drives geometry.
    """
    if len(image_shape) < 2 or image_shape[0] < 1 or image_shape[1] < 1:
        raise ValueError("image_shape must contain positive height and width.")
    height, width = int(image_shape[0]), int(image_shape[1])
    empty_mask = np.zeros((height, width), dtype=np.float32)
    material_mask = np.zeros((height, width), dtype=np.float32)
    reasons: list[str] = []
    if len(output.raw_results) != 1:
        reasons.append(
            "missing_inference_result"
            if not output.raw_results
            else "ambiguous_multiple_inference_results"
        )
        return SemanticObservation(
            material_mask=material_mask,
            empty_mask=empty_mask,
            material_confidences=(),
            empty_confidences=(),
            material_instance_count=0,
            empty_instance_count=0,
            material_present=False,
            empty_present=False,
            estimation_mode=profile.estimation_mode,
            source_path=source_path,
            source_sha256=source_sha256,
            extraction_reasons=tuple(reasons),
        )

    result = output.raw_results[0]
    boxes = getattr(result, "boxes", None)
    classes = _array(getattr(boxes, "cls", None)).reshape(-1)
    confidences = _array(getattr(boxes, "conf", None)).reshape(-1)
    if classes.size != confidences.size:
        reasons.append("inconsistent_class_confidence_metadata")
    count = min(classes.size, confidences.size)
    classes = classes[:count].astype(np.int64, copy=False)
    confidences = confidences[:count].astype(np.float64, copy=False)
    material_indices = np.flatnonzero(classes == output.class_map.material_id)
    empty_indices = np.flatnonzero(classes == output.class_map.empty_id)
    selected_material = _highest_confidence_index(material_indices, confidences)
    selected_empty = _highest_confidence_index(empty_indices, confidences)

    if profile.estimation_mode == "segmentation_interface":
        masks_object = getattr(result, "masks", None)
        mask_data = _array(getattr(masks_object, "data", None))
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, ...]
        if mask_data.ndim != 3:
            mask_data = np.empty((0, height, width), dtype=np.float32)
            reasons.append("missing_segmentation_masks")
        if mask_data.shape[0] != classes.size:
            reasons.append("inconsistent_mask_class_metadata")
        material_mask = _selected_resized_mask(mask_data, selected_material, (height, width))
        empty_mask = _selected_resized_mask(mask_data, selected_empty, (height, width))
        material_present = bool(material_indices.size)
        empty_present = bool(empty_indices.size)
        if material_present and not np.any(material_mask):
            reasons.append("missing_material_mask")
        if empty_present and not np.any(empty_mask):
            reasons.append("missing_empty_mask")
    elif profile.estimation_mode == "bbox_vertical_extent":
        xyxy = _array(getattr(boxes, "xyxy", None))
        if xyxy.ndim != 2 or xyxy.shape[1] < 4 or xyxy.shape[0] != classes.size:
            xyxy = np.empty((0, 4), dtype=np.float64)
            reasons.append("inconsistent_bbox_class_metadata")
        material_mask, material_box_reasons = _boxes_to_support(
            xyxy,
            selected_material,
            (height, width),
            role="material",
        )
        empty_mask, empty_box_reasons = _boxes_to_support(
            xyxy,
            selected_empty,
            (height, width),
            role="empty",
        )
        reasons.extend(material_box_reasons)
        reasons.extend(empty_box_reasons)
        material_present = bool(material_indices.size)
        empty_present = bool(empty_indices.size)
    else:
        raise ValueError(f"Unsupported estimation mode: {profile.estimation_mode}")

    return SemanticObservation(
        material_mask=material_mask,
        empty_mask=empty_mask,
        material_confidences=tuple(float(confidences[index]) for index in material_indices if index < count),
        empty_confidences=tuple(float(confidences[index]) for index in empty_indices if index < count),
        material_instance_count=int(material_indices.size),
        empty_instance_count=int(empty_indices.size),
        material_present=material_present,
        empty_present=empty_present,
        estimation_mode=profile.estimation_mode,
        source_path=source_path,
        source_sha256=source_sha256,
        extraction_reasons=tuple(dict.fromkeys(reasons)),
    )


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return np.asarray(candidate)


def _highest_confidence_index(
    indices: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    if not indices.size:
        return np.asarray([], dtype=np.int64)
    usable = [int(index) for index in indices if int(index) < confidences.size]
    if not usable:
        return np.asarray([], dtype=np.int64)
    selected = max(usable, key=lambda index: (float(confidences[index]), -index))
    return np.asarray([selected], dtype=np.int64)


def _selected_resized_mask(
    masks: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    selected = np.zeros((height, width), dtype=np.float32)
    for index in indices:
        if int(index) >= masks.shape[0]:
            continue
        mask = np.asarray(masks[int(index)], dtype=np.float32)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        selected = mask
        break
    return selected


def _boxes_to_support(
    boxes: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, int],
    *,
    role: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    height, width = shape
    support = np.zeros((height, width), dtype=np.float32)
    reasons: list[str] = []
    for index in indices:
        if int(index) >= boxes.shape[0]:
            continue
        x1, y1, x2, y2 = (float(value) for value in boxes[int(index), :4])
        if not np.all(np.isfinite((x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
            reasons.append(f"invalid_{role}_bbox")
            continue
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(width, int(np.ceil(x2)))
        bottom = min(height, int(np.ceil(y2)))
        if left >= right or top >= bottom:
            reasons.append(f"{role}_bbox_outside_image")
            continue
        support[top:bottom, left:right] = 1.0
    return support, tuple(dict.fromkeys(reasons))
