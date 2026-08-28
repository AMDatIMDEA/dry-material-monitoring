from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MorphologyConfig, PairingConfig
from .models import BBox, MaskInstance, PairCandidate


def clean_instance(instance: MaskInstance, config: MorphologyConfig) -> MaskInstance:
    mask = instance.mask.astype(np.uint8)
    if config.open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.open_kernel, config.open_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if config.close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return MaskInstance(
        class_name=instance.class_name,
        mask=mask.astype(bool),
        confidence=instance.confidence,
        source_index=instance.source_index,
    )


def union_bbox(*masks: np.ndarray) -> BBox:
    union = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_size(bbox: BBox) -> Tuple[int, int]:
    return (max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1]))


def _interface_distance(empty: np.ndarray, material: np.ndarray) -> Optional[float]:
    columns = np.flatnonzero(empty.any(axis=0) & material.any(axis=0))
    if len(columns) == 0:
        return None
    distances = []
    for x in columns:
        empty_rows = np.flatnonzero(empty[:, x])
        material_rows = np.flatnonzero(material[:, x])
        if len(empty_rows) and len(material_rows):
            distances.append(float(material_rows.min() - empty_rows.max() - 1))
    return float(np.median(distances)) if distances else None


def score_pair(
    empty: MaskInstance,
    material: MaskInstance,
    empty_index: int,
    material_index: int,
    config: PairingConfig,
) -> Optional[PairCandidate]:
    ex1, _, ex2, _ = empty.bbox
    mx1, _, mx2, _ = material.bbox
    ew = max(1, ex2 - ex1)
    mw = max(1, mx2 - mx1)
    overlap_width = max(0, min(ex2, mx2) - max(ex1, mx1))
    horizontal_overlap = overlap_width / min(ew, mw)
    if horizontal_overlap < config.min_horizontal_overlap:
        return None

    center_distance = abs(empty.center[0] - material.center[0])
    center_ratio = center_distance / max(1.0, (ew + mw) / 2.0)
    if center_ratio > config.max_center_x_distance_ratio:
        return None
    center_score = 1.0 - center_ratio / max(config.max_center_x_distance_ratio, 1e-9)

    signed_interface_distance = _interface_distance(empty.mask, material.mask)
    if signed_interface_distance is None:
        return None
    _, eh = bbox_size(empty.bbox)
    _, mh = bbox_size(material.bbox)
    allowed_interface = max(
        config.max_interface_distance_px,
        config.max_interface_distance_ratio * max(eh, mh),
    )
    interface_distance = abs(signed_interface_distance)
    if interface_distance > allowed_interface:
        return None
    interface_score = 1.0 - interface_distance / max(allowed_interface, 1e-9)

    vertical_delta = material.center[1] - empty.center[1]
    if vertical_delta < -config.max_wrong_order_px:
        return None
    order_score = float(np.clip((vertical_delta + config.max_wrong_order_px) / max(
        (eh + mh) / 2.0 + config.max_wrong_order_px, 1.0
    ), 0.0, 1.0))

    weights = np.array(
        [
            config.overlap_weight,
            config.center_weight,
            config.interface_weight,
            config.order_weight,
        ],
        dtype=float,
    )
    terms = np.array(
        [horizontal_overlap, center_score, interface_score, order_score], dtype=float
    )
    score = float(np.dot(weights, terms) / weights.sum())
    if score < config.min_pair_score:
        return None
    return PairCandidate(
        empty_index=empty_index,
        material_index=material_index,
        score=score,
        horizontal_overlap=horizontal_overlap,
        center_score=center_score,
        interface_score=interface_score,
        order_score=order_score,
        interface_distance_px=signed_interface_distance,
    )


def effective_areas(empty: np.ndarray, material: np.ndarray) -> Tuple[float, float, float]:
    """Return overlap-safe empty area, material area, and overlap fraction.

    A minor overlap has no class-independent perfect answer. Splitting its pixels
    equally prevents double counting and avoids systematically favoring a class.
    """

    overlap = empty & material
    overlap_area = float(np.count_nonzero(overlap))
    empty_only = float(np.count_nonzero(empty & ~material))
    material_only = float(np.count_nonzero(material & ~empty))
    empty_area = empty_only + overlap_area / 2.0
    material_area = material_only + overlap_area / 2.0
    smaller = max(1.0, min(float(np.count_nonzero(empty)), float(np.count_nonzero(material))))
    return empty_area, material_area, overlap_area / smaller
