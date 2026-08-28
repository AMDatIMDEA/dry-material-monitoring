from __future__ import annotations

import cv2
import numpy as np

from .config import RenderingConfig
from .models import AnalysisResult


def annotate_image(
    image: np.ndarray, result: AnalysisResult, config: RenderingConfig
) -> np.ndarray:
    """Draw mask overlays, bounding boxes, percentages, and rejection labels."""

    annotated = image.copy()
    overlay = image.copy()
    for measurement in result.measurements:
        if measurement.valid:
            if measurement.empty_mask is not None:
                overlay[measurement.empty_mask] = config.empty_color_bgr
            if measurement.material_mask is not None:
                overlay[measurement.material_mask] = config.material_color_bgr
        else:
            combined = None
            if measurement.empty_mask is not None:
                combined = measurement.empty_mask.copy()
            if measurement.material_mask is not None:
                combined = (
                    measurement.material_mask.copy()
                    if combined is None
                    else combined | measurement.material_mask
                )
            if combined is not None:
                overlay[combined] = config.invalid_color_bgr

    cv2.addWeighted(
        overlay,
        config.mask_alpha,
        annotated,
        1.0 - config.mask_alpha,
        0.0,
        dst=annotated,
    )
    for measurement in result.measurements:
        x1, y1, x2, y2 = measurement.bbox
        color = (
            config.material_color_bgr if measurement.valid else config.invalid_color_bgr
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if measurement.valid:
            label = (
                f"{measurement.tube_id} {measurement.percentage:.1f}% "
                f"c={measurement.confidence:.2f}"
            )
        else:
            label = f"{measurement.tube_id} INVALID: {measurement.rejection_reason}"
        _draw_label(annotated, label, x1, y1, color)
    return annotated


def _draw_label(
    image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - baseline - 6)
    right = min(image.shape[1] - 1, x + width + 6)
    cv2.rectangle(image, (x, top), (right, y), color, -1)
    cv2.putText(
        image,
        text,
        (x + 3, max(height + 2, y - baseline - 3)),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

