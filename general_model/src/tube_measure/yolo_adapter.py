from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import cv2
import numpy as np

from .models import MaskInstance


def instances_from_result(result: Any) -> List[MaskInstance]:
    """Convert one Ultralytics Results object into image-sized masks."""

    if result.masks is None or result.boxes is None:
        return []
    masks = _to_numpy(result.masks.data)
    classes = _to_numpy(result.boxes.cls).astype(int)
    confidences = _to_numpy(result.boxes.conf)
    if len(masks) != len(classes):
        raise ValueError("Ultralytics result has inconsistent mask and box counts")

    target_height, target_width = result.orig_shape
    instances: List[MaskInstance] = []
    for index, (mask, class_id, confidence) in enumerate(
        zip(masks, classes, confidences)
    ):
        if mask.shape != (target_height, target_width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
        instances.append(
            MaskInstance(
                class_name=_class_name(result.names, int(class_id)),
                mask=mask > 0.5,
                confidence=float(confidence),
                source_index=index,
            )
        )
    return instances


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _class_name(names: Mapping[int, str] | Sequence[str], class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names[class_id])
    return str(names[class_id])

