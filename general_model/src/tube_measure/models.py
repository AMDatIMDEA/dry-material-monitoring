from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]


@dataclass
class MaskInstance:
    """One model segmentation instance in image coordinates."""

    class_name: str
    mask: np.ndarray
    confidence: float = 1.0
    source_index: int = -1

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("MaskInstance.mask must be a 2-D array")
        self.mask = self.mask.astype(bool, copy=False)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def bbox(self) -> BBox:
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class PairCandidate:
    empty_index: int
    material_index: int
    score: float
    horizontal_overlap: float
    center_score: float
    interface_score: float
    order_score: float
    interface_distance_px: float


@dataclass
class TubeMeasurement:
    tube_id: str
    percentage: Optional[float]
    confidence: float
    valid: bool
    rejection_reason: Optional[str]
    bbox: BBox
    empty_area_px: float
    material_area_px: float
    pair_score: Optional[float]
    source: str
    empty_mask: Optional[np.ndarray] = field(default=None, repr=False)
    material_mask: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tube_id": self.tube_id,
            "percentage": None if self.percentage is None else round(self.percentage, 4),
            "confidence": round(self.confidence, 4),
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
            },
            "empty_area_px": round(self.empty_area_px, 2),
            "material_area_px": round(self.material_area_px, 2),
            "pair_score": None if self.pair_score is None else round(self.pair_score, 4),
            "source": self.source,
        }


@dataclass
class AnalysisResult:
    measurements: List[TubeMeasurement]
    ignored_instances: int = 0
    reference_height_px: Optional[float] = None
    reference_width_px: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_height_px": self.reference_height_px,
            "reference_width_px": self.reference_width_px,
            "ignored_instances": self.ignored_instances,
            "tubes": [measurement.to_dict() for measurement in self.measurements],
        }
