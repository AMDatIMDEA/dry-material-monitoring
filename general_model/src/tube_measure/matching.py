from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PairingConfig
from .geometry import score_pair
from .models import MaskInstance, PairCandidate


def match_instances(
    empty_instances: Sequence[MaskInstance],
    material_instances: Sequence[MaskInstance],
    config: PairingConfig,
) -> Tuple[List[PairCandidate], Dict[int, int], Dict[int, int]]:
    """Find a maximum-quality one-to-one assignment of compatible masks."""

    if not empty_instances or not material_instances:
        return [], {}, {}

    candidates: Dict[Tuple[int, int], PairCandidate] = {}
    scores = np.full((len(empty_instances), len(material_instances)), -1.0, dtype=float)
    empty_candidate_counts: Dict[int, int] = {}
    material_candidate_counts: Dict[int, int] = {}

    for empty_index, empty in enumerate(empty_instances):
        for material_index, material in enumerate(material_instances):
            candidate = score_pair(
                empty, material, empty_index, material_index, config
            )
            if candidate is None:
                continue
            candidates[(empty_index, material_index)] = candidate
            scores[empty_index, material_index] = candidate.score
            empty_candidate_counts[empty_index] = empty_candidate_counts.get(empty_index, 0) + 1
            material_candidate_counts[material_index] = (
                material_candidate_counts.get(material_index, 0) + 1
            )

    if not candidates:
        return [], empty_candidate_counts, material_candidate_counts

    # Invalid edges are deliberately much more expensive than leaving their
    # filtered assignment unused after solving the rectangular matrix.
    cost = np.where(scores >= 0.0, 1.0 - scores, 1_000_000.0)
    rows, columns = linear_sum_assignment(cost)
    matches = [
        candidates[(int(row), int(column))]
        for row, column in zip(rows, columns)
        if (int(row), int(column)) in candidates
    ]
    return matches, empty_candidate_counts, material_candidate_counts

