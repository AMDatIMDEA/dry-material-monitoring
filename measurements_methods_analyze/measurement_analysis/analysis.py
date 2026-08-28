"""Traceable matching, reference construction, and descriptive statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import AnalysisConfig
from .io import discover_measurement_files, read_measurement_file


METHODS = ("Human", "YOLO", "D405")


@dataclass(frozen=True, slots=True)
class Comparison:
    experiment_id: str
    measurement_id: str
    measurement_index: int
    method: str
    reference_volume_ml: float
    estimate_volume_ml: float
    error_ml: float
    absolute_error_ml: float
    reference_source: str
    source_status: str


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    experiment_id: str
    measurement_id: str
    measurement_index: int
    reference_volume_ml: float
    reference_source: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    references: tuple[ReferencePoint, ...]
    comparisons: tuple[Comparison, ...]
    summary: tuple[dict[str, Any], ...]
    input_files: dict[str, Path]
    excluded: tuple[dict[str, str], ...]
    reference_source_counts: dict[str, int]


def analyze_experiment(config: AnalysisConfig) -> AnalysisResult:
    files = discover_measurement_files(config.experiment_directory, config.input_preference)
    rows_by_method = {name: read_measurement_file(path) for name, path in files.items()}
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for expected_method, rows in rows_by_method.items():
        for row in rows:
            measurement_id = _text(row.get("measurement_id"))
            actual_method = _text(row.get("method")).casefold()
            if not measurement_id or actual_method != expected_method:
                continue
            if actual_method in grouped.setdefault(measurement_id, {}):
                raise ValueError(f"Duplicate {actual_method} row for measurement_id {measurement_id}")
            grouped[measurement_id][actual_method] = row

    comparisons: list[Comparison] = []
    references: list[ReferencePoint] = []
    excluded: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    for measurement_id, method_rows in sorted(grouped.items(), key=_measurement_sort_key):
        shared_rows = tuple(method_rows.values())
        if not shared_rows:
            continue
        if not _shared_fields_consistent(shared_rows):
            excluded.append({"measurement_id": measurement_id, "method": "all", "reason": "conflicting_shared_fields"})
            continue
        base = shared_rows[0]
        reference, reference_source = _gravimetric_reference(base, config)
        if reference is None:
            for method in METHODS:
                excluded.append({"measurement_id": measurement_id, "method": method, "reason": "gravimetric_reference_unavailable"})
            continue
        source_counts[reference_source] = source_counts.get(reference_source, 0) + 1
        experiment_id = _text(base.get("experiment_id"))
        index = int(float(base.get("measurement_index")))
        references.append(
            ReferencePoint(
                experiment_id=experiment_id,
                measurement_id=measurement_id,
                measurement_index=index,
                reference_volume_ml=reference,
                reference_source=reference_source,
            )
        )

        manual = _manual_volume(base)
        if manual is not None:
            comparisons.append(_comparison(experiment_id, measurement_id, index, "Human", reference, manual, reference_source, "operator_supplied"))
        else:
            excluded.append({"measurement_id": measurement_id, "method": "Human", "reason": "manual_estimate_unavailable"})

        for source_name, label in (("yolo", "YOLO"), ("depth", "D405")):
            row = method_rows.get(source_name)
            if row is None:
                excluded.append({"measurement_id": measurement_id, "method": label, "reason": "method_row_unavailable"})
                continue
            if not _is_complete_valid(row):
                excluded.append({"measurement_id": measurement_id, "method": label, "reason": f"not_complete_valid:{_text(row.get('status')) or 'blank'}"})
                continue
            estimate = _finite(row.get("estimated_material_volume_ml"))
            if estimate is None:
                excluded.append({"measurement_id": measurement_id, "method": label, "reason": "valid_row_missing_estimate"})
                continue
            comparisons.append(_comparison(experiment_id, measurement_id, index, label, reference, estimate, reference_source, _text(row.get("status"))))

    return AnalysisResult(
        references=tuple(references),
        comparisons=tuple(comparisons),
        summary=tuple(_summary(comparisons, method) for method in METHODS),
        input_files=files,
        excluded=tuple(excluded),
        reference_source_counts=source_counts,
    )


def _gravimetric_reference(row: dict[str, Any], config: AnalysisConfig) -> tuple[float | None, str]:
    explicit = _finite(row.get("reference_volume_from_weight_ml"))
    if explicit is not None:
        return explicit, "reference_volume_from_weight_ml"
    if not config.allow_full_mass_fraction_fallback:
        return None, "unavailable"
    remaining = _finite(row.get("reference_material_weight_g"))
    full_mass = _finite(row.get("total_possible_weight_g"))
    capacity = _finite(row.get("total_capacity_ml"))
    if remaining is None or full_mass is None or capacity is None or full_mass <= 0 or capacity <= 0:
        return None, "unavailable"
    if remaining < 0 or remaining > full_mass:
        return None, "unavailable"
    return capacity * remaining / full_mass, "full_mass_fraction_of_capacity"


def _manual_volume(row: dict[str, Any]) -> float | None:
    explicit = _finite(row.get("reference_volume_from_manual_ml"))
    if explicit is not None:
        return explicit
    percent = _finite(row.get("manual_material_percent"))
    capacity = _finite(row.get("total_capacity_ml"))
    if percent is None or capacity is None or not 0 <= percent <= 100 or capacity <= 0:
        return None
    return capacity * percent / 100.0


def _is_complete_valid(row: dict[str, Any]) -> bool:
    valid = row.get("valid")
    if isinstance(valid, bool):
        is_valid = valid
    else:
        is_valid = _text(valid).casefold() == "true"
    return is_valid and _text(row.get("status")).casefold() == "complete_valid"


def _shared_fields_consistent(rows: tuple[dict[str, Any], ...]) -> bool:
    fields = ("experiment_id", "measurement_id", "measurement_index", "total_capacity_ml", "bulk_density_g_per_ml", "total_possible_weight_g", "reference_material_weight_g", "reference_volume_from_weight_ml", "manual_material_percent", "reference_volume_from_manual_ml")
    return all(
        all(_shared_value_equal(rows[0].get(field), row.get(field)) for field in fields)
        for row in rows[1:]
    )


def _shared_value_equal(left: Any, right: Any) -> bool:
    left_number, right_number = _finite(left), _finite(right)
    if left_number is not None or right_number is not None:
        return (
            left_number is not None
            and right_number is not None
            and math.isclose(left_number, right_number, rel_tol=1e-12, abs_tol=1e-9)
        )
    return _text(left) == _text(right)


def _comparison(experiment_id: str, measurement_id: str, index: int, method: str, reference: float, estimate: float, reference_source: str, status: str) -> Comparison:
    error = estimate - reference
    return Comparison(experiment_id, measurement_id, index, method, reference, estimate, error, abs(error), reference_source, status)


def _summary(comparisons: list[Comparison], method: str) -> dict[str, Any]:
    items = [item for item in comparisons if item.method == method]
    if not items:
        return {"method": method, "mae_ml": None, "rmse_ml": None, "mean_bias_ml": None, "error_sd_ml": None, "r_squared": None, "valid_measurement_count": 0}
    references = np.asarray([item.reference_volume_ml for item in items], dtype=float)
    estimates = np.asarray([item.estimate_volume_ml for item in items], dtype=float)
    errors = estimates - references
    sst = float(np.sum((references - np.mean(references)) ** 2))
    r_squared = None if len(items) < 2 or sst == 0 else 1.0 - float(np.sum(errors ** 2)) / sst
    return {
        "method": method,
        "mae_ml": float(np.mean(np.abs(errors))),
        "rmse_ml": float(np.sqrt(np.mean(errors ** 2))),
        "mean_bias_ml": float(np.mean(errors)),
        "error_sd_ml": None if len(items) < 2 else float(np.std(errors, ddof=1)),
        "r_squared": r_squared,
        "valid_measurement_count": len(items),
    }


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _measurement_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    measurement_id = item[0]
    prefix = measurement_id.split("_", 1)[0]
    return (int(prefix) if prefix.isdigit() else 2**31, measurement_id)
