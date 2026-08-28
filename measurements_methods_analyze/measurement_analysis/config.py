"""Strict YAML configuration loading with paths relative to the YAML file."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    config_path: Path
    experiment_directory: Path
    output_directory: Path
    input_preference: str
    allow_full_mass_fraction_fallback: bool
    grouping_tolerance_ml: float
    dpi: int
    formats: tuple[str, ...]
    style: str
    figure_width_inches: float
    figure_height_inches: float


def load_config(path: Path) -> AnalysisConfig:
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")
    allowed = {
        "experiment_directory", "output_directory", "input_preference",
        "reference", "repeatability", "plots",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    experiment = _path(raw.get("experiment_directory"), config_path.parent)
    output = _path(raw.get("output_directory", "outputs"), config_path.parent)
    preference = str(raw.get("input_preference", "csv")).strip().casefold()
    if preference not in {"csv", "xlsx"}:
        raise ValueError("input_preference must be csv or xlsx.")
    reference = _mapping(raw.get("reference"), "reference")
    repeatability = _mapping(raw.get("repeatability"), "repeatability")
    plots = _mapping(raw.get("plots"), "plots")
    formats = tuple(str(item).strip().casefold() for item in plots.get("formats", ["png", "pdf"]))
    if not formats or any(item not in {"png", "pdf"} for item in formats):
        raise ValueError("plots.formats must contain only png and/or pdf.")
    tolerance = _positive(repeatability.get("grouping_tolerance_ml", 0.25), "grouping_tolerance_ml", allow_zero=True)
    dpi = int(plots.get("dpi", 600))
    if dpi < 72:
        raise ValueError("plots.dpi must be at least 72.")
    return AnalysisConfig(
        config_path=config_path,
        experiment_directory=experiment,
        output_directory=output,
        input_preference=preference,
        allow_full_mass_fraction_fallback=bool(reference.get("allow_full_mass_fraction_fallback", True)),
        grouping_tolerance_ml=tolerance,
        dpi=dpi,
        formats=formats,
        style=str(plots.get("style", "seaborn-v0_8-whitegrid")),
        figure_width_inches=_positive(plots.get("figure_width_inches", 7.2), "figure_width_inches"),
        figure_height_inches=_positive(plots.get("figure_height_inches", 5.2), "figure_height_inches"),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _path(value: Any, base: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("experiment_directory cannot be blank.")
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "greater than zero"
        raise ValueError(f"{label} must be finite and {qualifier}.")
    return result

