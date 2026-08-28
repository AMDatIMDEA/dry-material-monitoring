from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ClassConfig:
    empty: Tuple[str, ...] = ("Empty",)
    material: Tuple[str, ...] = ("Polymer", "Material")


@dataclass(frozen=True)
class MorphologyConfig:
    close_kernel: int = 3
    open_kernel: int = 0


@dataclass(frozen=True)
class PairingConfig:
    min_mask_area_px: int = 40
    min_horizontal_overlap: float = 0.55
    max_center_x_distance_ratio: float = 0.45
    max_interface_distance_px: float = 12.0
    max_interface_distance_ratio: float = 0.20
    max_wrong_order_px: float = 4.0
    min_pair_score: float = 0.50
    overlap_weight: float = 0.35
    center_weight: float = 0.20
    interface_weight: float = 0.35
    order_weight: float = 0.10


@dataclass(frozen=True)
class CompletenessConfig:
    expected_full_height_px: Optional[float] = None
    height_tolerance_ratio: float = 0.22
    width_tolerance_ratio: float = 0.35
    infer_from_valid_pairs: bool = True
    min_reference_pairs: int = 1


@dataclass(frozen=True)
class MeasurementConfig:
    max_overlap_fraction: float = 0.12
    min_combined_area_px: int = 100


@dataclass(frozen=True)
class InferenceConfig:
    confidence: float = 0.25
    image_size: int = 1280
    device: Optional[str] = "auto"
    image_extensions: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    )
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass(frozen=True)
class RenderingConfig:
    empty_color_bgr: Tuple[int, int, int] = (255, 160, 40)
    material_color_bgr: Tuple[int, int, int] = (40, 190, 70)
    invalid_color_bgr: Tuple[int, int, int] = (40, 40, 230)
    mask_alpha: float = 0.30


@dataclass(frozen=True)
class AppConfig:
    classes: ClassConfig = field(default_factory=ClassConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    completeness: CompletenessConfig = field(default_factory=CompletenessConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)

    @property
    def empty_names(self) -> frozenset[str]:
        return frozenset(name.casefold() for name in self.classes.empty)

    @property
    def material_names(self) -> frozenset[str]:
        return frozenset(name.casefold() for name in self.classes.material)


def _section(raw: Mapping[str, Any], name: str) -> Dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return dict(value)


def _tuple_fields(values: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
    for name in names:
        if name in values:
            values[name] = tuple(values[name])
    return values


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping")

    classes = ClassConfig(**_tuple_fields(_section(raw, "classes"), ["empty", "material"]))
    morphology = MorphologyConfig(**_section(raw, "morphology"))
    pairing = PairingConfig(**_section(raw, "pairing"))
    completeness = CompletenessConfig(**_section(raw, "completeness"))
    measurement = MeasurementConfig(**_section(raw, "measurement"))
    inference = InferenceConfig(
        **_tuple_fields(
            _section(raw, "inference"), ["image_extensions", "video_extensions"]
        )
    )
    rendering = RenderingConfig(
        **_tuple_fields(
            _section(raw, "rendering"),
            ["empty_color_bgr", "material_color_bgr", "invalid_color_bgr"],
        )
    )

    config = AppConfig(
        classes=classes,
        morphology=morphology,
        pairing=pairing,
        completeness=completeness,
        measurement=measurement,
        inference=inference,
        rendering=rendering,
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    device = "auto" if config.inference.device is None else config.inference.device.strip().lower()
    if device not in {"auto", "cpu", "cuda"} and not (
        device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError("inference.device must be auto, cpu, cuda, or cuda:<index>")
    for name, kernel in (
        ("close_kernel", config.morphology.close_kernel),
        ("open_kernel", config.morphology.open_kernel),
    ):
        if kernel < 0 or (kernel > 1 and kernel % 2 == 0):
            raise ValueError(f"morphology.{name} must be 0, 1, or a positive odd integer")
    if not config.classes.empty or not config.classes.material:
        raise ValueError("Both classes.empty and classes.material must contain names")
    if config.empty_names & config.material_names:
        raise ValueError("Empty and material class aliases must not overlap")
    bounded = {
        "pairing.min_horizontal_overlap": config.pairing.min_horizontal_overlap,
        "pairing.min_pair_score": config.pairing.min_pair_score,
        "completeness.height_tolerance_ratio": config.completeness.height_tolerance_ratio,
        "completeness.width_tolerance_ratio": config.completeness.width_tolerance_ratio,
        "measurement.max_overlap_fraction": config.measurement.max_overlap_fraction,
        "inference.confidence": config.inference.confidence,
        "rendering.mask_alpha": config.rendering.mask_alpha,
    }
    for name, value in bounded.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    weights = (
        config.pairing.overlap_weight
        + config.pairing.center_weight
        + config.pairing.interface_weight
        + config.pairing.order_weight
    )
    if weights <= 0:
        raise ValueError("Pairing weights must sum to a positive value")
