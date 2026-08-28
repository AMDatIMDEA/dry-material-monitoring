"""Typed domain models for offline YOLO configuration and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import WeightsError


@dataclass(frozen=True, slots=True)
class SemanticRole:
    """One configured scientific role resolved against model metadata at runtime."""

    class_name: str
    explicit_id: int | None = None


@dataclass(frozen=True, slots=True)
class NmsSettings:
    agnostic: bool
    class_filter_only_semantic: bool


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    confidence: float
    iou: float
    image_size: int
    device: str
    max_detections: int
    nms: NmsSettings


@dataclass(frozen=True, slots=True)
class TubeRoi:
    """Normalized image ROI plus an optional external calibration reference."""

    left: float
    top: float
    right: float
    bottom: float
    axis: str
    calibration_reference: Path | None
    frozen: bool = False
    usage_mode: str = "configured"


@dataclass(frozen=True, slots=True)
class AggregationThresholds:
    minimum_valid_images: int
    manual_minimum_valid_images: int
    maximum_spread_percentage_points: float
    maximum_overlap_fraction: float
    maximum_unclassified_fraction: float
    minimum_material_confidence: float
    minimum_empty_confidence: float


@dataclass(frozen=True, slots=True)
class EstimationThresholds:
    mask_threshold: float
    row_occupancy_threshold: float
    row_smoothing_window: int
    minimum_dominant_run_rows: int
    maximum_instances_per_role: int
    minimum_column_classified_fraction: float
    minimum_single_class_lateral_fraction: float
    minimum_single_class_axial_fraction: float
    single_class_end_occupancy_threshold: float
    minimum_whole_image_detection_lateral_fraction: float
    minimum_whole_image_detection_axial_fraction: float
    minimum_whole_image_detection_area_fraction: float


@dataclass(frozen=True, slots=True)
class CameraSettings:
    device_index: int
    backend: str
    width: int
    height: int
    fps: float
    warmup_frames: int
    capture_count: int
    capture_interval_seconds: float
    capture_duration_seconds: float | None
    capture_key: str
    quit_key: str
    window_name: str
    preview_wait_ms: int
    mirror_preview: bool
    close_on_capture: bool
    freeze_after_capture_ms: int
    device_name_contains: str | None = None


@dataclass(frozen=True, slots=True)
class OutputSettings:
    root: Path
    storage_profile: str
    image_format: str
    save_overlays: bool
    save_masks: bool
    overwrite: bool
    log_level: str


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Fully resolved settings for one material/model combination."""

    name: str
    recorded_material_names: tuple[str, ...]
    weights_path: Path
    material_role: SemanticRole
    empty_role: SemanticRole
    expected_task: str
    estimation_mode: str
    inference: InferenceSettings
    tube_roi: TubeRoi
    estimation: EstimationThresholds
    aggregation: AggregationThresholds
    camera: CameraSettings
    output: OutputSettings

    def require_weights(self) -> Path:
        """Return a usable weights file or fail before model construction."""
        if not self.weights_path.exists():
            raise WeightsError(
                f"Weights for profile {self.name!r} were not found: {self.weights_path}"
            )
        if not self.weights_path.is_file():
            raise WeightsError(
                f"Weights for profile {self.name!r} are not a file: {self.weights_path}"
            )
        return self.weights_path

    def require_frozen_roi(self) -> TubeRoi:
        """Require either a frozen calibration or an explicit whole-image choice."""
        from .errors import ConfigurationError

        if self.tube_roi.usage_mode == "whole_image":
            if (
                self.tube_roi.left,
                self.tube_roi.top,
                self.tube_roi.right,
                self.tube_roi.bottom,
            ) != (0.0, 0.0, 1.0, 1.0):
                raise ConfigurationError(
                    "Whole-image ROI mode must use normalized bounds 0,0,1,1."
                )
            return self.tube_roi
        if self.tube_roi.usage_mode != "configured":
            raise ConfigurationError(
                f"Unsupported tube ROI usage mode: {self.tube_roi.usage_mode!r}"
            )
        if not self.tube_roi.frozen:
            raise ConfigurationError(
                f"Profile {self.name!r} has no frozen tube ROI. Run "
                "Material_level_using_yolo/configure_roi.py with a representative image, "
                "or deliberately select --roi-mode whole_image."
            )
        reference = self.tube_roi.calibration_reference
        if reference is None or not reference.is_file():
            raise ConfigurationError(
                f"Frozen tube ROI calibration reference is missing: {reference}"
            )
        return self.tube_roi


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    schema_version: int
    config_path: Path
    default_profile: str
    profiles: Mapping[str, ModelProfile]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))

    def select_profile(self, name: str | None = None) -> ModelProfile:
        from .errors import ProfileNotFoundError

        selected = self.default_profile if name is None else name
        try:
            return self.profiles[selected]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles))
            raise ProfileNotFoundError(
                f"Unknown model profile {selected!r}. Available profiles: {available}"
            ) from exc

    def select_profile_for_material(
        self,
        material_name: str | None,
        requested_profile: str | None = None,
    ) -> ModelProfile:
        """Resolve a configured material alias or an explicit recovery profile."""
        from .errors import ConfigurationError

        if not isinstance(material_name, str) or not material_name.strip():
            if requested_profile is not None:
                return self.select_profile(requested_profile)
            raise ConfigurationError(
                "Synchronized offline inference requires a recorded material_name or "
                "an explicit --profile selection."
            )
        normalized = material_name.strip().casefold()
        matches = [
            profile
            for profile in self.profiles.values()
            if normalized in {alias.casefold() for alias in profile.recorded_material_names}
        ]
        if requested_profile is not None:
            selected = self.select_profile(requested_profile)
            if selected not in matches:
                raise ConfigurationError(
                    f"Recorded material {material_name!r} is not configured for profile "
                    f"{selected.name!r}; configured aliases: "
                    + ", ".join(selected.recorded_material_names)
                )
            return selected
        if not matches:
            aliases = ", ".join(
                f"{profile.name}=[{', '.join(profile.recorded_material_names)}]"
                for profile in self.profiles.values()
            )
            raise ConfigurationError(
                f"No YOLO profile is configured for recorded material {material_name!r}. "
                f"Configured aliases: {aliases}"
            )
        if len(matches) != 1:
            raise ConfigurationError(
                f"Recorded material {material_name!r} ambiguously matches profiles: "
                + ", ".join(sorted(profile.name for profile in matches))
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class SemanticClassMap:
    material_id: int
    material_name: str
    empty_id: int
    empty_name: str
    model_names: Mapping[int, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_names", MappingProxyType(dict(self.model_names)))


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    """Raw adapter result plus the semantic mapping frozen before inference."""

    raw_results: tuple[Any, ...]
    class_map: SemanticClassMap
    model_task: str
    weights_sha256: str
    inference_device: str = "unreported"
