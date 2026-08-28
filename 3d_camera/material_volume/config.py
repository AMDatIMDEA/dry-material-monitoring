"""Typed YAML configuration and physical setup validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, tan, radians
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError


@dataclass(slots=True)
class TubeConfig:
    outer_diameter_mm: float = 58.0
    wall_thickness_mm: float = 3.0
    inner_diameter_mm: float = 55.0
    usable_height_mm: float = 115.0
    wall_edge_exclusion_mm: float = 2.0
    geometry_warning_tolerance_mm: float = 0.5

    @property
    def inner_radius_mm(self) -> float:
        return self.inner_diameter_mm / 2.0

    @property
    def capacity_ml(self) -> float:
        return pi * self.inner_radius_mm**2 * self.usable_height_mm / 1000.0


@dataclass(slots=True)
class CameraConfig:
    serial_number: str | None = None
    distance_to_rim_mm: float = 65.0
    depth_width: int = 640
    depth_height: int = 360
    fps: int = 30
    fallback_profiles: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(640, 360, 30), (480, 270, 30)]
    )
    enable_color: bool = True
    color_width: int = 640
    color_height: int = 480
    color_fps: int = 30
    enable_infrared_diagnostics: bool = False
    warmup_frames: int = 30
    frame_timeout_ms: int = 5000
    auto_exposure: bool = True
    manual_exposure: float | None = None
    manual_gain: float | None = None
    visual_preset: float | None = None

    @property
    def requested_profile(self) -> tuple[int, int, int]:
        return self.depth_width, self.depth_height, self.fps


@dataclass(slots=True)
class FilterConfig:
    threshold_enabled: bool = True
    min_distance_mm: float = 45.0
    max_distance_mm: float = 250.0
    decimation_enabled: bool = False
    decimation_magnitude: int = 1
    disparity_domain_enabled: bool = True
    spatial_enabled: bool = True
    spatial_magnitude: int = 2
    spatial_alpha: float = 0.5
    spatial_delta: int = 20
    spatial_holes_fill: int = 0
    temporal_enabled: bool = True
    temporal_alpha: float = 0.4
    temporal_delta: int = 20
    temporal_persistence: int = 2
    hole_filling_enabled: bool = False
    hole_filling_mode: int = 0


@dataclass(slots=True)
class FusionConfig:
    burst_frames: int = 45
    mad_sigma: float = 3.5
    minimum_inlier_band_mm: float = 0.35
    per_pixel_min_valid_fraction: float = 0.5


@dataclass(slots=True)
class CalibrationConfig:
    file: str = "calibration/empty_tube_calibration.npz"
    center_x_px: float | None = None
    center_y_px: float | None = None
    plane_fit_radius_fraction: float = 0.72
    minimum_plane_coverage: float = 0.55
    maximum_plane_rms_mm: float = 1.25
    maximum_tilt_degrees: float = 8.0
    plane_trim_sigma: float = 3.5
    plane_fit_iterations: int = 4
    reference_target_thickness_mm: float = 0.0
    maximum_rim_distance_error_mm: float = 12.0


@dataclass(slots=True)
class DetectionRoiConfig:
    """Manual circle selection in the processed native depth-image coordinates."""

    center_x_px: float | None = None
    center_y_px: float | None = None
    diameter_mm: float | None = None
    free_image_margin_percent: float = 10.0
    frame_width_px: int | None = None
    frame_height_px: int | None = None
    fps: int | None = None
    stream: str = "depth"
    format: str = "z16"
    camera_serial_number: str | None = None
    selected_utc: str | None = None


@dataclass(slots=True)
class ReconstructionConfig:
    grid_cell_size_mm: float = 1.0
    minimum_surface_coverage: float = 0.70
    maximum_internal_hole_radius_mm: float = 4.0
    local_median_kernel: int = 3
    spatial_outlier_threshold_mm: float = 6.0
    physical_height_tolerance_mm: float = 4.0


@dataclass(slots=True)
class DecisionConfig:
    refill_threshold_percent: float = 10.0
    clear_threshold_percent: float = 12.0
    confirmations_required: int = 2
    minimum_quality_score: float = 0.65


@dataclass(slots=True)
class RuntimeConfig:
    update_interval_seconds: float = 5.0
    continue_after_invalid_measurement: bool = True


@dataclass(slots=True)
class OutputConfig:
    directory: str = "output"
    save_json: bool = True
    save_height_map_npy: bool = True
    save_height_map_png: bool = True
    save_surface_ply: bool = True
    save_color_snapshot: bool = True


@dataclass(slots=True)
class ResearchConfig:
    """Defaults for opt-in session-aware research records."""

    directory: str = "research_records"
    storage_profile: str = "research"


@dataclass(slots=True)
class SyntheticConfig:
    random_seed: int = 7
    fill_percent: float = 35.0
    depth_noise_std_mm: float = 0.30
    invalid_pixel_fraction: float = 0.04
    outlier_pixel_fraction: float = 0.01
    surface_slope_x_mm: float = 0.0
    surface_slope_y_mm: float = 0.0


@dataclass(slots=True)
class AppConfig:
    schema_version: int
    tube: TubeConfig
    camera: CameraConfig
    filters: FilterConfig
    fusion: FusionConfig
    calibration: CalibrationConfig
    detection_roi: DetectionRoiConfig
    reconstruction: ReconstructionConfig
    decision: DecisionConfig
    runtime: RuntimeConfig
    output: OutputConfig
    research: ResearchConfig
    synthetic: SyntheticConfig
    config_path: Path

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent

    @property
    def calibration_path(self) -> Path:
        return _resolve_path(self.config_dir, self.calibration.file)

    @property
    def output_path(self) -> Path:
        return _resolve_path(self.config_dir, self.output.directory)

    @property
    def research_output_path(self) -> Path:
        return _resolve_path(self.config_dir, self.research.directory)

    def validate(self) -> list[str]:
        """Raise for unsafe values and return non-fatal setup warnings."""
        t = self.tube
        c = self.camera
        f = self.filters
        roi = self.detection_roi
        r = self.reconstruction
        d = self.decision

        if self.research.storage_profile not in {"compact", "research", "full_raw"}:
            raise ConfigurationError(
                "research.storage_profile must be compact, research, or full_raw."
            )

        positive = {
            "tube.inner_diameter_mm": t.inner_diameter_mm,
            "tube.usable_height_mm": t.usable_height_mm,
            "camera.distance_to_rim_mm": c.distance_to_rim_mm,
            "camera.depth_width": c.depth_width,
            "camera.depth_height": c.depth_height,
            "camera.fps": c.fps,
            "fusion.burst_frames": self.fusion.burst_frames,
            "reconstruction.grid_cell_size_mm": r.grid_cell_size_mm,
            "runtime.update_interval_seconds": self.runtime.update_interval_seconds,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0.0]
        if invalid:
            raise ConfigurationError(f"Values must be positive: {', '.join(invalid)}")
        if t.wall_edge_exclusion_mm < 0 or t.wall_edge_exclusion_mm >= t.inner_radius_mm:
            raise ConfigurationError(
                "tube.wall_edge_exclusion_mm must be non-negative and smaller than the inner radius."
            )
        if not 0.0 < self.fusion.per_pixel_min_valid_fraction <= 1.0:
            raise ConfigurationError("fusion.per_pixel_min_valid_fraction must be in (0, 1].")
        if self.fusion.mad_sigma <= 0.0:
            raise ConfigurationError("fusion.mad_sigma must be positive.")
        if self.fusion.minimum_inlier_band_mm < 0.0:
            raise ConfigurationError("fusion.minimum_inlier_band_mm must be non-negative.")
        if not 0.0 < self.calibration.plane_fit_radius_fraction <= 1.0:
            raise ConfigurationError("calibration.plane_fit_radius_fraction must be in (0, 1].")
        if not 0.0 < self.calibration.minimum_plane_coverage <= 1.0:
            raise ConfigurationError("calibration.minimum_plane_coverage must be in (0, 1].")
        if not 0.0 <= self.calibration.reference_target_thickness_mm < t.usable_height_mm:
            raise ConfigurationError(
                "calibration.reference_target_thickness_mm must be non-negative and below "
                "tube.usable_height_mm."
            )
        if not 0.0 < r.minimum_surface_coverage <= 1.0:
            raise ConfigurationError("reconstruction.minimum_surface_coverage must be in (0, 1].")
        if r.local_median_kernel < 1 or r.local_median_kernel % 2 == 0:
            raise ConfigurationError("reconstruction.local_median_kernel must be an odd positive integer.")
        if not 0.0 <= d.refill_threshold_percent <= 100.0:
            raise ConfigurationError("decision.refill_threshold_percent must be between 0 and 100.")
        if d.clear_threshold_percent < d.refill_threshold_percent:
            raise ConfigurationError("decision.clear_threshold_percent must not be below refill threshold.")
        if f.min_distance_mm >= f.max_distance_mm:
            raise ConfigurationError("filters.min_distance_mm must be below filters.max_distance_mm.")
        if not 0.0 <= d.minimum_quality_score <= 1.0:
            raise ConfigurationError("decision.minimum_quality_score must be between 0 and 1.")
        if d.confirmations_required < 1:
            raise ConfigurationError("decision.confirmations_required must be at least 1.")
        if (roi.center_x_px is None) != (roi.center_y_px is None):
            raise ConfigurationError(
                "detection_roi.center_x_px and center_y_px must both be set or both be null."
            )
        if (roi.frame_width_px is None) != (roi.frame_height_px is None):
            raise ConfigurationError(
                "detection_roi.frame_width_px and frame_height_px must both be set or both be null."
            )
        if roi.diameter_mm is not None and roi.diameter_mm <= 0.0:
            raise ConfigurationError("detection_roi.diameter_mm must be positive when set.")
        if not 0.0 <= roi.free_image_margin_percent <= 100.0:
            raise ConfigurationError(
                "detection_roi.free_image_margin_percent must be between 0 and 100."
            )
        if roi.stream != "depth":
            raise ConfigurationError("detection_roi.stream must be 'depth'.")
        if roi.frame_width_px is not None:
            if roi.frame_width_px <= 0 or roi.frame_height_px is None or roi.frame_height_px <= 0:
                raise ConfigurationError("Detection ROI frame dimensions must be positive.")
            if roi.center_x_px is not None and not (
                0.0 <= roi.center_x_px < roi.frame_width_px
                and 0.0 <= float(roi.center_y_px) < roi.frame_height_px
            ):
                raise ConfigurationError("The detection ROI centre is outside its saved depth frame.")
        if roi.fps is not None and roi.fps <= 0:
            raise ConfigurationError("detection_roi.fps must be positive when set.")
        if (
            roi.diameter_mm is not None
            and abs(roi.diameter_mm - t.inner_diameter_mm) > 0.01
        ):
            raise ConfigurationError(
                "detection_roi.diameter_mm must match tube.inner_diameter_mm because both "
                "describe the measurement ROI."
            )

        warnings: list[str] = []
        if (
            roi.frame_width_px is not None
            and not f.decimation_enabled
            and (roi.frame_width_px, roi.frame_height_px)
            != (c.depth_width, c.depth_height)
        ):
            warnings.append(
                "The saved detection ROI uses depth coordinates at "
                f"{roi.frame_width_px}x{roi.frame_height_px}, but the requested measurement "
                f"profile is {c.depth_width}x{c.depth_height}. Rerun the centre setup tool "
                "before calibration or measurement."
            )
        implied_inner = t.outer_diameter_mm - 2.0 * t.wall_thickness_mm
        if abs(implied_inner - t.inner_diameter_mm) > t.geometry_warning_tolerance_mm:
            warnings.append(
                "Tube geometry conflict: outer diameter and wall thickness imply "
                f"{implied_inner:.2f} mm inner diameter, but inner_diameter_mm is "
                f"{t.inner_diameter_mm:.2f} mm. Volume uses inner_diameter_mm."
            )

        min_z = approximate_d405_min_z_mm(c.depth_width, c.depth_height)
        if min_z is not None and c.distance_to_rim_mm < min_z:
            warnings.append(
                f"Configured {c.depth_width}x{c.depth_height} depth has an approximate "
                f"D405 minimum-Z of {min_z:.0f} mm, but the rim distance is "
                f"{c.distance_to_rim_mm:.1f} mm. Use a lower-resolution profile or move the camera."
            )
        bottom_reference_distance_mm = (
            c.distance_to_rim_mm
            + t.usable_height_mm
            - self.calibration.reference_target_thickness_mm
        )
        if bottom_reference_distance_mm >= f.max_distance_mm:
            warnings.append(
                f"The empty-bottom reference is nominally {bottom_reference_distance_mm:.1f} mm "
                f"from the camera, but filters.max_distance_mm is {f.max_distance_mm:.1f} mm. "
                "Calibration may reject the bottom; increase the validated far threshold or "
                "reduce the camera-to-bottom distance."
            )
        if c.distance_to_rim_mm < 70.0:
            warnings.append(
                f"The highest possible material surface is {c.distance_to_rim_mm:.1f} mm from "
                "the camera, below the D405's 70 mm ideal-range boundary. The selected profile "
                "may still output depth, but move the camera to at least 70 mm when practical "
                "and validate accuracy with real fill levels."
            )

        fov_width_mm = 2.0 * c.distance_to_rim_mm * tan(radians(87.0 / 2.0))
        if fov_width_mm < t.outer_diameter_mm * 1.10:
            warnings.append(
                f"The nominal horizontal view at the rim is only {fov_width_mm:.1f} mm; "
                "leave at least 10% margin around the tube."
            )
        return warnings


def approximate_d405_min_z_mm(width: int, height: int) -> float | None:
    """Published D401/D405 minimum-Z values by depth profile."""
    values = {
        (1280, 720): 100.0,
        (848, 480): 70.0,
        # The D405 datasheet does not specify a minimum-Z for 640x480.
        (640, 360): 55.0,
        (480, 270): 45.0,
        (424, 240): 40.0,
    }
    return values.get((int(width), int(height)))


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read YAML configuration: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("The YAML root must be a mapping.")

    camera_data = _mapping(data, "camera")
    fallback_raw = camera_data.get("fallback_profiles", CameraConfig().fallback_profiles)
    try:
        fallback = [tuple(int(v) for v in profile) for profile in fallback_raw]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("camera.fallback_profiles must contain [width, height, fps] rows.") from exc
    if any(len(profile) != 3 for profile in fallback):
        raise ConfigurationError("Each camera fallback profile must contain exactly three integers.")
    camera_data["fallback_profiles"] = fallback

    try:
        cfg = AppConfig(
            schema_version=int(data.get("schema_version", 1)),
            tube=TubeConfig(**_mapping(data, "tube")),
            camera=CameraConfig(**camera_data),
            filters=FilterConfig(**_mapping(data, "filters")),
            fusion=FusionConfig(**_mapping(data, "fusion")),
            calibration=CalibrationConfig(**_mapping(data, "calibration")),
            detection_roi=DetectionRoiConfig(**_mapping(data, "detection_roi")),
            reconstruction=ReconstructionConfig(**_mapping(data, "reconstruction")),
            decision=DecisionConfig(**_mapping(data, "decision")),
            runtime=RuntimeConfig(**_mapping(data, "runtime")),
            output=OutputConfig(**_mapping(data, "output")),
            research=ResearchConfig(**_mapping(data, "research")),
            synthetic=SyntheticConfig(**_mapping(data, "synthetic")),
            config_path=config_path,
        )
    except TypeError as exc:
        raise ConfigurationError(f"Unknown or missing configuration field: {exc}") from exc
    cfg.validate()
    return cfg


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"'{key}' must be a YAML mapping.")
    return dict(value)


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()
