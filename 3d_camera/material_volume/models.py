"""Shared data models for capture, calibration, and measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import TubeConfig
from .errors import CalibrationError


@dataclass(slots=True, frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str = "none"
    coefficients: tuple[float, ...] = ()

    @classmethod
    def from_realsense(cls, value: Any) -> "Intrinsics":
        coeffs = tuple(float(v) for v in getattr(value, "coeffs", ()))
        return cls(
            width=int(value.width),
            height=int(value.height),
            fx=float(value.fx),
            fy=float(value.fy),
            ppx=float(value.ppx),
            ppy=float(value.ppy),
            distortion_model=str(getattr(value, "model", "unknown")),
            coefficients=coeffs,
        )


@dataclass(slots=True)
class DepthBurst:
    frames_m: list[np.ndarray]
    intrinsics: Intrinsics
    depth_scale_m: float
    color_bgr: np.ndarray | None = None
    infrared_left: np.ndarray | None = None
    infrared_right: np.ndarray | None = None
    source_name: str = "unknown"
    raw_frames_z16: list[np.ndarray] | None = None
    frame_timestamps_ms: list[float] = field(default_factory=list)
    camera_serial_number: str | None = None
    active_stream_profile: tuple[int, int, int] | None = None
    sensor_settings: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.frames_m:
            raise ValueError("A depth burst must contain at least one frame.")
        expected = (self.intrinsics.height, self.intrinsics.width)
        bad = [frame.shape for frame in self.frames_m if frame.shape != expected]
        if bad:
            raise ValueError(f"Depth-frame shape does not match intrinsics {expected}: {bad[0]}")
        if self.raw_frames_z16 is not None:
            if len(self.raw_frames_z16) != len(self.frames_m):
                raise ValueError("Raw Z16 frames must contain one frame per processed depth frame.")
            raw_shapes = {frame.shape for frame in self.raw_frames_z16}
            if len(raw_shapes) != 1 or any(frame.dtype != np.uint16 for frame in self.raw_frames_z16):
                raise ValueError("Raw depth frames must share one shape and use lossless uint16 Z16 data.")


@dataclass(slots=True)
class FusedDepth:
    depth_m: np.ndarray
    valid_fraction: np.ndarray
    temporal_mad_m: np.ndarray
    raw_valid_fraction: np.ndarray
    frame_count: int


@dataclass(slots=True)
class CalibrationData:
    intrinsics: Intrinsics
    bottom_plane_point_m: np.ndarray
    tube_axis_toward_camera: np.ndarray
    basis_x: np.ndarray
    basis_y: np.ndarray
    bottom_center_m: np.ndarray
    inner_diameter_mm: float
    usable_height_mm: float
    center_x_px: float
    center_y_px: float
    plane_rms_mm: float
    plane_coverage: float
    measured_rim_distance_mm: float
    calibration_method: str = "automatic_plane_fit"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            schema_version=np.array(1, dtype=np.int32),
            intrinsics=np.array(
                [
                    self.intrinsics.width,
                    self.intrinsics.height,
                    self.intrinsics.fx,
                    self.intrinsics.fy,
                    self.intrinsics.ppx,
                    self.intrinsics.ppy,
                ],
                dtype=np.float64,
            ),
            distortion_model=np.array(self.intrinsics.distortion_model),
            distortion_coefficients=np.array(self.intrinsics.coefficients, dtype=np.float64),
            bottom_plane_point_m=np.asarray(self.bottom_plane_point_m, dtype=np.float64),
            tube_axis_toward_camera=np.asarray(self.tube_axis_toward_camera, dtype=np.float64),
            basis_x=np.asarray(self.basis_x, dtype=np.float64),
            basis_y=np.asarray(self.basis_y, dtype=np.float64),
            bottom_center_m=np.asarray(self.bottom_center_m, dtype=np.float64),
            inner_diameter_mm=np.array(self.inner_diameter_mm, dtype=np.float64),
            usable_height_mm=np.array(self.usable_height_mm, dtype=np.float64),
            center_x_px=np.array(self.center_x_px, dtype=np.float64),
            center_y_px=np.array(self.center_y_px, dtype=np.float64),
            plane_rms_mm=np.array(self.plane_rms_mm, dtype=np.float64),
            plane_coverage=np.array(self.plane_coverage, dtype=np.float64),
            measured_rim_distance_mm=np.array(self.measured_rim_distance_mm, dtype=np.float64),
            calibration_method=np.array(self.calibration_method),
            created_utc=np.array(self.created_utc),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationData":
        source = Path(path)
        if not source.is_file():
            raise CalibrationError(
                f"Calibration file not found: {source}. Run calibrate_empty.py with an empty tube first."
            )
        try:
            with np.load(source, allow_pickle=False) as values:
                raw_intrinsics = values["intrinsics"].astype(float)
                intrinsics = Intrinsics(
                    width=int(raw_intrinsics[0]),
                    height=int(raw_intrinsics[1]),
                    fx=float(raw_intrinsics[2]),
                    fy=float(raw_intrinsics[3]),
                    ppx=float(raw_intrinsics[4]),
                    ppy=float(raw_intrinsics[5]),
                    distortion_model=str(values["distortion_model"].item()),
                    coefficients=tuple(values["distortion_coefficients"].astype(float).tolist()),
                )
                return cls(
                    intrinsics=intrinsics,
                    bottom_plane_point_m=values["bottom_plane_point_m"].astype(float),
                    tube_axis_toward_camera=values["tube_axis_toward_camera"].astype(float),
                    basis_x=values["basis_x"].astype(float),
                    basis_y=values["basis_y"].astype(float),
                    bottom_center_m=values["bottom_center_m"].astype(float),
                    inner_diameter_mm=float(values["inner_diameter_mm"]),
                    usable_height_mm=float(values["usable_height_mm"]),
                    center_x_px=float(values["center_x_px"]),
                    center_y_px=float(values["center_y_px"]),
                    plane_rms_mm=float(values["plane_rms_mm"]),
                    plane_coverage=float(values["plane_coverage"]),
                    measured_rim_distance_mm=float(values["measured_rim_distance_mm"]),
                    calibration_method=(
                        str(values["calibration_method"].item())
                        if "calibration_method" in values.files
                        else "automatic_plane_fit"
                    ),
                    created_utc=str(values["created_utc"].item()),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise CalibrationError(f"Calibration file is invalid or incompatible: {source}: {exc}") from exc

    def compatibility_errors(
        self,
        intrinsics: Intrinsics,
        tube: TubeConfig,
        *,
        center_x_px: float | None = None,
        center_y_px: float | None = None,
    ) -> list[str]:
        errors: list[str] = []
        if (intrinsics.width, intrinsics.height) != (
            self.intrinsics.width,
            self.intrinsics.height,
        ):
            errors.append(
                "Depth resolution changed since calibration: "
                f"{self.intrinsics.width}x{self.intrinsics.height} -> "
                f"{intrinsics.width}x{intrinsics.height}."
            )
        for name in ("fx", "fy", "ppx", "ppy"):
            old = float(getattr(self.intrinsics, name))
            new = float(getattr(intrinsics, name))
            if abs(old - new) > max(0.25, abs(old) * 0.002):
                errors.append(f"Camera intrinsic {name} changed since calibration ({old:.3f} -> {new:.3f}).")
        if self.intrinsics.distortion_model != intrinsics.distortion_model:
            errors.append(
                "Camera distortion model changed since calibration "
                f"({self.intrinsics.distortion_model} -> {intrinsics.distortion_model})."
            )
        old_coefficients = np.asarray(self.intrinsics.coefficients, dtype=np.float64)
        new_coefficients = np.asarray(intrinsics.coefficients, dtype=np.float64)
        if old_coefficients.shape != new_coefficients.shape or not np.allclose(
            old_coefficients,
            new_coefficients,
            rtol=1e-5,
            atol=1e-8,
        ):
            errors.append("Camera distortion coefficients changed since calibration.")
        if abs(self.inner_diameter_mm - tube.inner_diameter_mm) > 0.01:
            errors.append("tube.inner_diameter_mm changed since calibration.")
        if abs(self.usable_height_mm - tube.usable_height_mm) > 0.01:
            errors.append("tube.usable_height_mm changed since calibration.")
        if center_x_px is not None and abs(self.center_x_px - center_x_px) > 0.01:
            errors.append("detection_roi.center_x_px changed since calibration.")
        if center_y_px is not None and abs(self.center_y_px - center_y_px) > 0.01:
            errors.append("detection_roi.center_y_px changed since calibration.")
        return errors


@dataclass(slots=True)
class MeasurementQuality:
    """Quality gates plus a backward-compatible heuristic variability field.

    ``uncertainty_percent`` is not a metrological expanded uncertainty. It is a
    conservative internal indicator assembled from temporal dispersion, plane
    residual, grid sampling, and missing-data penalties.
    """
    valid: bool
    score: float
    surface_coverage: float
    median_temporal_valid_fraction: float
    temporal_mad_mm: float
    maximum_hole_radius_mm: float
    calibration_plane_rms_mm: float
    spatial_outliers_replaced: int
    uncertainty_percent: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MeasurementResult:
    timestamp_utc: str
    source: str
    capacity_ml: float
    material_volume_ml: float
    empty_volume_ml: float
    fill_percent: float
    mean_level_mm: float
    minimum_level_mm: float
    maximum_level_mm: float
    refill_threshold_percent: float
    refill_required: bool
    warning_state: str
    quality: MeasurementQuality

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass(slots=True)
class VolumeEstimate:
    result: MeasurementResult
    height_map_mm: np.ndarray
    inside_mask: np.ndarray
    observed_mask: np.ndarray
    x_centers_mm: np.ndarray
    y_centers_mm: np.ndarray
    color_bgr: np.ndarray | None = None
