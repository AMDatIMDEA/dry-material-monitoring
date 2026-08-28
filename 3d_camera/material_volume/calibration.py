"""Empty-tube calibration of the bottom plane, center, and tube coordinate frame."""

from __future__ import annotations

from math import acos, degrees

import numpy as np

from .config import AppConfig
from .errors import CalibrationError
from .geometry import (
    deproject_pixels,
    depth_image_to_points,
    fit_plane_robust,
    plane_basis,
    ray_plane_intersection,
)
from .models import CalibrationData, DepthBurst, FusedDepth, Intrinsics
from .processing import fuse_depth_burst


def create_virtual_bottom_calibration(
    intrinsics: Intrinsics,
    config: AppConfig,
) -> CalibrationData:
    """Create a camera-aligned virtual bottom solely from configured geometry.

    No observed depth values are used. The configured camera-to-rim distance and
    usable height place the plane along camera Z, while the configured detection
    centre locates the tube axis in X/Y. This deliberately assumes that the tube
    axis is parallel to the camera optical axis.
    """
    tube = config.tube
    roi = config.detection_roi
    settings = config.calibration
    configured_center_x = (
        roi.center_x_px if roi.center_x_px is not None else settings.center_x_px
    )
    configured_center_y = (
        roi.center_y_px if roi.center_y_px is not None else settings.center_y_px
    )
    center_x = intrinsics.ppx if configured_center_x is None else float(configured_center_x)
    center_y = intrinsics.ppy if configured_center_y is None else float(configured_center_y)

    if (
        roi.frame_width_px is not None
        and (roi.frame_width_px, roi.frame_height_px)
        != (intrinsics.width, intrinsics.height)
    ):
        raise CalibrationError(
            "The manually selected detection centre was saved for "
            f"{roi.frame_width_px}x{roi.frame_height_px}, but the active depth "
            f"stream is {intrinsics.width}x{intrinsics.height}. Rerun the centre setup tool."
        )
    if not (0 <= center_x < intrinsics.width and 0 <= center_y < intrinsics.height):
        raise CalibrationError("The configured tube center is outside the depth image.")

    bottom_depth_m = (
        config.camera.distance_to_rim_mm + tube.usable_height_mm
    ) / 1000.0
    bottom_center = deproject_pixels(
        np.array([center_x], dtype=np.float64),
        np.array([center_y], dtype=np.float64),
        np.array([bottom_depth_m], dtype=np.float64),
        intrinsics,
    )[0]
    normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    basis_x, basis_y = plane_basis(normal)

    return CalibrationData(
        intrinsics=intrinsics,
        bottom_plane_point_m=bottom_center.copy(),
        tube_axis_toward_camera=normal,
        basis_x=basis_x,
        basis_y=basis_y,
        bottom_center_m=bottom_center,
        inner_diameter_mm=tube.inner_diameter_mm,
        usable_height_mm=tube.usable_height_mm,
        center_x_px=center_x,
        center_y_px=center_y,
        plane_rms_mm=0.0,
        plane_coverage=1.0,
        measured_rim_distance_mm=config.camera.distance_to_rim_mm,
        calibration_method="virtual_from_config",
    )


def calibrate_empty_burst(burst: DepthBurst, config: AppConfig) -> CalibrationData:
    fused = fuse_depth_burst(burst, config.fusion, config.filters)
    return calibrate_empty_depth(fused, burst.intrinsics, config)


def calibrate_empty_depth(
    fused: FusedDepth,
    intrinsics: Intrinsics,
    config: AppConfig,
) -> CalibrationData:
    """Fit a matte reference at the inner bottom of an empty, fixed tube."""
    tube = config.tube
    camera = config.camera
    settings = config.calibration
    roi = config.detection_roi
    configured_center_x = (
        roi.center_x_px if roi.center_x_px is not None else settings.center_x_px
    )
    configured_center_y = (
        roi.center_y_px if roi.center_y_px is not None else settings.center_y_px
    )
    center_x = intrinsics.ppx if configured_center_x is None else float(configured_center_x)
    center_y = intrinsics.ppy if configured_center_y is None else float(configured_center_y)

    if (
        roi.frame_width_px is not None
        and (roi.frame_width_px, roi.frame_height_px)
        != (intrinsics.width, intrinsics.height)
    ):
        raise CalibrationError(
            "The manually selected detection centre was saved for "
            f"{roi.frame_width_px}x{roi.frame_height_px}, but the measurement depth "
            f"stream is {intrinsics.width}x{intrinsics.height}. Rerun the centre setup tool."
        )

    if not (0 <= center_x < intrinsics.width and 0 <= center_y < intrinsics.height):
        raise CalibrationError("The configured tube center is outside the depth image.")

    reference_depth_mm = (
        camera.distance_to_rim_mm
        + tube.usable_height_mm
        - settings.reference_target_thickness_mm
    )
    fit_radius_mm = tube.inner_radius_mm * settings.plane_fit_radius_fraction
    radius_x_px = intrinsics.fx * fit_radius_mm / reference_depth_mm
    radius_y_px = intrinsics.fy * fit_radius_mm / reference_depth_mm
    yy, xx = np.indices(fused.depth_m.shape, dtype=np.float64)
    candidate_mask = (
        ((xx - center_x) / radius_x_px) ** 2
        + ((yy - center_y) / radius_y_px) ** 2
        <= 1.0
    )
    candidate_count = int(np.count_nonzero(candidate_mask))
    valid_mask = candidate_mask & np.isfinite(fused.depth_m) & (fused.depth_m > 0.0)
    valid_count = int(np.count_nonzero(valid_mask))
    coverage = valid_count / max(1, candidate_count)
    if coverage < settings.minimum_plane_coverage:
        raise CalibrationError(
            "The empty-bottom reference has only "
            f"{coverage:.1%} valid depth coverage; at least "
            f"{settings.minimum_plane_coverage:.1%} is required. Place a flat matte disk "
            "inside the bottom, improve lighting/texture, and keep the transparent wall out of the ROI."
        )

    points, _, _ = depth_image_to_points(fused.depth_m, intrinsics, valid_mask)
    plane = fit_plane_robust(
        points,
        trim_sigma=settings.plane_trim_sigma,
        iterations=settings.plane_fit_iterations,
    )
    plane_rms_mm = plane.rms_m * 1000.0
    if plane_rms_mm > settings.maximum_plane_rms_mm:
        raise CalibrationError(
            f"Bottom reference plane RMS is {plane_rms_mm:.3f} mm, above the "
            f"{settings.maximum_plane_rms_mm:.3f} mm limit."
        )

    normal = plane.normal_toward_camera
    optical_toward_camera = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tilt_degrees = degrees(acos(float(np.clip(np.dot(normal, optical_toward_camera), -1.0, 1.0))))
    if tilt_degrees > settings.maximum_tilt_degrees:
        raise CalibrationError(
            f"Tube/camera tilt is {tilt_degrees:.2f} degrees, above the configured "
            f"{settings.maximum_tilt_degrees:.2f} degree limit. Level the camera or raise the limit deliberately."
        )

    reference_center = ray_plane_intersection(
        center_x,
        center_y,
        intrinsics,
        plane.point_m,
        normal,
    )
    # A calibration insert sits toward the camera relative to the true bottom.
    bottom_center = reference_center - normal * (settings.reference_target_thickness_mm / 1000.0)
    bottom_plane_point = plane.point_m - normal * (settings.reference_target_thickness_mm / 1000.0)
    basis_x, basis_y = plane_basis(normal)
    rim_center = bottom_center + normal * (tube.usable_height_mm / 1000.0)
    measured_rim_distance_mm = float(rim_center[2] * 1000.0)
    if measured_rim_distance_mm <= 0.0:
        raise CalibrationError("The calculated rim is at or behind the camera; check height and orientation.")
    if abs(measured_rim_distance_mm - camera.distance_to_rim_mm) > settings.maximum_rim_distance_error_mm:
        raise CalibrationError(
            f"Calibration predicts a rim distance of {measured_rim_distance_mm:.1f} mm, "
            f"but config.yaml says {camera.distance_to_rim_mm:.1f} mm. Check the usable "
            "internal height, camera distance, calibration target thickness, and tube center."
        )

    return CalibrationData(
        intrinsics=intrinsics,
        bottom_plane_point_m=bottom_plane_point,
        tube_axis_toward_camera=normal,
        basis_x=basis_x,
        basis_y=basis_y,
        bottom_center_m=bottom_center,
        inner_diameter_mm=tube.inner_diameter_mm,
        usable_height_mm=tube.usable_height_mm,
        center_x_px=center_x,
        center_y_px=center_y,
        plane_rms_mm=plane_rms_mm,
        plane_coverage=coverage,
        measured_rim_distance_mm=measured_rim_distance_mm,
    )
