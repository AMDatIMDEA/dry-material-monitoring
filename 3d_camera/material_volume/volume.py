"""Tube-coordinate height-map reconstruction and bulk-volume integration."""

from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt

import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.stats import binned_statistic_2d

from .config import AppConfig
from .errors import MeasurementQualityError
from .geometry import camera_points_to_tube, depth_image_to_points
from .models import (
    CalibrationData,
    FusedDepth,
    Intrinsics,
    MeasurementQuality,
    MeasurementResult,
    VolumeEstimate,
)


def reconstruct_volume(
    fused: FusedDepth,
    intrinsics: Intrinsics,
    calibration: CalibrationData,
    config: AppConfig,
    source_name: str,
    color_bgr: np.ndarray | None = None,
) -> VolumeEstimate:
    """Create a metric surface grid and integrate its bulk height over the tube."""
    tube = config.tube
    reconstruction = config.reconstruction
    tolerance_mm = reconstruction.physical_height_tolerance_mm

    points_m, pixel_u, pixel_v = depth_image_to_points(fused.depth_m, intrinsics)
    if points_m.shape[0] < 30:
        raise MeasurementQualityError("Fewer than 30 valid 3D points were captured.")
    tube_points_mm = camera_points_to_tube(
        points_m,
        calibration.bottom_center_m,
        calibration.basis_x,
        calibration.basis_y,
        calibration.tube_axis_toward_camera,
    ) * 1000.0
    x_mm = tube_points_mm[:, 0]
    y_mm = tube_points_mm[:, 1]
    height_mm = tube_points_mm[:, 2]
    radial_mm = np.hypot(x_mm, y_mm)
    reliable_radius_mm = tube.inner_radius_mm - tube.wall_edge_exclusion_mm
    physical = (
        (radial_mm <= reliable_radius_mm)
        & (height_mm >= -tolerance_mm)
        & (height_mm <= tube.usable_height_mm + tolerance_mm)
    )
    if np.count_nonzero(physical) < 30:
        raise MeasurementQualityError(
            "Too few depth points lie inside the physically valid inner-tube region."
        )

    x_mm = x_mm[physical]
    y_mm = y_mm[physical]
    height_mm = np.clip(height_mm[physical], 0.0, tube.usable_height_mm)
    pixel_u = pixel_u[physical]
    pixel_v = pixel_v[physical]

    requested_cell_mm = reconstruction.grid_cell_size_mm
    cell_count = max(8, int(np.ceil(tube.inner_diameter_mm / requested_cell_mm)))
    edges_mm = np.linspace(
        -tube.inner_radius_mm,
        tube.inner_radius_mm,
        cell_count + 1,
        dtype=np.float64,
    )
    cell_size_mm = float(edges_mm[1] - edges_mm[0])
    centers_mm = (edges_mm[:-1] + edges_mm[1:]) / 2.0
    grid_x, grid_y = np.meshgrid(centers_mm, centers_mm)
    radial_grid = np.hypot(grid_x, grid_y)
    inside_mask = radial_grid <= tube.inner_radius_mm
    reliable_mask = radial_grid <= reliable_radius_mm

    # y first makes the resulting array image-like: rows are Y and columns are X.
    statistic = binned_statistic_2d(
        y_mm,
        x_mm,
        height_mm,
        statistic="median",
        bins=(edges_mm, edges_mm),
    ).statistic
    height_grid = np.asarray(statistic, dtype=np.float64)
    observed_mask = np.isfinite(height_grid) & reliable_mask
    observed_count = int(np.count_nonzero(observed_mask))
    if observed_count == 0:
        raise MeasurementQualityError("No valid surface grid cells were reconstructed.")
    surface_coverage = observed_count / max(1, int(np.count_nonzero(reliable_mask)))

    missing_distance_cells, nearest_indices = distance_transform_edt(
        ~observed_mask,
        return_distances=True,
        return_indices=True,
    )
    missing_reliable = reliable_mask & ~observed_mask
    maximum_hole_radius_mm = (
        float(np.max(missing_distance_cells[missing_reliable])) * cell_size_mm
        if np.any(missing_reliable)
        else 0.0
    )

    nearest_filled = height_grid[tuple(nearest_indices)]
    local_median = median_filter(
        nearest_filled,
        size=reconstruction.local_median_kernel,
        mode="nearest",
    )
    spatial_outliers = observed_mask & (
        np.abs(height_grid - local_median) > reconstruction.spatial_outlier_threshold_mm
    )
    cleaned = height_grid.copy()
    cleaned[spatial_outliers] = local_median[spatial_outliers]
    cleaned_observed = np.isfinite(cleaned) & reliable_mask
    _, final_nearest_indices = distance_transform_edt(
        ~cleaned_observed,
        return_distances=True,
        return_indices=True,
    )
    complete_grid = cleaned[tuple(final_nearest_indices)]
    complete_grid = np.clip(complete_grid, 0.0, tube.usable_height_mm)
    complete_grid[~inside_mask] = np.nan

    heights_inside = complete_grid[inside_mask]
    mean_height_mm = float(np.mean(heights_inside))
    material_volume_ml = tube.capacity_ml * mean_height_mm / tube.usable_height_mm
    material_volume_ml = float(np.clip(material_volume_ml, 0.0, tube.capacity_ml))
    empty_volume_ml = float(tube.capacity_ml - material_volume_ml)
    fill_percent = float(100.0 * material_volume_ml / tube.capacity_ml)

    point_valid_fraction = fused.valid_fraction[pixel_v, pixel_u]
    point_temporal_mad_mm = fused.temporal_mad_m[pixel_v, pixel_u] * 1000.0
    median_temporal_valid_fraction = _finite_median(point_valid_fraction, default=0.0)
    temporal_mad_mm = _finite_median(point_temporal_mad_mm, default=float("inf"))

    coverage_score = min(1.0, surface_coverage / reconstruction.minimum_surface_coverage)
    temporal_score = float(np.clip(median_temporal_valid_fraction, 0.0, 1.0))
    plane_score = float(
        np.clip(
            1.0 - calibration.plane_rms_mm / max(config.calibration.maximum_plane_rms_mm, 1e-9),
            0.0,
            1.0,
        )
    )
    hole_score = float(
        np.clip(
            1.0
            - maximum_hole_radius_mm
            / max(reconstruction.maximum_internal_hole_radius_mm, 1e-9),
            0.0,
            1.0,
        )
    )
    quality_score = float(
        0.40 * coverage_score
        + 0.30 * temporal_score
        + 0.15 * plane_score
        + 0.15 * hole_score
    )

    reasons: list[str] = []
    if surface_coverage < reconstruction.minimum_surface_coverage:
        reasons.append(
            f"surface coverage {surface_coverage:.1%} is below "
            f"{reconstruction.minimum_surface_coverage:.1%}"
        )
    if median_temporal_valid_fraction < config.fusion.per_pixel_min_valid_fraction:
        reasons.append(
            f"temporal validity {median_temporal_valid_fraction:.1%} is below "
            f"{config.fusion.per_pixel_min_valid_fraction:.1%}"
        )
    if maximum_hole_radius_mm > reconstruction.maximum_internal_hole_radius_mm:
        reasons.append(
            f"largest internal data hole {maximum_hole_radius_mm:.2f} mm exceeds "
            f"{reconstruction.maximum_internal_hole_radius_mm:.2f} mm"
        )
    if quality_score < config.decision.minimum_quality_score:
        reasons.append(
            f"quality score {quality_score:.3f} is below {config.decision.minimum_quality_score:.3f}"
        )

    quantization_mm = cell_size_mm / sqrt(12.0)
    base_uncertainty_mm = sqrt(
        (1.4826 * temporal_mad_mm) ** 2
        + calibration.plane_rms_mm**2
        + quantization_mm**2
    )
    observed_spread_mm = float(np.std(height_grid[observed_mask]))
    missing_penalty_mm = (1.0 - surface_coverage) * min(
        observed_spread_mm,
        tube.usable_height_mm * 0.10,
    )
    uncertainty_percent = float(
        100.0 * (base_uncertainty_mm + missing_penalty_mm) / tube.usable_height_mm
    )

    quality = MeasurementQuality(
        valid=not reasons,
        score=quality_score,
        surface_coverage=surface_coverage,
        median_temporal_valid_fraction=median_temporal_valid_fraction,
        temporal_mad_mm=temporal_mad_mm,
        maximum_hole_radius_mm=maximum_hole_radius_mm,
        calibration_plane_rms_mm=calibration.plane_rms_mm,
        spatial_outliers_replaced=int(np.count_nonzero(spatial_outliers)),
        uncertainty_percent=uncertainty_percent,
        reasons=reasons,
    )
    refill_required = bool(
        quality.valid and fill_percent < config.decision.refill_threshold_percent
    )
    warning_state = (
        "REFILL_REQUIRED"
        if refill_required
        else ("OK" if quality.valid else "INVALID_MEASUREMENT")
    )
    result = MeasurementResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source=source_name,
        capacity_ml=tube.capacity_ml,
        material_volume_ml=material_volume_ml,
        empty_volume_ml=empty_volume_ml,
        fill_percent=fill_percent,
        mean_level_mm=mean_height_mm,
        minimum_level_mm=float(np.min(heights_inside)),
        maximum_level_mm=float(np.max(heights_inside)),
        refill_threshold_percent=config.decision.refill_threshold_percent,
        refill_required=refill_required,
        warning_state=warning_state,
        quality=quality,
    )
    return VolumeEstimate(
        result=result,
        height_map_mm=complete_grid.astype(np.float32),
        inside_mask=inside_mask,
        observed_mask=observed_mask,
        x_centers_mm=centers_mm.astype(np.float32),
        y_centers_mm=centers_mm.astype(np.float32),
        color_bgr=color_bgr,
    )


def _finite_median(values: np.ndarray, default: float) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float(default)
