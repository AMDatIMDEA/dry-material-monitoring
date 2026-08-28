"""Optional empirical D405 flat-target repeatability check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from material_volume.geometry import deproject_pixels, fit_plane_robust
from material_volume.models import Intrinsics
from set_up.setup_optimizer import CameraProfile


@dataclass(slots=True, frozen=True)
class PrecisionTestResult:
    frame_count: int
    roi_fraction: float
    median_distance_mm: float
    frame_median_repeatability_mm: float
    median_pixel_robust_std_mm: float
    plane_residual_rms_mm: float
    valid_depth_coverage: float
    known_reference_distance_mm: float | None
    absolute_error_mm: float | None


def run_flat_target_precision_test(
    profile: CameraProfile,
    *,
    serial_number: str | None = None,
    frame_count: int = 60,
    warmup_frames: int = 30,
    timeout_ms: int = 5000,
    roi_fraction: float = 0.50,
    known_reference_distance_mm: float | None = None,
) -> PrecisionTestResult:
    """Capture a central matte plane and report empirical, non-absolute metrics."""
    if frame_count < 10:
        raise ValueError("Precision test requires at least 10 frames.")
    if warmup_frames < 0:
        raise ValueError("Warm-up frame count cannot be negative.")
    if not 0.1 <= roi_fraction <= 1.0:
        raise ValueError("Precision-test ROI fraction must be between 0.1 and 1.0.")
    if known_reference_distance_mm is not None and known_reference_distance_mm <= 0.0:
        raise ValueError("Known reference distance must be greater than zero.")
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is unavailable; precision test cannot run.") from exc

    pipeline: Any = rs.pipeline()
    stream_config: Any = rs.config()
    if serial_number:
        stream_config.enable_device(serial_number)
    stream_config.enable_stream(
        rs.stream.depth,
        profile.width,
        profile.height,
        rs.format.z16,
        profile.fps,
    )
    started = False
    try:
        active = pipeline.start(stream_config)
        started = True
        sensor = active.get_device().first_depth_sensor()
        depth_scale_mm = float(sensor.get_depth_scale()) * 1000.0
        for _ in range(warmup_frames):
            pipeline.wait_for_frames(timeout_ms)

        x0, x1, y0, y1 = _central_roi(
            profile.width, profile.height, roi_fraction
        )
        frames_mm: list[np.ndarray] = []
        active_intrinsics: Intrinsics | None = None
        for _ in range(frame_count):
            frames = pipeline.wait_for_frames(timeout_ms)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError("The D405 returned a frame set without depth.")
            video = depth_frame.profile.as_video_stream_profile()
            value = (
                video.get_intrinsics()
                if hasattr(video, "get_intrinsics")
                else video.intrinsics
            )
            active_intrinsics = Intrinsics.from_realsense(value)
            raw = np.asanyarray(depth_frame.get_data())
            crop = raw[y0:y1, x0:x1].astype(np.float32) * depth_scale_mm
            valid = (
                (raw[y0:y1, x0:x1] > 0)
                & np.isfinite(crop)
                & (crop >= profile.minimum_usable_depth_mm)
                & (crop <= profile.maximum_reliable_depth_mm)
            )
            frames_mm.append(np.where(valid, crop, np.nan))
    finally:
        if started:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

    if active_intrinsics is None or not frames_mm:
        raise RuntimeError("No usable depth frames were captured for the precision test.")
    stack = np.stack(frames_mm).astype(np.float32, copy=False)
    valid_coverage = float(np.count_nonzero(np.isfinite(stack)) / stack.size)
    if valid_coverage < 0.25:
        raise RuntimeError(
            f"Flat-target ROI has only {valid_coverage:.1%} valid depth; "
            "use a flat matte target within the selected depth range."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_image_mm = np.nanmedian(stack, axis=0)
        pixel_mad_mm = np.nanmedian(
            np.abs(stack - median_image_mm[None, :, :]), axis=0
        )
        frame_medians_mm = np.nanmedian(stack, axis=(1, 2))
    frame_center = float(np.nanmedian(frame_medians_mm))
    frame_repeatability = float(
        1.4826 * np.nanmedian(np.abs(frame_medians_mm - frame_center))
    )
    pixel_robust_std = float(np.nanmedian(1.4826 * pixel_mad_mm))

    valid_plane = np.isfinite(median_image_mm) & (median_image_mm > 0.0)
    local_v, local_u = np.nonzero(valid_plane)
    points_m = deproject_pixels(
        local_u + x0,
        local_v + y0,
        median_image_mm[valid_plane] / 1000.0,
        active_intrinsics,
    )
    plane = fit_plane_robust(points_m, trim_sigma=3.5, iterations=4)
    absolute_error = (
        frame_center - known_reference_distance_mm
        if known_reference_distance_mm is not None
        else None
    )
    return PrecisionTestResult(
        frame_count=frame_count,
        roi_fraction=roi_fraction,
        median_distance_mm=frame_center,
        frame_median_repeatability_mm=frame_repeatability,
        median_pixel_robust_std_mm=pixel_robust_std,
        plane_residual_rms_mm=plane.rms_m * 1000.0,
        valid_depth_coverage=valid_coverage,
        known_reference_distance_mm=known_reference_distance_mm,
        absolute_error_mm=absolute_error,
    )


def _central_roi(
    width: int, height: int, fraction: float
) -> tuple[int, int, int, int]:
    roi_width = max(2, int(round(width * fraction)))
    roi_height = max(2, int(round(height * fraction)))
    x0 = (width - roi_width) // 2
    y0 = (height - roi_height) // 2
    return x0, x0 + roi_width, y0, y0 + roi_height
