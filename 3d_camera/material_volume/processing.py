"""Depth burst validation and robust temporal fusion."""

from __future__ import annotations

import warnings

import numpy as np

from .config import FilterConfig, FusionConfig
from .errors import FrameCaptureError
from .models import DepthBurst, FusedDepth


def fuse_depth_burst(
    burst: DepthBurst,
    fusion: FusionConfig,
    filters: FilterConfig,
) -> FusedDepth:
    """Fuse a static frame burst using per-pixel median and MAD rejection."""
    burst.validate()
    try:
        stack = np.stack(burst.frames_m).astype(np.float32, copy=False)
    except (ValueError, MemoryError) as exc:
        raise FrameCaptureError(f"Could not stack the captured depth frames: {exc}") from exc

    valid = (
        np.isfinite(stack)
        & (stack > float(filters.min_distance_mm) / 1000.0)
        & (stack < float(filters.max_distance_mm) / 1000.0)
    )
    raw_valid_fraction = np.mean(valid, axis=0, dtype=np.float32)
    stack = np.where(valid, stack, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        first_median = np.nanmedian(stack, axis=0)
        deviations = np.abs(stack - first_median[None, :, :])
        mad = np.nanmedian(deviations, axis=0)

    inlier_band = np.maximum(
        float(fusion.minimum_inlier_band_mm) / 1000.0,
        float(fusion.mad_sigma) * 1.4826 * mad,
    )
    inliers = valid & (deviations <= inlier_band[None, :, :])
    valid_fraction = np.mean(inliers, axis=0, dtype=np.float32)
    cleaned = np.where(inliers, stack, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fused = np.nanmedian(cleaned, axis=0).astype(np.float32)
        temporal_mad = np.nanmedian(
            np.abs(cleaned - fused[None, :, :]), axis=0
        ).astype(np.float32)

    accepted = valid_fraction >= float(fusion.per_pixel_min_valid_fraction)
    fused[~accepted] = np.nan
    temporal_mad[~accepted] = np.nan
    return FusedDepth(
        depth_m=fused,
        valid_fraction=valid_fraction,
        temporal_mad_m=temporal_mad,
        raw_valid_fraction=raw_valid_fraction,
        frame_count=stack.shape[0],
    )
