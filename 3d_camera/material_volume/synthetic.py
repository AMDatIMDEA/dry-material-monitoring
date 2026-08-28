"""Deterministic D405-like synthetic depth source for hardware-free validation."""

from __future__ import annotations

from math import radians, tan

import numpy as np

from .config import AppConfig
from .models import DepthBurst, Intrinsics


class SyntheticDepthSource:
    """Generate noisy flat/sloped polymer surfaces inside the configured tube."""

    def __init__(self, config: AppConfig, fill_percent: float | None = None) -> None:
        self.config = config
        self.fill_percent = (
            config.synthetic.fill_percent if fill_percent is None else float(fill_percent)
        )
        if not 0.0 <= self.fill_percent <= 100.0:
            raise ValueError("Synthetic fill percent must be between 0 and 100.")
        self._capture_index = 0
        self.intrinsics = synthetic_intrinsics(
            config.camera.depth_width,
            config.camera.depth_height,
        )

    def __enter__(self) -> "SyntheticDepthSource":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def capture_burst(self, frame_count: int, *, retain_raw: bool = False) -> DepthBurst:
        seed = self.config.synthetic.random_seed + 1009 * self._capture_index
        rng = np.random.default_rng(seed)
        frames = [self._make_frame(rng) for _ in range(int(frame_count))]
        self._capture_index += 1
        raw_frames = None
        if retain_raw:
            raw_frames = [
                np.rint(np.nan_to_num(frame, nan=0.0) / 0.001).astype(np.uint16)
                for frame in frames
            ]
        return DepthBurst(
            frames_m=frames,
            intrinsics=self.intrinsics,
            depth_scale_m=0.001,
            color_bgr=self._make_color_image(),
            source_name=f"synthetic:{self.fill_percent:.2f}%",
            raw_frames_z16=raw_frames,
            camera_serial_number="synthetic",
            active_stream_profile=(
                self.config.camera.depth_width,
                self.config.camera.depth_height,
                self.config.camera.fps,
            ),
            sensor_settings={"source": "deterministic_synthetic"},
        )

    def _make_frame(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        intr = self.intrinsics
        tube = cfg.tube
        syn = cfg.synthetic
        bottom_depth_m = (cfg.camera.distance_to_rim_mm + tube.usable_height_mm) / 1000.0
        base_height_mm = tube.usable_height_mm * self.fill_percent / 100.0

        yy, xx = np.indices((intr.height, intr.width), dtype=np.float64)
        surface_depth_m = bottom_depth_m - base_height_mm / 1000.0
        x_mm = (xx - intr.ppx) * surface_depth_m / intr.fx * 1000.0
        y_mm = (yy - intr.ppy) * surface_depth_m / intr.fy * 1000.0
        normalized_x = x_mm / max(tube.inner_radius_mm, 1e-9)
        normalized_y = y_mm / max(tube.inner_radius_mm, 1e-9)
        height_mm = (
            base_height_mm
            + syn.surface_slope_x_mm * normalized_x
            + syn.surface_slope_y_mm * normalized_y
        )
        height_mm = np.clip(height_mm, 0.0, tube.usable_height_mm)
        depth_m = bottom_depth_m - height_mm / 1000.0

        # Recompute physical radial position after applying the sloped surface depth.
        x_mm = (xx - intr.ppx) * depth_m / intr.fx * 1000.0
        y_mm = (yy - intr.ppy) * depth_m / intr.fy * 1000.0
        radius_mm = np.hypot(x_mm, y_mm)
        inside = radius_mm <= tube.inner_radius_mm
        frame = np.full((intr.height, intr.width), np.nan, dtype=np.float32)
        noise_m = rng.normal(0.0, syn.depth_noise_std_mm / 1000.0, frame.shape)
        frame[inside] = (depth_m[inside] + noise_m[inside]).astype(np.float32)

        # Transparent-wall corruption is concentrated in the annulus intentionally
        # excluded by the reconstruction stage.
        edge = inside & (radius_mm >= tube.inner_radius_mm - tube.wall_edge_exclusion_mm)
        edge_invalid = edge & (rng.random(frame.shape) < 0.45)
        frame[edge_invalid] = np.nan

        invalid = inside & (rng.random(frame.shape) < syn.invalid_pixel_fraction)
        frame[invalid] = np.nan
        outliers = inside & np.isfinite(frame) & (rng.random(frame.shape) < syn.outlier_pixel_fraction)
        outlier_count = int(np.count_nonzero(outliers))
        if outlier_count:
            signs = rng.choice(np.array([-1.0, 1.0]), size=outlier_count)
            frame[outliers] += (signs * rng.uniform(0.003, 0.012, outlier_count)).astype(np.float32)
        return frame

    def _make_color_image(self) -> np.ndarray:
        height, width = self.intrinsics.height, self.intrinsics.width
        image = np.full((height, width, 3), (235, 235, 235), dtype=np.uint8)
        yy, xx = np.indices((height, width), dtype=np.float64)
        depth_m = (
            self.config.camera.distance_to_rim_mm
            + self.config.tube.usable_height_mm * (1.0 - self.fill_percent / 100.0)
        ) / 1000.0
        x_mm = (xx - self.intrinsics.ppx) * depth_m / self.intrinsics.fx * 1000.0
        y_mm = (yy - self.intrinsics.ppy) * depth_m / self.intrinsics.fy * 1000.0
        inside = np.hypot(x_mm, y_mm) <= self.config.tube.inner_radius_mm
        image[inside] = (65, 135, 210)
        return image


def synthetic_intrinsics(width: int, height: int) -> Intrinsics:
    fx = (width / 2.0) / tan(radians(87.0 / 2.0))
    fy = (height / 2.0) / tan(radians(58.0 / 2.0))
    return Intrinsics(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        ppx=(width - 1) / 2.0,
        ppy=(height - 1) / 2.0,
        distortion_model="rectified_synthetic",
        coefficients=(),
    )
