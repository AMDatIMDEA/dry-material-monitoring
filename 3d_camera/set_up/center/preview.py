"""Native processed-depth preview and physical circle projection helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from material_volume.capture import RealSenseDepthSource
from material_volume.config import AppConfig
from material_volume.errors import CameraUnavailableError, FrameCaptureError
from material_volume.models import Intrinsics


WINDOW_NAME = "D405 detection-circle centre"
CENTER_DEPTH_PROFILE = (1280, 720, 30)


class PreviewCancelled(RuntimeError):
    """The user closed or cancelled the preview without selecting a centre."""


@dataclass(slots=True, frozen=True)
class PreviewSelection:
    center_x_px: float
    center_y_px: float
    frame_width_px: int
    frame_height_px: int
    fps: int
    depth_width: int
    depth_height: int
    camera_serial_number: str | None


def projected_radii_px(
    diameter_mm: float,
    distance_mm: float,
    intrinsics: Intrinsics,
) -> tuple[float, float]:
    """Project a physical circle at a fronto-parallel plane into depth pixels."""
    if diameter_mm <= 0.0 or distance_mm <= 0.0:
        raise ValueError("Circle diameter and projection distance must be positive.")
    radius_mm = diameter_mm / 2.0
    return (
        intrinsics.fx * radius_mm / distance_mm,
        intrinsics.fy * radius_mm / distance_mm,
    )


def margin_radii_px(
    radii_px: tuple[float, float], margin_percent: float
) -> tuple[float, float]:
    if not 0.0 <= margin_percent <= 100.0:
        raise ValueError("Free image margin must be between 0 and 100 percent.")
    # Project convention: the percentage is a fraction of the physical circle
    # diameter added independently on each side. A 10% margin therefore makes
    # the safety diameter 120% of the physical diameter.
    scale = 1.0 + 2.0 * margin_percent / 100.0
    return radii_px[0] * scale, radii_px[1] * scale


def ellipse_fits_frame(
    center: tuple[float, float],
    radii_px: tuple[float, float],
    width: int,
    height: int,
) -> bool:
    x, y = center
    rx, ry = radii_px
    return (
        x - rx >= -0.5
        and y - ry >= -0.5
        and x + rx <= width - 0.5
        and y + ry <= height - 0.5
    )


def select_center(
    config: AppConfig,
    diameter_mm: float,
    margin_percent: float,
) -> PreviewSelection:
    """Open the measurement depth stream and return an accepted mouse selection."""
    try:
        import cv2
    except ImportError as exc:
        raise CameraUnavailableError(
            "OpenCV is not installed. Install 3d_camera/requirements.txt in the project environment."
        ) from exc

    clicked: list[tuple[int, int] | None] = [None]
    clicked_depth_mm: list[float | None] = [None]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[0] = (x, y)
            clicked_depth_mm[0] = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked[0] = None
            clicked_depth_mm[0] = None

    preview_config = deepcopy(config)
    depth_width, depth_height, depth_fps = CENTER_DEPTH_PROFILE
    preview_config.camera.depth_width = depth_width
    preview_config.camera.depth_height = depth_height
    preview_config.camera.fps = depth_fps
    preview_config.camera.fallback_profiles = [CENTER_DEPTH_PROFILE]
    preview_config.camera.enable_color = True
    preview_config.camera.enable_infrared_diagnostics = True
    source = RealSenseDepthSource(preview_config, align_color_to_depth=True)
    try:
        with source:
            # Keep mouse coordinates one-to-one with native depth pixels.
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)
            intrinsics: Intrinsics | None = None
            radii: tuple[float, float] | None = None
            safety_radii: tuple[float, float] | None = None
            view_mode = 0
            fallback_depth_mm = (
                config.camera.distance_to_rim_mm
                + config.tube.usable_height_mm
                - config.calibration.reference_target_thickness_mm
            )
            while True:
                burst = source.capture_burst(1)
                intrinsics = burst.intrinsics
                depth_m = burst.frames_m[0]
                if view_mode == 0:
                    image = _colorize_depth(depth_m, config, cv2)
                    view_label = "1/3 DEPTH MAP"
                elif view_mode == 1:
                    if burst.infrared_left is not None:
                        image = _infrared_view(
                            burst.infrared_left,
                            intrinsics.width,
                            intrinsics.height,
                            cv2,
                        )
                        view_label = "2/3 LEFT INFRARED IMAGER"
                    else:
                        image = _colorize_depth(depth_m, config, cv2)
                        view_label = "2/3 INFRARED UNAVAILABLE - showing depth map"
                elif burst.color_bgr is not None:
                    image = _normal_view(
                        burst.color_bgr,
                        intrinsics.width,
                        intrinsics.height,
                        cv2,
                    )
                    view_label = "3/3 NORMAL CAMERA (aligned to depth)"
                else:
                    image = _colorize_depth(depth_m, config, cv2)
                    view_label = "3/3 NORMAL CAMERA UNAVAILABLE - showing depth map"
                _draw_text(
                    image,
                    "Left-click centre | S switch view | Right-click reset | Enter accept | Esc cancel",
                    (12, 25),
                    cv2,
                )
                _draw_text(
                    image,
                    f"{view_label} | depth coordinates {intrinsics.width}x{intrinsics.height}",
                    (12, 50),
                    cv2,
                )
                if clicked[0] is not None:
                    if clicked_depth_mm[0] is None:
                        clicked_depth_mm[0] = local_depth_mm(depth_m, clicked[0])
                    projection_depth_mm = clicked_depth_mm[0] or fallback_depth_mm
                    radii = projected_radii_px(
                        diameter_mm,
                        projection_depth_mm,
                        intrinsics,
                    )
                    safety_radii = margin_radii_px(radii, margin_percent)
                    fits = ellipse_fits_frame(
                        clicked[0], safety_radii, intrinsics.width, intrinsics.height
                    )
                    _draw_overlay(image, clicked[0], radii, safety_radii, fits, cv2)
                    depth_source = (
                        "local depth" if clicked_depth_mm[0] is not None else "configured bottom fallback"
                    )
                    _draw_text(
                        image,
                        f"Circle {diameter_mm:g} mm at {projection_depth_mm:.1f} mm ({depth_source}); margin {margin_percent:g}%",
                        (12, 75),
                        cv2,
                    )
                    if not fits:
                        _draw_text(
                            image,
                            "Safety margin crosses the depth-frame edge; choose another centre.",
                            (12, 100),
                            cv2,
                            color=(0, 0, 255),
                        )
                else:
                    _draw_text(
                        image,
                        f"Circle {diameter_mm:g} mm; margin {margin_percent:g}%",
                        (12, 75),
                        cv2,
                    )
                cv2.imshow(WINDOW_NAME, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    raise PreviewCancelled("Preview cancelled; configuration was not changed.")
                if key in (10, 13, 32) and clicked[0] is not None:
                    if safety_radii is None:
                        continue
                    if ellipse_fits_frame(
                        clicked[0], safety_radii, intrinsics.width, intrinsics.height
                    ):
                        break
                if key in (ord("s"), ord("S")):
                    view_mode = (view_mode + 1) % 3
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    raise PreviewCancelled("Preview closed; configuration was not changed.")

            if intrinsics is None or clicked[0] is None:
                raise FrameCaptureError("No depth-frame centre was selected.")
            active = source.active_profile
            fps = active[2] if active is not None else config.camera.fps
            depth_width = active[0] if active is not None else intrinsics.width
            depth_height = active[1] if active is not None else intrinsics.height
            serial = _serial_from_source_name(burst.source_name)
            return PreviewSelection(
                center_x_px=float(clicked[0][0]),
                center_y_px=float(clicked[0][1]),
                frame_width_px=intrinsics.width,
                frame_height_px=intrinsics.height,
                fps=fps,
                depth_width=depth_width,
                depth_height=depth_height,
                camera_serial_number=serial,
            )
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass


def _colorize_depth(depth_m: np.ndarray, config: AppConfig, cv2: Any) -> np.ndarray:
    minimum = config.filters.min_distance_mm / 1000.0
    maximum = config.filters.max_distance_mm / 1000.0
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth_m[valid], minimum, maximum)
        normalized[valid] = np.rint(
            255.0 * (1.0 - (clipped - minimum) / max(maximum - minimum, 1e-9))
        ).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def _normal_view(
    color_bgr: np.ndarray,
    width: int,
    height: int,
    cv2: Any,
) -> np.ndarray:
    image = np.asarray(color_bgr).copy()
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def _infrared_view(
    infrared: np.ndarray,
    width: int,
    height: int,
    cv2: Any,
) -> np.ndarray:
    image = np.asarray(infrared)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def local_depth_mm(
    depth_m: np.ndarray,
    center: tuple[int, int],
    *,
    window_radius: int = 3,
) -> float | None:
    """Return robust local depth for circle projection, or None for an invalid patch."""
    if depth_m.ndim != 2:
        raise ValueError("Depth preview must be a two-dimensional image.")
    x, y = center
    height, width = depth_m.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    x0 = max(0, x - window_radius)
    x1 = min(width, x + window_radius + 1)
    y0 = max(0, y - window_radius)
    y1 = min(height, y + window_radius + 1)
    values = depth_m[y0:y1, x0:x1]
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid) * 1000.0)


def _draw_overlay(
    image: np.ndarray,
    center: tuple[int, int],
    radii: tuple[float, float],
    safety_radii: tuple[float, float],
    fits: bool,
    cv2: Any,
) -> None:
    circle_axes = tuple(max(1, int(round(value))) for value in radii)
    safety_axes = tuple(max(1, int(round(value))) for value in safety_radii)
    safety_color = (0, 255, 255) if fits else (0, 0, 255)
    cv2.ellipse(image, center, safety_axes, 0, 0, 360, (0, 0, 0), 5)
    cv2.ellipse(image, center, safety_axes, 0, 0, 360, safety_color, 2)
    cv2.ellipse(image, center, circle_axes, 0, 0, 360, (0, 0, 0), 5)
    cv2.ellipse(image, center, circle_axes, 0, 0, 360, (0, 255, 0), 2)
    cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 17, 2)


def _draw_text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    cv2: Any,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def _serial_from_source_name(source_name: str) -> str | None:
    prefix = "D405:"
    return source_name[len(prefix) :] if source_name.startswith(prefix) else None
