"""Intel RealSense D405 interactive capability lab.

Run with a connected camera::

    python Tests/intel_camera.py

Run without hardware, using a synthetic moving depth scene::

    python Tests/intel_camera.py --demo

Only ``numpy``, ``opencv-python`` (or ``opencv-contrib-python``), and Intel's
``pyrealsense2`` wrapper are needed.  The latter is deliberately imported only
when real hardware is requested so that demo mode is always available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


APP_NAME = "D405 DEPTH LAB"
WINDOW_NAME = "Intel RealSense D405 - Depth Lab"
PANEL_W, PANEL_H = 424, 318
MARGIN, GAP, HEADER_H = 16, 12, 58
CANVAS_W = MARGIN * 2 + PANEL_W * 3 + GAP * 2
CANVAS_H = 820

BG = (16, 20, 28)
PANEL_BG = (23, 29, 39)
PANEL_BORDER = (53, 65, 82)
TEXT = (224, 231, 239)
MUTED = (139, 153, 172)
CYAN = (244, 196, 55)
GREEN = (112, 211, 113)
ORANGE = (55, 157, 255)
RED = (89, 93, 246)


@dataclass(frozen=True)
class Intrinsics:
    """Minimal pinhole camera model, independent of the RealSense package."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass
class FrameBundle:
    color: np.ndarray
    depth_m: np.ndarray
    intrinsics: Intrinsics
    frame_number: int
    timestamp_ms: float


class CameraError(RuntimeError):
    """A camera or SDK problem with a message suitable for the operator."""


class RealSenseBackend:
    """Small RealSense pipeline wrapper specialized for the close-range D405."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        serial: str | None,
        filters_enabled: bool,
    ) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise CameraError(
                "Intel RealSense Python support is not installed.\n\n"
                "Install it in this project's virtual environment:\n"
                r"  .\.venv\Scripts\python.exe -m pip install pyrealsense2"
                "\n\nThen reconnect the D405 and run this file again."
            ) from exc

        self.rs = rs
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.filters_enabled = filters_enabled
        self.pipeline: Any = None
        self.profile: Any = None
        self.align: Any = None
        self.depth_scale = 0.001
        self.frame_number = 0
        self.info: dict[str, str] = {}
        self._filters: list[Any] = []
        self._sensors: list[Any] = []

    @staticmethod
    def list_devices() -> list[dict[str, str]]:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise CameraError("pyrealsense2 is not installed.") from exc
        devices: list[dict[str, str]] = []
        for device in rs.context().query_devices():
            row: dict[str, str] = {}
            for key, label in (
                (rs.camera_info.name, "name"),
                (rs.camera_info.serial_number, "serial"),
                (rs.camera_info.firmware_version, "firmware"),
                (rs.camera_info.usb_type_descriptor, "usb"),
            ):
                try:
                    row[label] = device.get_info(key)
                except RuntimeError:
                    row[label] = "n/a"
            devices.append(row)
        return devices

    def _new_pipeline(self, explicit_streams: bool) -> tuple[Any, Any]:
        rs = self.rs
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        if explicit_streams:
            config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
        return pipeline, config

    def start(self) -> None:
        devices = self.list_devices()
        if not devices:
            raise CameraError(
                "No RealSense camera was found. Check the USB cable, close RealSense "
                "Viewer, and try again. Use --demo to explore without hardware."
            )

        first_error = ""
        try:
            self.pipeline, config = self._new_pipeline(explicit_streams=True)
            self.profile = self.pipeline.start(config)
        except RuntimeError as exc:
            first_error = str(exc)
            self.pipeline, config = self._new_pipeline(explicit_streams=False)
            try:
                self.profile = self.pipeline.start(config)
            except RuntimeError as fallback_exc:
                raise CameraError(
                    "The D405 was detected but streaming could not start. Close any "
                    "other camera application and check its firmware/USB connection.\n"
                    f"Requested profile error: {first_error}\n"
                    f"Default profile error: {fallback_exc}"
                ) from fallback_exc

        rs = self.rs
        device = self.profile.get_device()
        self._sensors = list(device.query_sensors())
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self.align = rs.align(rs.stream.color)
        self._filters = [
            rs.disparity_transform(True),
            rs.spatial_filter(),
            rs.temporal_filter(),
            rs.disparity_transform(False),
            rs.hole_filling_filter(),
        ]

        for key, label in (
            (rs.camera_info.name, "Product"),
            (rs.camera_info.serial_number, "Serial"),
            (rs.camera_info.firmware_version, "Firmware"),
            (rs.camera_info.usb_type_descriptor, "USB"),
        ):
            try:
                self.info[label] = device.get_info(key)
            except RuntimeError:
                self.info[label] = "n/a"
        self.info["Depth scale"] = f"{self.depth_scale * 1000:.4f} mm/unit"

        # Auto exposure is the safest starting point under changing lab lighting.
        for sensor in self._sensors:
            try:
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1.0)
            except RuntimeError:
                pass

        # Allow exposure to settle before measurements are shown.
        for _ in range(8):
            self.pipeline.wait_for_frames(3000)

    def read(self) -> FrameBundle:
        if self.pipeline is None:
            raise CameraError("Camera pipeline is not running.")
        try:
            frames = self.pipeline.wait_for_frames(5000)
            frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                raise CameraError("A depth or color frame was missing.")
            if self.filters_enabled:
                for depth_filter in self._filters:
                    depth_frame = depth_filter.process(depth_frame)
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth *= self.depth_scale
            color = np.asanyarray(color_frame.get_data()).copy()
            if color_frame.profile.format() == self.rs.format.rgb8:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            profile = depth_frame.profile.as_video_stream_profile()
            raw = profile.intrinsics
            intrinsics = Intrinsics(
                raw.width, raw.height, raw.fx, raw.fy, raw.ppx, raw.ppy
            )
            self.frame_number = int(depth_frame.get_frame_number())
            return FrameBundle(
                color=color,
                depth_m=depth,
                intrinsics=intrinsics,
                frame_number=self.frame_number,
                timestamp_ms=float(depth_frame.get_timestamp()),
            )
        except RuntimeError as exc:
            raise CameraError(f"Frame capture failed: {exc}") from exc

    def toggle_filters(self) -> bool:
        self.filters_enabled = not self.filters_enabled
        return self.filters_enabled

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None


class DemoBackend:
    """Animated synthetic scene for UI exploration and automated smoke tests."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_number = 0
        self.started = 0.0
        focal = width * 0.92
        self.intrinsics = Intrinsics(
            width, height, focal, focal, width / 2.0, height / 2.0
        )
        self.filters_enabled = True
        self.info = {
            "Product": "D405 simulated scene",
            "Serial": "DEMO-0405",
            "Firmware": "simulation",
            "USB": "virtual",
            "Depth scale": "0.1000 mm/unit",
        }

    def start(self) -> None:
        self.started = time.perf_counter()

    def read(self) -> FrameBundle:
        h, w = self.height, self.width
        t = time.perf_counter() - self.started
        yy, xx = np.mgrid[0:h, 0:w]

        # A tilted workbench plus two close objects creates a useful 3-D scene.
        depth = 0.43 + (xx - w / 2) * 0.00010 + (yy - h / 2) * 0.00016
        cx = int(w * (0.33 + 0.035 * math.sin(t * 0.7)))
        cy = int(h * (0.52 + 0.025 * math.cos(t * 0.9)))
        radius = max(18, int(min(w, h) * 0.14))
        sphere_mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
        sphere_shape = np.sqrt(
            np.maximum(0.0, radius**2 - (xx - cx) ** 2 - (yy - cy) ** 2)
        )
        depth = np.where(sphere_mask, 0.265 - sphere_shape * 0.00028, depth)

        x1, x2 = int(w * 0.57), int(w * 0.84)
        y1, y2 = int(h * 0.31), int(h * 0.69)
        block = (xx > x1) & (xx < x2) & (yy > y1) & (yy < y2)
        block_depth = 0.325 + (xx - x1) * 0.00006
        depth = np.where(block, block_depth, depth)

        rng = np.random.default_rng(self.frame_number // 3 + 405)
        noise = rng.normal(0.0, 0.00035, (h, w)).astype(np.float32)
        depth = (depth + noise).astype(np.float32)
        # Small invalid patches make the coverage meter meaningful.
        holes = ((xx * 13 + yy * 7 + self.frame_number) % 997 == 0)
        depth[holes] = 0.0

        color = np.empty((h, w, 3), dtype=np.uint8)
        color[:] = (44, 48, 54)
        grid = ((xx % 48 < 2) | (yy % 48 < 2))
        color[grid] = (58, 64, 71)
        color[sphere_mask] = (196, 108, 45)
        color[block] = (55, 164, 224)
        cv2.circle(color, (cx, cy), radius, (238, 158, 72), 2, cv2.LINE_AA)
        cv2.rectangle(color, (x1, y1), (x2, y2), (88, 205, 255), 2)
        cv2.putText(
            color,
            "SYNTHETIC CLOSE-RANGE SCENE",
            (18, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (215, 220, 225),
            1,
            cv2.LINE_AA,
        )
        self.frame_number += 1
        return FrameBundle(
            color,
            depth,
            self.intrinsics,
            self.frame_number,
            time.perf_counter() * 1000.0,
        )

    def toggle_filters(self) -> bool:
        self.filters_enabled = not self.filters_enabled
        return self.filters_enabled

    def stop(self) -> None:
        pass


def put_text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float = 0.48,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox an image without changing its aspect ratio."""
    ih, iw = image.shape[:2]
    scale = min(width / iw, height / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    output = np.full((height, width, 3), PANEL_BG, np.uint8)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    output[oy : oy + nh, ox : ox + nw] = resized
    return output


def colorize_depth(
    depth_m: np.ndarray, near_m: float, far_m: float, colormap: int
) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    clipped = np.clip(depth_m, near_m, far_m)
    # Near objects are warm and distant objects are cool.
    normalized = ((far_m - clipped) / max(1e-6, far_m - near_m) * 255).astype(
        np.uint8
    )
    colored = cv2.applyColorMap(normalized, colormap)
    colored[~valid] = 0
    return colored


def robust_depth(depth: np.ndarray, x: int, y: int, radius: int = 3) -> float:
    """Median depth around a pixel, resistant to stereo holes and outliers."""
    h, w = depth.shape
    x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
    patch = depth[
        max(0, y - radius) : min(h, y + radius + 1),
        max(0, x - radius) : min(w, x + radius + 1),
    ]
    values = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(values)) if values.size else 0.0


def deproject(x: float, y: float, z: float, intr: Intrinsics) -> np.ndarray:
    return np.array(
        [(x - intr.ppx) / intr.fx * z, (y - intr.ppy) / intr.fy * z, z],
        dtype=np.float32,
    )


def panel_rect(index: int) -> tuple[int, int, int, int]:
    x = MARGIN + index * (PANEL_W + GAP)
    return x, HEADER_H + 18, PANEL_W, PANEL_H


def draw_panel(
    canvas: np.ndarray, index: int, title: str, image: np.ndarray
) -> tuple[int, int, int, int]:
    x, y, w, h = panel_rect(index)
    cv2.rectangle(canvas, (x - 1, y - 25), (x + w, y + h), PANEL_BORDER, 1)
    put_text(canvas, title.upper(), (x + 8, y - 8), 0.43, MUTED, 1)
    canvas[y : y + h, x : x + w] = fit_image(image, w, h)
    return x, y, w, h


def source_to_panel(
    x: int, y: int, source_shape: tuple[int, int], rect: tuple[int, int, int, int]
) -> tuple[int, int]:
    sh, sw = source_shape
    rx, ry, rw, rh = rect
    scale = min(rw / sw, rh / sh)
    ox, oy = rx + (rw - sw * scale) / 2, ry + (rh - sh * scale) / 2
    return int(ox + x * scale), int(oy + y * scale)


def panel_to_source(
    px: int, py: int, source_shape: tuple[int, int], rect: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    sh, sw = source_shape
    rx, ry, rw, rh = rect
    scale = min(rw / sw, rh / sh)
    ox, oy = rx + (rw - sw * scale) / 2, ry + (rh - sh * scale) / 2
    if not (ox <= px < ox + sw * scale and oy <= py < oy + sh * scale):
        return None
    return (
        int(np.clip((px - ox) / scale, 0, sw - 1)),
        int(np.clip((py - oy) / scale, 0, sh - 1)),
    )


def render_point_cloud(
    color: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    zoom: float,
    near_m: float,
    far_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render an RGB point cloud and return the sampled XYZ/color arrays."""
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[:] = (10, 13, 19)
    # Roughly 75k samples at VGA keeps interaction fluid while retaining detail.
    stride = max(1, math.ceil(max(depth.shape) / 320))
    z = depth[::stride, ::stride]
    yy, xx = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    valid = np.isfinite(z) & (z >= near_m) & (z <= far_m)
    if not np.any(valid):
        put_text(output, "NO VALID DEPTH", (width // 2 - 65, height // 2), 0.55, RED)
        return output, np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    zv = z[valid]
    xv = (xx[valid] - intr.ppx) / intr.fx * zv
    yv = (yy[valid] - intr.ppy) / intr.fy * zv
    points = np.column_stack((xv, yv, zv)).astype(np.float32)
    colors = color[::stride, ::stride][valid]

    pivot = float(np.median(zv))
    x = xv
    y = yv
    zc = zv - pivot
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    xr = cy * x + sy * zc
    zr = -sy * x + cy * zc
    yr = cp * y - sp * zr
    zr = sp * y + cp * zr + pivot
    focal = min(width, height) * 1.25 * zoom
    good = zr > 0.01
    u = np.rint(width / 2 + xr[good] / zr[good] * focal).astype(np.int32)
    v = np.rint(height / 2 + yr[good] / zr[good] * focal).astype(np.int32)
    projected_z = zr[good]
    projected_colors = colors[good]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[inside], v[inside]
    projected_z = projected_z[inside]
    projected_colors = projected_colors[inside]
    # Painter's algorithm: near points overwrite far points.
    order = np.argsort(projected_z)[::-1]
    output[v[order], u[order]] = projected_colors[order]
    output = cv2.dilate(output, np.ones((2, 2), np.uint8))

    cv2.line(output, (width // 2 - 10, height // 2), (width // 2 + 10, height // 2), (70, 80, 92), 1)
    cv2.line(output, (width // 2, height // 2 - 10), (width // 2, height // 2 + 10), (70, 80, 92), 1)
    return output, points, colors


def clipped_roi(
    roi: tuple[int, int, int, int] | None, shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    h, w = shape
    x1, y1, x2, y2 = roi
    x1, x2 = sorted((int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))))
    y1, y2 = sorted((int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))))
    return (x1, y1, x2, y2) if x2 - x1 >= 3 and y2 - y1 >= 3 else None


def roi_metrics(
    depth: np.ndarray,
    intr: Intrinsics,
    roi: tuple[int, int, int, int] | None,
) -> dict[str, float]:
    h, w = depth.shape
    if roi is None:
        half = max(8, min(h, w) // 20)
        cx, cy = w // 2, h // 2
        roi = (cx - half, cy - half, cx + half, cy + half)
    x1, y1, x2, y2 = roi
    patch = depth[y1 : y2 + 1, x1 : x2 + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if not valid.size:
        return {"count": 0.0}
    median = float(np.median(valid))
    p10, p90 = np.percentile(valid, [10, 90])
    width_mm = abs(x2 - x1) * median / intr.fx * 1000.0
    height_mm = abs(y2 - y1) * median / intr.fy * 1000.0
    return {
        "count": float(valid.size),
        "median": median,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "spread_mm": float((p90 - p10) * 1000.0),
        "width_mm": width_mm,
        "height_mm": height_mm,
        "coverage": float(valid.size / patch.size * 100.0),
    }


def draw_analytics(
    canvas: np.ndarray,
    frame: FrameBundle,
    metrics: dict[str, float],
    cursor: tuple[int, int],
    fps: float,
    device_info: dict[str, str],
    filters_enabled: bool,
    is_recording: bool,
) -> None:
    top = HEADER_H + 18 + PANEL_H + 42
    bottom = CANVAS_H - 45
    cv2.rectangle(canvas, (MARGIN, top), (CANVAS_W - MARGIN, bottom), PANEL_BG, -1)
    cv2.rectangle(canvas, (MARGIN, top), (CANVAS_W - MARGIN, bottom), PANEL_BORDER, 1)

    # Left: live measurements and camera calibration.
    x = MARGIN + 18
    put_text(canvas, "LIVE METROLOGY", (x, top + 27), 0.55, CYAN, 1)
    probe_z = robust_depth(frame.depth_m, cursor[0], cursor[1])
    probe = deproject(cursor[0], cursor[1], probe_z, frame.intrinsics)
    values = [
        ("PROBE DISTANCE", f"{probe_z * 1000:8.2f} mm" if probe_z else "       n/a"),
        ("X / Y / Z", f"{probe[0]*1000:+.1f} / {probe[1]*1000:+.1f} / {probe[2]*1000:.1f} mm" if probe_z else "n/a"),
        ("ROI SIZE (approx.)", f"{metrics.get('width_mm', 0):.1f} x {metrics.get('height_mm', 0):.1f} mm"),
        ("ROI DEPTH SPREAD", f"{metrics.get('spread_mm', 0):.2f} mm (P10-P90)"),
        ("ROI VALID PIXELS", f"{metrics.get('coverage', 0):.1f}%"),
        ("CALIBRATED FOCAL", f"fx {frame.intrinsics.fx:.1f}  fy {frame.intrinsics.fy:.1f}"),
    ]
    for row, (label, value) in enumerate(values):
        yy = top + 58 + row * 34
        put_text(canvas, label, (x, yy), 0.37, MUTED)
        put_text(canvas, value, (x + 150, yy), 0.46, TEXT)

    # Middle: depth distribution and a horizontal surface profile.
    graph_x = 440
    graph_w = 430
    put_text(canvas, "DEPTH DISTRIBUTION + CENTERLINE PROFILE", (graph_x, top + 27), 0.47, CYAN)
    valid = frame.depth_m[np.isfinite(frame.depth_m) & (frame.depth_m > 0)]
    gx1, gy1, gx2, gy2 = graph_x, top + 47, graph_x + graph_w, top + 138
    cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (15, 19, 26), -1)
    if valid.size:
        lo, hi = np.percentile(valid, [1, 99])
        if hi > lo:
            hist, _ = np.histogram(valid, bins=80, range=(lo, hi))
            hist = hist.astype(np.float32) / max(1, hist.max())
            for i, amount in enumerate(hist):
                bx = int(gx1 + i / len(hist) * graph_w)
                bh = int(amount * (gy2 - gy1 - 8))
                cv2.line(canvas, (bx, gy2 - 3), (bx, gy2 - 3 - bh), CYAN, 2)
            put_text(canvas, f"{lo*1000:.0f} mm", (gx1, gy2 + 17), 0.35, MUTED)
            put_text(canvas, f"{hi*1000:.0f} mm", (gx2 - 58, gy2 + 17), 0.35, MUTED)

    py1, py2 = top + 176, top + 258
    cv2.rectangle(canvas, (gx1, py1), (gx2, py2), (15, 19, 26), -1)
    row = frame.depth_m[int(np.clip(cursor[1], 0, frame.depth_m.shape[0] - 1))]
    xs = np.linspace(0, len(row) - 1, graph_w).astype(np.int32)
    sampled = row[xs]
    good = sampled > 0
    if np.any(good):
        lo, hi = np.percentile(sampled[good], [1, 99])
        span = max(0.01, hi - lo)
        plot_y = py2 - 5 - np.clip((sampled - lo) / span, 0, 1) * (py2 - py1 - 10)
        last: tuple[int, int] | None = None
        for i in range(graph_w):
            if good[i]:
                point = (gx1 + i, int(plot_y[i]))
                if last is not None:
                    cv2.line(canvas, last, point, GREEN, 1, cv2.LINE_AA)
                last = point
            else:
                last = None
        put_text(canvas, f"scanline y={cursor[1]} | range {lo*1000:.1f}-{hi*1000:.1f} mm", (gx1, py2 + 17), 0.35, MUTED)

    # Right: health telemetry and compact controls.
    x = 900
    put_text(canvas, "CAMERA HEALTH", (x, top + 27), 0.55, CYAN)
    coverage = 100.0 * valid.size / frame.depth_m.size
    health = [
        ("DEVICE", device_info.get("Product", "unknown")),
        ("SERIAL", device_info.get("Serial", "n/a")),
        ("FIRMWARE", device_info.get("Firmware", "n/a")),
        ("USB", device_info.get("USB", "n/a")),
        ("DEPTH COVERAGE", f"{coverage:.1f}%"),
        ("STREAM", f"{frame.depth_m.shape[1]}x{frame.depth_m.shape[0]}  {fps:.1f} FPS"),
        ("FILTER CHAIN", "ON" if filters_enabled else "OFF"),
    ]
    for row_index, (label, value) in enumerate(health):
        yy = top + 57 + row_index * 25
        put_text(canvas, label, (x, yy), 0.35, MUTED)
        put_text(canvas, str(value)[:34], (x + 112, yy), 0.40, TEXT)

    put_text(canvas, "CONTROLS", (x, top + 244), 0.43, CYAN)
    put_text(canvas, "drag RGB: measure   drag cloud: orbit   wheel: zoom", (x, top + 266), 0.34, MUTED)
    put_text(canvas, "F filters   C colors   M fusion   S snapshot   P PLY", (x, top + 285), 0.34, MUTED)
    put_text(canvas, "V video   R reset view   SPACE freeze   Q quit", (x, top + 304), 0.34, MUTED)
    if is_recording:
        cv2.circle(canvas, (CANVAS_W - 33, 31), 7, RED, -1)
        put_text(canvas, "REC", (CANVAS_W - 78, 36), 0.42, RED, 1)


def save_ply(path: Path, points: np.ndarray, colors_bgr: np.ndarray) -> None:
    """Write a compact binary PLY that opens in CloudCompare, MeshLab, or Blender."""
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[valid]
    colors_rgb = colors_bgr[valid][:, ::-1]
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors_rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


class DepthLab:
    """Rendering, interaction, capture, and measurement state."""

    def __init__(
        self,
        backend: RealSenseBackend | DemoBackend,
        near_m: float,
        far_m: float,
        output_dir: Path,
        headless: bool,
    ) -> None:
        self.backend = backend
        self.near_m = near_m
        self.far_m = far_m
        self.output_dir = output_dir
        self.headless = headless
        self.cursor = (backend.width // 2, backend.height // 2)
        self.roi: tuple[int, int, int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.cloud_drag: tuple[int, int] | None = None
        self.yaw, self.pitch, self.zoom = -0.22, -0.12, 1.0
        self.frozen = False
        self.fusion_mode = 0
        self.colormaps = [cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_INFERNO]
        self.colormap_index = 0
        self.last_frame: FrameBundle | None = None
        self.last_canvas: np.ndarray | None = None
        self.last_points = np.empty((0, 3), np.float32)
        self.last_colors = np.empty((0, 3), np.uint8)
        self.status = "Ready"
        self.status_until = 0.0
        self.video_writer: cv2.VideoWriter | None = None
        self.video_path: Path | None = None
        self.fps_value = 0.0
        self._last_tick = time.perf_counter()
        self._rgb_rect = panel_rect(0)
        self._cloud_rect = panel_rect(2)

    def set_status(self, message: str, seconds: float = 2.5) -> None:
        self.status = message
        self.status_until = time.perf_counter() + seconds

    def _mouse(self, event: int, x: int, y: int, flags: int, _: Any) -> None:
        if self.last_frame is None:
            return
        shape = self.last_frame.depth_m.shape
        rgb_point = panel_to_source(x, y, shape, self._rgb_rect)
        cx, cy, cw, ch = self._cloud_rect
        in_cloud = cx <= x < cx + cw and cy <= y < cy + ch

        if rgb_point is not None:
            self.cursor = rgb_point
        if event == cv2.EVENT_LBUTTONDOWN:
            if rgb_point is not None:
                self.drag_start = rgb_point
                self.roi = None
            elif in_cloud:
                self.cloud_drag = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None and rgb_point is not None:
                self.roi = (*self.drag_start, *rgb_point)
            if self.cloud_drag is not None:
                dx, dy = x - self.cloud_drag[0], y - self.cloud_drag[1]
                self.yaw += dx * 0.008
                self.pitch = float(np.clip(self.pitch + dy * 0.008, -1.45, 1.45))
                self.cloud_drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_start is not None and rgb_point is not None:
                self.roi = clipped_roi((*self.drag_start, *rgb_point), shape)
            self.drag_start = None
            self.cloud_drag = None
        elif event == cv2.EVENT_RBUTTONDOWN and rgb_point is not None:
            self.roi = None
            self.set_status("Measurement ROI cleared")
        elif event == cv2.EVENT_MOUSEWHEEL and in_cloud:
            self.zoom = float(np.clip(self.zoom * (1.12 if flags > 0 else 0.89), 0.35, 3.0))

    def _decorate_rgb(self, frame: FrameBundle, depth_color: np.ndarray) -> np.ndarray:
        view = frame.color.copy()
        if self.fusion_mode == 1:
            view = cv2.addWeighted(view, 0.52, depth_color, 0.48, 0)
        elif self.fusion_mode == 2:
            valid = (frame.depth_m >= self.near_m) & (frame.depth_m <= self.far_m)
            dimmed = (view.astype(np.float32) * 0.20).astype(np.uint8)
            dimmed[valid] = view[valid]
            edges = cv2.Canny((valid.astype(np.uint8) * 255), 50, 150)
            dimmed[edges > 0] = (60, 230, 255)
            view = dimmed

        x, y = self.cursor
        cv2.drawMarker(view, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
        z = robust_depth(frame.depth_m, x, y)
        label = f"{z * 1000:.2f} mm" if z else "NO DEPTH"
        lx, ly = min(x + 12, view.shape[1] - 130), max(22, y - 12)
        cv2.rectangle(view, (lx - 4, ly - 17), (lx + 112, ly + 5), (12, 16, 22), -1)
        put_text(view, label, (lx, ly), 0.48, CYAN, 1)
        roi = clipped_roi(self.roi, frame.depth_m.shape)
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(view, (x1, y1), (x2, y2), GREEN, 2, cv2.LINE_AA)
            m = roi_metrics(frame.depth_m, frame.intrinsics, roi)
            label = f"{m.get('width_mm', 0):.1f} x {m.get('height_mm', 0):.1f} mm"
            put_text(view, label, (x1, max(18, y1 - 7)), 0.47, GREEN, 1)
        return view

    def render(self, frame: FrameBundle) -> np.ndarray:
        now = time.perf_counter()
        elapsed = now - self._last_tick
        instant_fps = self.backend.fps if elapsed < 0.005 else 1.0 / elapsed
        self._last_tick = now
        self.fps_value = instant_fps if self.fps_value == 0 else self.fps_value * 0.9 + instant_fps * 0.1

        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
        put_text(canvas, APP_NAME, (MARGIN, 34), 0.78, TEXT, 2)
        put_text(canvas, "calibrated close-range vision / metrology / 3D", (244, 34), 0.46, MUTED)
        put_text(canvas, f"FRAME {frame.frame_number}", (CANVAS_W - 170, 34), 0.40, MUTED)

        depth_color = colorize_depth(frame.depth_m, self.near_m, self.far_m, self.colormaps[self.colormap_index])
        rgb_view = self._decorate_rgb(frame, depth_color)
        cloud, self.last_points, self.last_colors = render_point_cloud(
            frame.color,
            frame.depth_m,
            frame.intrinsics,
            PANEL_W,
            PANEL_H,
            self.yaw,
            self.pitch,
            self.zoom,
            self.near_m,
            self.far_m,
        )
        fusion_names = ["CALIBRATED RGB", "RGB + DEPTH FUSION", "DEPTH RANGE ISOLATION"]
        self._rgb_rect = draw_panel(canvas, 0, fusion_names[self.fusion_mode], rgb_view)
        draw_panel(canvas, 1, f"DEPTH HEATMAP  {self.near_m*1000:.0f}-{self.far_m*1000:.0f} MM", depth_color)
        self._cloud_rect = draw_panel(canvas, 2, "LIVE RGB POINT CLOUD", cloud)

        # Mirror the crosshair into the depth panel.
        dx, dy = source_to_panel(self.cursor[0], self.cursor[1], frame.depth_m.shape, panel_rect(1))
        cv2.drawMarker(canvas, (dx, dy), (255, 255, 255), cv2.MARKER_CROSS, 15, 1, cv2.LINE_AA)

        metrics = roi_metrics(frame.depth_m, frame.intrinsics, clipped_roi(self.roi, frame.depth_m.shape))
        draw_analytics(
            canvas,
            frame,
            metrics,
            self.cursor,
            self.fps_value,
            self.backend.info,
            self.backend.filters_enabled,
            self.video_writer is not None,
        )
        if self.frozen:
            cv2.rectangle(canvas, (CANVAS_W // 2 - 54, 12), (CANVAS_W // 2 + 54, 43), ORANGE, -1)
            put_text(canvas, "FROZEN", (CANVAS_W // 2 - 37, 34), 0.48, (10, 15, 20), 1)
        if time.perf_counter() < self.status_until:
            put_text(canvas, self.status, (MARGIN, CANVAS_H - 15), 0.43, GREEN)
        else:
            put_text(canvas, "Q quit  |  H controls  |  move pointer over RGB to probe depth", (MARGIN, CANVAS_H - 15), 0.40, MUTED)
        return canvas

    def _capture_folder(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        folder = self.output_dir / f"capture_{stamp}"
        folder.mkdir(parents=True, exist_ok=False)
        return folder

    def save_snapshot(self) -> None:
        if self.last_frame is None or self.last_canvas is None:
            return
        folder = self._capture_folder()
        frame = self.last_frame
        depth_mm = np.clip(frame.depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(folder / "color.png"), frame.color)
        cv2.imwrite(str(folder / "depth_mm.png"), depth_mm)
        cv2.imwrite(str(folder / "dashboard.png"), self.last_canvas)
        np.save(folder / "depth_m.npy", frame.depth_m)
        metadata = {
            "frame_number": frame.frame_number,
            "timestamp_ms": frame.timestamp_ms,
            "intrinsics": frame.intrinsics.__dict__,
            "device": self.backend.info,
            "depth_range_m": [self.near_m, self.far_m],
            "probe_pixel": self.cursor,
            "probe_depth_m": robust_depth(frame.depth_m, *self.cursor),
            "roi": clipped_roi(self.roi, frame.depth_m.shape),
            "roi_metrics": roi_metrics(frame.depth_m, frame.intrinsics, clipped_roi(self.roi, frame.depth_m.shape)),
        }
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.set_status(f"Snapshot bundle saved: {folder}", 4.0)

    def export_ply(self) -> None:
        if not len(self.last_points):
            self.set_status("No valid points to export", 3.0)
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"pointcloud_{stamp}.ply"
        save_ply(path, self.last_points, self.last_colors)
        self.set_status(f"3D point cloud saved: {path}", 4.0)

    def toggle_video(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.set_status(f"Dashboard video saved: {self.video_path}", 4.0)
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.output_dir / f"depth_lab_{stamp}.mp4"
        writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(10.0, self.fps_value), (CANVAS_W, CANVAS_H)
        )
        if not writer.isOpened():
            self.set_status("Video encoder could not be opened", 3.0)
            return
        self.video_writer = writer
        self.set_status("Dashboard recording started", 2.0)

    def handle_key(self, key: int) -> bool:
        if key in (27, ord("q"), ord("Q")):
            return False
        if key == ord(" "):
            self.frozen = not self.frozen
            self.set_status("Frame frozen" if self.frozen else "Live capture resumed")
        elif key in (ord("f"), ord("F")):
            enabled = self.backend.toggle_filters()
            self.set_status(f"Depth post-processing filters {'enabled' if enabled else 'disabled'}")
        elif key in (ord("c"), ord("C")):
            self.colormap_index = (self.colormap_index + 1) % len(self.colormaps)
            self.set_status("Depth color palette changed")
        elif key in (ord("m"), ord("M")):
            self.fusion_mode = (self.fusion_mode + 1) % 3
            self.set_status("RGB/depth visualization mode changed")
        elif key in (ord("r"), ord("R")):
            self.yaw, self.pitch, self.zoom = -0.22, -0.12, 1.0
            self.set_status("3D view reset")
        elif key in (ord("s"), ord("S")):
            self.save_snapshot()
        elif key in (ord("p"), ord("P")):
            self.export_ply()
        elif key in (ord("v"), ord("V")):
            self.toggle_video()
        elif key in (ord("h"), ord("H")):
            self.set_status("Drag RGB=ROI | right-click=clear | drag cloud=orbit | wheel=zoom | F/C/M/S/P/V/R/SPACE/Q", 8.0)
        return True

    def run(self, max_frames: int = 0) -> None:
        self.backend.start()
        if not self.headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
            cv2.setMouseCallback(WINDOW_NAME, self._mouse)
        count = 0
        try:
            while True:
                if not self.frozen or self.last_frame is None:
                    self.last_frame = self.backend.read()
                self.last_canvas = self.render(self.last_frame)
                if self.video_writer is not None:
                    self.video_writer.write(self.last_canvas)
                count += 1
                if max_frames and count >= max_frames:
                    break
                if self.headless:
                    continue
                cv2.imshow(WINDOW_NAME, self.last_canvas)
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
            self.backend.stop()
            if not self.headless:
                cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive close-range RGB-D, metrology, and point-cloud lab for the RealSense D405."
    )
    parser.add_argument("--demo", action="store_true", help="use a synthetic scene instead of camera hardware")
    parser.add_argument("--list", action="store_true", help="list connected RealSense devices and exit")
    parser.add_argument("--serial", help="use a specific camera serial number")
    parser.add_argument("--width", type=int, default=640, help="requested stream width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="requested stream height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="requested frame rate (default: 30)")
    parser.add_argument("--near", type=float, default=0.07, help="visualization near plane in metres")
    parser.add_argument("--far", type=float, default=0.70, help="visualization far plane in metres")
    parser.add_argument("--no-filters", action="store_true", help="start with depth post-processing disabled")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "d405_depth_lab", help="capture directory")
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.near <= 0 or args.far <= args.near:
        print("Error: --near must be positive and --far must be greater than --near.", file=sys.stderr)
        return 2
    try:
        if args.list:
            devices = RealSenseBackend.list_devices()
            if not devices:
                print("No RealSense devices found.")
                return 1
            for index, device in enumerate(devices, 1):
                print(f"[{index}] {device['name']} | serial {device['serial']} | firmware {device['firmware']} | USB {device['usb']}")
            return 0

        backend: RealSenseBackend | DemoBackend
        if args.demo:
            backend = DemoBackend(args.width, args.height, args.fps)
        else:
            backend = RealSenseBackend(
                args.width, args.height, args.fps, args.serial, not args.no_filters
            )
        app = DepthLab(backend, args.near, args.far, args.output, args.headless)
        app.run(max_frames=args.frames)
        return 0
    except CameraError as exc:
        print(f"\n{APP_NAME} could not start:\n\n{exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
