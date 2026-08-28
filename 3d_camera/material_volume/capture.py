"""RealSense D405 capture with optional playback and native SDK filters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import AppConfig
from .errors import CameraUnavailableError, FrameCaptureError
from .models import DepthBurst, Intrinsics


class DepthSource(Protocol):
    def __enter__(self) -> "DepthSource": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    def capture_burst(self, frame_count: int, *, retain_raw: bool = False) -> DepthBurst: ...


class RealSenseDepthSource:
    """Own a RealSense pipeline and capture processed static depth bursts."""

    def __init__(
        self,
        config: AppConfig,
        bag_path: str | Path | None = None,
        *,
        align_color_to_depth: bool = False,
    ) -> None:
        self.config = config
        self.bag_path = Path(bag_path).expanduser().resolve() if bag_path else None
        self.align_color_to_depth = align_color_to_depth
        self.rs: Any = None
        self.pipeline: Any = None
        self.profile: Any = None
        self.depth_scale_m = 0.0
        self.filter_chain: _RealSenseFilterChain | None = None
        self.color_alignment: Any = None
        self.active_profile: tuple[int, int, int] | None = None
        self.reported_sensor_settings: dict[str, float | None] = {}
        self._started = False

    def __enter__(self) -> "RealSenseDepthSource":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraUnavailableError(
                "pyrealsense2 is not installed in this Python environment. Activate the project .venv first."
            ) from exc
        self.rs = rs

        if self.bag_path is not None and not self.bag_path.is_file():
            raise CameraUnavailableError(f"RealSense bag recording not found: {self.bag_path}")
        if self.bag_path is None:
            devices = rs.context().query_devices()
            if len(devices) == 0:
                raise CameraUnavailableError(
                    "No Intel RealSense camera was detected. Connect the D405 through USB 3, "
                    "close RealSense Viewer, and retry. Use --source synthetic for a hardware-free test."
                )

        requested = self.config.camera.requested_profile
        profiles = [requested] + [
            p for p in self.config.camera.fallback_profiles if p != requested
        ]
        errors: list[str] = []
        for width, height, fps in profiles:
            try:
                self._start_profile(width, height, fps)
                self.active_profile = (width, height, fps)
                break
            except RuntimeError as exc:
                errors.append(f"{width}x{height}@{fps}: {exc}")
                self.stop()
        if self.profile is None:
            raise CameraUnavailableError(
                "No configured RealSense stream profile could start. Attempts: " + " | ".join(errors)
            )

        try:
            device = self.profile.get_device()
            if self.bag_path is not None:
                playback = device.as_playback()
                playback.set_real_time(False)
            depth_sensor = device.first_depth_sensor()
            self.depth_scale_m = float(depth_sensor.get_depth_scale())
            self._configure_depth_sensor(depth_sensor)
            self.filter_chain = _RealSenseFilterChain(rs, self.config)
            self.color_alignment = (
                rs.align(rs.stream.depth)
                if self.align_color_to_depth and self.config.camera.enable_color
                else None
            )
            self._started = True

            # Warm auto-exposure and the stateful temporal filter.
            for _ in range(self.config.camera.warmup_frames):
                frames = self.pipeline.wait_for_frames(self.config.camera.frame_timeout_ms)
                depth = frames.get_depth_frame()
                if depth:
                    self.filter_chain.process(depth)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        self.pipeline = None
        self.profile = None
        self.filter_chain = None
        self.color_alignment = None
        self._started = False

    def capture_burst(self, frame_count: int, *, retain_raw: bool = False) -> DepthBurst:
        if not self._started or self.pipeline is None or self.filter_chain is None:
            raise CameraUnavailableError("RealSense source has not been started.")
        frames_m: list[np.ndarray] = []
        intrinsics: Intrinsics | None = None
        latest_color: np.ndarray | None = None
        latest_ir_left: np.ndarray | None = None
        latest_ir_right: np.ndarray | None = None
        raw_frames_z16: list[np.ndarray] | None = [] if retain_raw else None
        frame_timestamps_ms: list[float] = []

        for index in range(int(frame_count)):
            try:
                frameset = self.pipeline.wait_for_frames(self.config.camera.frame_timeout_ms)
            except RuntimeError as exc:
                raise FrameCaptureError(
                    f"Timed out while waiting for depth frame {index + 1}/{frame_count}: {exc}"
                ) from exc
            depth_frame = frameset.get_depth_frame()
            if not depth_frame:
                raise FrameCaptureError(f"Frame set {index + 1}/{frame_count} has no depth frame.")
            if raw_frames_z16 is not None:
                raw_frames_z16.append(np.asanyarray(depth_frame.get_data()).copy())
            frame_timestamps_ms.append(float(depth_frame.get_timestamp()))
            processed = self.filter_chain.process(depth_frame)
            if not processed:
                raise FrameCaptureError("A RealSense post-processing filter returned an empty depth frame.")

            profile = processed.profile.as_video_stream_profile()
            current_intrinsics = Intrinsics.from_realsense(profile.intrinsics)
            if intrinsics is None:
                intrinsics = current_intrinsics
            elif (intrinsics.width, intrinsics.height) != (
                current_intrinsics.width,
                current_intrinsics.height,
            ):
                raise FrameCaptureError("Processed depth resolution changed inside one burst.")
            raw = np.asanyarray(processed.get_data())
            depth_m = raw.astype(np.float32) * self.depth_scale_m
            depth_m[(raw == 0) | ~np.isfinite(depth_m)] = np.nan
            frames_m.append(depth_m)

            if self.config.camera.enable_color:
                color_frameset = frameset
                if self.color_alignment is not None:
                    try:
                        color_frameset = self.color_alignment.process(frameset)
                    except RuntimeError as exc:
                        raise FrameCaptureError(
                            f"Could not align the normal camera view to the depth frame: {exc}"
                        ) from exc
                color = color_frameset.get_color_frame()
                if color:
                    latest_color = np.asanyarray(color.get_data()).copy()
            if self.config.camera.enable_infrared_diagnostics:
                left = frameset.get_infrared_frame(1)
                if left:
                    latest_ir_left = np.asanyarray(left.get_data()).copy()

        if intrinsics is None:
            raise FrameCaptureError("No valid depth frames were captured.")
        serial = (
            "recording"
            if self.bag_path is not None
            else self.profile.get_device().get_info(self.rs.camera_info.serial_number)
        )
        return DepthBurst(
            frames_m=frames_m,
            intrinsics=intrinsics,
            depth_scale_m=self.depth_scale_m,
            color_bgr=latest_color,
            infrared_left=latest_ir_left,
            infrared_right=latest_ir_right,
            source_name=(f"bag:{self.bag_path.name}" if self.bag_path else f"D405:{serial}"),
            raw_frames_z16=raw_frames_z16,
            frame_timestamps_ms=frame_timestamps_ms,
            camera_serial_number=str(serial),
            active_stream_profile=self.active_profile,
            sensor_settings={
                "configured": {
                    "auto_exposure": self.config.camera.auto_exposure,
                    "manual_exposure": self.config.camera.manual_exposure,
                    "manual_gain": self.config.camera.manual_gain,
                    "visual_preset": self.config.camera.visual_preset,
                },
                "reported_by_sensor": self.reported_sensor_settings,
                "filter_configuration": {
                    name: getattr(self.config.filters, name)
                    for name in self.config.filters.__dataclass_fields__
                },
            },
        )

    def _start_profile(self, width: int, height: int, fps: int) -> None:
        rs = self.rs
        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.bag_path is not None:
            rs.config.enable_device_from_file(
                stream_config,
                str(self.bag_path),
                repeat_playback=False,
            )
        elif self.config.camera.serial_number:
            stream_config.enable_device(self.config.camera.serial_number)
        stream_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if self.config.camera.enable_color:
            # The D405 exposes color and depth through the same stereo module.
            # For an aligned preview, matching their profiles avoids unsupported
            # combinations such as depth 640x360 plus color 640x480.
            color_width = width if self.align_color_to_depth else self.config.camera.color_width
            color_height = height if self.align_color_to_depth else self.config.camera.color_height
            color_fps = fps if self.align_color_to_depth else self.config.camera.color_fps
            stream_config.enable_stream(
                rs.stream.color,
                color_width,
                color_height,
                rs.format.bgr8,
                color_fps,
            )
        if self.config.camera.enable_infrared_diagnostics:
            stream_config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        self.profile = pipeline.start(stream_config)
        self.pipeline = pipeline

    def _configure_depth_sensor(self, sensor: Any) -> None:
        rs = self.rs
        settings = self.config.camera
        if settings.visual_preset is not None:
            _set_option_if_supported(sensor, rs.option.visual_preset, float(settings.visual_preset))
        _set_option_if_supported(sensor, rs.option.enable_auto_exposure, 1.0 if settings.auto_exposure else 0.0)
        if not settings.auto_exposure and settings.manual_exposure is not None:
            _set_option_if_supported(sensor, rs.option.exposure, float(settings.manual_exposure))
        if not settings.auto_exposure and settings.manual_gain is not None:
            _set_option_if_supported(sensor, rs.option.gain, float(settings.manual_gain))
        self.reported_sensor_settings = {
            "visual_preset": _get_option_if_supported(sensor, rs.option.visual_preset),
            "auto_exposure": _get_option_if_supported(sensor, rs.option.enable_auto_exposure),
            "exposure": _get_option_if_supported(sensor, rs.option.exposure),
            "gain": _get_option_if_supported(sensor, rs.option.gain),
        }


class _RealSenseFilterChain:
    def __init__(self, rs: Any, config: AppConfig) -> None:
        self.rs = rs
        settings = config.filters
        self.settings = settings
        self.threshold = (
            rs.threshold_filter(settings.min_distance_mm / 1000.0, settings.max_distance_mm / 1000.0)
            if settings.threshold_enabled
            else None
        )
        self.decimation = rs.decimation_filter() if settings.decimation_enabled else None
        if self.decimation is not None:
            _set_option_if_supported(
                self.decimation,
                rs.option.filter_magnitude,
                float(settings.decimation_magnitude),
            )
        self.to_disparity = rs.disparity_transform(True) if settings.disparity_domain_enabled else None
        self.to_depth = rs.disparity_transform(False) if settings.disparity_domain_enabled else None
        self.spatial = rs.spatial_filter() if settings.spatial_enabled else None
        if self.spatial is not None:
            _set_option_if_supported(self.spatial, rs.option.filter_magnitude, settings.spatial_magnitude)
            _set_option_if_supported(self.spatial, rs.option.filter_smooth_alpha, settings.spatial_alpha)
            _set_option_if_supported(self.spatial, rs.option.filter_smooth_delta, settings.spatial_delta)
            _set_option_if_supported(self.spatial, rs.option.holes_fill, settings.spatial_holes_fill)
        self.temporal = rs.temporal_filter() if settings.temporal_enabled else None
        if self.temporal is not None:
            _set_option_if_supported(self.temporal, rs.option.filter_smooth_alpha, settings.temporal_alpha)
            _set_option_if_supported(self.temporal, rs.option.filter_smooth_delta, settings.temporal_delta)
            _set_option_if_supported(
                self.temporal,
                rs.option.holes_fill,
                settings.temporal_persistence,
            )
        self.hole_filling = rs.hole_filling_filter() if settings.hole_filling_enabled else None
        if self.hole_filling is not None:
            _set_option_if_supported(self.hole_filling, rs.option.holes_fill, settings.hole_filling_mode)

    def process(self, depth_frame: Any) -> Any:
        frame = depth_frame
        if self.threshold is not None:
            frame = self.threshold.process(frame)
        if self.decimation is not None:
            frame = self.decimation.process(frame)
        if self.to_disparity is not None:
            frame = self.to_disparity.process(frame)
        if self.spatial is not None:
            frame = self.spatial.process(frame)
        if self.temporal is not None:
            frame = self.temporal.process(frame)
        if self.to_depth is not None:
            frame = self.to_depth.process(frame)
        if self.hole_filling is not None:
            frame = self.hole_filling.process(frame)
        return frame.as_depth_frame()


def _set_option_if_supported(target: Any, option: Any, value: float) -> None:
    """Safely configure sensors and processing filters across SDK versions."""
    try:
        get_supported_options = getattr(target, "get_supported_options", None)

        if callable(get_supported_options):
            supported_options = get_supported_options()
            if option not in supported_options:
                return

        target.set_option(option, float(value))
    except (AttributeError, RuntimeError, TypeError):
        # Unsupported SDK or firmware options should not stop capture.
        return


def _get_option_if_supported(target: Any, option: Any) -> float | None:
    """Read back an effective sensor option without making capture depend on it."""
    try:
        supports = getattr(target, "supports", None)
        if callable(supports) and not supports(option):
            return None
        return float(target.get_option(option))
    except (AttributeError, RuntimeError, TypeError):
        return None
