"""C920 preview/capture primitives with no model or inference imports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import platform
from typing import Any, Protocol

import cv2
import numpy as np

from .camera_discovery import CameraDiscoveryError, list_directshow_camera_names
from .domain import CameraSettings
from .errors import AcquisitionError


class PreviewEvent(StrEnum):
    NONE = "none"
    CAPTURE = "capture"
    CANCEL = "cancel"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class CameraSelection:
    device_index: int
    backend: str
    friendly_name: str | None


class CameraPreviewProtocol(Protocol):
    def open(self, settings: CameraSettings) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def show(self, frame: np.ndarray, status: str) -> None: ...
    def poll_event(self, wait_ms: int) -> PreviewEvent: ...
    def close(self) -> None: ...


class OpenCVCameraPreview:
    """OpenCV C920 preview. This module cannot load or run a YOLO model."""

    def __init__(self) -> None:
        self._capture: Any = None
        self._settings: CameraSettings | None = None
        self._selection: CameraSelection | None = None
        self._clicked = False

    def open(self, settings: CameraSettings) -> None:
        if self._capture is not None:
            raise AcquisitionError("Camera preview is already open.")
        selection = resolve_camera_selection(settings)
        backend = opencv_backend(selection.backend)
        if selection.friendly_name is not None:
            print(
                "Opening camera "
                f"{selection.friendly_name!r} at enumerated index "
                f"{selection.device_index} using backend {selection.backend!r}."
            )
        capture = (
            cv2.VideoCapture(selection.device_index)
            if backend is None
            else cv2.VideoCapture(selection.device_index, backend)
        )
        if not capture.isOpened():
            capture.release()
            description = (
                f"{selection.friendly_name!r} at enumerated index "
                f"{selection.device_index}"
                if selection.friendly_name is not None
                else f"index {selection.device_index}"
            )
            raise AcquisitionError(
                f"Could not open C920 camera {description} "
                f"with backend {selection.backend!r}."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        capture.set(cv2.CAP_PROP_FPS, settings.fps)
        self._capture = capture
        self._settings = settings
        self._selection = selection
        if selection.friendly_name is not None:
            print(
                "Selected camera "
                f"{selection.friendly_name!r} at enumerated index "
                f"{selection.device_index} using backend {selection.backend!r}."
            )
        cv2.namedWindow(settings.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(settings.window_name, self._mouse_callback)

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            raise AcquisitionError("Camera preview is not open.")
        ok, frame = self._capture.read()
        if not ok or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        return frame

    def show(self, frame: np.ndarray, status: str) -> None:
        if self._settings is None:
            raise AcquisitionError("Camera preview is not open.")
        displayed = cv2.flip(frame, 1) if self._settings.mirror_preview else frame.copy()
        cv2.putText(
            displayed,
            status,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self._settings.window_name, displayed)

    def poll_event(self, wait_ms: int) -> PreviewEvent:
        if self._settings is None:
            raise AcquisitionError("Camera preview is not open.")
        key = cv2.waitKey(wait_ms) & 0xFF
        if self._clicked:
            self._clicked = False
            return PreviewEvent.CAPTURE
        if key == ord(self._settings.capture_key):
            return PreviewEvent.CAPTURE
        if key == ord(self._settings.quit_key) or key == 27:
            return PreviewEvent.CANCEL
        try:
            if cv2.getWindowProperty(self._settings.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return PreviewEvent.CLOSED
        except cv2.error:
            return PreviewEvent.CLOSED
        return PreviewEvent.NONE

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._settings is not None:
            try:
                cv2.destroyWindow(self._settings.window_name)
            except cv2.error:
                pass
            self._settings = None
            self._selection = None

    def _mouse_callback(
        self,
        event: int,
        _x: int,
        _y: int,
        _flags: int,
        _data: Any,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked = True


def warm_up_preview(preview: CameraPreviewProtocol, settings: CameraSettings) -> str | None:
    """Warm the C920 while remaining responsive to cancel/window close."""
    for index in range(settings.warmup_frames):
        frame = preview.read()
        if frame is None:
            raise AcquisitionError(f"Camera read failed during warm-up frame {index + 1}.")
        preview.show(
            frame,
            f"Warming camera {index + 1}/{settings.warmup_frames} - "
            f"{settings.quit_key}: cancel",
        )
        event = preview.poll_event(settings.preview_wait_ms)
        if event in {PreviewEvent.CANCEL, PreviewEvent.CLOSED}:
            return "window_closed" if event is PreviewEvent.CLOSED else "operator_cancelled"
    return None


def opencv_backend(name: str) -> int | None:
    if name == "auto":
        return None
    values = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "v4l2": getattr(cv2, "CAP_V4L2", None),
    }
    value = values.get(name)
    if value is None:
        raise AcquisitionError(f"OpenCV backend {name!r} is unavailable on this platform.")
    if name in {"dshow", "msmf"} and platform.system() != "Windows":
        raise AcquisitionError(f"OpenCV backend {name!r} is Windows-only; use backend: auto.")
    return int(value)


def resolve_camera_selection(
    settings: CameraSettings,
    *,
    directshow_names: tuple[str, ...] | None = None,
) -> CameraSelection:
    """Resolve an optional Windows friendly-name match to a DirectShow index."""
    requested = settings.device_name_contains
    if requested is None:
        return CameraSelection(settings.device_index, settings.backend, None)
    needle = requested.strip()
    if not needle:
        raise AcquisitionError("camera.device_name_contains cannot be blank.")
    if platform.system() != "Windows":
        raise AcquisitionError(
            "camera.device_name_contains currently requires Windows DirectShow; "
            "set it to null and configure device_index on this platform."
        )
    if settings.backend == "v4l2":
        raise AcquisitionError(
            "Windows friendly-name camera selection cannot use the V4L2 backend."
        )
    try:
        names = (
            list_directshow_camera_names()
            if directshow_names is None
            else directshow_names
        )
    except CameraDiscoveryError as exc:
        raise AcquisitionError(str(exc)) from exc
    matches = [
        (index, name)
        for index, name in enumerate(names)
        if needle.casefold() in name.casefold()
    ]
    available = ", ".join(
        f"{index}: {name}" for index, name in enumerate(names)
    ) or "<none>"
    if not matches:
        raise AcquisitionError(
            f"No DirectShow camera name contains {needle!r}. "
            f"Available cameras: {available}."
        )
    if len(matches) > 1:
        matched = ", ".join(f"{index}: {name}" for index, name in matches)
        raise AcquisitionError(
            f"Camera name match {needle!r} is ambiguous: {matched}. "
            "Use a more specific device_name_contains value."
        )
    index, friendly_name = matches[0]
    return CameraSelection(index, settings.backend, friendly_name)
