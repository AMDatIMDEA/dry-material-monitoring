"""Adapter around the existing prepared D405 method; no numerical changes."""

from __future__ import annotations

from typing import Any

from material_volume.capture import RealSenseDepthSource
from material_volume.config import AppConfig
from material_volume.models import CalibrationData
from material_volume.session import (
    CapturedDepthBurst,
    DepthMeasurementOutcome,
    DepthMeasurementRequest,
    DepthMeasurementReservation,
    PreparedDepthMethod,
)


class PreparedDepthController:
    """Own one opened/warmed RealSense source and delegate the tested method API."""

    def __init__(
        self,
        config: AppConfig,
        calibration: CalibrationData,
        *,
        source: Any | None = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.source = source or RealSenseDepthSource(config)
        self.prepared: PreparedDepthMethod | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        start = getattr(self.source, "start", None)
        if callable(start):
            start()
        else:
            self.source.__enter__()
        self.prepared = PreparedDepthMethod(self.config, self.calibration, self.source)
        self._started = True

    def reserve(self, request: DepthMeasurementRequest) -> DepthMeasurementReservation:
        if self.prepared is None:
            raise RuntimeError("Depth controller has not been started/warmed.")
        return self.prepared.reserve(request)

    def capture(self, request: DepthMeasurementRequest, *, now_utc: Any = None) -> CapturedDepthBurst:
        if self.prepared is None:
            raise RuntimeError("Depth controller has not been started/warmed.")
        return self.prepared.capture(request, now_utc=now_utc)

    def process(
        self,
        captured: CapturedDepthBurst,
        request: DepthMeasurementRequest,
        reservation: DepthMeasurementReservation,
        *,
        now_utc: Any = None,
    ) -> DepthMeasurementOutcome:
        if self.prepared is None:
            raise RuntimeError("Depth controller has not been started/warmed.")
        return self.prepared.process(
            captured,
            request,
            now_utc=now_utc,
            reservation=reservation,
            manage_capture_manifest=False,
        )

    def close(self) -> None:
        if not self._started:
            return
        stop = getattr(self.source, "stop", None)
        if callable(stop):
            stop()
        else:
            self.source.__exit__(None, None, None)
        self._started = False
        self.prepared = None
