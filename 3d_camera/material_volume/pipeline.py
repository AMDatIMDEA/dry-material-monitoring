"""High-level calibration, measurement, and refill-warning state."""

from __future__ import annotations

from .calibration import calibrate_empty_burst
from .config import AppConfig
from .errors import CalibrationError
from .models import CalibrationData, DepthBurst, VolumeEstimate
from .processing import fuse_depth_burst
from .volume import reconstruct_volume


class MaterialVolumePipeline:
    def __init__(self, config: AppConfig, calibration: CalibrationData) -> None:
        self.config = config
        self.calibration = calibration

    @staticmethod
    def calibrate(burst: DepthBurst, config: AppConfig) -> CalibrationData:
        return calibrate_empty_burst(burst, config)

    def measure(self, burst: DepthBurst) -> VolumeEstimate:
        compatibility = self.calibration.compatibility_errors(
            burst.intrinsics,
            self.config.tube,
            center_x_px=self.config.detection_roi.center_x_px,
            center_y_px=self.config.detection_roi.center_y_px,
        )
        if compatibility:
            raise CalibrationError(
                "Calibration is not compatible with this capture: " + " ".join(compatibility)
            )
        fused = fuse_depth_burst(burst, self.config.fusion, self.config.filters)
        return reconstruct_volume(
            fused=fused,
            intrinsics=burst.intrinsics,
            calibration=self.calibration,
            config=self.config,
            source_name=burst.source_name,
            color_bgr=burst.color_bgr,
        )


class RefillWarningLatch:
    """Confirm low level repeatedly and add hysteresis before clearing the alarm."""

    def __init__(self, config: AppConfig) -> None:
        self.settings = config.decision
        self._low_count = 0
        self.active = False

    def update(self, estimate: VolumeEstimate) -> str:
        result = estimate.result
        if not result.quality.valid:
            result.warning_state = "INVALID_MEASUREMENT"
            return result.warning_state

        if result.fill_percent < self.settings.refill_threshold_percent:
            self._low_count += 1
            if self._low_count >= self.settings.confirmations_required:
                self.active = True
                result.warning_state = "REFILL_WARNING_ACTIVE"
            else:
                result.warning_state = (
                    f"LOW_LEVEL_PENDING_{self._low_count}_OF_{self.settings.confirmations_required}"
                )
        elif result.fill_percent >= self.settings.clear_threshold_percent:
            self._low_count = 0
            self.active = False
            result.warning_state = "OK"
        else:
            result.warning_state = "REFILL_WARNING_ACTIVE" if self.active else "OK_HYSTERESIS_BAND"
        return result.warning_state
